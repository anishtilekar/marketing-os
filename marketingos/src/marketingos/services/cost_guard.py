"""Budget enforcement for tool invocations.

This service is the choke point for spend. :class:`CostGuard` holds a
:class:`~marketingos.models.cost.CostLedger` — which carries the applicable
``max_budget`` — and the :func:`cost_guarded` decorator wraps a tool's
``invoke`` so that every call:

1. is **priced before it leaves the process** via ``tool.cost_estimate()``
   and refused with
   :class:`~marketingos.exceptions.budget.InsufficientBudgetError` if it
   would take the run past its ceiling, and
2. is **recorded after it succeeds** via ``tool.cost_actual()``, appended to
   the ledger as a completed
   :class:`~marketingos.models.cost.CostEntry`.

The budget is injected, never read from config here: ``CostLedger`` already
documents that "callers (e.g. services/cost_guard.py) supply the applicable
budget", so the ceiling arrives as ``CostLedger.max_budget`` and this module
stays independent of the config chain.

Fail-closed
-----------
A tool whose ``invoke`` is decorated but which has no guard attached raises
:class:`~marketingos.exceptions.budget.CostTrackingError` rather than calling
the provider. An unpriced call is treated as a bug, not as a free one.
"""

from __future__ import annotations

from decimal import Decimal
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from loguru import logger

from marketingos.agents.qa import BudgetSnapshot
from marketingos.exceptions.budget import (
    CostTrackingError,
    InsufficientBudgetError,
)
from marketingos.models.cost import CostEntry, CostLedger, CostStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from loguru import Logger

    from marketingos.tools.base import Tool

__all__ = ["CostGuard", "CostGuardBudgetLedger", "GuardedTool", "cost_guarded"]


@runtime_checkable
class GuardedTool(Protocol):
    """Structural contract for a tool that :func:`cost_guarded` can wrap.

    Satisfied by any :class:`~marketingos.tools.base.Tool` that exposes a
    :class:`CostGuard` as ``cost_guard``. Declared structurally so this
    service does not depend on concrete tool classes.
    """

    @property
    def cost_guard(self) -> CostGuard | None:
        """The guard enforcing this tool's budget, if one is attached."""
        ...


class CostGuard:
    """Prices tool calls against a ledger's remaining budget.

    A guard is scoped to one run: every entry it records carries ``run_id``.
    It is the only component that appends to the ledger, so the ledger's
    totals and the run's real spend cannot drift apart.
    """

    def __init__(self, ledger: CostLedger, *, run_id: UUID) -> None:
        """Initialise the guard.

        Args:
            ledger: The run's ledger. Its ``max_budget`` is the ceiling this
                guard enforces — supply it when constructing the ledger.
            run_id: Identifier of the run, recorded on every cost entry.
        """
        self._ledger = ledger
        self._run_id = run_id
        self._logger: Logger = logger.bind(component="CostGuard", run_id=str(run_id))

    @property
    def ledger(self) -> CostLedger:
        """The ledger this guard records to."""
        return self._ledger

    @property
    def spent(self) -> Decimal:
        """Total actual cost of completed entries recorded so far."""
        return sum(
            (
                entry.actual_cost
                for entry in self._ledger.entries
                if entry.status is CostStatus.COMPLETED
            ),
            Decimal("0"),
        )

    @property
    def remaining(self) -> Decimal:
        """Budget still available before the ceiling is reached."""
        return self._ledger.max_budget - self.spent

    def check(self, tool: Tool[Any, Any], payload: Any) -> Decimal:
        """Price a pending call and authorise it, or refuse it.

        Args:
            tool: The tool about to be invoked.
            payload: The input it would be invoked with.

        Returns:
            The estimated cost of the call.

        Raises:
            CostTrackingError: If the tool prices the call as negative.
            InsufficientBudgetError: If the estimate exceeds the remaining
                budget. Raised *before* the provider is contacted.
        """
        estimate = tool.cost_estimate(payload)
        if estimate < Decimal("0"):
            raise CostTrackingError(
                f"Tool {tool.name!r} produced a negative cost estimate: {estimate}"
            )

        remaining = self.remaining
        if estimate > remaining:
            self._logger.bind(
                event="cost_guard.blocked",
                tool=tool.name,
                estimate=str(estimate),
                remaining=str(remaining),
                max_budget=str(self._ledger.max_budget),
            ).warning("Call refused: estimated cost exceeds remaining budget")
            raise InsufficientBudgetError(
                f"Call to {tool.name!r} would cost {estimate} but only "
                f"{remaining} of the {self._ledger.max_budget} budget remains."
            )

        self._logger.bind(
            event="cost_guard.authorised",
            tool=tool.name,
            estimate=str(estimate),
            remaining=str(remaining),
        ).debug("Call authorised")
        return estimate

    def record(
        self,
        tool: Tool[Any, Any],
        payload: Any,
        result: Any,
        *,
        estimate: Decimal,
    ) -> CostEntry:
        """Record the real cost of a completed call on the ledger.

        Args:
            tool: The tool that was invoked.
            payload: The input it was invoked with.
            result: The output it produced.
            estimate: The pre-flight estimate, kept alongside the actual for
                later estimate-accuracy analysis.

        Returns:
            The recorded entry.

        Raises:
            CostTrackingError: If the tool prices the completed call as
                negative, or if appending would breach the ledger's own
                budget invariant.
        """
        actual = tool.cost_actual(payload, result)
        if actual < Decimal("0"):
            raise CostTrackingError(
                f"Tool {tool.name!r} reported a negative actual cost: {actual}"
            )

        entry = CostEntry(
            run_id=self._run_id,
            category=tool.cost_category,
            status=CostStatus.COMPLETED,
            provider=tool.provider,
            tool_name=tool.name,
            estimated_cost=estimate,
            actual_cost=actual,
        )
        try:
            # Assigned rather than appended so CostLedger re-runs its own
            # budget invariant: the ledger is the last line of defence if an
            # actual cost overshoots its estimate.
            self._ledger.entries = [*self._ledger.entries, entry]
        except ValueError as exc:
            raise CostTrackingError(
                f"Recording {actual} for {tool.name!r} breached the ledger "
                f"budget of {self._ledger.max_budget}: {exc}"
            ) from exc

        self._logger.bind(
            event="cost_guard.recorded",
            tool=tool.name,
            estimate=str(estimate),
            actual=str(actual),
            spent=str(self.spent),
            remaining=str(self.remaining),
        ).info("Cost recorded")
        return entry


class CostGuardBudgetLedger:
    """Adapts a :class:`CostGuard` to the QA agent's ``BudgetLedgerPort``.

    QA only needs to *read* a spend snapshot; it never records, adjusts, or
    refunds spend. ``currency`` and ``warning_threshold`` are not carried on
    the guard or its ledger, so they are supplied here from the run's
    :class:`~marketingos.config.settings.BudgetSettings`.
    """

    def __init__(
        self,
        guard: CostGuard,
        *,
        currency: str,
        warning_threshold: Decimal | None = None,
    ) -> None:
        """Initialise the adapter.

        Args:
            guard: The run's shared cost guard.
            currency: ISO currency code for the run's budget.
            warning_threshold: Absolute spend amount QA should flag as
                close to the ceiling, if any.
        """
        self._guard = guard
        self._currency = currency
        self._warning_threshold = warning_threshold

    async def snapshot(self) -> BudgetSnapshot:
        """Return the run's current spend position for QA to audit."""
        ledger = self._guard.ledger
        return BudgetSnapshot(
            total_spend=self._guard.spent,
            max_budget=ledger.max_budget,
            warning_threshold=self._warning_threshold,
            currency=self._currency,
            entry_count=len(ledger.entries),
        )


def cost_guarded[**P, R](
    func: Callable[..., Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """Enforce the run's budget around a tool's ``invoke``.

    Applied to a concrete tool's ``invoke`` method, so that calling
    ``tool.invoke(payload)`` — directly or through an adapter such as
    ``GeminiClient.complete()`` — cannot reach the provider without first
    being priced and authorised.

    Tools do not normally apply this by hand:
    :meth:`marketingos.tools.base.Tool.__init_subclass__` applies it to every
    subclass's ``invoke`` automatically, so an unguarded tool cannot be
    defined. Applying it explicitly remains safe — the wrapper is marked with
    ``__cost_guarded__`` and the auto-wrap skips anything already guarded.

    Args:
        func: The tool's ``async def invoke(self, payload)``.

    Returns:
        The wrapped coroutine function, marked ``__cost_guarded__``.

    Raises:
        CostTrackingError: At call time, if the tool has no guard attached.
            Unpriced calls fail closed rather than proceeding for free.
        InsufficientBudgetError: At call time, if the estimated cost exceeds
            the remaining budget. The wrapped function is never awaited.
    """

    @wraps(func)
    async def wrapper(self: GuardedTool, payload: Any, *args: Any, **kwargs: Any) -> R:
        guard = getattr(self, "cost_guard", None)
        if guard is None:
            raise CostTrackingError(
                f"Tool {getattr(self, 'name', type(self).__name__)!r} was invoked "
                "without a CostGuard; refusing to call an unpriced provider."
            )
        estimate = guard.check(self, payload)
        result = await func(self, payload, *args, **kwargs)
        guard.record(self, payload, result, estimate=estimate)
        return result

    wrapper.__cost_guarded__ = True  # type: ignore[attr-defined]
    return wrapper
