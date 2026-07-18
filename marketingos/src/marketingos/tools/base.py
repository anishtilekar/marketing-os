"""Foundational Tool abstraction for MarketingOS.

Every external-world integration MarketingOS calls — LLM completion, image
generation, video assembly, website/Instagram/search access — is a
:class:`Tool` subclass. This module defines that single abstraction and
nothing else: concrete providers live under ``tools/llm``, ``tools/image``,
``tools/video`` and ``tools/web``; capability-keyed lookup lives in
:mod:`marketingos.tools.registry`.

Contract
--------
A concrete tool declares, via read-only properties:

* ``name`` — a human/log-facing identifier (e.g. a model id).
* ``capability`` — the key it registers under in the
  :class:`~marketingos.tools.registry.ToolRegistry` (``"text_generation"``,
  ``"image_generation"``, ...). Agents ask for a *capability*, never a
  vendor, so swapping providers is a wiring change, not an agent-code
  change.
* ``provider`` — the vendor name, recorded on every
  :class:`~marketingos.models.cost.CostEntry`.
* ``cost_category`` — the :class:`~marketingos.models.cost.CostCategory`
  spend from this tool falls under.
* ``input_schema`` / ``output_schema`` — the Pydantic request/response
  models, exposed for introspection (contract tests, API docs) even though
  Python's own typing already enforces them at call sites.
* ``cost_guard`` — the :class:`~marketingos.services.cost_guard.CostGuard`
  enforcing this tool's budget. The base implementation returns ``None``
  deliberately: an unconfigured tool must fail closed rather than call a
  provider unpriced. Every concrete tool accepts a ``cost_guard`` at
  construction (see ``GeminiClient`` in ``tools/llm/gemini_client.py`` for
  the reference shape) and overrides this property to return it.

and implements two methods:

* ``cost_estimate(payload)`` — price a call *before* it is sent.
* ``invoke(payload)`` — perform the call.

Automatic cost enforcement
---------------------------
:meth:`Tool.__init_subclass__` wraps every subclass's own ``invoke`` with
:func:`marketingos.services.cost_guard.cost_guarded` the moment the
subclass is defined. This is what makes "every tool call is priced and
recorded" a structural guarantee instead of a convention each tool author
has to remember — there is no code path to a provider that skips the
guard, including zero-cost tools (they still record a ``Decimal("0")``
entry, keeping the ledger a complete account of every call). The wrapper
is idempotent (marked ``__cost_guarded__``) and only applied to methods a
subclass actually defines, so intermediate abstract subclasses and
subclasses that don't override ``invoke`` are left untouched — their
inherited ``invoke`` was already wrapped when its defining class was
built.

``cost_actual`` is intentionally *not* abstract: most tools cannot know
the true cost of a call until after it completes (token usage, generated
asset size, ...), so subclasses override it when they have better
information; the default simply reuses the pre-flight estimate, which is
correct for flat-rate and free tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from marketingos.models.cost import CostCategory
from marketingos.services.cost_guard import CostGuard, cost_guarded

__all__ = ["Tool"]


class Tool[InputT: BaseModel, OutputT: BaseModel](ABC):
    """Abstract base for every external-world tool MarketingOS calls.

    Type parameters:
        InputT: Pydantic model accepted by :meth:`invoke`.
        OutputT: Pydantic model produced by :meth:`invoke`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap a freshly-defined subclass's ``invoke`` with the cost guard.

        Only methods the subclass itself defines are touched: looking the
        method up in ``cls.__dict__`` (rather than with ``getattr``)
        excludes inherited implementations, so a subclass that doesn't
        override ``invoke`` is left alone — its parent's ``invoke`` was
        already wrapped when *that* class was defined, and wrapping it
        again here would double-charge every call. Callables already
        marked ``__cost_guarded__`` are skipped too, so a tool author who
        applies the decorator by hand stays safe.
        """
        super().__init_subclass__(**kwargs)
        invoke = cls.__dict__.get("invoke")
        if invoke is None or getattr(invoke, "__cost_guarded__", False):
            return
        # setattr, not `cls.invoke = ...`: Pyright checks a direct attribute
        # assignment against `invoke`'s declared `Coroutine` return type, which
        # `cost_guarded`'s `Awaitable`-typed wrapper can never satisfy.
        setattr(cls, "invoke", cost_guarded(invoke))

    # -- identity --------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human/log-facing identifier for this tool instance."""

    @property
    @abstractmethod
    def capability(self) -> str:
        """The capability key this tool registers under in the ToolRegistry."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """The vendor name, recorded on every ``CostEntry``."""

    @property
    @abstractmethod
    def cost_category(self) -> CostCategory:
        """The spend category this tool's calls fall under."""

    @property
    @abstractmethod
    def input_schema(self) -> type[InputT]:
        """The Pydantic model :meth:`invoke` accepts."""

    @property
    @abstractmethod
    def output_schema(self) -> type[OutputT]:
        """The Pydantic model :meth:`invoke` returns."""

    @property
    def cost_guard(self) -> CostGuard | None:
        """The guard enforcing this tool's budget.

        ``None`` by default. :func:`~marketingos.services.cost_guard.cost_guarded`
        treats an unset guard as a configuration bug and refuses to call
        the provider — fail closed — rather than proceeding unpriced (see
        ``CostTrackingError``). Concrete tools accept a ``CostGuard`` at
        construction and override this property to return it; this is
        true even for zero-cost tools, so every call still produces a
        ledger entry.
        """
        return None

    # -- cost --------------------------------------------------------------

    @abstractmethod
    def cost_estimate(self, payload: InputT) -> Decimal:
        """Price ``payload`` *before* :meth:`invoke` sends it anywhere.

        Called by the cost guard pre-flight; must never perform I/O. Free
        or local-compute tools return ``Decimal("0")``.
        """

    def cost_actual(self, payload: InputT, result: OutputT) -> Decimal:
        """Price a completed call, given its request and result.

        The default reuses :meth:`cost_estimate`, which is correct for
        tools whose cost doesn't vary with the outcome (flat-rate or free
        calls). Tools priced on usage the provider only reports after the
        fact (token counts, output size, ...) override this.
        """
        return self.cost_estimate(payload)

    # -- invocation ----------------------------------------------------------

    @abstractmethod
    async def invoke(self, payload: InputT) -> OutputT:
        """Perform one call, given a validated request.

        Automatically wrapped with budget enforcement by
        :meth:`__init_subclass__` — every concrete override is priced
        before it runs and recorded after it succeeds, with no way to opt
        out.

        Raises:
            InsufficientBudgetError: If the estimated cost would exceed
                the run's remaining budget. Raised by the cost-guard
                wrapper before this method's body runs.
            CostTrackingError: If this tool has no ``cost_guard`` attached.
        """