"""
approval_gates.py
==================

Approval Gate Management for MarketingOS
-----------------------------------------

This module provides a self-contained, dependency-free subsystem for managing
Human-in-the-Loop (HITL) approval checkpoints within the MarketingOS workflow
system. MarketingOS is built on LangGraph, and this module is designed to
integrate cleanly with LangGraph's ``interrupt`` / resume model without
importing or depending on LangGraph itself.

Scope
-----
This module is responsible ONLY for approval gate *state management*:

- Tracking gate state over the approval stages :class:`ApprovalStage`
  (defined in :mod:`.state`, the single source of truth for what stages
  exist, shared with :mod:`.edges`).
- Defining what decisions can be made about a stage (:class:`ApprovalDecision`).
- Modeling a single approval checkpoint's state (:class:`ApprovalGate`).
- Coordinating a collection of approval gates for a workflow run
  (:class:`ApprovalManager`).

This module explicitly does NOT contain:

- LangGraph graph construction, nodes, or ``interrupt()`` calls.
- Workflow execution or orchestration logic.
- API routes, UI logic, or database/persistence code.
- Business validation or agent logic.

Orchestration code (elsewhere in MarketingOS) is expected to:

1. Attach an :class:`ApprovalManager` instance to a shared workflow state
   object (e.g. a future ``MarketingState`` TypedDict/BaseModel) under some
   field name, such as ``state.approvals``.
2. Call :meth:`ApprovalManager.requires_approval` before executing a node
   that gates on human approval, and raise ``langgraph.types.interrupt(...)``
   in the calling code if approval is still pending.
3. On resume (after a human approves/rejects out-of-band), call
   :meth:`ApprovalManager.approve` or :meth:`ApprovalManager.reject` to
   update gate state before the graph continues.

Because LangGraph checkpoints the entire state object between interrupts,
every model in this module is designed to be:

- Fully (de)serializable via Pydantic (``model_dump`` / ``model_validate``)
  and via plain-dict helpers (:meth:`ApprovalManager.to_dict` /
  :meth:`ApprovalManager.from_dict`) for checkpoint-friendly storage.
- Safe to round-trip repeatedly across many interrupt/resume cycles
  (supports resets and revisions for repeated approval after changes).

Thread Safety
-------------
:class:`ApprovalManager` guards its internal mutable state with a
``threading.RLock`` so that concurrent orchestration code (e.g. multiple
LangGraph node executions inspecting approval status concurrently) does not
corrupt internal state. This is a practical safety net, not a substitute for
proper workflow-level concurrency control.

Extensibility
-------------
New approval stages can be added by adding a new member to
:class:`~marketingos.orchestration.state.ApprovalStage` in ``state.py`` --
no other code in this module needs to change. New decision types can
similarly be added to :class:`ApprovalDecision`. The :class:`ApprovalGate`
model accepts free-form ``metadata`` and ``tags`` for forward-compatible,
schema-light extension without requiring migrations.
"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .edges import NodeName
from .state import ApprovalStage, MarketingState

__all__ = [
    "ApprovalStage",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalManager",
    "ApprovalGateError",
    "UnknownApprovalStageError",
    "InvalidApprovalTransitionError",
    "DuplicateApprovalError",
    "ApprovalAlreadyFinalizedError",
    "build_approval_gate_nodes",
    "list_approval_gate_names",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware ``datetime``.

    Centralizing "now" generation makes the module easier to test (a single
    seam to monkeypatch) and guarantees all timestamps are timezone-aware,
    avoiding naive/aware ``datetime`` comparison bugs.
    """
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ApprovalGateError(Exception):
    """Base exception for all approval-gate-related errors.

    Catching this exception type is sufficient to handle any error raised
    by this module's public API.
    """


class UnknownApprovalStageError(ApprovalGateError):
    """Raised when an operation references an :class:`ApprovalStage` that
    the :class:`ApprovalManager` has no gate registered for.

    This should not normally occur since ``ApprovalManager`` seeds a gate
    for every known ``ApprovalStage`` member at construction time, but is
    retained as a defensive guard for custom/partial managers.
    """

    def __init__(self, stage: Any) -> None:
        self.stage = stage
        super().__init__(f"No approval gate is registered for stage: {stage!r}")


class InvalidApprovalTransitionError(ApprovalGateError):
    """Raised when a requested decision transition is not permitted from the
    gate's current decision state.

    For example, this is raised if an internal transition rule is violated
    in a way that is not covered by the more specific exceptions below.
    """

    def __init__(self, stage: Any, current: Any, attempted: Any) -> None:
        self.stage = stage
        self.current = current
        self.attempted = attempted
        super().__init__(
            f"Cannot transition stage {stage!r} from decision {current!r} "
            f"to {attempted!r}."
        )


class DuplicateApprovalError(ApprovalGateError):
    """Raised when attempting to approve a stage that has already been
    approved, without an intervening :meth:`ApprovalManager.reset`.

    This guards against accidental double-submission (e.g. a UI double
    click or a retried request) silently incrementing revision counters.
    """

    def __init__(self, stage: Any) -> None:
        self.stage = stage
        super().__init__(
            f"Stage {stage!r} has already been approved. Call reset() first "
            "if a re-approval cycle is intended."
        )


class ApprovalAlreadyFinalizedError(ApprovalGateError):
    """Raised when attempting to approve or reject a stage whose decision is
    already finalized (approved or rejected) and a reset was not performed.

    "Finalized" is intentionally distinct from "duplicate approval" so
    callers can distinguish "you already said yes" from "you already said
    no, and now you're trying to say yes" -- both are errors, but callers
    may wish to handle them differently.
    """

    def __init__(self, stage: Any, current: Any) -> None:
        self.stage = stage
        self.current = current
        super().__init__(
            f"Stage {stage!r} is already finalized with decision {current!r}. "
            "Call reset() to start a new approval cycle."
        )


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------
#
# ``ApprovalStage`` itself is defined in ``state.py`` (imported above) rather
# than here, so this module and ``edges.py`` -- which already imports it from
# the same place -- always agree on what approval checkpoints exist. The
# helpers below used to be methods on a module-local ``ApprovalStage``; they
# stay here as plain functions since a ``StrEnum`` we don't own can't have
# methods added to it from outside its defining module.


def _ordered_stages() -> tuple[ApprovalStage, ...]:
    """Return all stages in their canonical pipeline order.

    This reflects declaration order on :class:`~marketingos.orchestration.
    state.ApprovalStage`, which corresponds to the natural progression of
    the MarketingOS content pipeline.
    """
    return tuple(ApprovalStage)


def _coerce_stage(value: "str | ApprovalStage") -> ApprovalStage:
    """Coerce a raw string or existing member into an ``ApprovalStage``.

    Parameters
    ----------
    value:
        Either an ``ApprovalStage`` member (returned as-is) or a raw string
        matching one of the enum's values.

    Raises
    ------
    UnknownApprovalStageError
        If ``value`` does not correspond to any known stage.
    """
    if isinstance(value, ApprovalStage):
        return value
    try:
        return ApprovalStage(value)
    except ValueError as exc:
        raise UnknownApprovalStageError(value) from exc


def is_terminal_stage(stage: ApprovalStage) -> bool:
    """Return whether ``stage`` is the final gate in the pipeline (i.e. no
    further approval stages follow it)."""
    ordered = _ordered_stages()
    return ordered[-1] == stage


def next_stage(stage: ApprovalStage) -> Optional[ApprovalStage]:
    """Return the stage that canonically follows ``stage``, or ``None`` if
    ``stage`` is the last one in the pipeline."""
    ordered = _ordered_stages()
    idx = ordered.index(stage)
    if idx + 1 < len(ordered):
        return ordered[idx + 1]
    return None


class ApprovalDecision(str, Enum):
    """Enumerates the possible outcomes of an approval gate.

    Like :class:`ApprovalStage`, this subclasses ``str`` for clean
    serialization. Future statuses (e.g. ``NEEDS_REVISION`` or
    ``ESCALATED``) can be added as new members; update
    :meth:`ApprovalDecision.is_final` if a new status should also be
    treated as a terminal state.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

    def is_final(self) -> bool:
        """Return whether this decision represents a finalized outcome
        (i.e. not awaiting further human input)."""
        return self in (ApprovalDecision.APPROVED, ApprovalDecision.REJECTED)


# ---------------------------------------------------------------------------
# ApprovalGate model
# ---------------------------------------------------------------------------

class ApprovalGate(BaseModel):
    """Represents the state of a single approval checkpoint.

    An ``ApprovalGate`` is a pure data model: it holds state but does not
    itself enforce transition rules. Transition rules (e.g. "cannot approve
    an already-approved gate") are enforced by :class:`ApprovalManager`,
    keeping this model simple, immutable-friendly, and easy to serialize for
    LangGraph checkpointing.

    Attributes
    ----------
    stage:
        Which approval checkpoint this gate represents.
    decision:
        The current decision outcome. Defaults to
        :attr:`ApprovalDecision.PENDING`.
    approver:
        Identifier (name, email, user ID, etc.) of the human who most
        recently approved this gate. ``None`` until an approval occurs.
    reviewer:
        Identifier of the human currently assigned to review this gate,
        which may be set before a decision is made (e.g. for routing an
        approval request to a specific person). Distinct from ``approver``,
        which records who actually made the decision.
    comments:
        Free-text comments supplied by the approver/reviewer, e.g.
        rejection rationale or approval notes.
    created_at:
        Timestamp (UTC) when this gate was first created.
    updated_at:
        Timestamp (UTC) of the most recent modification to this gate,
        of any kind (decision change, metadata update, reset, etc.).
    approved_at:
        Timestamp (UTC) of the most recent approval, if any.
    rejected_at:
        Timestamp (UTC) of the most recent rejection, if any.
    reset_count:
        Number of times this gate has been reset back to
        :attr:`ApprovalDecision.PENDING` after a prior decision. Useful for
        auditability (how many revision cycles a stage went through).
    revision_number:
        Monotonically increasing counter representing which "version" of
        the content this gate's current decision applies to. Incremented
        automatically whenever :meth:`ApprovalManager.reset` starts a new
        approval cycle, so approvals/rejections can be tied to a specific
        revision of the underlying content.
    metadata:
        Free-form dictionary for forward-compatible extension (e.g.
        links to the content being approved, model-generated confidence
        scores, source agent identifiers). Not validated or interpreted
        by this module.
    tags:
        Free-form list of string tags for categorization/filtering (e.g.
        ``["urgent", "client-facing"]``).

    Notes
    -----
    Instances are created and mutated exclusively through
    :class:`ApprovalManager`. Direct field mutation is possible (Pydantic
    models are mutable by default) but is discouraged outside of the
    manager, since the manager is responsible for keeping timestamps,
    counters, and validation in sync.
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    stage: ApprovalStage = Field(..., description="Which approval checkpoint this gate represents.")
    decision: ApprovalDecision = Field(
        default=ApprovalDecision.PENDING,
        description="Current decision outcome for this gate.",
    )
    approver: Optional[str] = Field(
        default=None, description="Identifier of the human who most recently approved this gate."
    )
    reviewer: Optional[str] = Field(
        default=None,
        description="Identifier of the human currently assigned to review this gate.",
    )
    comments: Optional[str] = Field(
        default=None, description="Free-text comments supplied alongside the decision."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="UTC timestamp when this gate was first created."
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="UTC timestamp of the most recent modification."
    )
    approved_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp of the most recent approval, if any."
    )
    rejected_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp of the most recent rejection, if any."
    )
    reset_count: int = Field(
        default=0, ge=0, description="Number of times this gate has been reset to PENDING."
    )
    revision_number: int = Field(
        default=1, ge=1, description="Which content revision this gate's decision applies to."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form metadata for forward-compatible extension."
    )
    tags: list[str] = Field(
        default_factory=list, description="Free-form tags for categorization/filtering."
    )

    @field_validator("comments")
    @classmethod
    def _blank_comments_to_none(cls, value: Optional[str]) -> Optional[str]:
        """Normalize whitespace-only comment strings to ``None`` so empty
        strings don't masquerade as meaningful content."""
        if value is not None and value.strip() == "":
            return None
        return value

    @property
    def is_pending(self) -> bool:
        """Return whether this gate is awaiting a decision."""
        return self.decision == ApprovalDecision.PENDING

    @property
    def is_approved(self) -> bool:
        """Return whether this gate's current decision is approved."""
        return self.decision == ApprovalDecision.APPROVED

    @property
    def is_rejected(self) -> bool:
        """Return whether this gate's current decision is rejected."""
        return self.decision == ApprovalDecision.REJECTED

    @property
    def is_finalized(self) -> bool:
        """Return whether this gate's decision is in a terminal state."""
        return self.decision.is_final()


# ---------------------------------------------------------------------------
# ApprovalManager
# ---------------------------------------------------------------------------

@dataclass
class ApprovalManager:
    """Manages the full set of :class:`ApprovalGate` objects for a single
    MarketingOS workflow run.

    This class is the primary public API that orchestration code (and,
    indirectly, LangGraph nodes) should interact with. It is intentionally
    implemented as a plain ``dataclass`` rather than a Pydantic model:
    Pydantic v2 models are less ergonomic for classes that need a
    non-serialized internal lock and custom mutation methods, whereas the
    *contents* it manages (``ApprovalGate`` instances) are Pydantic models
    and remain fully serializable via :meth:`to_dict`.

    Design Notes
    ------------
    - **No hard dependency on MarketingState.** This class does not import
      or reference any workflow state type. Orchestration code is expected
      to attach an instance of this manager to a shared state object, e.g.::

          class MarketingState(BaseModel):
              approvals: ApprovalManager = Field(default_factory=ApprovalManager)
              ...

      Because ``ApprovalManager`` supports :meth:`to_dict`/:meth:`from_dict`,
      it can also be stored as a plain dict field on a state object if the
      state schema prefers pure-dict serialization over embedding this class
      directly (both patterns are supported).

    - **LangGraph interrupt/resume friendliness.** A typical usage pattern
      inside a LangGraph node looks like::

          if manager.requires_approval(ApprovalStage.PLANNING_REVIEW):
              payload = interrupt({"stage": "planning_review", "reason": "awaiting human review"})
              # On resume, the graph re-enters here with the human's decision:
              if payload["decision"] == "approved":
                  manager.approve(ApprovalStage.PLANNING_REVIEW, approver=payload["user"])
              else:
                  manager.reject(ApprovalStage.PLANNING_REVIEW, approver=payload["user"],
                                  comments=payload.get("comments"))

      This module does not call ``interrupt()`` itself -- that is the
      calling node's responsibility -- but its state model is shaped
      specifically to make that calling code simple.

    - **Repeated approval cycles.** :meth:`reset` supports the case where
      content is revised after rejection (or after approval, if a stage
      needs to be redone) and must go through the approval gate again. Each
      reset increments :attr:`ApprovalGate.reset_count` and
      :attr:`ApprovalGate.revision_number`.

    - **Thread safety.** All mutating and reading methods acquire an
      internal ``RLock``. This makes the manager safe to use from
      concurrent contexts (e.g. multiple threads inspecting status), though
      LangGraph workflows are typically single-threaded per run.

    Parameters
    ----------
    stages:
        Optional iterable of :class:`ApprovalStage` members to manage. If
        omitted, defaults to every member of :class:`ApprovalStage`
        (see :func:`_ordered_stages`). Supplying a subset is supported for
        specialized workflows that don't require every stage.
    """

    stages: Optional[Iterable[ApprovalStage]] = None
    _gates: dict[ApprovalStage, ApprovalGate] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        selected_stages = tuple(self.stages) if self.stages is not None else _ordered_stages()
        for stage in selected_stages:
            self._gates[stage] = ApprovalGate(stage=stage)

    # -- internal helpers ----------------------------------------------

    def _get_gate(self, stage: "ApprovalStage | str") -> ApprovalGate:
        """Resolve and return the internal gate for ``stage``.

        Raises
        ------
        UnknownApprovalStageError
            If no gate is registered for the resolved stage.
        """
        resolved = _coerce_stage(stage)
        gate = self._gates.get(resolved)
        if gate is None:
            raise UnknownApprovalStageError(resolved)
        return gate

    def _touch(self, gate: ApprovalGate) -> None:
        """Update ``gate.updated_at`` to the current time in-place."""
        gate.updated_at = _utcnow()

    # -- core public API --------------------------------------------------

    def requires_approval(self, stage: "ApprovalStage | str") -> bool:
        """Return ``True`` if ``stage`` is still awaiting a decision.

        This is the primary check orchestration code should perform before
        allowing a gated node to execute -- if ``True``, the calling code
        should raise a LangGraph ``interrupt()`` rather than proceeding.
        """
        with self._lock:
            return self._get_gate(stage).is_pending

    def approve(
        self,
        stage: "ApprovalStage | str",
        approver: str,
        comments: Optional[str] = None,
    ) -> ApprovalGate:
        """Record an approval decision for ``stage``.

        Parameters
        ----------
        stage:
            The stage being approved.
        approver:
            Identifier of the human granting approval. Required -- an
            approval without an identified approver is not permitted, for
            auditability.
        comments:
            Optional free-text comments to attach to the approval.

        Returns
        -------
        ApprovalGate
            The updated gate, for convenience.

        Raises
        ------
        UnknownApprovalStageError
            If ``stage`` is not recognized.
        DuplicateApprovalError
            If the gate is already approved.
        ApprovalAlreadyFinalizedError
            If the gate is already rejected (a reset is required first).
        ValueError
            If ``approver`` is empty/blank.
        """
        if not approver or not approver.strip():
            raise ValueError("approver must be a non-empty identifier.")

        with self._lock:
            gate = self._get_gate(stage)
            if gate.decision == ApprovalDecision.APPROVED:
                raise DuplicateApprovalError(gate.stage)
            if gate.decision == ApprovalDecision.REJECTED:
                raise ApprovalAlreadyFinalizedError(gate.stage, gate.decision)

            now = _utcnow()
            gate.decision = ApprovalDecision.APPROVED
            gate.approver = approver
            gate.comments = comments
            gate.approved_at = now
            gate.updated_at = now
            return gate

    def reject(
        self,
        stage: "ApprovalStage | str",
        approver: str,
        comments: Optional[str] = None,
    ) -> ApprovalGate:
        """Record a rejection decision for ``stage``.

        Parameters mirror :meth:`approve`; ``approver`` here identifies the
        human issuing the rejection (naming remains ``approver`` for API
        consistency/symmetry with ``approve``, representing "the human who
        acted on this gate").

        Raises
        ------
        UnknownApprovalStageError
            If ``stage`` is not recognized.
        ApprovalAlreadyFinalizedError
            If the gate is already approved or already rejected (a reset
            is required first).
        ValueError
            If ``approver`` is empty/blank.
        """
        if not approver or not approver.strip():
            raise ValueError("approver must be a non-empty identifier.")

        with self._lock:
            gate = self._get_gate(stage)
            if gate.decision.is_final():
                raise ApprovalAlreadyFinalizedError(gate.stage, gate.decision)

            now = _utcnow()
            gate.decision = ApprovalDecision.REJECTED
            gate.approver = approver
            gate.comments = comments
            gate.rejected_at = now
            gate.updated_at = now
            return gate

    def reset(self, stage: "ApprovalStage | str") -> ApprovalGate:
        """Reset ``stage`` back to :attr:`ApprovalDecision.PENDING`, starting
        a new approval cycle.

        This is used when content gated by a stage is revised after a
        rejection (or needs to be re-approved after an approved stage's
        upstream content changes). Increments both ``reset_count`` and
        ``revision_number``; clears ``approver``, ``comments``,
        ``approved_at``, and ``rejected_at`` since those pertain to the
        prior cycle.

        Returns
        -------
        ApprovalGate
            The reset gate.

        Raises
        ------
        UnknownApprovalStageError
            If ``stage`` is not recognized.
        """
        with self._lock:
            gate = self._get_gate(stage)
            gate.decision = ApprovalDecision.PENDING
            gate.approver = None
            gate.comments = None
            gate.approved_at = None
            gate.rejected_at = None
            gate.reset_count += 1
            gate.revision_number += 1
            gate.updated_at = _utcnow()
            return gate

    def is_complete(self) -> bool:
        """Return ``True`` if every managed gate has reached a finalized
        (approved or rejected) decision.

        Note this does not mean every gate was *approved* -- use
        :meth:`list_rejected` to check for any rejections if "complete
        success" (all approved) is what's actually required.
        """
        with self._lock:
            return all(gate.decision.is_final() for gate in self._gates.values())

    def all_approved(self) -> bool:
        """Return ``True`` if every managed gate is specifically approved.

        Convenience method distinguishing "all decisions made" (see
        :meth:`is_complete`) from "all decisions were approvals".
        """
        with self._lock:
            return all(gate.is_approved for gate in self._gates.values())

    def get(self, stage: "ApprovalStage | str") -> ApprovalGate:
        """Return a defensive copy of the gate for ``stage``.

        A copy is returned (rather than the internal instance) so callers
        cannot mutate manager-internal state except through the manager's
        own methods, preserving invariants like timestamp/counter
        consistency.
        """
        with self._lock:
            return self._get_gate(stage).model_copy(deep=True)

    def list_pending(self) -> list[ApprovalGate]:
        """Return copies of all gates currently awaiting a decision, in
        pipeline order."""
        with self._lock:
            return [g.model_copy(deep=True) for g in self._gates.values() if g.is_pending]

    def list_completed(self) -> list[ApprovalGate]:
        """Return copies of all gates that have reached any finalized
        decision (approved or rejected), in pipeline order."""
        with self._lock:
            return [g.model_copy(deep=True) for g in self._gates.values() if g.is_finalized]

    def list_rejected(self) -> list[ApprovalGate]:
        """Return copies of all gates currently in a rejected state, in
        pipeline order."""
        with self._lock:
            return [g.model_copy(deep=True) for g in self._gates.values() if g.is_rejected]

    def list_approved(self) -> list[ApprovalGate]:
        """Return copies of all gates currently in an approved state, in
        pipeline order."""
        with self._lock:
            return [g.model_copy(deep=True) for g in self._gates.values() if g.is_approved]

    def get_decision(self, stage: "ApprovalStage | str") -> ApprovalDecision:
        """Return the current :class:`ApprovalDecision` for ``stage``."""
        with self._lock:
            return self._get_gate(stage).decision

    def get_comments(self, stage: "ApprovalStage | str") -> Optional[str]:
        """Return the current comments string for ``stage``, if any."""
        with self._lock:
            return self._get_gate(stage).comments

    def get_metadata(self, stage: "ApprovalStage | str") -> dict[str, Any]:
        """Return a deep copy of the metadata dict for ``stage``.

        A copy is returned to prevent callers from mutating manager-internal
        state without going through :meth:`update_metadata`.
        """
        with self._lock:
            return deepcopy(self._get_gate(stage).metadata)

    def update_metadata(
        self,
        stage: "ApprovalStage | str",
        updates: dict[str, Any],
        *,
        replace: bool = False,
    ) -> ApprovalGate:
        """Update the metadata dict for ``stage``.

        Parameters
        ----------
        stage:
            The stage whose metadata should be updated.
        updates:
            Key/value pairs to merge into (or replace) the gate's metadata.
        replace:
            If ``True``, ``updates`` entirely replaces the existing
            metadata dict. If ``False`` (default), ``updates`` is shallow
            merged into the existing metadata, with keys in ``updates``
            taking precedence.

        Returns
        -------
        ApprovalGate
            The updated gate.
        """
        with self._lock:
            gate = self._get_gate(stage)
            if replace:
                gate.metadata = dict(updates)
            else:
                merged = dict(gate.metadata)
                merged.update(updates)
                gate.metadata = merged
            self._touch(gate)
            return gate

    def set_reviewer(self, stage: "ApprovalStage | str", reviewer: Optional[str]) -> ApprovalGate:
        """Assign (or clear, if ``None``) the reviewer for ``stage`` without
        affecting its decision state.

        Useful for routing an approval request to a specific human before
        a decision has been made.
        """
        with self._lock:
            gate = self._get_gate(stage)
            gate.reviewer = reviewer
            self._touch(gate)
            return gate

    def add_tags(self, stage: "ApprovalStage | str", tags: Iterable[str]) -> ApprovalGate:
        """Add one or more tags to ``stage``'s gate, avoiding duplicates
        while preserving existing tag order."""
        with self._lock:
            gate = self._get_gate(stage)
            existing = list(gate.tags)
            for tag in tags:
                if tag not in existing:
                    existing.append(tag)
            gate.tags = existing
            self._touch(gate)
            return gate

    def clear(self) -> None:
        """Reset every managed gate back to a fresh, pending state.

        Unlike :meth:`reset`, which increments counters to preserve audit
        history for a single stage, ``clear()`` re-initializes the entire
        manager's gates from scratch (as if newly constructed). This is
        useful for starting an entirely new workflow run while reusing an
        existing ``ApprovalManager`` instance.
        """
        with self._lock:
            stages = list(self._gates.keys())
            self._gates = {stage: ApprovalGate(stage=stage) for stage in stages}

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manager's full state to a plain, JSON-compatible
        dictionary.

        The returned structure is::

            {
                "gates": {
                    "<stage-value>": { ...ApprovalGate fields... },
                    ...
                }
            }

        This shape is suitable for LangGraph checkpoint storage (any
        serializer capable of handling nested dicts/lists/primitives) and
        for reconstruction via :meth:`from_dict`.
        """
        with self._lock:
            return {
                "gates": {
                    stage.value: gate.model_dump(mode="json")
                    for stage, gate in self._gates.items()
                }
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalManager":
        """Reconstruct an :class:`ApprovalManager` from a dictionary
        previously produced by :meth:`to_dict`.

        Parameters
        ----------
        data:
            A dictionary matching the shape produced by :meth:`to_dict`,
            i.e. containing a top-level ``"gates"`` key mapping stage
            string values to serialized ``ApprovalGate`` field dicts.

        Returns
        -------
        ApprovalManager
            A new manager instance populated with the deserialized gates.

        Raises
        ------
        UnknownApprovalStageError
            If ``data`` references a stage value not recognized by
            :class:`ApprovalStage`.
        """
        gates_data = data.get("gates", {})
        stages: list[ApprovalStage] = []
        parsed_gates: dict[ApprovalStage, ApprovalGate] = {}

        for stage_value, gate_dict in gates_data.items():
            stage = _coerce_stage(stage_value)
            stages.append(stage)
            parsed_gates[stage] = ApprovalGate.model_validate(gate_dict)

        manager = cls(stages=stages or None)
        with manager._lock:
            manager._gates.update(parsed_gates)
        return manager

    # -- dunder helpers ------------------------------------------------

    def __contains__(self, stage: object) -> bool:
        """Support ``stage in manager`` membership checks."""
        try:
            resolved = _coerce_stage(stage)  # type: ignore[arg-type]
        except UnknownApprovalStageError:
            return False
        return resolved in self._gates

    def __len__(self) -> int:
        """Return the number of gates managed by this instance."""
        return len(self._gates)

    def __iter__(self):
        """Iterate over managed gates in pipeline order (copies, not
        internal references)."""
        with self._lock:
            return iter([g.model_copy(deep=True) for g in self._gates.values()])


# ---------------------------------------------------------------------------
# LangGraph node factories
# ---------------------------------------------------------------------------
#
# ``ApprovalManager`` above models approval-gate state in the abstract, with
# no dependency on ``MarketingState`` or LangGraph. ``graph.py`` needs
# concrete node callables it can register on the ``StateGraph`` and pause
# execution before (via ``interrupt_before``); the functions below adapt this
# module's stage/gate concepts to that narrower contract.
#
# Gate node names are sourced from ``edges.NodeName`` (rather than
# ``ApprovalStage``) so the node names used here, the ``interrupt_before``
# targets in ``graph.py``, and the routing targets in ``edges.py`` all stay
# in lockstep.

_GATE_NODE_NAMES: tuple[str, ...] = (
    NodeName.BUSINESS_REVIEW_GATE,
    NodeName.PLANNING_REVIEW_GATE,
    NodeName.CREATIVE_REVIEW_GATE,
    NodeName.FINAL_APPROVAL_GATE,
)


def list_approval_gate_names() -> list[str]:
    """Return the names of every node execution should pause before for
    human approval.

    Returns
    -------
    list[str]
        Node names, in pipeline order, matching ``graph.py``'s
        ``interrupt_before`` targets and ``edges.py``'s gate routing.
    """
    return list(_GATE_NODE_NAMES)


def build_approval_gate_nodes() -> dict[str, Callable[[MarketingState], MarketingState]]:
    """Build the LangGraph node callables for every human approval gate.

    Each gate node is a no-op passthrough: the actual pause happens because
    ``graph.py`` registers these node names in ``interrupt_before``, so
    LangGraph suspends execution *before* the node runs, giving an external
    actor a chance to call :meth:`MarketingState.approve_stage` before the
    graph resumes. The node itself only needs to exist so LangGraph has
    something to interrupt in front of.

    Returns
    -------
    dict[str, Callable[[MarketingState], MarketingState]]
        Mapping of gate node name to node callable, suitable for
        ``GraphBuilder.with_approval_gates``.
    """

    def _passthrough(state: MarketingState) -> MarketingState:
        return state

    return {name: _passthrough for name in _GATE_NODE_NAMES}