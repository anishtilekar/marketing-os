"""Human-approval gates for the four checkpoints the architecture doc
requires: after Business Analysis, after Planner, after creative drafts,
and before Packaging.

This service owns the *decision record* — who approved or rejected what,
and when — and coordinates with :class:`~marketingos.services.run_manager.RunManager`
so a pending approval is reflected in the run's status. It does not decide
*what happens* on rejection (loop back for revision vs. abort); that is the
orchestration graph's job, reading :meth:`ApprovalService.get_decision` to
route.

``ApprovalGate``/``ApprovalDecision``/``ApprovalRecord`` are defined here for
the same reason ``RunRecord`` lives in ``run_manager.py``: the corresponding
model file does not exist yet. Move them to ``models/approval.py`` unchanged
when it does.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from marketingos.exceptions.workflow import InvalidWorkflowStateError
from marketingos.services.run_manager import RunHandle, RunManager

if TYPE_CHECKING:
    from loguru import Logger

__all__ = [
    "APPROVALS_SUBDIR",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalRecord",
    "ApprovalService",
]

#: Subdirectory (relative to the run root) approval records are stored under.
APPROVALS_SUBDIR = "approvals"


class ApprovalGate(StrEnum):
    """The four human-approval checkpoints in the pipeline, in order."""

    BUSINESS_CONTEXT = "business_context"
    PLAN = "plan"
    CREATIVES = "creatives"
    PRE_PACKAGING = "pre_packaging"


class ApprovalDecision(StrEnum):
    """Outcome of a review at a gate."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRecord(BaseModel):
    """One gate's review outcome for one run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: UUID
    gate: ApprovalGate
    decision: ApprovalDecision = ApprovalDecision.PENDING
    reviewer: str | None = Field(default=None, max_length=200)
    comment: str | None = Field(default=None, max_length=2000)
    requested_at: datetime
    decided_at: datetime | None = None


class ApprovalService:
    """Requests, records, and resolves human-approval decisions per run.

    Stateless aside from its configured root and collaborator: every method
    takes the run explicitly, so one instance safely serves every run.
    """

    def __init__(self, *, runs_root: Path = Path("data/runs"), run_manager: RunManager) -> None:
        """Initialise the service.

        Args:
            runs_root: Directory under which per-run working directories
                live; must match the root the given ``run_manager`` uses.
            run_manager: Manager used to reflect pending/resolved approvals
                in the run's overall status.
        """
        self._runs_root = runs_root
        self._run_manager = run_manager
        self._logger: Logger = logger.bind(component="ApprovalService")

    # -- paths --------------------------------------------------------------

    def _record_path(self, run_id: UUID, gate: ApprovalGate) -> Path:
        return self._runs_root / str(run_id) / APPROVALS_SUBDIR / f"{gate.value}.json"

    # -- requesting -----------------------------------------------------------

    def request(self, handle: RunHandle, *, gate: ApprovalGate) -> ApprovalRecord:
        """Open a pending approval at ``gate`` and pause the run for it.

        Args:
            handle: The run reaching this checkpoint.
            gate: Which of the four checkpoints is being requested.

        Returns:
            The new, pending approval record.

        Raises:
            InvalidWorkflowStateError: If a decision already exists for this
                gate on this run — gates are requested once each.
        """
        path = self._record_path(handle.run_id, gate)
        if path.is_file():
            existing = self._load(path)
            raise InvalidWorkflowStateError(
                f"Gate {gate.value} for run {handle.run_id} was already "
                f"requested (decision: {existing.decision.value})."
            )

        record = ApprovalRecord(
            run_id=handle.run_id, gate=gate, requested_at=datetime.now(UTC)
        )
        self._save(record)
        self._run_manager.await_approval(handle)

        self._logger.bind(
            event="approval.requested", run_id=str(handle.run_id), gate=gate.value
        ).info("Approval requested")
        return record

    # -- deciding -------------------------------------------------------------

    def decide(
        self,
        handle: RunHandle,
        *,
        gate: ApprovalGate,
        decision: ApprovalDecision,
        reviewer: str | None = None,
        comment: str | None = None,
    ) -> ApprovalRecord:
        """Resolve a pending approval and resume the run.

        The run resumes regardless of outcome — rejection is a routing
        signal for the orchestration graph (loop back for revision, or
        abort), not a reason to leave the run paused.

        Args:
            handle: The run whose gate is being decided.
            gate: Which checkpoint is being resolved.
            decision: ``APPROVED`` or ``REJECTED``. ``PENDING`` is invalid.
            reviewer: Optional identifier of who made the call.
            comment: Optional free-text rationale.

        Returns:
            The resolved approval record.

        Raises:
            InvalidWorkflowStateError: If no pending approval exists for
                this gate, it was already decided, or ``decision`` is
                ``PENDING``.
        """
        if decision is ApprovalDecision.PENDING:
            raise InvalidWorkflowStateError("decision must be APPROVED or REJECTED.")

        path = self._record_path(handle.run_id, gate)
        if not path.is_file():
            raise InvalidWorkflowStateError(
                f"No approval was requested for gate {gate.value} on run "
                f"{handle.run_id}."
            )
        record = self._load(path)
        if record.decision is not ApprovalDecision.PENDING:
            raise InvalidWorkflowStateError(
                f"Gate {gate.value} for run {handle.run_id} was already "
                f"decided: {record.decision.value}."
            )

        record = record.model_copy(
            update={
                "decision": decision,
                "reviewer": reviewer,
                "comment": comment,
                "decided_at": datetime.now(UTC),
            }
        )
        self._save(record)
        self._run_manager.resume_run(handle)

        self._logger.bind(
            event="approval.decided",
            run_id=str(handle.run_id),
            gate=gate.value,
            decision=decision.value,
            reviewer=reviewer,
        ).info("Approval decided")
        return record

    # -- reading --------------------------------------------------------------

    def get_decision(self, run_id: UUID, gate: ApprovalGate) -> ApprovalRecord | None:
        """Return the approval record for ``gate``, or ``None`` if unrequested."""
        path = self._record_path(run_id, gate)
        return self._load(path) if path.is_file() else None

    def history(self, run_id: UUID) -> list[ApprovalRecord]:
        """Return every approval record for a run, in gate-declaration order."""
        records = []
        for gate in ApprovalGate:
            record = self.get_decision(run_id, gate)
            if record is not None:
                records.append(record)
        return records

    def is_approved(self, run_id: UUID, gate: ApprovalGate) -> bool:
        """Whether ``gate`` was resolved as ``APPROVED``."""
        record = self.get_decision(run_id, gate)
        return record is not None and record.decision is ApprovalDecision.APPROVED

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> ApprovalRecord:
        return ApprovalRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, record: ApprovalRecord) -> None:
        path = self._record_path(record.run_id, record.gate)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.loads(record.model_dump_json())
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
