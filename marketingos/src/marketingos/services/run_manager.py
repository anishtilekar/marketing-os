"""Run lifecycle management: identity, directory structure, and the
cost ledger/guard pair every agent in a run shares.

:class:`RunManager` is the single place a run comes into existence. It
generates the ``run_id``, lays out the numbered working directory from the
architecture doc's output-organization section, and wires a fresh
:class:`~marketingos.services.cost_ledger.CostLedgerService`-backed ledger
to a :class:`~marketingos.services.cost_guard.CostGuard` so every agent in
the run enforces the same budget instance. It also persists a small run
record (status, timestamps, checkpoints) so a run's progress survives a
process restart.

``RunRecord``, ``RunSection``, and ``RunStatus`` live in ``models/run.py``;
this module imports them rather than defining them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

from loguru import logger

from marketingos.exceptions.workflow import InvalidWorkflowStateError, WorkflowExecutionError
from marketingos.models.run import RunRecord, RunSection, RunStatus
from marketingos.services.cost_guard import CostGuard
from marketingos.services.cost_ledger import CostLedgerService

if TYPE_CHECKING:
    from loguru import Logger

__all__ = [
    "RUN_RECORD_FILENAME",
    "RunHandle",
    "RunManager",
]

#: File name of the persisted run record, at the run's root directory.
RUN_RECORD_FILENAME = "run.json"

#: Numbered working directories created for every run, per the architecture
#: doc's output-organization section. Order matches pipeline order.
_RUN_SUBDIRS: Final[tuple[str, ...]] = (
    "00_source_pack",
    "01_business_context",
    "02_strategy",
    "03_plan",
    "04_creatives/posts",
    "04_creatives/videos",
    "05_qa",
    "06_cost",
    "07_logs",
    "package",
)

#: Ledger file location relative to the run root, inside the cost folder.
_LEDGER_RELATIVE_PATH: Final[str] = "06_cost/cost_ledger.json"


class RunHandle:
    """A live run's identity plus its shared budget-enforcement pair.

    Returned by :meth:`RunManager.start_run` / :meth:`RunManager.load_run`
    and passed to every agent so they share one :class:`CostGuard` instance
    for the run.
    """

    __slots__ = ("guard", "record", "run_id")

    def __init__(self, *, run_id: UUID, guard: CostGuard, record: RunRecord) -> None:
        self.run_id = run_id
        self.guard = guard
        self.record = record


class RunManager:
    """Creates, resumes, checkpoints, and closes out MarketingOS runs.

    Stateless aside from its configured roots: every method takes the run
    (as a :class:`RunHandle` or ``run_id``) explicitly, so one instance
    safely serves every run in the process.
    """

    def __init__(
        self,
        *,
        runs_root: Path = Path("data/runs"),
        ledger_service: CostLedgerService | None = None,
    ) -> None:
        """Initialise the manager.

        Args:
            runs_root: Directory under which every run's working directory
                (``{runs_root}/{run_id}/``) is created.
            ledger_service: Ledger persistence service to use. Defaults to
                one rooted at ``runs_root`` storing the ledger at
                ``06_cost/cost_ledger.json`` inside the run directory.
        """
        self._runs_root = runs_root
        self._ledger_service = ledger_service or CostLedgerService(
            runs_root=runs_root, ledger_filename=_LEDGER_RELATIVE_PATH
        )
        self._logger: Logger = logger.bind(component="RunManager")

    # -- paths --------------------------------------------------------------

    def run_dir(self, run_id: UUID) -> Path:
        """Root working directory for a run."""
        return self._runs_root / str(run_id)

    def section_dir(self, run_id: UUID, section: RunSection) -> Path:
        """Path to one of a run's numbered pipeline-stage directories."""
        return self.run_dir(run_id) / section.value

    def _record_path(self, run_id: UUID) -> Path:
        return self.run_dir(run_id) / RUN_RECORD_FILENAME

    # -- lifecycle ------------------------------------------------------------

    def start_run(self, *, max_budget: Decimal) -> RunHandle:
        """Create a new run: directory tree, ledger, guard, and record.

        Args:
            max_budget: Spend ceiling for this run, typically
                ``BudgetSettings.max_budget``.

        Returns:
            A handle carrying the new run's id and shared ``CostGuard``.
        """
        run_id = uuid4()
        run_dir = self.run_dir(run_id)
        for subdir in _RUN_SUBDIRS:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        ledger = self._ledger_service.create(run_id=run_id, max_budget=max_budget)
        guard = CostGuard(ledger, run_id=run_id)

        now = datetime.now(UTC)
        record = RunRecord(
            run_id=run_id,
            max_budget=max_budget,
            started_at=now,
            updated_at=now,
        )
        self._save_record(record)

        self._logger.bind(event="run.started", run_id=str(run_id)).info(
            "Run started"
        )
        return RunHandle(run_id=run_id, guard=guard, record=record)

    def load_run(self, run_id: UUID) -> RunHandle:
        """Reconstruct a handle for an existing run, e.g. after a restart.

        Args:
            run_id: Identifier of the run to resume.

        Returns:
            A handle with the run's persisted record and a guard rebuilt
            from its persisted ledger.

        Raises:
            WorkflowExecutionError: If no run record exists for ``run_id``.
        """
        path = self._record_path(run_id)
        if not path.is_file():
            raise WorkflowExecutionError(f"No run found for {run_id} at {path}.")
        record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        ledger = self._ledger_service.load(run_id)
        guard = CostGuard(ledger, run_id=run_id)
        return RunHandle(run_id=run_id, guard=guard, record=record)

    def checkpoint(self, handle: RunHandle, *, node_name: str) -> RunRecord:
        """Persist the ledger's current state and record a checkpoint.

        Call after each agent node when ``WorkflowSettings.checkpoint_after_each_agent``
        is enabled, so a resumed run knows exactly how far it got.

        Args:
            handle: The run to checkpoint.
            node_name: Name of the agent/graph node just completed.

        Returns:
            The updated run record.

        Raises:
            InvalidWorkflowStateError: If the run is already finished.
        """
        self._require_running(handle.record)
        self._ledger_service.save(handle.guard.ledger, run_id=handle.run_id)
        handle.record = handle.record.model_copy(
            update={
                "checkpoints": [*handle.record.checkpoints, node_name],
                "updated_at": datetime.now(UTC),
            }
        )
        self._save_record(handle.record)
        self._logger.bind(
            event="run.checkpoint", run_id=str(handle.run_id), node=node_name
        ).debug("Run checkpointed")
        return handle.record

    def await_approval(self, handle: RunHandle) -> RunRecord:
        """Mark the run as paused on a human-approval interrupt."""
        return self._transition(handle, status=RunStatus.AWAITING_APPROVAL)

    def resume_run(self, handle: RunHandle) -> RunRecord:
        """Resume a run that was paused for approval."""
        if handle.record.status is not RunStatus.AWAITING_APPROVAL:
            raise InvalidWorkflowStateError(
                f"Run {handle.run_id} is {handle.record.status}, not "
                f"{RunStatus.AWAITING_APPROVAL}; cannot resume."
            )
        return self._transition(handle, status=RunStatus.RUNNING)

    def complete_run(self, handle: RunHandle) -> RunRecord:
        """Mark the run completed and persist its final ledger state."""
        self._require_running(handle.record, allow_awaiting=True)
        self._ledger_service.save(handle.guard.ledger, run_id=handle.run_id)
        return self._transition(handle, status=RunStatus.COMPLETED, finished=True)

    def fail_run(self, handle: RunHandle, *, error: str) -> RunRecord:
        """Mark the run failed, recording the error and final ledger state."""
        self._ledger_service.save(handle.guard.ledger, run_id=handle.run_id)
        handle.record = handle.record.model_copy(
            update={"error": error, "updated_at": datetime.now(UTC)}
        )
        return self._transition(handle, status=RunStatus.FAILED, finished=True)

    # -- internals ------------------------------------------------------------

    def _require_running(self, record: RunRecord, *, allow_awaiting: bool = False) -> None:
        allowed = {RunStatus.RUNNING} | ({RunStatus.AWAITING_APPROVAL} if allow_awaiting else set())
        if record.status not in allowed:
            raise InvalidWorkflowStateError(
                f"Run {record.run_id} is {record.status}; operation requires "
                f"one of {sorted(s.value for s in allowed)}."
            )

    def _transition(
        self, handle: RunHandle, *, status: RunStatus, finished: bool = False
    ) -> RunRecord:
        now = datetime.now(UTC)
        update: dict[str, object] = {"status": status, "updated_at": now}
        if finished:
            update["finished_at"] = now
        handle.record = handle.record.model_copy(update=update)
        self._save_record(handle.record)
        self._logger.bind(
            event="run.status_changed", run_id=str(handle.run_id), status=status.value
        ).info("Run status changed")
        return handle.record

    def _save_record(self, record: RunRecord) -> None:
        path = self._record_path(record.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.loads(record.model_dump_json())
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)