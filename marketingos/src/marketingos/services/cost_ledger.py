"""Per-run cost ledger persistence and summarisation.

Complements :mod:`marketingos.services.cost_guard`, which enforces the
budget in memory: this service owns the ledger's life outside the process.
It creates the run's :class:`~marketingos.models.cost.CostLedger` with the
injected budget ceiling, persists it as the append-only spend log the spec
requires (``data/runs/{run_id}/cost_ledger.json``), reloads it across
restarts, and rolls entries up into a
:class:`~marketingos.models.cost.CostSummary`.

Division of labour
------------------
* ``models/cost.py``  — data contracts + structural budget invariant.
* ``cost_guard.py``   — the only writer of entries; refuses over-budget calls.
* this module         — construction, disk persistence, aggregation. It never
  appends entries itself, so the guard remains the single choke point.

The budget is supplied by the caller (run manager reads it from
``BudgetSettings``); this module stays independent of the config chain,
mirroring the guard.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError

from marketingos.exceptions.budget import CostTrackingError
from marketingos.models.cost import CostLedger, CostStatus, CostSummary

if TYPE_CHECKING:
    from uuid import UUID

    from loguru import Logger

__all__ = ["CostLedgerService", "DEFAULT_LEDGER_FILENAME"]

#: File name of the persisted ledger inside a run directory.
DEFAULT_LEDGER_FILENAME = "cost_ledger.json"


class CostLedgerService:
    """Creates, persists, loads and summarises per-run cost ledgers.

    One instance serves all runs; every method takes the ``run_id`` (or the
    ledger itself) explicitly, so the service holds no per-run state and is
    safe to share.
    """

    def __init__(
        self,
        *,
        runs_root: Path = Path("data/runs"),
        ledger_filename: str = DEFAULT_LEDGER_FILENAME,
    ) -> None:
        """Initialise the service.

        Args:
            runs_root: Directory under which per-run working directories
                live; the ledger for a run is stored at
                ``{runs_root}/{run_id}/{ledger_filename}``.
            ledger_filename: File name of the persisted ledger.
        """
        self._runs_root = runs_root
        self._ledger_filename = ledger_filename
        self._logger: Logger = logger.bind(component="CostLedgerService")

    # -- paths ------------------------------------------------------------

    def ledger_path(self, run_id: UUID) -> Path:
        """Return the on-disk path of the ledger for ``run_id``."""
        return self._runs_root / str(run_id) / self._ledger_filename

    # -- lifecycle --------------------------------------------------------

    def create(self, *, run_id: UUID, max_budget: Decimal) -> CostLedger:
        """Create and persist an empty ledger for a new run.

        Args:
            run_id: Identifier of the run the ledger belongs to.
            max_budget: The run's spend ceiling, typically taken from
                ``BudgetSettings.max_budget``.

        Returns:
            The new, empty ledger.

        Raises:
            CostTrackingError: If a ledger already exists for this run —
                creating over an existing spend log would erase real spend.
        """
        path = self.ledger_path(run_id)
        if path.exists():
            raise CostTrackingError(
                f"A cost ledger already exists for run {run_id} at {path}; "
                "refusing to overwrite a spend log. Use load() instead."
            )
        ledger = CostLedger(max_budget=max_budget)
        self._write(ledger, path=path, run_id=run_id)
        self._logger.bind(
            event="cost_ledger.created",
            run_id=str(run_id),
            max_budget=str(max_budget),
        ).info("Ledger created")
        return ledger

    def load(self, run_id: UUID) -> CostLedger:
        """Load and validate the persisted ledger for a run.

        Validation re-runs the model's own budget invariant, so a ledger
        tampered with or corrupted on disk (spend past ceiling, malformed
        entries) is rejected rather than silently trusted.

        Args:
            run_id: Identifier of the run whose ledger to load.

        Returns:
            The validated ledger.

        Raises:
            CostTrackingError: If no ledger exists for the run, or the file
                is not valid JSON, or it fails model validation.
        """
        path = self.ledger_path(run_id)
        if not path.is_file():
            raise CostTrackingError(f"No cost ledger found for run {run_id} at {path}.")
        try:
            ledger = CostLedger.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError) as exc:
            raise CostTrackingError(
                f"Cost ledger for run {run_id} at {path} is corrupt or "
                f"invalid: {exc}"
            ) from exc
        self._logger.bind(
            event="cost_ledger.loaded",
            run_id=str(run_id),
            entries=len(ledger.entries),
        ).debug("Ledger loaded")
        return ledger

    def save(self, ledger: CostLedger, *, run_id: UUID) -> Path:
        """Persist the ledger's current state for a run.

        Called after the guard records entries (e.g. per checkpoint or at
        run end) so on-disk state tracks in-memory spend. The write is
        atomic — a temp file replaced over the target — so a crash
        mid-write cannot leave a truncated spend log.

        Args:
            ledger: The ledger to persist.
            run_id: Identifier of the run it belongs to.

        Returns:
            The path the ledger was written to.
        """
        path = self.ledger_path(run_id)
        self._write(ledger, path=path, run_id=run_id)
        return path

    # -- aggregation ------------------------------------------------------

    def summarize(self, ledger: CostLedger) -> CostSummary:
        """Roll the ledger's completed entries up into a summary.

        Only ``COMPLETED`` entries contribute cost — mirroring the guard's
        definition of spend — but ``entry_count`` reports all entries so
        failed/refunded attempts remain visible in reports.

        Args:
            ledger: The ledger to summarise.

        Returns:
            The aggregate summary. Currency is taken from the first entry;
            an empty ledger keeps the model default.
        """
        zero = Decimal("0")
        completed = [e for e in ledger.entries if e.status is CostStatus.COMPLETED]
        kwargs: dict[str, object] = {
            "total_estimated_cost": sum((e.estimated_cost for e in completed), zero),
            "total_actual_cost": sum((e.actual_cost for e in completed), zero),
            "entry_count": len(ledger.entries),
        }
        if ledger.entries:
            kwargs["currency"] = ledger.entries[0].currency
        return CostSummary.model_validate(kwargs)

    # -- internals --------------------------------------------------------

    def _write(self, ledger: CostLedger, *, path: Path, run_id: UUID) -> None:
        """Atomically serialise ``ledger`` to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.loads(ledger.model_dump_json())
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.replace(path)
        self._logger.bind(
            event="cost_ledger.saved",
            run_id=str(run_id),
            entries=len(ledger.entries),
            path=str(path),
        ).debug("Ledger persisted")
