"""Data contracts for a MarketingOS run's identity, lifecycle, and layout.

These were originally defined inline in ``services/run_manager.py`` because
this module did not yet exist; they have moved here unchanged, and
``run_manager.py`` now imports them instead of defining them. This module is
a leaf: pure data only, no I/O, no service logic. Directory creation,
persistence, and lifecycle transitions remain the responsibility of
:class:`~marketingos.services.run_manager.RunManager`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "RunRecord",
    "RunSection",
    "RunStatus",
]


class RunStatus(StrEnum):
    """Lifecycle state of a MarketingOS run."""

    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class RunSection(StrEnum):
    """Numbered pipeline stages a run's working directory is organised by.

    Values are the literal subdirectory names under a run's root
    (``data/runs/{run_id}/``), per the architecture doc's output-organization
    section. Member declaration order matches pipeline order.
    """

    SOURCE_PACK = "00_source_pack"
    BUSINESS_CONTEXT = "01_business_context"
    STRATEGY = "02_strategy"
    PLAN = "03_plan"
    CREATIVE_POSTS = "04_creatives/posts"
    CREATIVE_VIDEOS = "04_creatives/videos"
    QA = "05_qa"
    COST = "06_cost"
    LOGS = "07_logs"
    PACKAGE = "package"
    EVALUATION = "eval"


class RunRecord(BaseModel):
    """Persisted status and checkpoint history for one run.

    Deliberately thin: the cost ledger is the spend log, agent outputs live
    in the numbered directories, and this record only tracks what's needed
    to resume or audit a run's progress.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: UUID = Field(
        description="Identifier of the MarketingOS execution run this record belongs to.",
    )

    status: RunStatus = Field(
        default=RunStatus.RUNNING,
        description="Current lifecycle state of the run.",
    )

    max_budget: Decimal = Field(
        description="Spend ceiling for this run, enforced via the run's CostGuard.",
    )

    checkpoints: list[str] = Field(
        default_factory=list,
        description="Names of agent/graph nodes completed so far, in completion order.",
    )

    error: str | None = Field(
        default=None,
        description="Error message recorded if the run transitioned to FAILED.",
    )

    started_at: datetime = Field(
        description="Timestamp when the run was created.",
    )

    updated_at: datetime = Field(
        description="Timestamp of the most recent change to this record.",
    )

    finished_at: datetime | None = Field(
        default=None,
        description="Timestamp when the run reached COMPLETED or FAILED, if it has.",
    )

    @field_validator("max_budget")
    @classmethod
    def max_budget_not_negative(cls, value: Decimal) -> Decimal:
        if value < Decimal("0"):
            raise ValueError("max_budget cannot be negative.")
        return value

    @field_validator("started_at", "updated_at", "finished_at")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
