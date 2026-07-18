from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CostCategory(str, Enum):
    """Source category of a cost-incurring operation."""

    LLM_GENERATION = "llm_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    EMBEDDING = "embedding"
    WEB_TOOL = "web_tool"
    STORAGE = "storage"
    OTHER = "other"


class CostStatus(str, Enum):
    """Lifecycle state of a cost event."""

    ESTIMATED = "estimated"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class CostEntry(BaseModel):
    """
    A single cost transaction incurred by a paid tool, model, or
    service call within a MarketingOS run.

    This is a pure data contract. Budget enforcement, aggregation,
    and persistence are handled by services/cost_guard.py and
    services/cost_ledger.py, not by this model.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)

    run_id: UUID = Field(
        ...,
        description="Identifier of the MarketingOS execution run this cost belongs to.",
    )

    category: CostCategory = Field(
        ...,
        description="Category of the cost-incurring operation.",
    )

    status: CostStatus = Field(
        default=CostStatus.COMPLETED,
        description="Lifecycle status of this cost event.",
    )

    provider: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Name of the external provider, e.g. 'openai', 'stability', 'elevenlabs'.",
    )

    tool_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Specific tool or model invoked, e.g. 'gpt-4.1', 'image_generation'.",
    )

    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional human-readable description of the cost event.",
    )

    estimated_cost: Decimal = Field(
        default=Decimal("0"),
        description="Estimated cost of this operation prior to execution.",
    )

    actual_cost: Decimal = Field(
        default=Decimal("0"),
        description="Actual cost incurred, once known.",
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
        description="ISO-style currency code for this cost entry.",
    )

    input_tokens: int | None = Field(
        default=None,
        description="Number of input tokens consumed, if applicable.",
    )

    output_tokens: int | None = Field(
        default=None,
        description="Number of output tokens produced, if applicable.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional structured metadata.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this cost entry was created.",
    )

    @field_validator("provider", "tool_name")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace only.")
        return stripped

    @field_validator("estimated_cost", "actual_cost")
    @classmethod
    def cost_not_negative(cls, value: Decimal) -> Decimal:
        if value < Decimal("0"):
            raise ValueError("Cost values cannot be negative.")
        return value

    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def tokens_not_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Token counts cannot be negative.")
        return value

    @field_validator("currency")
    @classmethod
    def currency_must_be_uppercase_iso(cls, value: str) -> str:
        stripped = value.strip().upper()
        if not stripped.isalpha() or len(stripped) != 3:
            raise ValueError("currency must be a 3-letter uppercase ISO-style code.")
        return stripped

    @field_validator("created_at")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class CostSummary(BaseModel):
    """
    Aggregate cost representation summarizing multiple CostEntry
    records. Aggregation logic itself lives outside this model.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    total_estimated_cost: Decimal = Field(
        default=Decimal("0"),
        description="Sum of estimated costs across aggregated entries.",
    )

    total_actual_cost: Decimal = Field(
        default=Decimal("0"),
        description="Sum of actual costs across aggregated entries.",
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
        description="ISO-style currency code for this summary.",
    )

    entry_count: int = Field(
        default=0,
        description="Number of cost entries included in this summary.",
    )

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this summary was generated.",
    )

    @field_validator("total_estimated_cost", "total_actual_cost")
    @classmethod
    def cost_not_negative(cls, value: Decimal) -> Decimal:
        if value < Decimal("0"):
            raise ValueError("Cost values cannot be negative.")
        return value

    @field_validator("entry_count")
    @classmethod
    def entry_count_not_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("entry_count cannot be negative.")
        return value

    @field_validator("currency")
    @classmethod
    def currency_must_be_uppercase_iso(cls, value: str) -> str:
        stripped = value.strip().upper()
        if not stripped.isalpha() or len(stripped) != 3:
            raise ValueError("currency must be a 3-letter uppercase ISO-style code.")
        return stripped

    @field_validator("generated_at")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class CostLedger(BaseModel):
    """
    Collection of all cost entries for a MarketingOS run, with
    structural budget enforcement.

    The max_budget is configurable per instance and is never
    hardcoded; callers (e.g. services/cost_guard.py) supply the
    applicable budget when constructing or updating this ledger.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    entries: list[CostEntry] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="All cost entries recorded for this run.",
    )

    max_budget: Decimal = Field(
        default=Decimal("100"),
        description="Maximum allowed budget for this run.",
    )

    @field_validator("max_budget")
    @classmethod
    def max_budget_not_negative(cls, value: Decimal) -> Decimal:
        if value < Decimal("0"):
            raise ValueError("max_budget cannot be negative.")
        return value

    @model_validator(mode="after")
    def validate_budget(self) -> CostLedger:
        spent = sum(
            (entry.actual_cost for entry in self.entries if entry.status == CostStatus.COMPLETED),
            Decimal("0"),
        )
        if spent > self.max_budget:
            raise ValueError(
                f"Budget exceeded: spent {spent} exceeds maximum budget {self.max_budget}."
            )
        return self
