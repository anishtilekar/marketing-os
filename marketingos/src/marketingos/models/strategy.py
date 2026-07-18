"""
strategy.py

Domain model for StrategyOutput, the structured output of the Strategy
Agent (pipeline stage ``02_strategy``, between ``01_business_context``
and ``03_plan``).

A StrategyOutput translates a :class:`~marketingos.models.business_context.
BusinessContext` into an actionable content strategy: how the business
should position itself, what it should say, to whom, and on which
platforms. The Planner Agent consumes this output to produce a
:class:`~marketingos.models.plan.WeekPlan`.

This module is a pure domain model:
    - No database logic
    - No API logic
    - No LangGraph / agent orchestration logic
    - No file I/O

Instances of StrategyOutput are expected to be serialized to JSON and
persisted (by a separate service layer) under paths such as:
    data/runs/{run_id}/02_strategy/
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketingos.models.plan import Platform

__all__ = [
    "ContentPillar",
    "StrategyGoal",
    "StrategyOutput",
]


# ---------------------------------------------------------------------------
# Supporting models used only by StrategyOutput
# ---------------------------------------------------------------------------


class ContentPillar(BaseModel):
    """
    A single recurring content theme the strategy commits to, e.g.
    ('Behind the Scenes', 'Builds trust by showing how products are made').
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short label for this content pillar, e.g. 'Behind the Scenes'.",
    )

    rationale: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Why this pillar supports the business's marketing goals.",
    )

    @field_validator("name", "rationale")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("ContentPillar fields cannot be empty or whitespace only.")
        return stripped


class StrategyGoal(BaseModel):
    """A single measurable objective the strategy is designed to achieve."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Human-readable statement of the goal, e.g. 'Grow Instagram followers'.",
    )

    metric: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Metric used to track progress toward this goal, e.g. 'follower_count'.",
    )

    target_value: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Target value for the metric, if quantified, e.g. '+10% in 4 weeks'.",
    )

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("StrategyGoal.description cannot be empty or whitespace only.")
        return stripped

    @field_validator("metric", "target_value")
    @classmethod
    def optional_text_not_blank_if_provided(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("Field cannot be an empty string if provided.")
        return value


# ---------------------------------------------------------------------------
# StrategyOutput
# ---------------------------------------------------------------------------


class StrategyOutput(BaseModel):
    """
    Structured content strategy produced by the Strategy Agent from a
    BusinessContext.

    Deliberately does not reference BusinessContext directly (no
    cross-model coupling); orchestration code is responsible for passing
    the relevant BusinessContext to the Strategy Agent and recording the
    resulting run_id linkage here.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)

    run_id: str = Field(
        description="Identifier of the run this strategy output belongs to.",
    )

    positioning_statement: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="How the business should present itself relative to its market.",
    )

    audience_focus: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Narrative summary of who the content strategy is targeting.",
    )

    content_pillars: list[ContentPillar] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Recurring content themes the weekly plan should draw from.",
    )

    key_messages: list[str] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Core messages the content strategy should consistently reinforce.",
    )

    differentiators: list[str] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Points of competitive differentiation the strategy leans on.",
    )

    target_platforms: list[Platform] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Distribution platforms this strategy prioritizes.",
    )

    goals: list[StrategyGoal] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Measurable objectives this strategy is designed to achieve.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this strategy output was created.",
    )

    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last modification, if any.",
    )

    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional additional structured metadata.",
    )

    @field_validator("positioning_statement", "audience_focus")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace only.")
        return stripped

    @field_validator("key_messages", "differentiators")
    @classmethod
    def strip_and_drop_blanks(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @field_validator("target_platforms")
    @classmethod
    def platforms_unique(cls, value: list[Platform]) -> list[Platform]:
        deduped: list[Platform] = []
        seen: set[Platform] = set()
        for platform in value:
            if platform not in seen:
                seen.add(platform)
                deduped.append(platform)
        return deduped

    @field_validator("created_at", "updated_at")
    @classmethod
    def ensure_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return value
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def deduplicate_content_pillars(self) -> "StrategyOutput":
        """
        Removes exact duplicate content pillars based on name, preserving
        first-occurrence order, mirroring BusinessContext's dedup pattern
        for facts/assumptions.
        """
        seen_names: set[str] = set()
        deduped: list[ContentPillar] = []
        for pillar in self.content_pillars:
            key = pillar.name.lower()
            if key not in seen_names:
                seen_names.add(key)
                deduped.append(pillar)
        # Bypass __setattr__: with validate_assignment=True, a normal
        # assignment here would re-run every "after" validator (including
        # this one), recursing until the interpreter's stack limit.
        object.__setattr__(self, "content_pillars", deduped)
        return self
