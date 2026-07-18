"""
business_context.py

Domain model for BusinessContext, the structured output of the
Business Analysis Agent.

Architectural principle:
    FACTS != ASSUMPTIONS.

Facts and Assumptions are modeled as separate, non-interchangeable
Pydantic types stored in physically separate fields (`observed_facts`
and `assumptions`). There is no shared base class between them that
would allow a value to satisfy both fields' type constraints, and
runtime validators enforce that each list only ever contains the
correct type.

This module is a pure domain model:
    - No database logic
    - No API logic
    - No LangGraph / agent orchestration logic
    - No file I/O

Instances of BusinessContext are expected to be serialized to JSON
and persisted (by a separate service layer) under paths such as:
    data/runs/{run_id}/01_business_context/
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """Origin category of a Fact's source material."""

    WEBSITE = "website"
    NEWS_ARTICLE = "news_article"
    SOCIAL_MEDIA = "social_media"
    SEARCH_ENGINE = "search_engine"
    REPORT = "report"
    API = "api"
    DATABASE = "database"
    SURVEY = "survey"
    PUBLIC_RECORD = "public_record"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk level associated with an Assumption being incorrect."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------


class Fact(BaseModel):
    """
    An objectively observed piece of information extracted from an
    external source.

    A Fact must NEVER encode interpretation, inference, prediction,
    or opinion. If a statement requires reasoning to arrive at, it
    belongs in `Assumption`, not here.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,  # facts are immutable observations once recorded
    )

    id: UUID = Field(default_factory=uuid4)

    statement: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="The literal, objectively observed statement.",
    )

    source_reference: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="URL, citation, or identifier of the originating source.",
    )

    source_type: SourceType = Field(
        ...,
        description="Category of the source that produced this fact.",
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that this observation was extracted correctly (0.0-1.0).",
    )

    category: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional grouping label, e.g. 'pricing', 'audience', 'competitors'.",
    )

    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this fact was extracted from its source.",
    )

    @field_validator("statement")
    @classmethod
    def statement_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Fact.statement cannot be empty or whitespace only.")
        return stripped

    @field_validator("source_reference")
    @classmethod
    def source_reference_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Fact.source_reference cannot be empty.")
        return stripped

    @field_validator("extracted_at")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        # Normalize naive datetimes to UTC rather than silently accepting them.
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


# ---------------------------------------------------------------------------
# Assumption
# ---------------------------------------------------------------------------


class Assumption(BaseModel):
    """
    An inferred hypothesis or interpretation derived from one or more
    facts (or from general reasoning).

    Assumptions must always carry an explicit confidence score and
    risk level so downstream agents treat them as uncertain, never
    as ground truth.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,  # assumptions are immutable once recorded; superseding
                      # an assumption means creating a new one, not mutating it
    )

    id: UUID = Field(default_factory=uuid4)

    statement: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="The inferred/hypothesized statement.",
    )

    reasoning: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Explanation of how this assumption was derived.",
    )

    risk_level: RiskLevel = Field(
        ...,
        description="Risk level if this assumption turns out to be false.",
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this assumption's validity (0.0-1.0).",
    )

    category: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional grouping label, e.g. 'positioning', 'audience', 'strategy'.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this assumption was created.",
    )

    @field_validator("statement")
    @classmethod
    def statement_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Assumption.statement cannot be empty or whitespace only.")
        return stripped

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Assumption.reasoning cannot be empty.")
        return stripped

    @field_validator("created_at")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


# ---------------------------------------------------------------------------
# Supporting models used only by BusinessContext
# ---------------------------------------------------------------------------


class TargetAudience(BaseModel):
    """
    Structured description of who the business is trying to reach.

    Kept intentionally lightweight; deeper segmentation/persona logic
    belongs in a later pipeline stage (e.g. Plan), not in this model.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Narrative description of the target audience.",
    )

    demographics: list[str] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Discrete demographic attributes, e.g. 'age 25-34', 'urban'.",
    )

    interests: list[str] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Interests or affinities relevant to targeting.",
    )

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("TargetAudience.description cannot be empty.")
        return stripped


class BrandCharacteristic(BaseModel):
    """A single labeled brand trait, e.g. ('tone', 'playful')."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    trait: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1, max_length=500)

    @field_validator("trait", "value")
    @classmethod
    def not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("BrandCharacteristic fields cannot be empty.")
        return stripped


# ---------------------------------------------------------------------------
# BusinessContext
# ---------------------------------------------------------------------------


class BusinessContext(BaseModel):
    """
    Structured business understanding produced by the Business
    Analysis Agent by combining observed Facts with derived
    Assumptions.

    Facts and Assumptions are stored in physically separate,
    strongly-typed fields (`observed_facts: list[Fact]` and
    `assumptions: list[Assumption]`). Because Fact and Assumption
    are distinct, unrelated Pydantic models (no shared base class,
    no shared field shape), Pydantic's validation will reject an
    Assumption placed in `observed_facts` or a Fact placed in
    `assumptions` at construction time. The model_validator below
    provides an additional defensive runtime check for the same
    invariant.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: UUID = Field(default_factory=uuid4)

    run_id: str = Field(
        description="Identifier of the BusinessAnalysisAgent execution that "
        "produced this context."
    )

    business_name: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Name of the business this context describes.",
    )

    industry: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Industry or vertical the business operates in.",
    )

    description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Free-text summary of the business.",
    )

    # Physically separate fields: never merge these two lists.
    observed_facts: list[Fact] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Objectively observed facts. Must contain only Fact instances.",
    )

    assumptions: list[Assumption] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Inferred hypotheses. Must contain only Assumption instances.",
    )

    target_audience: Optional[TargetAudience] = Field(
        default=None,
        description="Structured description of the intended audience.",
    )

    brand_characteristics: list[BrandCharacteristic] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Labeled brand traits (tone, voice, values, etc.).",
    )

    business_goals: list[str] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Stated business objectives relevant to marketing strategy.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when this BusinessContext was created.",
    )

    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the last modification, if any.",
    )

    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional additional structured metadata.",
    )

    # -- field validators --------------------------------------------------

    @field_validator("business_name")
    @classmethod
    def business_name_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("BusinessContext.business_name cannot be empty.")
        return stripped

    @field_validator("business_goals")
    @classmethod
    def goals_not_blank(cls, value: list[str]) -> list[str]:
        cleaned = [goal.strip() for goal in value if goal.strip()]
        return cleaned

    @field_validator("created_at", "updated_at")
    @classmethod
    def ensure_timezone_aware(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return value
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    # -- cross-field / structural validators --------------------------------

    @model_validator(mode="after")
    def enforce_fact_assumption_separation(self) -> "BusinessContext":
        """
        Defensive runtime guard (in addition to static typing) that
        ensures observed_facts and assumptions never cross-contaminate,
        even if this model is constructed dynamically (e.g. from a
        dict via model_validate) where type coercion could otherwise
        mask a mistake.
        """
        for item in self.observed_facts:
            if isinstance(item, Assumption):
                raise ValueError("An Assumption cannot be stored in observed_facts.")

        for item in self.assumptions:
            if isinstance(item, Fact):
                raise ValueError("A Fact cannot be stored in assumptions.")

        return self

    @model_validator(mode="after")
    def deduplicate_facts_and_assumptions(self) -> "BusinessContext":
        """
        Removes exact duplicate facts/assumptions based on their
        semantic content (statement + source_reference for facts;
        statement + reasoning for assumptions), preserving first
        occurrence order. IDs are intentionally excluded from the
        dedup key since two independently-generated records with
        identical content but different UUIDs still represent the
        same underlying fact/assumption.
        """
        seen_facts: set[tuple[str, str]] = set()
        deduped_facts: list[Fact] = []
        for fact in self.observed_facts:
            key = (fact.statement.lower(), fact.source_reference.lower())
            if key not in seen_facts:
                seen_facts.add(key)
                deduped_facts.append(fact)
        # Bypass __setattr__: with validate_assignment=True, a normal
        # assignment here would re-run every "after" validator (including
        # this one), recursing until the interpreter's stack limit.
        object.__setattr__(self, "observed_facts", deduped_facts)

        seen_assumptions: set[tuple[str, str]] = set()
        deduped_assumptions: list[Assumption] = []
        for assumption in self.assumptions:
            key = (assumption.statement.lower(), assumption.reasoning.lower())
            if key not in seen_assumptions:
                seen_assumptions.add(key)
                deduped_assumptions.append(assumption)
        object.__setattr__(self, "assumptions", deduped_assumptions)

        return self