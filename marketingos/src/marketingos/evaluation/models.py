"""Data contracts for the per-agent Evaluation Framework.

These models describe the outcome of evaluating a single agent execution:
whether its output is structurally valid, how complete it is, and what
artifacts (JSON, prompts, images, videos) it produced. They deliberately
mirror the conventions established by :mod:`marketingos.agents.qa` — frozen
models, ``StrEnum`` for closed vocabularies, and a score that is *derived*
from the recorded checks rather than asserted, so a report can never claim a
score its own checks don't support.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "EvaluationCheck",
    "EvaluationReport",
    "ValidationIssue",
    "ValidationResult",
]


class ArtifactKind(StrEnum):
    """The kind of artifact an evaluated agent output referenced."""

    JSON = "json"
    PROMPT = "prompt"
    IMAGE = "image"
    VIDEO = "video"
    LOG = "log"


class ArtifactRecord(BaseModel):
    """One artifact discovered on an agent's output."""

    model_config = ConfigDict(frozen=True)

    kind: ArtifactKind
    field_path: str = Field(
        min_length=1,
        description="Dotted/indexed path to the field this artifact came "
        "from, e.g. 'videos[0].direction.script'.",
    )
    value: str = Field(
        min_length=1, description="The artifact's content or location."
    )
    description: str = Field(default="", max_length=500)


class ValidationIssue(BaseModel):
    """One structural or domain problem found on an agent's output."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(min_length=1)
    code: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable machine-readable identifier for this class of "
        "issue, e.g. 'missing_required_field'.",
    )
    message: str = Field(min_length=1, max_length=1000)


class ValidationResult(BaseModel):
    """The outcome of validating one agent output's structure."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    issues: tuple[ValidationIssue, ...] = ()


class EvaluationCheck(BaseModel):
    """One weighted rubric check contributing to the overall score."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    weight: int = Field(ge=0, le=100, description="Points this check is worth.")
    earned: int = Field(ge=0, le=100, description="Points actually awarded.")
    detail: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _validate_earned_within_weight(self) -> EvaluationCheck:
        if self.earned > self.weight:
            raise ValueError(
                f"check {self.name!r} earned {self.earned} points, exceeding "
                f"its weight of {self.weight}"
            )
        return self


class EvaluationReport(BaseModel):
    """The evaluation verdict for one agent execution.

    ``score`` is derived, not asserted: construction recomputes it from
    ``checks`` and rejects any report whose declared score disagrees, the
    same trick :class:`~marketingos.agents.qa.QAReport` uses for its status.
    """

    model_config = ConfigDict(frozen=True)

    agent_name: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    input_schema: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    validation: ValidationResult
    score: int = Field(ge=0, le=100)
    checks: tuple[EvaluationCheck, ...] = Field(min_length=1)
    artifacts: tuple[ArtifactRecord, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_score(self) -> EvaluationReport:
        derived = sum(check.earned for check in self.checks)
        if self.score != derived:
            raise ValueError(
                f"score {self.score} disagrees with the checks, which "
                f"derive {derived}"
            )
        return self
