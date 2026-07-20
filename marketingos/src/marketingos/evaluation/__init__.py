"""Evaluation Framework: per-agent input/output schema checks, structural
validation, deterministic 0-100 scoring, and artifact tracking.

See :func:`marketingos.evaluation.harness.evaluate_agent_output` for the
framework's single entrypoint.
"""

from __future__ import annotations

from marketingos.evaluation.harness import (
    evaluate_agent_output,
    persist_evaluation_report,
)
from marketingos.evaluation.models import (
    ArtifactKind,
    ArtifactRecord,
    EvaluationCheck,
    EvaluationReport,
    ValidationIssue,
    ValidationResult,
)
from marketingos.evaluation.spec import EvaluationSpec

__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "EvaluationCheck",
    "EvaluationReport",
    "EvaluationSpec",
    "ValidationIssue",
    "ValidationResult",
    "evaluate_agent_output",
    "persist_evaluation_report",
]
