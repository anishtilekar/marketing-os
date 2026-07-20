"""Per-agent evaluation specifications.

An :class:`EvaluationSpec` tells the evaluation harness what "good" looks
like for one agent's output type: which fields must be non-empty
(structural validity is handled generically, see ``validator.py``), which
completeness rules apply, and which domain-specific invariants to check.

This is a plain ``dataclass`` rather than a Pydantic model — the one
deliberate deviation from this codebase's Pydantic-everywhere convention —
because it holds callables (rule functions), which Pydantic v2 cannot
validate without ``arbitrary_types_allowed`` escape hatches that would add
noise without adding safety.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from marketingos.evaluation.models import ValidationIssue

__all__ = ["CompletenessRule", "DomainRule", "EvaluationSpec"]

#: A named predicate over an agent's output: True if the completeness
#: expectation it describes is satisfied.
CompletenessRule = tuple[str, Callable[[BaseModel], bool]]

#: A domain-specific check that returns the issues it found (empty if none).
DomainRule = Callable[[BaseModel], list[ValidationIssue]]


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation configuration for one agent's output type."""

    agent_name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    completeness_rules: tuple[CompletenessRule, ...] = field(default_factory=tuple)
    domain_rules: tuple[DomainRule, ...] = field(default_factory=tuple)
