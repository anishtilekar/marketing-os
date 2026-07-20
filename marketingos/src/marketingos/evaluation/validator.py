"""Generic structural validation for agent outputs.

The heavy lifting — type conformance, cross-field invariants, count
constraints — is already done by each agent's own Pydantic output model
(see e.g. ``WeekPlan``'s ``_validate_plan_invariants`` in
``agents/planner.py``): an instance simply cannot exist unless it satisfies
those rules. What's left for a *generic* structural pass is a defensive,
reflection-based sweep for the class of bug those per-field validators don't
always catch — a required string or collection field that is blank despite
passing Pydantic's own checks (e.g. constructed via ``model_construct``, or
a field pattern that permits whitespace-only values) — plus whatever
agent-specific domain rules the agent's :class:`~marketingos.evaluation.spec.
EvaluationSpec` supplies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from marketingos.evaluation.models import ValidationIssue, ValidationResult
from marketingos.evaluation.spec import EvaluationSpec

__all__ = ["find_provenance_fields", "validate_structure"]

#: Field names treated as run-provenance identifiers for the provenance
#: check in ``scorer.py``. Matches this codebase's convention of a
#: ``run_id`` field plus ``source_*_run_id`` fields (see e.g.
#: ``CampaignBundle`` in ``agents/qa.py``).
_PROVENANCE_SUFFIX = "run_id"


def _is_blank(value: Any) -> bool:
    """Return whether a required field's value counts as structurally empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return False


def validate_structure(output: BaseModel, spec: EvaluationSpec) -> ValidationResult:
    """Run the generic required-field sweep plus ``spec``'s domain rules.

    Args:
        output: The agent's typed output instance.
        spec: The agent's evaluation specification.

    Returns:
        A :class:`ValidationResult` recording every issue found; ``passed``
        is ``True`` only when no issues were found.
    """
    issues: list[ValidationIssue] = []

    for name, field_info in type(output).model_fields.items():
        if not field_info.is_required():
            continue
        value = getattr(output, name)
        if _is_blank(value):
            issues.append(
                ValidationIssue(
                    field=name,
                    code="missing_required_field",
                    message=f"Required field {name!r} is blank or empty.",
                )
            )

    for domain_rule in spec.domain_rules:
        issues.extend(domain_rule(output))

    return ValidationResult(passed=not issues, issues=tuple(issues))


def find_provenance_fields(output: BaseModel) -> Mapping[str, Any]:
    """Return every top-level field on ``output`` that looks provenance-like.

    A field counts as provenance-like when its name is ``run_id`` or ends
    with ``_run_id`` (e.g. ``source_plan_run_id``).
    """
    return {
        name: getattr(output, name)
        for name in type(output).model_fields
        if name == _PROVENANCE_SUFFIX or name.endswith(f"_{_PROVENANCE_SUFFIX}")
    }
