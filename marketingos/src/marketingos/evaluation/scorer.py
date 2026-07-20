"""Deterministic 0-100 scoring rubric for agent outputs.

No language model is involved — every point is earned from a reproducible,
inspectable calculation, mirroring the deterministic-checks-first
philosophy of :mod:`marketingos.agents.qa`. Four weighted checks sum to 100:

* ``structural_validity`` (40) — fraction of required fields that are
  present and non-blank (see ``validator.validate_structure``).
* ``completeness`` (30) — fraction of the agent's
  :class:`~marketingos.evaluation.spec.EvaluationSpec` completeness rules
  that pass.
* ``domain_rules`` (15) — fraction of the spec's domain rules that reported
  no issues.
* ``provenance`` (15) — fraction of ``run_id``/``source_*_run_id`` fields
  that are present and non-blank.

Each check's points are pro-rated by its pass ratio rather than all-or-
nothing, so a mostly-complete output scores meaningfully higher than an
empty one.
"""

from __future__ import annotations

from pydantic import BaseModel

from marketingos.evaluation.models import EvaluationCheck, ValidationResult
from marketingos.evaluation.spec import EvaluationSpec
from marketingos.evaluation.validator import find_provenance_fields

__all__ = ["score_output"]

_STRUCTURAL_VALIDITY_WEIGHT = 40
_COMPLETENESS_WEIGHT = 30
_DOMAIN_RULES_WEIGHT = 15
_PROVENANCE_WEIGHT = 15


def _prorated(weight: int, ratio: float) -> int:
    """Return ``weight`` scaled by ``ratio``, rounded to the nearest point."""
    return round(weight * max(0.0, min(1.0, ratio)))


def _structural_validity_check(
    output: BaseModel, validation: ValidationResult
) -> EvaluationCheck:
    required_fields = [
        name
        for name, field_info in type(output).model_fields.items()
        if field_info.is_required()
    ]
    missing = [
        issue
        for issue in validation.issues
        if issue.code == "missing_required_field"
    ]
    total = len(required_fields)
    ratio = 1.0 if total == 0 else (total - len(missing)) / total
    earned = _prorated(_STRUCTURAL_VALIDITY_WEIGHT, ratio)
    detail = (
        "all required fields present"
        if not missing
        else f"{len(missing)}/{total} required fields blank or missing"
    )
    return EvaluationCheck(
        name="structural_validity",
        weight=_STRUCTURAL_VALIDITY_WEIGHT,
        earned=earned,
        detail=detail,
    )


def _completeness_check(output: BaseModel, spec: EvaluationSpec) -> EvaluationCheck:
    rules = spec.completeness_rules
    if not rules:
        return EvaluationCheck(
            name="completeness",
            weight=_COMPLETENESS_WEIGHT,
            earned=_COMPLETENESS_WEIGHT,
            detail="no completeness rules configured",
        )
    passed = [name for name, rule in rules if rule(output)]
    ratio = len(passed) / len(rules)
    earned = _prorated(_COMPLETENESS_WEIGHT, ratio)
    failed = [name for name, _ in rules if name not in passed]
    detail = "all completeness rules passed" if not failed else f"failed: {failed}"
    return EvaluationCheck(
        name="completeness", weight=_COMPLETENESS_WEIGHT, earned=earned, detail=detail
    )


def _domain_rules_check(
    output: BaseModel, spec: EvaluationSpec, validation: ValidationResult
) -> EvaluationCheck:
    if not spec.domain_rules:
        return EvaluationCheck(
            name="domain_rules",
            weight=_DOMAIN_RULES_WEIGHT,
            earned=_DOMAIN_RULES_WEIGHT,
            detail="no domain rules configured",
        )
    clean_runs = sum(1 for rule in spec.domain_rules if not rule(output))
    ratio = clean_runs / len(spec.domain_rules)
    earned = _prorated(_DOMAIN_RULES_WEIGHT, ratio)
    domain_issue_count = sum(
        1 for issue in validation.issues if issue.code != "missing_required_field"
    )
    detail = (
        "all domain rules passed"
        if domain_issue_count == 0
        else f"{domain_issue_count} domain issue(s) found"
    )
    return EvaluationCheck(
        name="domain_rules", weight=_DOMAIN_RULES_WEIGHT, earned=earned, detail=detail
    )


def _provenance_check(output: BaseModel) -> EvaluationCheck:
    fields = find_provenance_fields(output)
    if not fields:
        return EvaluationCheck(
            name="provenance",
            weight=_PROVENANCE_WEIGHT,
            earned=_PROVENANCE_WEIGHT,
            detail="no provenance fields on this output",
        )
    present = [
        name
        for name, value in fields.items()
        if isinstance(value, str) and value.strip()
    ]
    ratio = len(present) / len(fields)
    earned = _prorated(_PROVENANCE_WEIGHT, ratio)
    missing = sorted(set(fields) - set(present))
    detail = (
        "all provenance fields present"
        if not missing
        else f"missing provenance fields: {missing}"
    )
    return EvaluationCheck(
        name="provenance", weight=_PROVENANCE_WEIGHT, earned=earned, detail=detail
    )


def score_output(
    output: BaseModel, spec: EvaluationSpec, validation: ValidationResult
) -> tuple[int, tuple[EvaluationCheck, ...]]:
    """Compute the 0-100 score and the checks that justify it.

    Args:
        output: The agent's typed output instance.
        spec: The agent's evaluation specification.
        validation: The result of ``validator.validate_structure`` for the
            same ``output``/``spec`` pair.

    Returns:
        A ``(score, checks)`` pair where ``score`` equals the sum of each
        check's earned points.
    """
    checks = (
        _structural_validity_check(output, validation),
        _completeness_check(output, spec),
        _domain_rules_check(output, spec, validation),
        _provenance_check(output),
    )
    return sum(check.earned for check in checks), checks
