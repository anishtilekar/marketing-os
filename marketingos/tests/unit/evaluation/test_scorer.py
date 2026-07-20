from __future__ import annotations

from conftest import make_week_plan

from marketingos.agents.planner import PlannerAgent, WeekPlan
from marketingos.agents.strategist import Strategy
from marketingos.evaluation.models import ValidationIssue
from marketingos.evaluation.scorer import score_output
from marketingos.evaluation.spec import EvaluationSpec
from marketingos.evaluation.validator import validate_structure


def _spec(**kwargs: object) -> EvaluationSpec:
    return EvaluationSpec(
        agent_name=PlannerAgent.__name__,
        input_model=Strategy,
        output_model=WeekPlan,
        **kwargs,  # type: ignore[arg-type]
    )


def test_perfect_output_scores_100() -> None:
    plan = make_week_plan()
    spec = _spec(completeness_rules=(("has_items", lambda p: len(p.items) > 0),))
    validation = validate_structure(plan, spec)
    score, checks = score_output(plan, spec, validation)
    assert score == 100
    assert sum(check.earned for check in checks) == score
    assert {check.name for check in checks} == {
        "structural_validity",
        "completeness",
        "domain_rules",
        "provenance",
    }


def test_failing_completeness_rule_reduces_score() -> None:
    plan = make_week_plan()
    spec = _spec(completeness_rules=(("never_passes", lambda _p: False),))
    validation = validate_structure(plan, spec)
    score, checks = score_output(plan, spec, validation)
    assert score == 70  # 100 - the full 30-point completeness weight
    completeness = next(c for c in checks if c.name == "completeness")
    assert completeness.earned == 0


def test_failing_domain_rule_reduces_score() -> None:
    def always_flags(_: WeekPlan) -> list[ValidationIssue]:
        return [ValidationIssue(field="items", code="bad", message="bad")]

    plan = make_week_plan()
    spec = _spec(domain_rules=(always_flags,))
    validation = validate_structure(plan, spec)
    score, checks = score_output(plan, spec, validation)
    assert score == 85  # 100 - the full 15-point domain_rules weight
    domain = next(c for c in checks if c.name == "domain_rules")
    assert domain.earned == 0


def test_no_rules_configured_awards_full_credit_for_those_checks() -> None:
    plan = make_week_plan()
    spec = _spec()
    validation = validate_structure(plan, spec)
    score, _checks = score_output(plan, spec, validation)
    assert score == 100
