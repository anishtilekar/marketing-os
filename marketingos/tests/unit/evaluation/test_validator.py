from __future__ import annotations

from conftest import make_week_plan

from marketingos.agents.planner import PlannerAgent, WeekPlan
from marketingos.agents.strategist import Strategy
from marketingos.evaluation.models import ValidationIssue
from marketingos.evaluation.spec import EvaluationSpec
from marketingos.evaluation.validator import find_provenance_fields, validate_structure


def _spec(*, domain_rules: tuple = ()) -> EvaluationSpec:
    return EvaluationSpec(
        agent_name=PlannerAgent.__name__,
        input_model=Strategy,
        output_model=WeekPlan,
        domain_rules=domain_rules,
    )


def test_validate_structure_passes_for_a_well_formed_plan() -> None:
    plan = make_week_plan()
    result = validate_structure(plan, _spec())
    assert result.passed
    assert result.issues == ()


def test_validate_structure_runs_domain_rules() -> None:
    def always_flags(_: WeekPlan) -> list[ValidationIssue]:
        return [ValidationIssue(field="items", code="always_flagged", message="nope")]

    plan = make_week_plan()
    result = validate_structure(plan, _spec(domain_rules=(always_flags,)))
    assert not result.passed
    assert result.issues[0].code == "always_flagged"


def test_find_provenance_fields_returns_run_id_fields() -> None:
    plan = make_week_plan(run_id="plan-run-42")
    fields = find_provenance_fields(plan)
    assert fields == {
        "run_id": "plan-run-42",
        "source_strategy_run_id": "strategy-run-1",
    }
