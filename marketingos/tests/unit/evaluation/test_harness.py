from __future__ import annotations

import json
from uuid import uuid4

from conftest import make_caption_package, make_research_result, make_week_plan

from marketingos.agents.copywriter import CopywriterAgent
from marketingos.agents.planner import PlannerAgent
from marketingos.evaluation.harness import (
    evaluate_agent_output,
    persist_evaluation_report,
)
from marketingos.evaluation.models import ArtifactKind
from marketingos.models.run import RunSection
from marketingos.services.run_manager import RunManager


def test_evaluate_agent_output_for_registered_agent_scores_and_reports() -> None:
    plan = make_week_plan()
    report = evaluate_agent_output(
        agent_name=PlannerAgent.__name__, output=plan, run_id="run-1"
    )
    assert report.agent_name == PlannerAgent.__name__
    assert report.output_schema == "WeekPlan"
    assert report.validation.passed
    assert report.score == 100
    assert any(a.kind == ArtifactKind.JSON for a in report.artifacts)


def test_evaluate_agent_output_for_unregistered_agent_never_raises() -> None:
    plan = make_week_plan()
    report = evaluate_agent_output(
        agent_name="TotallyUnknownAgent", output=plan, run_id="run-1"
    )
    assert report.score == 0
    assert not report.validation.passed
    assert report.validation.issues[0].code == "no_spec_registered"


def test_evaluate_agent_output_collects_prompt_artifacts() -> None:
    plan = make_week_plan()
    captions = make_caption_package(plan=plan)
    report = evaluate_agent_output(
        agent_name=CopywriterAgent.__name__, output=captions, run_id="run-1"
    )
    # Caption bodies aren't a recognised prompt/asset field, so only the
    # top-level JSON dump artifact is expected here; this asserts the walk
    # doesn't crash on a caption package and always yields at least the
    # JSON artifact.
    assert len(report.artifacts) >= 1
    assert report.artifacts[0].kind == ArtifactKind.JSON


def test_evaluate_agent_output_scores_research_result() -> None:
    research = make_research_result()
    report = evaluate_agent_output(
        agent_name="ResearchAgent", output=research, run_id="run-1"
    )
    assert report.score == 100


def test_persist_evaluation_report_writes_expected_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_manager = RunManager(runs_root=tmp_path)
    run_id = uuid4()
    plan = make_week_plan()
    report = evaluate_agent_output(
        agent_name=PlannerAgent.__name__, output=plan, run_id=str(run_id)
    )

    persist_evaluation_report(run_manager, run_id, report)

    eval_path = (
        run_manager.section_dir(run_id, RunSection.EVALUATION)
        / f"{PlannerAgent.__name__}_eval.json"
    )
    assert eval_path.exists()
    on_disk = json.loads(eval_path.read_text(encoding="utf-8"))
    assert on_disk["agent_name"] == PlannerAgent.__name__
    assert on_disk["score"] == 100
