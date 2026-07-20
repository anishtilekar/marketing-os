"""The Evaluation Framework's single reusable entrypoint.

:func:`evaluate_agent_output` is called once per agent node, immediately
after ``await agent.execute(payload)`` (see ``orchestration/nodes/nodes.py``).
It is observational: an unregistered agent or an internal scoring error
never raises and never blocks the pipeline, it just yields a low-confidence
report — the same "always record what was inspected" spirit as
:class:`~marketingos.agents.qa.QAAgent`'s check harness.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel

from marketingos.evaluation.models import (
    ArtifactKind,
    ArtifactRecord,
    EvaluationCheck,
    EvaluationReport,
    ValidationIssue,
    ValidationResult,
)
from marketingos.evaluation.registry import get_spec
from marketingos.evaluation.scorer import score_output
from marketingos.evaluation.validator import validate_structure
from marketingos.models.run import RunSection

if TYPE_CHECKING:
    from marketingos.services.run_manager import RunManager

__all__ = ["evaluate_agent_output", "persist_evaluation_report"]

#: Field names on nested models that, when populated with a non-blank
#: string, are treated as a generation prompt artifact.
_PROMPT_FIELD_NAMES = frozenset({"prompt", "prompt_used", "script"})

#: Field names that, together with a sibling ``media_type``, are treated as
#: a rendered image/video artifact (e.g. ``GeneratedImageRef.uri``,
#: ``GeneratedVideoRef.uri``, ``Creative.asset_path``).
_ASSET_LOCATION_FIELD_NAMES = frozenset({"uri", "asset_path"})

_MAX_ARTIFACT_WALK_DEPTH = 6


def _artifact_kind_for_asset(model: BaseModel) -> ArtifactKind:
    media_type = getattr(model, "media_type", None)
    if isinstance(media_type, str) and media_type.startswith("video/"):
        return ArtifactKind.VIDEO
    asset_format = getattr(model, "asset_format", None)
    if asset_format is not None and str(asset_format).lower() == "mp4":
        return ArtifactKind.VIDEO
    return ArtifactKind.IMAGE


def _walk_artifacts(
    value: object, *, path: str, depth: int, out: list[ArtifactRecord]
) -> None:
    if depth > _MAX_ARTIFACT_WALK_DEPTH:
        return
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            field_value = getattr(value, name)
            field_path = f"{path}.{name}" if path else name
            if isinstance(field_value, str) and field_value.strip():
                if name in _PROMPT_FIELD_NAMES:
                    out.append(
                        ArtifactRecord(
                            kind=ArtifactKind.PROMPT,
                            field_path=field_path,
                            value=field_value,
                        )
                    )
                    continue
                if name in _ASSET_LOCATION_FIELD_NAMES:
                    out.append(
                        ArtifactRecord(
                            kind=_artifact_kind_for_asset(value),
                            field_path=field_path,
                            value=field_value,
                        )
                    )
                    continue
            _walk_artifacts(field_value, path=field_path, depth=depth + 1, out=out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_artifacts(item, path=f"{path}[{index}]", depth=depth + 1, out=out)


def _collect_artifacts(output: BaseModel) -> tuple[ArtifactRecord, ...]:
    records: list[ArtifactRecord] = [
        ArtifactRecord(
            kind=ArtifactKind.JSON,
            field_path="$",
            value=json.dumps(output.model_dump(mode="json"), indent=2),
            description=f"Full {type(output).__name__} output.",
        )
    ]
    _walk_artifacts(output, path="", depth=0, out=records)
    return tuple(records)


def _unregistered_agent_report(
    *, agent_name: str, output: BaseModel, run_id: str
) -> EvaluationReport:
    issue = ValidationIssue(
        field="$",
        code="no_spec_registered",
        message=f"No EvaluationSpec is registered for agent {agent_name!r}.",
    )
    check = EvaluationCheck(
        name="spec_registered", weight=0, earned=0, detail="no spec registered"
    )
    return EvaluationReport(
        agent_name=agent_name,
        run_id=run_id,
        input_schema="unknown",
        output_schema=type(output).__name__,
        validation=ValidationResult(passed=False, issues=(issue,)),
        score=0,
        checks=(check,),
        artifacts=_collect_artifacts(output),
        created_at=datetime.now(UTC),
    )


def evaluate_agent_output(
    *, agent_name: str, output: BaseModel, run_id: str
) -> EvaluationReport:
    """Evaluate one agent's output and return its report.

    Never raises: an unregistered agent yields a zero-score report rather
    than an exception, so a missing spec can never break the pipeline this
    step is wired into.

    Args:
        agent_name: The executing agent's logical name (``agent.name``).
        output: The agent's typed output instance.
        run_id: Identifier of the execution being evaluated.

    Returns:
        The resulting :class:`EvaluationReport`.
    """
    spec = get_spec(agent_name)
    if spec is None:
        return _unregistered_agent_report(
            agent_name=agent_name, output=output, run_id=run_id
        )

    validation = validate_structure(output, spec)
    score, checks = score_output(output, spec, validation)
    return EvaluationReport(
        agent_name=agent_name,
        run_id=run_id,
        input_schema=spec.input_model.__name__,
        output_schema=type(output).__name__,
        validation=validation,
        score=score,
        checks=checks,
        artifacts=_collect_artifacts(output),
        created_at=datetime.now(UTC),
    )


def persist_evaluation_report(
    run_manager: RunManager, run_id: UUID, report: EvaluationReport
) -> None:
    """Write ``report`` to ``{run_dir}/eval/{agent_name}_eval.json``.

    Mirrors the QA-report persistence pattern in
    ``orchestration/nodes/nodes.py``'s ``qa_node``.
    """
    eval_dir = run_manager.section_dir(run_id, RunSection.EVALUATION)
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{report.agent_name}_eval.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
