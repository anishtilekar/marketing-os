"""Evaluation specs for every MarketingOS agent.

One entry per agent, keyed by the agent's logical name (``agent.name``,
which defaults to the class name — see
:meth:`marketingos.agents.base.BaseAgent.name`). Each spec adds a handful of
completeness/domain rules on top of the generic structural sweep every
output already gets for free.
"""

from __future__ import annotations

from marketingos.agents.business_analysis import BusinessAnalysisAgent
from marketingos.agents.copywriter import CaptionPackage, CopywriterAgent
from marketingos.agents.designer import CreativePackage, DesignBrief, DesignerAgent
from marketingos.agents.packaging import (
    CampaignPackage,
    PackagingAgent,
    PackagingRequest,
)
from marketingos.agents.planner import PlannerAgent, WeekPlan
from marketingos.agents.qa import CampaignBundle, QAAgent, QAReport
from marketingos.agents.research import ResearchAgent, ResearchInput, ResearchResult
from marketingos.agents.strategist import StrategistAgent, Strategy
from marketingos.agents.synthetic_resource import (
    SyntheticSourceAgent,
    SyntheticSourceMaterial,
)
from marketingos.agents.video_director import (
    VideoBrief,
    VideoDirectorAgent,
    VideoPackage,
)
from marketingos.evaluation.models import ValidationIssue
from marketingos.evaluation.spec import EvaluationSpec
from marketingos.models.business_context import BusinessContext

__all__ = ["EVALUATION_SPECS", "get_spec"]


def _week_plan_has_items(output: object) -> bool:
    assert isinstance(output, WeekPlan)
    return len(output.items) > 0


def _captions_cover_every_item(output: object) -> list[ValidationIssue]:
    assert isinstance(output, CaptionPackage)
    empty = [c.item_id for c in output.captions if not c.caption.strip()]
    if not empty:
        return []
    return [
        ValidationIssue(
            field="captions",
            code="blank_caption_body",
            message=f"Items with a blank caption body: {empty}",
        )
    ]


def _creatives_have_asset_uris(output: object) -> list[ValidationIssue]:
    assert isinstance(output, CreativePackage)
    missing = [c.item_id for c in output.creatives if not c.asset.uri.strip()]
    if not missing:
        return []
    return [
        ValidationIssue(
            field="creatives",
            code="missing_asset_uri",
            message=f"Creatives with no rendered asset uri: {missing}",
        )
    ]


def _videos_have_asset_uris(output: object) -> list[ValidationIssue]:
    assert isinstance(output, VideoPackage)
    missing = [v.item_id for v in output.videos if not v.asset.uri.strip()]
    if not missing:
        return []
    return [
        ValidationIssue(
            field="videos",
            code="missing_asset_uri",
            message=f"Videos with no rendered asset uri: {missing}",
        )
    ]


def _research_has_facts(output: object) -> bool:
    assert isinstance(output, ResearchResult)
    return len(output.facts) > 0


def _research_confidence_reasonable(output: object) -> list[ValidationIssue]:
    assert isinstance(output, ResearchResult)
    if output.confidence_score <= 0.0:
        return [
            ValidationIssue(
                field="confidence_score",
                code="zero_confidence",
                message="Research collected no confident observations.",
            )
        ]
    return []


def _synthetic_material_has_content(output: object) -> bool:
    assert isinstance(output, SyntheticSourceMaterial)
    return bool(output.facts or output.descriptions or output.brand_characteristics)


def _business_context_has_assumptions_or_facts(output: object) -> bool:
    assert isinstance(output, BusinessContext)
    return bool(output.observed_facts or output.assumptions)


def _strategy_has_grounded_pillars(output: object) -> list[ValidationIssue]:
    assert isinstance(output, Strategy)
    ungrounded = [p.name for p in output.content_pillars if not p.grounded_in]
    if not ungrounded:
        return []
    return [
        ValidationIssue(
            field="content_pillars",
            code="ungrounded_pillar",
            message=f"Content pillars with no grounding citations: {ungrounded}",
        )
    ]


def _qa_report_has_checks(output: object) -> bool:
    assert isinstance(output, QAReport)
    return len(output.checks) > 0


def _package_has_asset_index(output: object) -> bool:
    assert isinstance(output, CampaignPackage)
    return len(output.asset_index) > 0


EVALUATION_SPECS: dict[str, EvaluationSpec] = {
    ResearchAgent.__name__: EvaluationSpec(
        agent_name=ResearchAgent.__name__,
        input_model=ResearchInput,
        output_model=ResearchResult,
        completeness_rules=(("has_facts", _research_has_facts),),
        domain_rules=(_research_confidence_reasonable,),
    ),
    SyntheticSourceAgent.__name__: EvaluationSpec(
        agent_name=SyntheticSourceAgent.__name__,
        input_model=ResearchResult,
        output_model=SyntheticSourceMaterial,
        completeness_rules=(("has_content", _synthetic_material_has_content),),
    ),
    BusinessAnalysisAgent.__name__: EvaluationSpec(
        agent_name=BusinessAnalysisAgent.__name__,
        input_model=ResearchResult,
        output_model=BusinessContext,
        completeness_rules=(
            ("has_facts_or_assumptions", _business_context_has_assumptions_or_facts),
        ),
    ),
    StrategistAgent.__name__: EvaluationSpec(
        agent_name=StrategistAgent.__name__,
        input_model=BusinessContext,
        output_model=Strategy,
        domain_rules=(_strategy_has_grounded_pillars,),
    ),
    PlannerAgent.__name__: EvaluationSpec(
        agent_name=PlannerAgent.__name__,
        input_model=Strategy,
        output_model=WeekPlan,
        completeness_rules=(("has_items", _week_plan_has_items),),
    ),
    CopywriterAgent.__name__: EvaluationSpec(
        agent_name=CopywriterAgent.__name__,
        input_model=WeekPlan,
        output_model=CaptionPackage,
        domain_rules=(_captions_cover_every_item,),
    ),
    DesignerAgent.__name__: EvaluationSpec(
        agent_name=DesignerAgent.__name__,
        input_model=DesignBrief,
        output_model=CreativePackage,
        domain_rules=(_creatives_have_asset_uris,),
    ),
    VideoDirectorAgent.__name__: EvaluationSpec(
        agent_name=VideoDirectorAgent.__name__,
        input_model=VideoBrief,
        output_model=VideoPackage,
        domain_rules=(_videos_have_asset_uris,),
    ),
    QAAgent.__name__: EvaluationSpec(
        agent_name=QAAgent.__name__,
        input_model=CampaignBundle,
        output_model=QAReport,
        completeness_rules=(("has_checks", _qa_report_has_checks),),
    ),
    PackagingAgent.__name__: EvaluationSpec(
        agent_name=PackagingAgent.__name__,
        input_model=PackagingRequest,
        output_model=CampaignPackage,
        completeness_rules=(("has_asset_index", _package_has_asset_index),),
    ),
}


def get_spec(agent_name: str) -> EvaluationSpec | None:
    """Return the registered :class:`EvaluationSpec` for ``agent_name``, if any."""
    return EVALUATION_SPECS.get(agent_name)
