from __future__ import annotations

from typing import TYPE_CHECKING

from marketingos.agents.business_analysis import (
    BusinessAnalysisAgent,
    LanguageModelPort,
)
from marketingos.agents.copywriter import CopywriterAgent
from marketingos.agents.designer import DesignBrief, DesignerAgent, ImageGenerationPort
from marketingos.agents.packaging import (
    PackagingAgent,
    PackagingRequest,
    PackagingServicePort,
)
from marketingos.agents.planner import PlannerAgent
from marketingos.agents.qa import BudgetLedgerPort, CampaignBundle, QAAgent
from marketingos.agents.research import (
    BusinessSearchPort,
    InstagramReaderPort,
    ResearchAgent,
    ResearchInput,
    WebsiteScraperPort,
)
from marketingos.agents.strategist import StrategistAgent
from marketingos.agents.synthetic_resource import SyntheticSourceAgent
from marketingos.agents.video_director import (
    VideoBrief,
    VideoDirectorAgent,
    VideoGenerationPort,
)

from ..state import MarketingState

if TYPE_CHECKING:
    from ..graph import NodeAction


def make_research_node(
    *,
    website_scraper: WebsiteScraperPort | None = None,
    instagram_reader: InstagramReaderPort | None = None,
    search_tool: BusinessSearchPort | None = None,
) -> NodeAction:
    """Build the research node, backed by the given collection tools."""
    agent = ResearchAgent(
        website_scraper=website_scraper,
        instagram_reader=instagram_reader,
        search_tool=search_tool,
    )

    async def research_node(state: MarketingState) -> MarketingState:
        payload = ResearchInput(
            website_url=state.source_pack.get("website_url"),
            instagram_username=state.source_pack.get("instagram_username"),
            business_name=state.source_pack.get("business_name"),
        )
        state.research_result = await agent.execute(payload)
        return state

    return research_node


def make_synthetic_resource_node() -> NodeAction:
    """Build the synthetic-source node."""
    agent = SyntheticSourceAgent()

    async def synthetic_resource_node(state: MarketingState) -> MarketingState:
        assert state.research_result is not None, (
            "research_node must run before synthetic_resource_node"
        )
        state.synthetic_source = await agent.execute(state.research_result)
        return state

    return synthetic_resource_node


def make_business_analysis_node(*, llm: LanguageModelPort) -> NodeAction:
    """Build the business-analysis node, backed by the given LLM."""
    agent = BusinessAnalysisAgent(llm=llm)

    async def business_analysis_node(state: MarketingState) -> MarketingState:
        assert state.research_result is not None, (
            "research_node must run before business_analysis_node"
        )
        state.business_analysis = await agent.execute(state.research_result)
        return state

    return business_analysis_node


def make_strategist_node(*, llm: LanguageModelPort) -> NodeAction:
    """Build the strategist node, backed by the given LLM."""
    agent = StrategistAgent(llm=llm)

    async def strategist_node(state: MarketingState) -> MarketingState:
        assert state.business_analysis is not None, (
            "business_analysis_node must run before strategist_node"
        )
        state.strategy_output = await agent.execute(state.business_analysis)
        return state

    return strategist_node


def make_planner_node(*, llm: LanguageModelPort) -> NodeAction:
    """Build the planner node, backed by the given LLM."""
    agent = PlannerAgent(llm=llm)

    async def planner_node(state: MarketingState) -> MarketingState:
        assert state.strategy_output is not None, (
            "strategist_node must run before planner_node"
        )
        state.week_plan = await agent.execute(state.strategy_output)
        return state

    return planner_node


def make_copywriter_node(*, llm: LanguageModelPort) -> NodeAction:
    """Build the copywriter node, backed by the given LLM."""
    agent = CopywriterAgent(llm=llm)

    async def copywriter_node(state: MarketingState) -> MarketingState:
        assert state.week_plan is not None, "planner_node must run before copywriter_node"
        state.captions = await agent.execute(state.week_plan)
        return state

    return copywriter_node


def make_creative_node(*, image_generator: ImageGenerationPort) -> NodeAction:
    """Build the creative (design) node, backed by the given image generator."""
    agent = DesignerAgent(image_generator=image_generator)

    async def creative_node(state: MarketingState) -> MarketingState:
        assert state.week_plan is not None, "planner_node must run before creative_node"
        assert state.captions is not None, "copywriter_node must run before creative_node"
        payload = DesignBrief(week_plan=state.week_plan, captions=state.captions)
        state.creatives = await agent.execute(payload)
        return state

    return creative_node


def make_video_director_node(
    *, llm: LanguageModelPort, video_generator: VideoGenerationPort
) -> NodeAction:
    """Build the video-director node, backed by the given LLM and video generator."""
    agent = VideoDirectorAgent(llm=llm, video_generator=video_generator)

    async def video_director_node(state: MarketingState) -> MarketingState:
        assert state.week_plan is not None, "planner_node must run before video_director_node"
        assert state.captions is not None, (
            "copywriter_node must run before video_director_node"
        )
        assert state.creatives is not None, (
            "creative_node must run before video_director_node"
        )
        payload = VideoBrief(
            week_plan=state.week_plan, captions=state.captions, creatives=state.creatives
        )
        state.videos = await agent.execute(payload)
        return state

    return video_director_node


def make_qa_node(
    *, budget_ledger: BudgetLedgerPort | None = None, llm: LanguageModelPort | None = None
) -> NodeAction:
    """Build the QA node, backed by the given budget ledger and (optional) LLM."""
    agent = QAAgent(budget_ledger=budget_ledger, llm=llm)

    async def qa_node(state: MarketingState) -> MarketingState:
        assert state.business_analysis is not None, (
            "business_analysis_node must run before qa_node"
        )
        assert state.strategy_output is not None, "strategist_node must run before qa_node"
        assert state.week_plan is not None, "planner_node must run before qa_node"
        assert state.captions is not None, "copywriter_node must run before qa_node"
        assert state.creatives is not None, "creative_node must run before qa_node"
        assert state.videos is not None, "video_director_node must run before qa_node"
        payload = CampaignBundle(
            business_context=state.business_analysis,
            strategy=state.strategy_output,
            week_plan=state.week_plan,
            captions=state.captions,
            creatives=state.creatives,
            videos=state.videos,
        )
        state.qa_report = await agent.execute(payload)
        return state

    return qa_node


def make_packaging_node(*, packaging_service: PackagingServicePort) -> NodeAction:
    """Build the packaging node, backed by the given packaging service."""
    agent = PackagingAgent(packaging_service=packaging_service)

    async def packaging_node(state: MarketingState) -> MarketingState:
        assert state.business_analysis is not None, (
            "business_analysis_node must run before packaging_node"
        )
        assert state.strategy_output is not None, "strategist_node must run before packaging_node"
        assert state.week_plan is not None, "planner_node must run before packaging_node"
        assert state.captions is not None, "copywriter_node must run before packaging_node"
        assert state.creatives is not None, "creative_node must run before packaging_node"
        assert state.videos is not None, "video_director_node must run before packaging_node"
        assert state.qa_report is not None, "qa_node must run before packaging_node"
        bundle = CampaignBundle(
            business_context=state.business_analysis,
            strategy=state.strategy_output,
            week_plan=state.week_plan,
            captions=state.captions,
            creatives=state.creatives,
            videos=state.videos,
        )
        payload = PackagingRequest(bundle=bundle, qa_report=state.qa_report)
        state.campaign_package = await agent.execute(payload)
        return state

    return packaging_node
