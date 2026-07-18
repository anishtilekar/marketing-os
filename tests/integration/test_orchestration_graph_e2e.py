from __future__ import annotations

import json
from decimal import Decimal

import pytest
from langgraph.graph import END

from marketingos.agents.designer import GeneratedImageRef
from marketingos.agents.packaging import Checksum, PackageArchiveRef, StagedFile
from marketingos.agents.qa import BudgetSnapshot
from marketingos.agents.research import (
    ContactDetails,
    InstagramProfileSnapshot,
    WebsiteSnapshot,
)
from marketingos.agents.video_director import GeneratedVideoRef
from marketingos.models.cost import CostLedger
from marketingos.orchestration.graph import GraphBuilder
from marketingos.orchestration.nodes import (
    make_business_analysis_node,
    make_copywriter_node,
    make_creative_node,
    make_packaging_node,
    make_planner_node,
    make_qa_node,
    make_research_node,
    make_strategist_node,
    make_synthetic_resource_node,
    make_video_director_node,
)
from marketingos.orchestration.state import BudgetState, MarketingState

_PILLAR = "Pillar"


# ---- port mocks ------------------------------------------------------------
class MockWebsiteScraper:
    async def scrape(self, url: str) -> WebsiteSnapshot:
        return WebsiteSnapshot(
            url=url,
            title="Example Coffee Roasters",
            tagline="Small-batch craft coffee",
            about_text="A neighbourhood roastery serving single-origin beans.",
            products_services=("Single-origin beans", "Subscriptions"),
            contact=ContactDetails(emails=("hello@example.com",)),
        )


class MockInstagramReader:
    async def fetch_profile(self, username: str) -> InstagramProfileSnapshot:
        return InstagramProfileSnapshot(
            username=username, profile_url=f"https://instagram.com/{username}"
        )


class MockSearchTool:
    async def search(self, query: str, *, max_results: int):
        return ()


class MockLanguageModel:
    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        data = json.loads(user_prompt)
        if "approved_asset_ids" in data:
            return json.dumps(_video_direction())
        if "content_pillars" in data:
            return json.dumps(_week_plan(data["content_pillars"][0]["name"]))
        if "items" in data:
            return json.dumps(_captions([item["item_id"] for item in data["items"]]))
        if "assumptions" in data:
            return json.dumps(_strategy([f["id"] for f in data["facts"]]))
        return json.dumps({"assumptions": []})


class MockImageGenerator:
    def __init__(self) -> None:
        self._n = 0

    async def generate(
        self, *, prompt: str, negative_prompt: str, width: int, height: int
    ) -> GeneratedImageRef:
        self._n += 1
        return GeneratedImageRef(
            asset_id=f"img-{self._n}",
            uri="data://test-image",
            width=width,
            height=height,
            media_type="image/png",
        )


class MockVideoGenerator:
    def __init__(self) -> None:
        self._n = 0

    async def render(self, *, direction) -> GeneratedVideoRef:
        self._n += 1
        return GeneratedVideoRef(
            asset_id=f"vid-{self._n}", uri="data://test-video", media_type="video/mp4"
        )


class MockPackagingService:
    async def stage_asset(self, *, source_uri: str, target_path: str) -> StagedFile:
        return _staged(target_path, "application/octet-stream")

    async def stage_text(
        self, *, content: str, target_path: str, media_type: str
    ) -> StagedFile:
        return _staged(target_path, media_type)

    async def finalize(self, *, root_path: str) -> PackageArchiveRef:
        return PackageArchiveRef(
            uri=f"data://{root_path}.zip", size_bytes=1, checksum=_checksum()
        )


class MockBudgetLedger:
    async def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            total_spend=Decimal("0"), max_budget=Decimal("100"), currency="USD"
        )


# ---- mock payload builders -------------------------------------------------
def _checksum() -> Checksum:
    return Checksum(value="0" * 64)


def _staged(path: str, media_type: str) -> StagedFile:
    return StagedFile(
        path=path, size_bytes=1, checksum=_checksum(), media_type=media_type
    )


def _strategy(fact_ids: list[str]) -> dict:
    anchor = (fact_ids[0],)
    return {
        "goals": [{"statement": "Grow local awareness", "grounded_in": list(anchor)}],
        "target_audience": {
            "description": "Local coffee lovers",
            "needs": ["quality"],
            "grounded_in": list(anchor),
        },
        "positioning": "Craft roastery for the neighbourhood",
        "key_messages": ["Fresh small-batch coffee"],
        "content_pillars": [
            {
                "name": _PILLAR,
                "description": "Craft and origin stories",
                "grounded_in": list(anchor),
            }
        ],
        "success_metrics": [
            {"name": "Reach", "target": "1000", "measurement": "impressions"}
        ],
    }


def _week_plan(pillar: str) -> dict:
    items = []
    for day in range(1, 8):
        is_video = day >= 6
        items.append(
            {
                "id": f"C{day}",
                "day": day,
                "publish_time": "09:00",
                "format": "short_form_video" if is_video else "post",
                "platform": "instagram",
                "topic": f"Coffee topic for day {day}",
                "objective": "Engage the audience",
                "call_to_action": "Learn more today",
                "content_pillar": pillar,
                "depends_on": [],
            }
        )
    return {"items": items}


def _captions(item_ids: list[str]) -> dict:
    return {
        "brand_voice": "Warm and welcoming",
        "captions": [
            {
                "item_id": item_id,
                "headline": "Discover Our Craft Coffee",
                "caption": "Freshly roasted, small batch, served with care.",
                "call_to_action": "Visit us today",
                "hashtags": ["#coffee", "#craft"],
                "tone": "friendly",
                "keywords": ["coffee", "craft"],
            }
            for item_id in item_ids
        ],
    }


def _video_direction() -> dict:
    return {
        "script": "A short warm look at our roastery.",
        "scenes": [
            {
                "number": 1,
                "description": "Beans roasting close up",
                "voice_over": "Every batch is roasted with care.",
                "shots": [
                    {
                        "number": 1,
                        "description": "Close up of roasting beans",
                        "framing": "close-up",
                        "duration_seconds": 10.0,
                    }
                ],
            }
        ],
        "subtitles": [
            {"start_seconds": 0.0, "end_seconds": 10.0, "text": "Roasted with care."}
        ],
        "asset_references": [],
    }


@pytest.mark.asyncio
async def test_full_graph_execution():
    llm = MockLanguageModel()
    nodes = {
        "research": make_research_node(
            website_scraper=MockWebsiteScraper(),
            instagram_reader=MockInstagramReader(),
            search_tool=MockSearchTool(),
        ),
        "synthetic": make_synthetic_resource_node(),
        "business_analysis": make_business_analysis_node(llm=llm),
        "strategist": make_strategist_node(llm=llm),
        "planner": make_planner_node(llm=llm),
        "copywriter": make_copywriter_node(llm=llm),
        "creative": make_creative_node(image_generator=MockImageGenerator()),
        "video_director": make_video_director_node(
            llm=llm, video_generator=MockVideoGenerator()
        ),
        "qa": make_qa_node(budget_ledger=MockBudgetLedger()),
        "packaging": make_packaging_node(packaging_service=MockPackagingService()),
    }

    builder = GraphBuilder(entry_point="research").add_nodes(nodes)
    order = list(nodes)
    for source, target in zip(order, order[1:]):
        builder.add_edge(source, target)
    builder.add_edge("packaging", END)
    app = builder.compile()

    initial = MarketingState(
        workflow_id="wf-e2e",
        budget=BudgetState(cost_ledger=CostLedger()),
        source_pack={"website_url": "https://example.com"},
    )

    final = await app.ainvoke(initial)

    campaign_package = (
        final["campaign_package"]
        if isinstance(final, dict)
        else final.campaign_package
    )
    run_id = final["run_id"] if isinstance(final, dict) else final.run_id
    assert campaign_package is not None
    assert run_id is not None
