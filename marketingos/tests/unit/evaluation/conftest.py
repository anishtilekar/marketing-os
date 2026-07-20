"""Fixture builders for evaluation-framework tests.

These construct real, fully-valid agent output models (not mocks) so tests
exercise the evaluation framework exactly as it will see production data.
"""

from __future__ import annotations

from datetime import UTC, datetime, time

from marketingos.agents.copywriter import Caption, CaptionPackage
from marketingos.agents.planner import ContentFormat, PlannedContent, Platform, WeekPlan
from marketingos.agents.research import (
    FactCategory,
    ObservedFact,
    ResearchResult,
    SourceMetadata,
    SourceType,
)


def make_week_plan(*, run_id: str = "plan-run-1") -> WeekPlan:
    posts = [
        PlannedContent(
            id=f"C{day}",
            day=day,
            publish_time=time(9, 0),
            format=ContentFormat.POST,
            platform=Platform.INSTAGRAM,
            topic=f"topic {day}",
            objective=f"objective {day}",
            call_to_action="Learn more",
            content_pillar="Community",
        )
        for day in range(1, 6)
    ]
    videos = [
        PlannedContent(
            id=f"C{day}",
            day=day,
            publish_time=time(9, 0),
            format=ContentFormat.SHORT_FORM_VIDEO,
            platform=Platform.TIKTOK,
            topic=f"video topic {day}",
            objective=f"video objective {day}",
            call_to_action="Watch now",
            content_pillar="Community",
        )
        for day in range(6, 8)
    ]
    return WeekPlan(
        run_id=run_id,
        source_strategy_run_id="strategy-run-1",
        subject="Acme Coffee",
        items=tuple(posts + videos),
        created_at=datetime.now(UTC),
    )


def make_caption_package(*, plan: WeekPlan) -> CaptionPackage:
    captions = tuple(
        Caption(
            item_id=item.id,
            format=item.format,
            headline=f"Headline {item.id}",
            caption=f"Caption body for {item.id}",
            call_to_action=item.call_to_action,
            hashtags=("#coffee",),
            tone="warm",
            keywords=("coffee",),
        )
        for item in plan.items
    )
    return CaptionPackage(
        run_id="caption-run-1",
        source_plan_run_id=plan.run_id,
        subject=plan.subject,
        brand_voice="Warm and inviting",
        captions=captions,
        created_at=datetime.now(UTC),
    )


def make_research_result(*, run_id: str = "research-run-1") -> ResearchResult:
    source = SourceMetadata(
        source_type=SourceType.WEBSITE,
        tool_name="website_scraper",
        url="https://example.com",
        retrieved_at=datetime.now(UTC),
    )
    fact = ObservedFact(
        category=FactCategory.ABOUT,
        statement="The business sells coffee.",
        source=source,
        confidence=0.9,
    )
    return ResearchResult(
        run_id=run_id,
        subject="Acme Coffee",
        facts=(fact,),
        sources=(source,),
        urls_visited=("https://example.com",),
        collected_at=datetime.now(UTC),
        confidence_score=0.9,
    )
