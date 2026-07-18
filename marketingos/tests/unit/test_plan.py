from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from marketingos.models.plan import (
    ContentStatus,
    ContentType,
    Platform,
    PostItem,
    VideoItem,
    WeekPlan,
)


def make_post(day_number: int, **overrides):
    defaults = dict(
        day_number=day_number,
        title=f"Post Day {day_number}",
        description="A great post about our product.",
        platform=Platform.INSTAGRAM,
        objective="Increase engagement",
        caption="Check out our new feature!",
        hashtags=["#launch"],
    )
    defaults.update(overrides)
    return PostItem(**defaults)


def make_video(day_number: int, **overrides):
    defaults = dict(
        day_number=day_number,
        title=f"Video Day {day_number}",
        description="A short promotional video.",
        platform=Platform.YOUTUBE,
        objective="Drive awareness",
        script_outline="Intro, feature demo, call to action.",
        duration_seconds=30,
    )
    defaults.update(overrides)
    return VideoItem(**defaults)


def make_valid_week_plan(**overrides):
    posts = [make_post(day) for day in (1, 2, 3, 4, 5)]
    videos = [make_video(6), make_video(7)]
    defaults = dict(
        week_start_date=date(2026, 7, 20),
        posts=posts,
        videos=videos,
    )
    defaults.update(overrides)
    return WeekPlan(**defaults)


def test_post_item_defaults():
    post = make_post(1)
    assert post.content_type == ContentType.POST
    assert post.status == ContentStatus.DRAFT
    assert post.metadata == {}


def test_video_item_defaults():
    video = make_video(6)
    assert video.content_type == ContentType.VIDEO
    assert video.status == ContentStatus.DRAFT


def test_content_item_day_number_bounds():
    with pytest.raises(ValidationError):
        make_post(0)
    with pytest.raises(ValidationError):
        make_post(8)


def test_content_item_title_and_description_cannot_be_blank():
    with pytest.raises(ValidationError):
        make_post(1, title="   ")
    with pytest.raises(ValidationError):
        make_post(1, description="")


def test_post_item_caption_cannot_be_blank():
    with pytest.raises(ValidationError):
        make_post(1, caption="   ")


def test_post_item_hashtags_blank_entries_removed():
    post = make_post(1, hashtags=["#a", "  ", "#b"])
    assert post.hashtags == ["#a", "#b"]


def test_video_item_duration_must_be_positive():
    with pytest.raises(ValidationError):
        make_video(6, duration_seconds=0)
    with pytest.raises(ValidationError):
        make_video(6, duration_seconds=-10)


def test_video_item_script_outline_cannot_be_blank():
    with pytest.raises(ValidationError):
        make_video(6, script_outline="   ")


def test_week_plan_valid_construction():
    plan = make_valid_week_plan()
    assert len(plan.posts) == 5
    assert len(plan.videos) == 2
    assert plan.created_at.tzinfo is not None


def test_week_plan_requires_exactly_five_posts():
    posts = [make_post(day) for day in (1, 2, 3, 4)]
    videos = [make_video(6), make_video(7)]
    with pytest.raises(ValidationError):
        WeekPlan(week_start_date=date(2026, 7, 20), posts=posts, videos=videos)


def test_week_plan_requires_exactly_two_videos():
    posts = [make_post(day) for day in (1, 2, 3, 4, 5)]
    videos = [make_video(6)]
    with pytest.raises(ValidationError):
        WeekPlan(week_start_date=date(2026, 7, 20), posts=posts, videos=videos)


def test_week_plan_rejects_duplicate_ids():
    posts = [make_post(day) for day in (1, 2, 3, 4, 5)]
    videos = [make_video(6), make_video(7)]
    videos[1] = videos[1].model_copy(update={"id": videos[0].id})
    with pytest.raises(ValidationError):
        WeekPlan(week_start_date=date(2026, 7, 20), posts=posts, videos=videos)


def test_week_plan_requires_full_day_coverage():
    # Duplicate day 1 twice, skip day 5 entirely.
    posts = [make_post(1), make_post(1), make_post(2), make_post(3), make_post(4)]
    videos = [make_video(6), make_video(7)]
    with pytest.raises(ValidationError):
        WeekPlan(week_start_date=date(2026, 7, 20), posts=posts, videos=videos)


def test_week_plan_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        make_valid_week_plan(unexpected_field="oops")


def test_week_plan_serializes_to_json():
    plan = make_valid_week_plan()
    payload = plan.model_dump_json()
    assert isinstance(payload, str)
    assert "week_start_date" in payload


@pytest.mark.parametrize("platform", list(Platform))
def test_content_item_accepts_all_platforms(platform):
    post = make_post(1, platform=platform)
    assert post.platform == platform