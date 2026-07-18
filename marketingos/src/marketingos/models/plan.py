from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContentType(str, Enum):
    """Kind of content item within a weekly plan."""

    POST = "post"
    VIDEO = "video"


class Platform(str, Enum):
    """Target distribution platform for a content item."""

    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"


class ContentStatus(str, Enum):
    """Review status of a planned content item."""

    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class ContentItem(BaseModel):
    """
    Base representation of a single planned content item within a
    7-day marketing calendar. Concrete content (posts, videos) is
    represented by subclasses that pin content_type to a fixed value.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)

    day_number: int = Field(
        ...,
        ge=1,
        le=7,
        description="Day of the week this content item is scheduled for (1-7).",
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Human-readable title of the content item.",
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Description of the content item's purpose and content.",
    )

    content_type: ContentType = Field(
        ...,
        description="Whether this item is a post or a video.",
    )

    platform: Platform = Field(
        ...,
        description="Target distribution platform for this content item.",
    )

    objective: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Marketing objective this content item serves.",
    )

    status: ContentStatus = Field(
        default=ContentStatus.DRAFT,
        description="Current review status of this content item.",
    )

    target_audience: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional description of the intended audience for this item.",
    )

    metadata: dict[str, Any] = Field(  # type: ignore[assignment]
        default_factory=dict,
        description="Optional additional structured metadata for extensibility.",
    )

    @field_validator("title", "description", "objective")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be empty or whitespace only.")
        return stripped

    @field_validator("target_audience")
    @classmethod
    def target_audience_not_blank_if_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("target_audience cannot be an empty string if provided.")
        return value


class PostItem(ContentItem):
    """A single social media post within the weekly plan."""

    content_type: Literal[ContentType.POST] = Field(  # type: ignore[override]
        default=ContentType.POST,
        description="Fixed to POST for this content item type.",
    )

    caption: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Full caption text for the post.",
    )

    hashtags: list[str] = Field(  # type: ignore[assignment]
        default_factory=list,
        description="Hashtags associated with the post.",
    )

    call_to_action: str | None = Field(
        default=None,
        max_length=300,
        description="Optional call-to-action text for the post.",
    )

    @field_validator("caption")
    @classmethod
    def caption_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("caption cannot be empty or whitespace only.")
        return stripped

    @field_validator("hashtags")
    @classmethod
    def hashtags_not_blank(cls, value: list[str]) -> list[str]:
        cleaned = [tag.strip() for tag in value if tag.strip()]
        return cleaned

    @field_validator("call_to_action")
    @classmethod
    def cta_not_blank_if_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("call_to_action cannot be an empty string if provided.")
        return value


class VideoItem(ContentItem):
    """A single short video within the weekly plan."""

    content_type: Literal[ContentType.VIDEO] = Field(  # type: ignore[override]
        default=ContentType.VIDEO,
        description="Fixed to VIDEO for this content item type.",
    )

    script_outline: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Outline of the video's script/narrative structure.",
    )

    duration_seconds: int = Field(
        ...,
        gt=0,
        description="Planned duration of the video, in seconds. Must be positive.",
    )

    visual_direction: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional guidance on visual style/composition.",
    )

    audio_direction: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional guidance on audio/music/voiceover.",
    )

    @field_validator("script_outline")
    @classmethod
    def script_outline_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("script_outline cannot be empty or whitespace only.")
        return stripped

    @field_validator("visual_direction", "audio_direction")
    @classmethod
    def optional_text_not_blank_if_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Field cannot be an empty string if provided.")
        return value


class WeekPlan(BaseModel):
    """
    A complete 7-day marketing content plan produced by the Planner
    Agent, consisting of exactly 5 posts and 2 videos distributed
    across days 1-7.

    All hard constraints (item counts, day coverage, uniqueness) are
    enforced structurally via validators rather than relying on
    prompt instructions alone.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)

    week_start_date: date = Field(
        ...,
        description="Calendar date on which this weekly plan begins (day 1).",
    )

    posts: list[PostItem] = Field(
        ...,
        description="Exactly 5 planned social media posts.",
    )

    videos: list[VideoItem] = Field(
        ...,
        description="Exactly 2 planned short videos.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this weekly plan was created.",
    )

    @field_validator("posts")
    @classmethod
    def exactly_five_posts(cls, value: list[PostItem]) -> list[PostItem]:
        if len(value) != 5:
            raise ValueError(f"WeekPlan must contain exactly 5 posts; got {len(value)}.")
        return value

    @field_validator("videos")
    @classmethod
    def exactly_two_videos(cls, value: list[VideoItem]) -> list[VideoItem]:
        if len(value) != 2:
            raise ValueError(f"WeekPlan must contain exactly 2 videos; got {len(value)}.")
        return value

    @field_validator("created_at")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @model_validator(mode="after")
    def validate_total_item_count(self) -> WeekPlan:
        total = len(self.posts) + len(self.videos)
        if total != 7:
            raise ValueError(f"WeekPlan must contain exactly 7 content items total; got {total}.")
        return self

    @model_validator(mode="after")
    def validate_unique_ids(self) -> WeekPlan:
        all_items: list[ContentItem] = [*self.posts, *self.videos]
        ids = [item.id for item in all_items]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate content item IDs found within WeekPlan.")
        return self

    @model_validator(mode="after")
    def validate_day_coverage(self) -> WeekPlan:
        all_items: list[ContentItem] = [*self.posts, *self.videos]
        days_present = {item.day_number for item in all_items}
        missing_days = sorted(set(range(1, 8)) - days_present)
        if missing_days:
            raise ValueError(
                f"Every day from 1-7 must have at least one content item; "
                f"missing days: {missing_days}."
            )
        return self
