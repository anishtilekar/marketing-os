from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CreativeType(str, Enum):
    """Kind of marketing asset a Creative represents."""

    POST_IMAGE = "post_image"
    VIDEO = "video"


class CreativeStatus(str, Enum):
    """
    Lifecycle state of a Creative.

    Enforced transition order:
        DRAFT -> GENERATED -> QA_REVIEW -> QA_PASSED -> APPROVED
                                        -> REJECTED

    REJECTED may be reached from QA_REVIEW or QA_PASSED (a
    previously-passed creative can still be rejected on further
    review). Terminal states are APPROVED and REJECTED.
    """

    DRAFT = "draft"
    GENERATED = "generated"
    QA_REVIEW = "qa_review"
    QA_PASSED = "qa_passed"
    APPROVED = "approved"
    REJECTED = "rejected"


class AssetFormat(str, Enum):
    """File format of the generated artifact."""

    PNG = "png"
    JPG = "jpg"
    MP4 = "mp4"


_IMAGE_FORMATS = frozenset({AssetFormat.PNG, AssetFormat.JPG})
_VIDEO_FORMATS = frozenset({AssetFormat.MP4})

_ALLOWED_STATUS_TRANSITIONS: dict[CreativeStatus, frozenset[CreativeStatus]] = {
    CreativeStatus.DRAFT: frozenset({CreativeStatus.GENERATED}),
    CreativeStatus.GENERATED: frozenset({CreativeStatus.QA_REVIEW}),
    CreativeStatus.QA_REVIEW: frozenset(
        {CreativeStatus.QA_PASSED, CreativeStatus.REJECTED}
    ),
    CreativeStatus.QA_PASSED: frozenset(
        {CreativeStatus.APPROVED, CreativeStatus.REJECTED}
    ),
    CreativeStatus.APPROVED: frozenset(),
    CreativeStatus.REJECTED: frozenset(),
}


class Creative(BaseModel):
    """
    A single generated marketing asset (post image or video) tracked
    through the QA and approval lifecycle.

    This is a pure domain model: it holds no agent, storage, or
    generation logic. Status transitions are validated structurally
    so that impossible states (e.g. an APPROVED creative with no
    asset_path) cannot be constructed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)

    run_id: UUID = Field(
        ...,
        description="Identifier of the MarketingOS execution run this creative belongs to.",
    )

    creative_type: CreativeType = Field(
        ...,
        description="Whether this creative is a post image or a video.",
    )

    status: CreativeStatus = Field(
        default=CreativeStatus.DRAFT,
        description="Current lifecycle status of this creative.",
    )

    title: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Human-readable name of the creative.",
    )

    description: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional explanation of the creative's intent.",
    )

    asset_path: str | None = Field(
        default=None,
        description="Path or location of the generated artifact, once produced.",
    )

    asset_format: AssetFormat = Field(
        ...,
        description="File format of the generated artifact.",
    )

    prompt_used: str | None = Field(
        default=None,
        max_length=10000,
        description="Generation prompt used to produce this asset, for reproducibility.",
    )

    template_used: str | None = Field(
        default=None,
        max_length=500,
        description="Design template identifier used, if any.",
    )

    source_plan_item_id: UUID | None = Field(
        default=None,
        description="Identifier of the plan item (post/video slot) this creative fulfills.",
    )

    qa_feedback: str | None = Field(
        default=None,
        max_length=5000,
        description="Feedback recorded by the QA Agent, particularly on rejection.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this creative record was created.",
    )

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this creative record was last updated.",
    )

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("title cannot be empty or whitespace only.")
        return stripped

    @field_validator("asset_path")
    @classmethod
    def asset_path_not_blank_if_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("asset_path cannot be an empty string if provided.")
        return value

    @field_validator("prompt_used", "template_used", "qa_feedback")
    @classmethod
    def optional_text_not_blank_if_provided(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Field cannot be an empty string if provided.")
        return value

    @field_validator("created_at", "updated_at")
    @classmethod
    def ensure_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @model_validator(mode="after")
    def validate_format_matches_type(self) -> Creative:
        """Ensures image creatives use image formats and video creatives use video formats."""
        if self.creative_type == CreativeType.VIDEO and self.asset_format not in _VIDEO_FORMATS:
            raise ValueError(
                f"Video creatives must use a video-compatible format; got {self.asset_format}."
            )
        if (
            self.creative_type == CreativeType.POST_IMAGE
            and self.asset_format not in _IMAGE_FORMATS
        ):
            raise ValueError(
                f"Image creatives must use an image-compatible format; got {self.asset_format}."
            )
        return self

    @model_validator(mode="after")
    def validate_approved_state(self) -> Creative:
        """An APPROVED creative must have a completed asset and asset_path."""
        if self.status == CreativeStatus.APPROVED and not self.asset_path:
            raise ValueError("APPROVED creatives must have an asset_path present.")
        return self

    @model_validator(mode="after")
    def validate_rejected_state_has_feedback(self) -> Creative:
        """A REJECTED creative should record why, so QA decisions remain auditable."""
        if self.status == CreativeStatus.REJECTED and not self.qa_feedback:
            raise ValueError("REJECTED creatives must include qa_feedback.")
        return self

    def with_status(
        self, new_status: CreativeStatus, *, qa_feedback: str | None = None
    ) -> Creative:
        """
        Returns a copy of this Creative transitioned to new_status,
        enforcing the allowed lifecycle transitions. Does not mutate
        the original instance.
        """
        allowed = _ALLOWED_STATUS_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid status transition: {self.status} -> {new_status}."
            )

        update: dict[str, object] = {
            "status": new_status,
            "updated_at": datetime.now(UTC),
        }
        if qa_feedback is not None:
            update["qa_feedback"] = qa_feedback

        return self.model_copy(update=update)


class PostCreative(Creative):
    """A :class:`Creative` narrowed to the post-image type.

    Exists so orchestration state can type its post-image and video
    collections separately (``MarketingState.post_creatives`` vs
    ``.video_creatives``) while sharing all of ``Creative``'s validation.
    """

    creative_type: CreativeType = Field(default=CreativeType.POST_IMAGE, frozen=True)


class VideoCreative(Creative):
    """A :class:`Creative` narrowed to the video type.

    Exists so orchestration state can type its post-image and video
    collections separately (``MarketingState.post_creatives`` vs
    ``.video_creatives``) while sharing all of ``Creative``'s validation.
    """

    creative_type: CreativeType = Field(default=CreativeType.VIDEO, frozen=True)
