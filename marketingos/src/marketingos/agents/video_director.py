"""VideoDirectorAgent — short-form video direction and rendering coordination.

The :class:`VideoDirectorAgent` converts a :class:`VideoBrief` (a validated
triple of :class:`~marketingos.agents.planner.WeekPlan`,
:class:`~marketingos.agents.copywriter.CaptionPackage` and
:class:`~marketingos.agents.designer.CreativePackage`) into a
:class:`VideoPackage`: exactly
:data:`~marketingos.agents.planner.REQUIRED_VIDEOS` short-form videos, each
with a script, storyboard (scene sequence with per-scene shots), shot list,
voice-over text, timed subtitles and asset references.

Scope and guarantees
--------------------
* **Direction, not rendering.** The injected language model drafts the
  direction; the injected :class:`VideoGenerationPort` tool renders it. The
  agent implements neither.
* **Validated structure.** The :class:`VideoDirection` model enforces
  sequential scene and shot numbering and ordered, non-overlapping
  subtitles; the agent enforces the configured duration budget; the
  :class:`VideoPackage` model enforces the exact video count. Violations
  are retryable failures, so the base agent regenerates.
* **Anchored to approved creative.** Every ``asset_references`` entry must
  name an asset that exists in the approved :class:`CreativePackage`, so
  the videos reuse the campaign's visual identity instead of inventing a
  new one.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from marketingos.agents.base import (
    AgentConfig,
    BaseAgent,
    MemoryStore,
    PromptRepository,
    RetryableAgentError,
    ToolRegistry,
)
from marketingos.agents.business_analysis import (
    LanguageModelPort,
    extract_json_object,
)
from marketingos.agents.copywriter import Caption, CaptionPackage
from marketingos.agents.designer import CreativePackage
from marketingos.agents.planner import (
    REQUIRED_VIDEOS,
    ContentFormat,
    PlannedContent,
    WeekPlan,
)

__all__ = [
    "GeneratedVideoRef",
    "MalformedDirectionError",
    "Scene",
    "Shot",
    "SubtitleLine",
    "UnknownAssetReferenceError",
    "VideoBrief",
    "VideoCreative",
    "VideoDirection",
    "VideoDirectorAgent",
    "VideoDirectorAgentConfig",
    "VideoGenerationPort",
    "VideoPackage",
]


# ---------------------------------------------------------------------------
# Direction schema
# ---------------------------------------------------------------------------


class Shot(BaseModel):
    """One shot within a scene."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    description: str = Field(min_length=1)
    framing: str = Field(
        min_length=1, description="Camera framing, e.g. 'close-up'."
    )
    duration_seconds: float = Field(gt=0.0, le=30.0)


class Scene(BaseModel):
    """One scene in the storyboard: purpose, voice-over and its shots."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    description: str = Field(min_length=1)
    voice_over: str = Field(min_length=1)
    shots: tuple[Shot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_shot_numbering(self) -> Scene:
        """Shots must be numbered sequentially from 1 within the scene."""
        numbers = [shot.number for shot in self.shots]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"scene {self.number}: shots must be numbered 1.."
                f"{len(numbers)} in order, got {numbers}"
            )
        return self


class SubtitleLine(BaseModel):
    """One timed subtitle line."""

    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_interval(self) -> SubtitleLine:
        """A subtitle must end after it starts."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("subtitle end_seconds must exceed start_seconds")
        return self


class VideoDirection(BaseModel):
    """The complete direction for one short-form video.

    The storyboard is the ordered ``scenes`` tuple (the scene sequence);
    the flattened ``shot_list`` and the concatenated ``voice_over_text``
    are derived views, kept consistent by construction.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(pattern=r"^C\d+$")
    script: str = Field(
        min_length=1, description="The full narrative script of the video."
    )
    scenes: tuple[Scene, ...] = Field(min_length=1)
    subtitles: tuple[SubtitleLine, ...] = Field(min_length=1)
    asset_references: tuple[str, ...] = Field(
        default=(),
        description="Asset ids from the approved CreativePackage reused "
        "in this video.",
    )

    @model_validator(mode="after")
    def _validate_structure(self) -> VideoDirection:
        """Enforce scene numbering and subtitle ordering."""
        numbers = [scene.number for scene in self.scenes]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"scenes must be numbered 1..{len(numbers)} in order, "
                f"got {numbers}"
            )
        previous_end = 0.0
        for line in self.subtitles:
            if line.start_seconds < previous_end:
                raise ValueError(
                    "subtitles must be ordered and non-overlapping; line "
                    f"starting at {line.start_seconds}s overlaps the "
                    "previous line"
                )
            previous_end = line.end_seconds
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def shot_list(self) -> tuple[Shot, ...]:
        """All shots across the scene sequence, in playback order."""
        return tuple(shot for scene in self.scenes for shot in scene.shots)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def voice_over_text(self) -> str:
        """The full voice-over, scene by scene."""
        return "\n".join(scene.voice_over for scene in self.scenes)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_duration_seconds(self) -> float:
        """Planned duration: the sum of every shot's duration."""
        return round(sum(shot.duration_seconds for shot in self.shot_list), 3)


# ---------------------------------------------------------------------------
# Video tool contract (port)
# ---------------------------------------------------------------------------


class GeneratedVideoRef(BaseModel):
    """Asset metadata returned by the video-generation tool."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    duration_seconds: float | None = Field(default=None, gt=0.0)
    media_type: str = Field(default="video/mp4", min_length=1)


@runtime_checkable
class VideoGenerationPort(Protocol):
    """Structural contract for the video-generation tool.

    Satisfied by the video client in ``marketingos.tools``. The agent
    depends only on this protocol, keeping the video provider swappable.
    """

    async def render(self, *, direction: VideoDirection) -> GeneratedVideoRef:
        """Render one video from its direction and return its reference."""
        ...


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class VideoBrief(BaseModel):
    """Typed input tying together plan, captions and approved creatives.

    Construction validates the chain: the captions must belong to the plan
    and the creatives must belong to both the plan and the captions.
    """

    model_config = ConfigDict(frozen=True)

    week_plan: WeekPlan
    captions: CaptionPackage
    creatives: CreativePackage

    @model_validator(mode="after")
    def _validate_chain(self) -> VideoBrief:
        """Reject briefs assembled from mismatched pipeline runs."""
        if self.captions.source_plan_run_id != self.week_plan.run_id:
            raise ValueError(
                "caption package belongs to plan run "
                f"{self.captions.source_plan_run_id!r}, not "
                f"{self.week_plan.run_id!r}"
            )
        if self.creatives.source_plan_run_id != self.week_plan.run_id:
            raise ValueError(
                "creative package belongs to plan run "
                f"{self.creatives.source_plan_run_id!r}, not "
                f"{self.week_plan.run_id!r}"
            )
        if self.creatives.source_caption_run_id != self.captions.run_id:
            raise ValueError(
                "creative package was designed for caption run "
                f"{self.creatives.source_caption_run_id!r}, not "
                f"{self.captions.run_id!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class VideoCreative(BaseModel):
    """One finished short-form video: direction plus rendered asset."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(pattern=r"^C\d+$")
    direction: VideoDirection
    asset: GeneratedVideoRef

    @model_validator(mode="after")
    def _validate_direction_binding(self) -> VideoCreative:
        """The direction must describe this creative's plan item."""
        if self.direction.item_id != self.item_id:
            raise ValueError(
                f"direction is for item {self.direction.item_id!r}, "
                f"creative is for {self.item_id!r}"
            )
        return self


class VideoPackage(BaseModel):
    """All short-form videos for one week plan.

    Construction enforces exactly
    :data:`~marketingos.agents.planner.REQUIRED_VIDEOS` videos with unique
    item ids.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    source_plan_run_id: str
    source_caption_run_id: str
    source_creative_run_id: str
    subject: str
    videos: tuple[VideoCreative, ...]
    created_at: datetime

    @model_validator(mode="after")
    def _validate_video_count(self) -> VideoPackage:
        """Enforce the exact video count and id uniqueness."""
        if len(self.videos) != REQUIRED_VIDEOS:
            raise ValueError(
                f"package must contain exactly {REQUIRED_VIDEOS} videos, "
                f"got {len(self.videos)}"
            )
        item_ids = [video.item_id for video in self.videos]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("video item ids must be unique")
        return self


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MalformedDirectionError(RetryableAgentError):
    """The language model returned unusable or over-budget direction.

    Retryable: regeneration frequently produces valid direction.
    """


class UnknownAssetReferenceError(RetryableAgentError):
    """The direction references assets absent from the creative package.

    Retryable: the model hallucinated references; regeneration frequently
    fixes it.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class VideoDirectorAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`VideoDirectorAgent`."""

    max_duration_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=180.0,
        description="Duration budget for one short-form video.",
    )
    system_prompt_template: str = Field(
        default="video_director/system",
        description="PromptRepository template name for the system prompt; "
        "used when a repository is injected, otherwise the built-in "
        "default prompt applies.",
    )


#: Built-in system prompt, used when no PromptRepository is injected.
_DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a short-form video director inside an automated marketing "
    "system. You receive a JSON object describing one planned video: the "
    "business subject, the plan item (topic, objective, call to action, "
    "platform, content pillar), its approved caption, the campaign brand "
    "voice, and the ids of approved image assets you may reference.\n"
    "Direct the video.\n"
    "Respond with exactly one JSON object of the form:\n"
    '{"script": "...", "scenes": [{"number": 1, "description": "...", '
    '"voice_over": "...", "shots": [{"number": 1, "description": "...", '
    '"framing": "close-up", "duration_seconds": 2.5}]}], '
    '"subtitles": [{"start_seconds": 0.0, "end_seconds": 2.5, '
    '"text": "..."}], "asset_references": ["..."]}\n'
    "Rules:\n"
    "- Number scenes 1..n and, within each scene, shots 1..n.\n"
    "- Keep subtitles ordered, non-overlapping, and matched to the "
    "voice-over.\n"
    "- Reference only the provided approved asset ids, and reuse them "
    "where stills fit the story.\n"
    "- Stay aligned with the caption, brand voice, topic, objective, and "
    "call to action; introduce no new claims.\n"
    "- Output JSON only, with no surrounding text."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class VideoDirectorAgent(BaseAgent[VideoBrief, VideoPackage]):
    """Directs and coordinates rendering of the plan's short-form videos.

    Workflow, per video item (both items run concurrently):

    1. Serialise the item, its caption, the brand voice and the approved
       asset ids into the user prompt.
    2. Obtain the direction from the injected language model and validate
       it into :class:`VideoDirection` (structure) —
       :class:`MalformedDirectionError` on failure.
    3. Enforce the configured duration budget and verify every asset
       reference against the approved creative package —
       :class:`UnknownAssetReferenceError` on hallucinated references.
    4. Delegate rendering to the injected :class:`VideoGenerationPort`.

    The final :class:`VideoPackage` enforces the exact video count on
    construction.
    """

    def __init__(
        self,
        *,
        llm: LanguageModelPort,
        video_generator: VideoGenerationPort,
        name: str | None = None,
        config: VideoDirectorAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            llm: Language model client used to draft the direction.
            video_generator: Tool that renders videos from direction.
            name: Logical agent name; defaults to the class name.
            config: Video-director-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository; overrides the built-in
                system prompt when it provides
                ``config.system_prompt_template``.
        """
        settings = config or VideoDirectorAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings
        self._llm = llm
        self._video_generator = video_generator

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: VideoBrief, *, run_id: str) -> VideoPackage:
        """Direct and render every planned short-form video.

        Args:
            payload: The validated plan/captions/creatives triple.
            run_id: Identifier of this execution.

        Returns:
            A :class:`VideoPackage` with exactly the required videos.

        Raises:
            MalformedDirectionError: If the model direction is unusable or
                exceeds the duration budget.
            UnknownAssetReferenceError: If the direction references assets
                absent from the approved creative package.
        """
        captions = {
            caption.item_id: caption for caption in payload.captions.captions
        }
        approved_assets = tuple(
            creative.asset.asset_id for creative in payload.creatives.creatives
        )
        video_items = sorted(
            (
                item
                for item in payload.week_plan.items
                if item.format is ContentFormat.SHORT_FORM_VIDEO
            ),
            key=lambda item: (item.day, item.publish_time),
        )
        videos = await asyncio.gather(
            *(
                self._direct_video(
                    item,
                    caption=captions[item.id],
                    brand_voice=payload.captions.brand_voice,
                    subject=payload.week_plan.subject,
                    approved_assets=approved_assets,
                    run_id=run_id,
                )
                for item in video_items
            )
        )
        package = VideoPackage(
            run_id=run_id,
            source_plan_run_id=payload.week_plan.run_id,
            source_caption_run_id=payload.captions.run_id,
            source_creative_run_id=payload.creatives.run_id,
            subject=payload.week_plan.subject,
            videos=tuple(videos),
            created_at=datetime.now(UTC),
        )
        self._logger.bind(
            run_id=run_id,
            event="video_director.directed",
            videos=len(package.videos),
        ).debug("Video package directed and rendered")
        return package

    # -- per-item direction ----------------------------------------------------------

    async def _direct_video(
        self,
        item: PlannedContent,
        *,
        caption: Caption,
        brand_voice: str,
        subject: str,
        approved_assets: tuple[str, ...],
        run_id: str,
    ) -> VideoCreative:
        """Draft, validate and render the direction for one video item."""
        raw = await self._llm.complete(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(
                item,
                caption=caption,
                brand_voice=brand_voice,
                subject=subject,
                approved_assets=approved_assets,
            ),
        )
        try:
            data = extract_json_object(raw)
            direction = VideoDirection.model_validate(
                {**data, "item_id": item.id}
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedDirectionError(
                f"Language model returned unusable direction for item "
                f"{item.id}: {exc}",
                agent_name=self.name,
                run_id=run_id,
            ) from exc

        if direction.total_duration_seconds > self._settings.max_duration_seconds:
            raise MalformedDirectionError(
                f"Direction for item {item.id} runs "
                f"{direction.total_duration_seconds}s, exceeding the "
                f"{self._settings.max_duration_seconds}s budget.",
                agent_name=self.name,
                run_id=run_id,
            )
        if unknown := set(direction.asset_references) - set(approved_assets):
            raise UnknownAssetReferenceError(
                f"Direction for item {item.id} references unknown assets: "
                f"{sorted(unknown)}",
                agent_name=self.name,
                run_id=run_id,
            )

        asset = await self._video_generator.render(direction=direction)
        return VideoCreative(item_id=item.id, direction=direction, asset=asset)

    # -- prompt construction ---------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Return the repository-provided system prompt, or the built-in one."""
        base = (
            self.load_prompt(self._settings.system_prompt_template)
            if self.prompts is not None
            else _DEFAULT_SYSTEM_PROMPT
        )
        return (
            f"{base}\nKeep the total shot duration at or under "
            f"{self._settings.max_duration_seconds} seconds."
        )

    @staticmethod
    def _user_prompt(
        item: PlannedContent,
        *,
        caption: Caption,
        brand_voice: str,
        subject: str,
        approved_assets: tuple[str, ...],
    ) -> str:
        """Serialise the direction input as compact JSON."""
        return json.dumps(
            {
                "subject": subject,
                "item": {
                    "item_id": item.id,
                    "day": item.day,
                    "platform": item.platform.value,
                    "topic": item.topic,
                    "objective": item.objective,
                    "call_to_action": item.call_to_action,
                    "content_pillar": item.content_pillar,
                },
                "caption": {
                    "headline": caption.headline,
                    "caption": caption.caption,
                    "call_to_action": caption.call_to_action,
                    "tone": caption.tone,
                    "keywords": list(caption.keywords),
                },
                "brand_voice": brand_voice,
                "approved_asset_ids": list(approved_assets),
            },
            ensure_ascii=False,
        )
