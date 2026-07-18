"""DesignerAgent — post creatives via coordinated image generation.

The :class:`DesignerAgent` converts a :class:`DesignBrief` (a validated
pairing of :class:`~marketingos.agents.planner.WeekPlan` and
:class:`~marketingos.agents.copywriter.CaptionPackage`, plus optional brand
styling) into a :class:`CreativePackage`: exactly
:data:`~marketingos.agents.planner.REQUIRED_POSTS` post creatives, each with
a full creative specification, generation prompt, branding information and
asset metadata.

Scope and guarantees
--------------------
* **Coordination only.** The agent never generates pixels. It composes
  creative specifications deterministically from the plan, the captions,
  the brand style and a reusable template, then delegates rendering to an
  injected tool satisfying :class:`ImageGenerationPort`.
* **Reusable templates.** Creative dimensions, layout and base styling come
  from configured :class:`CreativeTemplate` records, assigned to plan items
  in schedule order (round-robin), so campaigns stay visually coherent and
  templates are reusable across runs.
* **Brand consistency.** A single :class:`BrandStyle` (from the brief, or
  the configured default) is applied to every creative in the package.
* **Deterministic inputs, validated output.** Prompt composition uses only
  validated upstream content — nothing is invented — and the
  :class:`CreativePackage` model enforces the exact creative count.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marketingos.agents.base import (
    AgentConfig,
    BaseAgent,
    MemoryStore,
    PromptRepository,
    ToolRegistry,
)
from marketingos.agents.copywriter import Caption, CaptionPackage
from marketingos.agents.planner import (
    REQUIRED_POSTS,
    ContentFormat,
    PlannedContent,
    WeekPlan,
)

__all__ = [
    "BrandStyle",
    "CreativePackage",
    "CreativeSpec",
    "CreativeTemplate",
    "DesignBrief",
    "DesignerAgent",
    "DesignerAgentConfig",
    "GeneratedImageRef",
    "ImageGenerationPort",
    "PostCreative",
]


# ---------------------------------------------------------------------------
# Branding and templates
# ---------------------------------------------------------------------------


class BrandStyle(BaseModel):
    """Visual branding applied consistently to every creative in a package."""

    model_config = ConfigDict(frozen=True)

    primary_color: str = Field(default="#111827", pattern=r"^#[0-9A-Fa-f]{6}$")
    secondary_color: str = Field(default="#F9FAFB", pattern=r"^#[0-9A-Fa-f]{6}$")
    font_family: str = Field(default="Inter", min_length=1)
    logo_asset_id: str | None = Field(
        default=None,
        description="Asset id of the brand logo, when one is available.",
    )
    style_keywords: tuple[str, ...] = Field(
        default=("modern", "minimal"), min_length=1
    )


class CreativeTemplate(BaseModel):
    """A reusable layout template for post creatives."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    width: int = Field(ge=64, le=8192)
    height: int = Field(ge=64, le=8192)
    layout: str = Field(
        min_length=1,
        description="Composition guidance baked into the generation prompt.",
    )
    style_keywords: tuple[str, ...] = Field(min_length=1)


def _default_templates() -> tuple[CreativeTemplate, ...]:
    """Built-in templates used when none are configured."""
    return (
        CreativeTemplate(
            name="square_feed_post",
            width=1080,
            height=1080,
            layout=(
                "single focal subject, centered composition, generous "
                "negative space in the upper third for overlay text"
            ),
            style_keywords=("clean", "high contrast", "professional photography"),
        ),
        CreativeTemplate(
            name="portrait_feed_post",
            width=1080,
            height=1350,
            layout=(
                "vertical composition, subject in the lower two thirds, "
                "headline space at the top"
            ),
            style_keywords=("vibrant", "editorial", "natural lighting"),
        ),
    )


# ---------------------------------------------------------------------------
# Image tool contract (port)
# ---------------------------------------------------------------------------


class GeneratedImageRef(BaseModel):
    """Asset metadata returned by the image-generation tool."""

    model_config = ConfigDict(frozen=True)

    asset_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    media_type: str = Field(default="image/png", min_length=1)


@runtime_checkable
class ImageGenerationPort(Protocol):
    """Structural contract for the image-generation tool.

    Satisfied by the image client in ``marketingos.tools``. The agent depends
    only on this protocol, keeping the image provider swappable.
    """

    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
    ) -> GeneratedImageRef:
        """Render one image and return its asset reference."""
        ...


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class DesignBrief(BaseModel):
    """Typed input pairing a week plan with its caption package.

    Construction validates the pairing: the captions must have been written
    for exactly this plan, and every plan item must have a caption. The
    optional brand style overrides the agent's configured default.
    """

    model_config = ConfigDict(frozen=True)

    week_plan: WeekPlan
    captions: CaptionPackage
    brand: BrandStyle | None = None

    @model_validator(mode="after")
    def _validate_pairing(self) -> DesignBrief:
        """Reject briefs whose captions do not belong to the plan."""
        if self.captions.source_plan_run_id != self.week_plan.run_id:
            raise ValueError(
                "caption package was written for plan run "
                f"{self.captions.source_plan_run_id!r}, not "
                f"{self.week_plan.run_id!r}"
            )
        plan_ids = {item.id for item in self.week_plan.items}
        caption_ids = {caption.item_id for caption in self.captions.captions}
        if plan_ids != caption_ids:
            raise ValueError(
                "captions must cover exactly the plan's item ids; "
                f"missing {sorted(plan_ids - caption_ids)}, "
                f"extra {sorted(caption_ids - plan_ids)}"
            )
        return self


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class CreativeSpec(BaseModel):
    """The full generation specification for one creative."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    negative_prompt: str = Field(min_length=1)
    width: int = Field(ge=64, le=8192)
    height: int = Field(ge=64, le=8192)
    style_keywords: tuple[str, ...] = Field(min_length=1)


class PostCreative(BaseModel):
    """One finished post creative: spec, asset reference and metadata."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(pattern=r"^C\d+$")
    template_name: str = Field(min_length=1)
    spec: CreativeSpec
    asset: GeneratedImageRef
    alt_text: str = Field(
        min_length=1,
        description="Accessibility description derived from the caption "
        "headline and the item topic.",
    )


class CreativePackage(BaseModel):
    """All post creatives for one week plan, in a single brand style.

    Construction enforces exactly
    :data:`~marketingos.agents.planner.REQUIRED_POSTS` creatives with unique
    item ids and unique asset ids.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    source_plan_run_id: str = Field(
        description="run_id of the PlannerAgent execution this package "
        "serves."
    )
    source_caption_run_id: str = Field(
        description="run_id of the CopywriterAgent execution whose copy "
        "informed the creatives."
    )
    subject: str
    brand: BrandStyle
    creatives: tuple[PostCreative, ...]
    created_at: datetime

    @model_validator(mode="after")
    def _validate_creative_count(self) -> CreativePackage:
        """Enforce the exact creative count and id uniqueness."""
        if len(self.creatives) != REQUIRED_POSTS:
            raise ValueError(
                f"package must contain exactly {REQUIRED_POSTS} post "
                f"creatives, got {len(self.creatives)}"
            )
        item_ids = [creative.item_id for creative in self.creatives]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("creative item ids must be unique")
        asset_ids = [creative.asset.asset_id for creative in self.creatives]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("creative asset ids must be unique")
        return self


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DesignerAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`DesignerAgent`."""

    templates: tuple[CreativeTemplate, ...] = Field(
        default_factory=_default_templates,
        min_length=1,
        description="Reusable creative templates, assigned to post items "
        "in schedule order (round-robin).",
    )
    default_brand: BrandStyle = Field(
        default_factory=BrandStyle,
        description="Brand style applied when the brief provides none.",
    )
    negative_prompt: str = Field(
        default=(
            "text, watermark, logo artifacts, low quality, blurry, "
            "distorted anatomy"
        ),
        min_length=1,
        description="Baseline negative prompt for every generation.",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DesignerAgent(BaseAgent[DesignBrief, CreativePackage]):
    """Produces the post creatives for a validated design brief.

    Workflow:

    1. Select the plan's post items in schedule order and pair each with
       its caption and a reusable template (round-robin).
    2. Compose each creative specification deterministically from the item
       topic, the caption's headline and keywords, the template layout and
       the brand style — no generation, no invention.
    3. Delegate rendering to the injected :class:`ImageGenerationPort`,
       concurrently across all creatives; any tool failure propagates
       through the base agent's error normalisation (transient failures
       retry the whole run, keeping the package atomic).
    4. Assemble the :class:`CreativePackage`, whose construction enforces
       the exact creative count.
    """

    def __init__(
        self,
        *,
        image_generator: ImageGenerationPort,
        name: str | None = None,
        config: DesignerAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            image_generator: Tool that renders images from specifications.
            name: Logical agent name; defaults to the class name.
            config: Designer-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository (unused by the default
                deterministic spec composer; available to subclasses).
        """
        settings = config or DesignerAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings
        self._image_generator = image_generator

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: DesignBrief, *, run_id: str) -> CreativePackage:
        """Design and render every post creative in the brief.

        Args:
            payload: The validated plan/caption pairing to design for.
            run_id: Identifier of this execution.

        Returns:
            A :class:`CreativePackage` with exactly the required creatives.
        """
        brand = payload.brand or self._settings.default_brand
        captions = {
            caption.item_id: caption for caption in payload.captions.captions
        }
        post_items = sorted(
            (
                item
                for item in payload.week_plan.items
                if item.format is ContentFormat.POST
            ),
            key=lambda item: (item.day, item.publish_time),
        )
        creatives = await asyncio.gather(
            *(
                self._design_post(
                    item,
                    caption=captions[item.id],
                    template=self._settings.templates[
                        index % len(self._settings.templates)
                    ],
                    brand=brand,
                    subject=payload.week_plan.subject,
                )
                for index, item in enumerate(post_items)
            )
        )
        package = CreativePackage(
            run_id=run_id,
            source_plan_run_id=payload.week_plan.run_id,
            source_caption_run_id=payload.captions.run_id,
            subject=payload.week_plan.subject,
            brand=brand,
            creatives=tuple(creatives),
            created_at=datetime.now(UTC),
        )
        self._logger.bind(
            run_id=run_id,
            event="designer.rendered",
            creatives=len(package.creatives),
            templates=len(self._settings.templates),
        ).debug("Creative package rendered and validated")
        return package

    # -- per-item design -----------------------------------------------------------------

    async def _design_post(
        self,
        item: PlannedContent,
        *,
        caption: Caption,
        template: CreativeTemplate,
        brand: BrandStyle,
        subject: str,
    ) -> PostCreative:
        """Compose the spec for one post and delegate rendering to the tool."""
        spec = self._compose_spec(
            item, caption=caption, template=template, brand=brand, subject=subject
        )
        asset = await self._image_generator.generate(
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt,
            width=spec.width,
            height=spec.height,
        )
        return PostCreative(
            item_id=item.id,
            template_name=template.name,
            spec=spec,
            asset=asset,
            alt_text=f"{caption.headline} — {item.topic}",
        )

    def _compose_spec(
        self,
        item: PlannedContent,
        *,
        caption: Caption,
        template: CreativeTemplate,
        brand: BrandStyle,
        subject: str,
    ) -> CreativeSpec:
        """Deterministically compose one generation specification.

        Every prompt fragment comes from validated upstream content: the
        plan item, its caption, the template and the brand style.
        """
        prompt = "; ".join(
            (
                f"social media image for {subject}",
                f"topic: {item.topic}",
                f"visual mood conveying: {caption.headline}",
                f"key motifs: {', '.join(caption.keywords)}",
                template.layout,
                f"style: {', '.join(template.style_keywords + brand.style_keywords)}",
                f"brand color accents {brand.primary_color} and "
                f"{brand.secondary_color}",
            )
        )
        return CreativeSpec(
            prompt=prompt,
            negative_prompt=self._settings.negative_prompt,
            width=template.width,
            height=template.height,
            style_keywords=template.style_keywords + brand.style_keywords,
        )
