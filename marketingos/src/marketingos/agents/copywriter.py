"""CopywriterAgent — captions for every planned content item.

The :class:`CopywriterAgent` converts a
:class:`~marketingos.agents.planner.WeekPlan` into a
:class:`CaptionPackage`: one caption per planned item — exactly
:data:`~marketingos.agents.planner.REQUIRED_POSTS` posts and
:data:`~marketingos.agents.planner.REQUIRED_VIDEOS` short-form videos — each
with a headline, caption body, call to action, hashtags, tone and keywords.

Scope and guarantees
--------------------
* **One caption per plan item, enforced by validation.** The agent requires
  the model's captions to cover exactly the plan's item ids, and the
  :class:`CaptionPackage` model itself re-enforces the 5-post/2-video mix.
  Violations are retryable failures, so the base agent regenerates.
* **Consistent brand voice.** All captions are produced in a single model
  call together with an explicit ``brand_voice`` statement that downstream
  agents can reuse; each caption's ``tone`` describes its variation within
  that one voice.
* **Copy only.** The agent writes text. It produces no image prompts, no
  video prompts and no creative specifications — those belong to the
  downstream designer and video director.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from marketingos.agents.planner import (
    REQUIRED_POSTS,
    REQUIRED_VIDEOS,
    ContentFormat,
    WeekPlan,
)

__all__ = [
    "Caption",
    "CaptionAlignmentError",
    "CaptionPackage",
    "CopywriterAgent",
    "CopywriterAgentConfig",
    "MalformedCaptionsError",
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class Caption(BaseModel):
    """The complete copy for one planned content item."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(
        pattern=r"^C\d+$",
        description="Id of the WeekPlan item this caption belongs to.",
    )
    format: ContentFormat = Field(
        description="Format of the plan item, carried over from the plan "
        "(never taken from the language model)."
    )
    headline: str = Field(min_length=1, max_length=200)
    caption: str = Field(min_length=1, max_length=3000)
    call_to_action: str = Field(min_length=1, max_length=200)
    hashtags: tuple[str, ...] = Field(min_length=1, max_length=15)
    tone: str = Field(
        min_length=1,
        max_length=100,
        description="This caption's tonal variation within the package's "
        "single brand voice.",
    )
    keywords: tuple[str, ...] = Field(min_length=1, max_length=15)

    @field_validator("hashtags")
    @classmethod
    def _normalize_hashtags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Normalise to '#tag' form, rejecting empty or multi-word tags."""
        normalized: list[str] = []
        for tag in value:
            cleaned = tag.strip().lstrip("#")
            if not cleaned or any(char.isspace() for char in cleaned):
                raise ValueError(f"invalid hashtag: {tag!r}")
            normalized.append(f"#{cleaned}")
        return tuple(dict.fromkeys(normalized))

    @field_validator("keywords")
    @classmethod
    def _normalize_keywords(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Lower-case, strip and deduplicate keywords."""
        cleaned = tuple(
            dict.fromkeys(
                keyword.strip().lower() for keyword in value if keyword.strip()
            )
        )
        if not cleaned:
            raise ValueError("keywords must contain at least one non-empty entry")
        return cleaned


class CaptionPackage(BaseModel):
    """All captions for one week plan, written in a single brand voice.

    Construction enforces the content mix: exactly
    :data:`~marketingos.agents.planner.REQUIRED_POSTS` post captions and
    :data:`~marketingos.agents.planner.REQUIRED_VIDEOS` video captions, with
    unique item ids.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    source_plan_run_id: str = Field(
        description="run_id of the PlannerAgent execution this package "
        "was written for."
    )
    subject: str
    brand_voice: str = Field(
        min_length=1,
        max_length=500,
        description="The single brand voice all captions are written in; "
        "reusable by downstream creative agents.",
    )
    captions: tuple[Caption, ...]
    created_at: datetime

    @model_validator(mode="after")
    def _validate_content_mix(self) -> CaptionPackage:
        """Enforce unique item ids and the 5-post/2-video caption mix."""
        item_ids = [caption.item_id for caption in self.captions]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("caption item ids must be unique")
        posts = sum(
            1 for caption in self.captions if caption.format is ContentFormat.POST
        )
        videos = sum(
            1
            for caption in self.captions
            if caption.format is ContentFormat.SHORT_FORM_VIDEO
        )
        if posts != REQUIRED_POSTS or videos != REQUIRED_VIDEOS:
            raise ValueError(
                f"package must contain exactly {REQUIRED_POSTS} post captions "
                f"and {REQUIRED_VIDEOS} video captions, got {posts} and {videos}"
            )
        return self


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MalformedCaptionsError(RetryableAgentError):
    """The language model returned output that fails caption validation.

    Retryable: regeneration frequently produces valid output.
    """


class CaptionAlignmentError(RetryableAgentError):
    """The model's captions do not cover exactly the plan's item ids.

    Retryable: the model mislabelled or skipped items; regeneration
    frequently fixes it.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CopywriterAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`CopywriterAgent`."""

    system_prompt_template: str = Field(
        default="copywriter/system",
        description="PromptRepository template name for the system prompt; "
        "used when a repository is injected, otherwise the built-in "
        "default prompt applies.",
    )


#: Built-in system prompt, used when no PromptRepository is injected.
_DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a social media copywriter inside an automated marketing "
    "system. You receive a JSON object with a business subject and a "
    "seven-day content plan; each plan item has an id, day, format, "
    "platform, topic, objective, call to action, and content pillar.\n"
    "Write the copy for every item, all in one consistent brand voice.\n"
    "Respond with exactly one JSON object of the form:\n"
    '{"brand_voice": "...", "captions": [{"item_id": "C1", '
    '"headline": "...", "caption": "...", "call_to_action": "...", '
    '"hashtags": ["#..."], "tone": "...", "keywords": ["..."]}]}\n'
    "Rules:\n"
    "- Produce exactly one caption per plan item, echoing its item_id.\n"
    "- State the shared brand voice once in brand_voice and stay inside it; "
    "each caption's tone describes only its variation within that voice.\n"
    "- Adapt wording to each item's platform and format.\n"
    "- Keep every caption aligned with the item's topic, objective, "
    "call to action, and content pillar; introduce no new claims.\n"
    "- Write copy only: no image prompts, no video prompts, no visual or "
    "scene descriptions.\n"
    "- Output JSON only, with no surrounding text."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CopywriterAgent(BaseAgent[WeekPlan, CaptionPackage]):
    """Writes the caption package for a validated week plan.

    Workflow:

    1. Serialise the plan items into the user prompt.
    2. Obtain all captions in a single model call so they share one brand
       voice.
    3. Verify the captions cover exactly the plan's item ids
       (:class:`CaptionAlignmentError` otherwise), attaching each item's
       format from the plan — never from the model.
    4. Validate the assembled :class:`CaptionPackage`, whose construction
       re-enforces the 5-post/2-video mix.
    """

    def __init__(
        self,
        *,
        llm: LanguageModelPort,
        name: str | None = None,
        config: CopywriterAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            llm: Language model client used to write the captions.
            name: Logical agent name; defaults to the class name.
            config: Copywriter-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository; overrides the built-in
                system prompt when it provides
                ``config.system_prompt_template``.
        """
        settings = config or CopywriterAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings
        self._llm = llm

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: WeekPlan, *, run_id: str) -> CaptionPackage:
        """Write and validate captions for every plan item.

        Args:
            payload: The output of a ``PlannerAgent`` execution.
            run_id: Identifier of this execution.

        Returns:
            A validated :class:`CaptionPackage` covering every plan item.

        Raises:
            CaptionAlignmentError: If the model's captions do not map
                one-to-one onto the plan's item ids.
            MalformedCaptionsError: If the model output cannot be validated
                into a :class:`CaptionPackage`.
        """
        raw = await self._llm.complete(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(payload),
        )
        plan_items = {item.id: item for item in payload.items}
        try:
            data = extract_json_object(raw)
            entries = data.get("captions")
            if not isinstance(entries, list):
                raise ValueError("'captions' must be a JSON array")
            provided_ids = [
                str(entry.get("item_id", "")) if isinstance(entry, dict) else ""
                for entry in entries
            ]
            self._check_alignment(provided_ids, plan_items, run_id=run_id)
            captions = tuple(
                Caption.model_validate(
                    {**entry, "format": plan_items[entry["item_id"]].format}
                )
                for entry in entries
            )
            package = CaptionPackage(
                run_id=run_id,
                source_plan_run_id=payload.run_id,
                subject=payload.subject,
                brand_voice=str(data.get("brand_voice", "")).strip(),
                captions=captions,
                created_at=datetime.now(UTC),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedCaptionsError(
                f"Language model returned unusable caption output: {exc}",
                agent_name=self.name,
                run_id=run_id,
            ) from exc

        self._logger.bind(
            run_id=run_id,
            event="copywriter.written",
            captions=len(package.captions),
        ).debug("Caption package written and validated")
        return package

    # -- alignment enforcement ---------------------------------------------------------

    def _check_alignment(
        self,
        provided_ids: list[str],
        plan_items: Mapping[str, object],
        *,
        run_id: str,
    ) -> None:
        """Require a one-to-one mapping between captions and plan items.

        Raises:
            CaptionAlignmentError: On missing, unknown, or duplicated ids.
        """
        expected = set(plan_items)
        provided = set(provided_ids)
        problems: list[str] = []
        if missing := expected - provided:
            problems.append(f"missing captions for items {sorted(missing)}")
        if unknown := provided - expected:
            problems.append(f"captions for unknown items {sorted(unknown)}")
        if len(provided_ids) != len(provided):
            problems.append("duplicate caption item ids")
        if problems:
            raise CaptionAlignmentError(
                "Captions do not align with the week plan: "
                + "; ".join(problems),
                agent_name=self.name,
                run_id=run_id,
            )

    # -- prompt construction ---------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Return the repository-provided system prompt, or the built-in one."""
        if self.prompts is not None:
            return self.load_prompt(self._settings.system_prompt_template)
        return _DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _user_prompt(plan: WeekPlan) -> str:
        """Serialise the copywriting input as compact JSON."""
        return json.dumps(
            {
                "subject": plan.subject,
                "items": [
                    {
                        "item_id": item.id,
                        "day": item.day,
                        "format": item.format.value,
                        "platform": item.platform.value,
                        "topic": item.topic,
                        "objective": item.objective,
                        "call_to_action": item.call_to_action,
                        "content_pillar": item.content_pillar,
                    }
                    for item in plan.items
                ],
            },
            ensure_ascii=False,
        )
