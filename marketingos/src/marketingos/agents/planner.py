"""PlannerAgent — a concrete first-week content plan from a strategy.

The :class:`PlannerAgent` converts a
:class:`~marketingos.agents.strategist.Strategy` into a :class:`WeekPlan`:
exactly five social media posts and two short-form videos, one item per day
across the seven days of the week, each with an objective, platform, topic,
call to action, dependencies and publishing schedule.

Scope and guarantees
--------------------
* **Constraints are enforced by validation, not by prompt wording.** The
  :class:`WeekPlan` model rejects any plan that does not contain exactly
  :data:`REQUIRED_POSTS` posts and :data:`REQUIRED_VIDEOS` short-form
  videos, does not place exactly one item on each of the seven days, or has
  broken dependency ordering. A model response that violates any constraint
  is rejected as a retryable failure so the base agent regenerates it.
* **Anchored to the strategy.** Every item's ``content_pillar`` must exactly
  match a pillar name from the input strategy; the agent validates this
  against the strategy in code.
* **Planning only.** The plan names topics, objectives and CTAs — it
  contains no creative assets, captions or scripts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, time
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from marketingos.agents.strategist import Strategy

__all__ = [
    "ContentFormat",
    "InvalidPlanError",
    "PlannedContent",
    "PlannerAgent",
    "PlannerAgentConfig",
    "Platform",
    "REQUIRED_POSTS",
    "REQUIRED_VIDEOS",
    "WEEK_DAYS",
    "WeekPlan",
]


#: Hard content-mix constraints for a first-week plan. These are product
#: invariants, therefore module constants rather than configuration.
REQUIRED_POSTS: Final[int] = 5
REQUIRED_VIDEOS: Final[int] = 2
WEEK_DAYS: Final[int] = 7


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ContentFormat(StrEnum):
    """The two content formats a first-week plan may contain."""

    POST = "post"
    SHORT_FORM_VIDEO = "short_form_video"


class Platform(StrEnum):
    """Publishing platforms the planner may schedule content on."""

    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    LINKEDIN = "linkedin"
    X = "x"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class PlannedContent(BaseModel):
    """One scheduled content item in the week plan."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^C\d+$")
    day: int = Field(
        ge=1, le=WEEK_DAYS, description="Day of the week, 1 = first day."
    )
    publish_time: time = Field(description="Publishing time on that day.")
    format: ContentFormat
    platform: Platform
    topic: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    call_to_action: str = Field(min_length=1)
    content_pillar: str = Field(
        min_length=1,
        description="Exact name of the strategy content pillar this item "
        "belongs to.",
    )
    depends_on: tuple[str, ...] = Field(
        default=(),
        description="Ids of plan items that must be published before this "
        "one (e.g. a post teasing a video).",
    )


class WeekPlan(BaseModel):
    """A validated seven-day content plan.

    Construction enforces the plan invariants, so an instance of this model
    is guaranteed to satisfy them:

    * exactly :data:`REQUIRED_POSTS` posts and :data:`REQUIRED_VIDEOS`
      short-form videos;
    * exactly one item on each of the :data:`WEEK_DAYS` days;
    * unique item ids;
    * dependencies reference existing items and are strictly earlier in the
      publishing schedule (which also excludes cycles and self-references).
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    source_strategy_run_id: str = Field(
        description="run_id of the StrategistAgent execution this plan "
        "was derived from."
    )
    subject: str
    items: tuple[PlannedContent, ...]
    created_at: datetime

    @model_validator(mode="after")
    def _validate_plan_invariants(self) -> "WeekPlan":
        """Enforce the content mix, daily distribution and dependency order."""
        posts = sum(
            1 for item in self.items if item.format is ContentFormat.POST
        )
        videos = sum(
            1
            for item in self.items
            if item.format is ContentFormat.SHORT_FORM_VIDEO
        )
        if posts != REQUIRED_POSTS or videos != REQUIRED_VIDEOS:
            raise ValueError(
                f"plan must contain exactly {REQUIRED_POSTS} posts and "
                f"{REQUIRED_VIDEOS} short-form videos, got {posts} posts "
                f"and {videos} videos"
            )

        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError("plan item ids must be unique")

        days = sorted(item.day for item in self.items)
        if days != list(range(1, WEEK_DAYS + 1)):
            raise ValueError(
                "plan must schedule exactly one item on each of the "
                f"{WEEK_DAYS} days (days 1..{WEEK_DAYS}), got days {days}"
            )

        schedule = {item.id: (item.day, item.publish_time) for item in self.items}
        for item in self.items:
            for dependency_id in item.depends_on:
                if dependency_id not in schedule:
                    raise ValueError(
                        f"item {item.id} depends on unknown item "
                        f"{dependency_id!r}"
                    )
                if schedule[dependency_id] >= (item.day, item.publish_time):
                    raise ValueError(
                        f"item {item.id} depends on {dependency_id}, which "
                        "is not scheduled strictly earlier"
                    )
        return self


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InvalidPlanError(RetryableAgentError):
    """The language model returned a plan that violates the plan invariants.

    Retryable: regeneration frequently produces a valid plan.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PlannerAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`PlannerAgent`."""

    system_prompt_template: str = Field(
        default="planner/week_plan_system",
        description="PromptRepository template name for the system prompt; "
        "used when a repository is injected, otherwise the built-in "
        "default prompt applies.",
    )


#: Built-in system prompt, used when no PromptRepository is injected. It
#: describes the required shape to raise generation success rates, but the
#: constraints themselves are enforced by WeekPlan validation.
_DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a content planner inside an automated marketing system. You "
    "receive a JSON object with a first-week marketing strategy: goals, "
    "target audience, positioning, key messages, content pillars, and "
    "success metrics. Produce the week's content plan.\n"
    "Respond with exactly one JSON object of the form:\n"
    '{"items": [{"id": "C1", "day": 1, "publish_time": "09:00", '
    '"format": "post", "platform": "instagram", "topic": "...", '
    '"objective": "...", "call_to_action": "...", '
    '"content_pillar": "...", "depends_on": []}]}\n'
    "Rules:\n"
    f"- Produce exactly {REQUIRED_POSTS + REQUIRED_VIDEOS} items: "
    f'{REQUIRED_POSTS} with format "post" and {REQUIRED_VIDEOS} with '
    'format "short_form_video".\n'
    f"- Schedule exactly one item per day, days 1 through {WEEK_DAYS}.\n"
    f"- Use ids C1 through C{REQUIRED_POSTS + REQUIRED_VIDEOS}, in "
    "schedule order.\n"
    "- depends_on may only reference items published strictly earlier.\n"
    '- platform must be one of: "instagram", "facebook", "tiktok", '
    '"youtube", "linkedin", "x".\n'
    "- content_pillar must exactly match one of the strategy's pillar "
    "names.\n"
    "- Plan topics, objectives, and calls to action only — no captions, "
    "scripts, or creative assets.\n"
    "- Output JSON only, with no surrounding text."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlannerAgent(BaseAgent[Strategy, WeekPlan]):
    """Generates a validated seven-day content plan from a strategy.

    Workflow:

    1. Serialise the strategy into the user prompt.
    2. Obtain the plan proposal from the injected language model.
    3. Validate the proposal into the typed :class:`WeekPlan` model, whose
       construction enforces the 5-post/2-video mix, the one-item-per-day
       distribution and dependency ordering.
    4. Verify that every item's ``content_pillar`` names a real strategy
       pillar.

    Any violation in steps 3–4 raises the retryable
    :class:`InvalidPlanError`, so :meth:`~marketingos.agents.base.BaseAgent.
    execute` regenerates the plan up to the configured retry budget.
    """

    def __init__(
        self,
        *,
        llm: LanguageModelPort,
        name: str | None = None,
        config: PlannerAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            llm: Language model client used to draft the plan.
            name: Logical agent name; defaults to the class name.
            config: Planner-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository; overrides the built-in
                system prompt when it provides
                ``config.system_prompt_template``.
        """
        settings = config or PlannerAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings
        self._llm = llm

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: Strategy, *, run_id: str) -> WeekPlan:
        """Draft and validate the seven-day content plan.

        Args:
            payload: The output of a ``StrategistAgent`` execution.
            run_id: Identifier of this execution.

        Returns:
            A :class:`WeekPlan` guaranteed to satisfy the plan invariants.

        Raises:
            InvalidPlanError: If the model output is unparseable, fails
                :class:`WeekPlan` validation, or references pillars that do
                not exist in the strategy.
        """
        raw = await self._llm.complete(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(payload),
        )
        try:
            data = extract_json_object(raw)
            plan = WeekPlan.model_validate(
                {
                    "run_id": run_id,
                    "source_strategy_run_id": payload.run_id,
                    "subject": payload.subject,
                    "items": data.get("items"),
                    "created_at": datetime.now(UTC),
                }
            )
            self._check_pillars(plan, payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidPlanError(
                f"Language model returned an invalid week plan: {exc}",
                agent_name=self.name,
                run_id=run_id,
            ) from exc

        self._logger.bind(
            run_id=run_id,
            event="planner.planned",
            items=len(plan.items),
            posts=REQUIRED_POSTS,
            videos=REQUIRED_VIDEOS,
        ).debug("Week plan drafted and validated")
        return plan

    # -- strategy anchoring ------------------------------------------------------------

    @staticmethod
    def _check_pillars(plan: WeekPlan, strategy: Strategy) -> None:
        """Verify every item belongs to a real strategy pillar.

        Raises:
            ValueError: If an item names a pillar absent from the strategy.
        """
        pillar_names = {pillar.name for pillar in strategy.content_pillars}
        unknown = {
            item.content_pillar
            for item in plan.items
            if item.content_pillar not in pillar_names
        }
        if unknown:
            raise ValueError(
                f"plan items reference unknown content pillars: "
                f"{sorted(unknown)}; strategy defines {sorted(pillar_names)}"
            )

    # -- prompt construction ---------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Return the repository-provided system prompt, or the built-in one."""
        if self.prompts is not None:
            return self.load_prompt(self._settings.system_prompt_template)
        return _DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _user_prompt(strategy: Strategy) -> str:
        """Serialise the planning input as compact JSON."""
        return json.dumps(
            {
                "subject": strategy.subject,
                "goals": [goal.statement for goal in strategy.goals],
                "target_audience": {
                    "description": strategy.target_audience.description,
                    "needs": list(strategy.target_audience.needs),
                },
                "positioning": strategy.positioning,
                "key_messages": list(strategy.key_messages),
                "content_pillars": [
                    {"name": pillar.name, "description": pillar.description}
                    for pillar in strategy.content_pillars
                ],
                "success_metrics": [
                    {"name": metric.name, "target": metric.target}
                    for metric in strategy.success_metrics
                ],
            },
            ensure_ascii=False,
        )