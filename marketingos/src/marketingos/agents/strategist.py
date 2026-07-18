"""StrategistAgent — first-week marketing strategy from business context.

The :class:`StrategistAgent` converts a
:class:`~marketingos.models.business_context.BusinessContext` into a
:class:`Strategy` for the business's first week of marketing: goals, target
audience, positioning, key messages, content pillars and success metrics.

Scope and guarantees
--------------------
* **Grounded, verifiably.** Every goal, the target audience and every
  content pillar must cite the ids of the context facts (``F*``) or labelled
  assumptions (``A*``) that justify it. The agent validates the citations
  against the context and rejects ungrounded output as a retryable failure —
  grounding is enforced by code, not by the prompt.
* **Strategy only.** The agent produces direction, not deliverables: no
  creative assets, no copy, no post ideas. Content planning belongs to the
  downstream ``PlannerAgent``.
* **Typed output.** The language model's JSON is validated into the frozen
  :class:`Strategy` model; anything that fails validation is rejected as
  :class:`MalformedStrategyError` (retryable — regeneration usually fixes
  malformed output).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

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
from marketingos.models.business_context import BusinessContext
from marketingos.prompts.registry import get_prompt_registry

__all__ = [
    "ContentPillar",
    "MalformedStrategyError",
    "MarketingGoal",
    "Strategy",
    "StrategistAgent",
    "StrategistAgentConfig",
    "SuccessMetric",
    "TargetAudience",
    "UngroundedStrategyError",
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class MarketingGoal(BaseModel):
    """One first-week goal, grounded in context fact/assumption ids."""

    model_config = ConfigDict(frozen=True)

    statement: str = Field(min_length=1)
    grounded_in: tuple[str, ...] = Field(min_length=1)


class TargetAudience(BaseModel):
    """The audience the first week should reach, grounded in the context."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(min_length=1)
    needs: tuple[str, ...] = ()
    grounded_in: tuple[str, ...] = Field(min_length=1)


class ContentPillar(BaseModel):
    """A recurring content theme, grounded in the context.

    Pillar names are the vocabulary the downstream planner uses to attach
    content items to the strategy, so they must be unique within a strategy.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    grounded_in: tuple[str, ...] = Field(min_length=1)


class SuccessMetric(BaseModel):
    """How the first week's outcome will be measured."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    measurement: str = Field(min_length=1)


class Strategy(BaseModel):
    """A complete first-week marketing strategy for one business."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    source_context_run_id: str = Field(
        description="run_id of the BusinessAnalysisAgent execution this "
        "strategy was derived from."
    )
    subject: str
    goals: tuple[MarketingGoal, ...] = Field(min_length=1, max_length=5)
    target_audience: TargetAudience
    positioning: str = Field(min_length=1)
    key_messages: tuple[str, ...] = Field(min_length=1, max_length=8)
    content_pillars: tuple[ContentPillar, ...] = Field(min_length=1, max_length=5)
    success_metrics: tuple[SuccessMetric, ...] = Field(min_length=1, max_length=8)
    created_at: datetime

    @model_validator(mode="after")
    def _validate_unique_pillar_names(self) -> Strategy:
        """Pillar names are planner vocabulary and must be unique."""
        names = [pillar.name for pillar in self.content_pillars]
        if len(set(names)) != len(names):
            raise ValueError("content pillar names must be unique")
        return self


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MalformedStrategyError(RetryableAgentError):
    """The language model returned output that fails strategy validation.

    Retryable: regeneration frequently produces valid output.
    """


class UngroundedStrategyError(RetryableAgentError):
    """The strategy cites context ids that do not exist.

    Retryable: the model hallucinated references; regeneration frequently
    fixes it.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class StrategistAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`StrategistAgent`."""

    system_prompt_template: str = Field(
        default="strategist/first_week_system",
        description="PromptRepository template reference for the system "
        "prompt, resolved against the injected repository. The version is "
        "omitted so the repository's default-version resolution applies.",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class StrategistAgent(BaseAgent[BusinessContext, Strategy]):
    """Generates a grounded first-week marketing strategy.

    Workflow:

    1. Serialise the business context (facts and assumptions with their ids)
       into the user prompt.
    2. Obtain the strategy proposal from the injected language model.
    3. Validate the proposal into the typed :class:`Strategy` model
       (structure, cardinality, unique pillar names).
    4. Verify that every ``grounded_in`` citation refers to a real context
       fact or assumption id.

    Steps 3 and 4 are code-level enforcement; a proposal that fails either
    is rejected with a retryable error so :meth:`~marketingos.agents.base.
    BaseAgent.execute` can regenerate it.
    """

    def __init__(
        self,
        *,
        llm: LanguageModelPort,
        name: str | None = None,
        config: StrategistAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            llm: Language model client used to draft the strategy.
            name: Logical agent name; defaults to the class name.
            config: Strategist-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Prompt repository used to resolve
                ``config.system_prompt_template``. Defaults to the
                process-wide :func:`~marketingos.prompts.registry.get_prompt_registry`
                instance, so the versioned template library is used unless a
                caller injects a different repository (for example a stub in
                tests).
        """
        settings = config or StrategistAgentConfig()
        super().__init__(
            name=name,
            config=settings,
            memory=memory,
            tools=tools,
            prompts=prompts if prompts is not None else get_prompt_registry(),
        )
        self._settings = settings
        self._llm = llm

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: BusinessContext, *, run_id: str) -> Strategy:
        """Draft and validate the first-week strategy.

        Args:
            payload: The output of a ``BusinessAnalysisAgent`` execution.
            run_id: Identifier of this execution.

        Returns:
            A validated, grounded :class:`Strategy`.

        Raises:
            MalformedStrategyError: If the model output cannot be validated
                into a :class:`Strategy`.
            UngroundedStrategyError: If the strategy cites nonexistent
                context ids.
        """
        raw = await self._llm.complete(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(payload),
        )
        try:
            data = extract_json_object(raw)
            source_context_run_id = (payload.metadata or {}).get(
                "business_analysis_run_id", str(payload.id)
            )
            strategy = Strategy.model_validate(
                {
                    **data,
                    "run_id": run_id,
                    "source_context_run_id": source_context_run_id,
                    "subject": payload.business_name,
                    "created_at": datetime.now(UTC),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedStrategyError(
                f"Language model returned unusable strategy output: {exc}",
                agent_name=self.name,
                run_id=run_id,
            ) from exc

        self._validate_grounding(strategy, payload, run_id=run_id)

        self._logger.bind(
            run_id=run_id,
            event="strategist.drafted",
            goals=len(strategy.goals),
            pillars=len(strategy.content_pillars),
            metrics=len(strategy.success_metrics),
        ).debug("Strategy drafted and validated")
        return strategy

    # -- grounding enforcement -------------------------------------------------------

    def _validate_grounding(
        self, strategy: Strategy, context: BusinessContext, *, run_id: str
    ) -> None:
        """Verify every citation against the context's real ids.

        Raises:
            UngroundedStrategyError: If any ``grounded_in`` id is unknown.
        """
        valid_ids = {str(fact.id) for fact in context.observed_facts} | {
            str(assumption.id) for assumption in context.assumptions
        }
        citations: list[tuple[str, tuple[str, ...]]] = [
            *(
                (f"goal {i}", goal.grounded_in)
                for i, goal in enumerate(strategy.goals, start=1)
            ),
            ("target_audience", strategy.target_audience.grounded_in),
            *(
                (f"content pillar {pillar.name!r}", pillar.grounded_in)
                for pillar in strategy.content_pillars
            ),
        ]
        problems = [
            f"{label} cites unknown ids {sorted(set(ids) - valid_ids)}"
            for label, ids in citations
            if set(ids) - valid_ids
        ]
        if problems:
            raise UngroundedStrategyError(
                "Strategy is not grounded in the business context: "
                + "; ".join(problems),
                agent_name=self.name,
                run_id=run_id,
            )

    # -- prompt construction ---------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Return the repository-provided system prompt for this execution."""
        return self.load_prompt(self._settings.system_prompt_template)

    @staticmethod
    def _user_prompt(context: BusinessContext) -> str:
        """Serialise the strategy input as compact JSON."""

        def _statements(category: str) -> list[str]:
            return list(
                dict.fromkeys(
                    fact.statement
                    for fact in context.observed_facts
                    if fact.category == category
                )
            )

        return json.dumps(
            {
                "subject": context.business_name,
                "facts": [
                    {
                        "id": str(fact.id),
                        "category": fact.category,
                        "statement": fact.statement,
                        "confidence": fact.confidence_score,
                    }
                    for fact in context.observed_facts
                ],
                "assumptions": [
                    {
                        "id": str(assumption.id),
                        "statement": assumption.statement,
                        "confidence": assumption.confidence_score,
                    }
                    for assumption in context.assumptions
                ],
                "offerings": _statements("products_services"),
                "brand_messages": _statements("brand_messaging"),
                "social_profiles": _statements("social_presence"),
            },
            ensure_ascii=False,
        )
