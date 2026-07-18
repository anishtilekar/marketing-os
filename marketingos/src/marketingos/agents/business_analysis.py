"""BusinessAnalysisAgent — structured business understanding from research.

The :class:`BusinessAnalysisAgent` converts a
:class:`~marketingos.agents.research.ResearchResult` into a
:class:`~marketingos.models.business_context.BusinessContext`: a structured
understanding of the business in which **observed facts** and **working
assumptions** are strictly separated. The output schemas themselves live in
:mod:`marketingos.models.business_context`; this module owns the agent that
builds them.

The system prompt is resolved through the injected ``PromptRepository``
(by default the process-wide :class:`~marketingos.prompts.registry.PromptRegistry`),
from the versioned template referenced by
``BusinessAnalysisAgentConfig.system_prompt_template``.

Scope and guarantees
--------------------
* **Facts are never generated.** The fact section of the output is built
  deterministically from the research result — every :class:`Fact` is
  an observed research fact carrying its own id.
* **Assumptions are always labelled.** Working assumptions (industry, likely
  customer base, business model, maturity) are produced by an injected
  language model, and each one must cite the fact ids it is based on and
  carry a confidence value. Assumptions never masquerade as facts.
* **Separation is validated, not promised.** The :class:`BusinessContext`
  model itself rejects any assumption that duplicates a fact or cites an
  unknown fact id, and the agent additionally rejects such model output as
  a retryable failure before constructing the context.
* **No strategy.** This agent produces understanding only; marketing
  strategy is the responsibility of the downstream ``StrategistAgent``.

This module also defines :class:`LanguageModelPort` — the structural
contract for the project's LLM completion client — and
:func:`extract_json_object`, the tolerant JSON reader for model responses.
Downstream agent modules (``strategist``, ``planner``) reuse both, following
the package's existing import chain.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Final, Protocol, runtime_checkable

from pydantic import Field

from marketingos.agents.base import (
    AgentConfig,
    BaseAgent,
    MemoryStore,
    PermanentAgentError,
    PromptRepository,
    RetryableAgentError,
    ToolRegistry,
)
from marketingos.agents.research import FactCategory, ResearchResult, SourceType
from marketingos.models.business_context import (
    Assumption,
    BusinessContext,
    Fact,
    RiskLevel,
)
from marketingos.models.business_context import SourceType as ContextSourceType
from marketingos.prompts.registry import get_prompt_registry

__all__ = [
    "BusinessAnalysisAgent",
    "BusinessAnalysisAgentConfig",
    "InsufficientResearchError",
    "LanguageModelPort",
    "MalformedAnalysisError",
    "extract_json_object",
]


# ---------------------------------------------------------------------------
# Language model port and response parsing
# ---------------------------------------------------------------------------


@runtime_checkable
class LanguageModelPort(Protocol):
    """Structural contract for the LLM completion client.

    Satisfied by the completion client in ``marketingos.tools`` /
    ``marketingos.services``. Agents depend only on this protocol, keeping
    the model provider swappable without touching agent code.
    """

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's completion for the given prompts."""
        ...


_JSON_FENCE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", re.DOTALL
)

#: Research source types have no direct equivalent in the business-context
#: model's (broader) SourceType vocabulary, so each is mapped to its closest
#: match.
_SOURCE_TYPE_MAP: Final[dict[SourceType, ContextSourceType]] = {
    SourceType.WEBSITE: ContextSourceType.WEBSITE,
    SourceType.INSTAGRAM: ContextSourceType.SOCIAL_MEDIA,
    SourceType.WEB_SEARCH: ContextSourceType.SEARCH_ENGINE,
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _fingerprint(statement: str) -> str:
    """Normalize a statement for content-based duplicate comparison."""
    return _WHITESPACE.sub(" ", statement.strip().lower())


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object contained in a model response.

    Tolerates markdown code fences and surrounding prose, since language
    models frequently wrap structured output despite instructions.

    Args:
        raw: The verbatim model response.

    Returns:
        The parsed top-level JSON object.

    Raises:
        ValueError: If the response contains no parseable JSON object.
    """
    text = raw.strip()
    fenced = _JSON_FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model response")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON value is not an object")
    return parsed


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InsufficientResearchError(PermanentAgentError):
    """The research result contains no facts to analyse (permanent)."""


class MalformedAnalysisError(RetryableAgentError):
    """The language model returned unusable analysis output.

    Retryable: regeneration frequently produces valid output.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class BusinessAnalysisAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`BusinessAnalysisAgent`."""

    max_assumptions: int = Field(
        default=8,
        ge=0,
        le=25,
        description="Upper bound on derived assumptions; 0 skips the "
        "language-model call and produces a fact-only context.",
    )
    system_prompt_template: str = Field(
        default="business_analysis/system",
        description="PromptRepository template reference for the system "
        "prompt, resolved against the injected repository. The version is "
        "omitted so the repository's default-version resolution applies.",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BusinessAnalysisAgent(BaseAgent[ResearchResult, BusinessContext]):
    """Builds a :class:`BusinessContext` from a research result.

    Workflow:

    1. **Carry facts over deterministically.** Observed research facts become
       :class:`Fact` records, verbatim — no generation involved.
    2. **Derive labelled assumptions.** The injected language model proposes
       assumptions citing fact ids; the agent validates every one against
       the facts and rejects duplication or unknown references as a
       retryable failure.
    3. **Validate separation.** The :class:`BusinessContext` model re-checks
       the separation invariant on construction as defence in depth.
    """

    def __init__(
        self,
        *,
        llm: LanguageModelPort,
        name: str | None = None,
        config: BusinessAnalysisAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            llm: Language model client used to derive assumptions.
            name: Logical agent name; defaults to the class name.
            config: Analysis-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Prompt repository used to resolve
                ``config.system_prompt_template``. Defaults to the
                process-wide :func:`~marketingos.prompts.registry.get_prompt_registry`
                instance, so the versioned template library is used unless a
                caller injects a different repository (for example a stub in
                tests).
        """
        settings = config or BusinessAnalysisAgentConfig()
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

    async def run(self, payload: ResearchResult, *, run_id: str) -> BusinessContext:
        """Analyse the research result into a structured business context.

        Args:
            payload: The output of a ``ResearchAgent`` execution.
            run_id: Identifier of this execution.

        Returns:
            A validated :class:`BusinessContext`.

        Raises:
            InsufficientResearchError: If the research contains no facts.
            MalformedAnalysisError: If the language model output cannot be
                turned into valid, properly separated assumptions.
        """
        if not payload.facts:
            raise InsufficientResearchError(
                f"Research result {payload.run_id} contains no facts to "
                "analyse.",
                agent_name=self.name,
                run_id=run_id,
            )

        facts = self._carry_over_facts(payload)
        if self._settings.max_assumptions > 0:
            assumptions = await self._derive_assumptions(
                facts, subject=payload.subject, run_id=run_id
            )
        else:
            assumptions = ()

        context = BusinessContext(
            run_id=run_id,
            business_name=payload.subject,
            description=self._description(payload),
            observed_facts=list(facts),
            assumptions=list(assumptions),
            created_at=datetime.now(UTC),
            metadata={
                "business_analysis_run_id": run_id,
                "source_research_run_id": payload.run_id,
                "source_confidence": payload.confidence_score,
            },
        )
        self._logger.bind(
            run_id=run_id,
            event="business_analysis.built",
            facts=len(context.observed_facts),
            assumptions=len(context.assumptions),
        ).debug("Business context built")
        return context

    # -- deterministic fact handling ----------------------------------------------

    @staticmethod
    def _carry_over_facts(payload: ResearchResult) -> tuple[Fact, ...]:
        """Convert research facts into context facts, verbatim."""
        return tuple(
            Fact(
                category=fact.category.value,
                statement=fact.statement,
                source_reference=fact.source.url or fact.source.tool_name,
                source_type=_SOURCE_TYPE_MAP[fact.source.source_type],
                confidence_score=fact.confidence,
                extracted_at=fact.source.retrieved_at,
            )
            for fact in payload.facts
        )

    @staticmethod
    def _description(payload: ResearchResult) -> str | None:
        """Join ABOUT-category statements into a free-text description."""
        about = [
            fact.statement
            for fact in payload.facts
            if fact.category is FactCategory.ABOUT
        ]
        if not about:
            return None
        return " ".join(about)[:5000]

    # -- assumption derivation ---------------------------------------------------------

    async def _derive_assumptions(
        self,
        facts: tuple[Fact, ...],
        *,
        subject: str,
        run_id: str,
    ) -> tuple[Assumption, ...]:
        """Ask the language model for assumptions and validate every one.

        Raises:
            MalformedAnalysisError: If the response is unparseable, fails
                schema validation, restates a fact, or cites unknown fact
                ids. Retryable — regeneration frequently fixes it.
        """
        raw = await self._llm.complete(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(facts, subject=subject),
        )
        try:
            data = extract_json_object(raw)
            entries = data.get("assumptions")
            if not isinstance(entries, list):
                raise ValueError("'assumptions' must be a JSON array")
            entries = entries[: self._settings.max_assumptions]
            fact_ids = {str(fact.id) for fact in facts}
            assumptions = tuple(
                self._build_assumption(entry, fact_ids=fact_ids)
                for entry in entries
            )
            self._check_separation(assumptions, facts)
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedAnalysisError(
                f"Language model returned unusable analysis output: {exc}",
                agent_name=self.name,
                run_id=run_id,
            ) from exc
        return assumptions

    @staticmethod
    def _build_assumption(entry: dict[str, Any], *, fact_ids: set[str]) -> Assumption:
        """Validate one model-proposed assumption entry and build it.

        Raises:
            ValueError: If the entry cites no known fact id.
        """
        cited: set[str] = set(entry.get("based_on_fact_ids") or ())
        unknown = cited - fact_ids
        if unknown:
            raise ValueError(f"assumption cites unknown fact ids: {sorted(unknown)}")
        if not cited:
            raise ValueError("assumption must cite at least one fact id")
        reasoning = str(entry["rationale"]).strip()
        reasoning = f"{reasoning} (grounded in facts: {', '.join(sorted(cited))})"
        return Assumption(
            statement=entry["statement"],
            reasoning=reasoning,
            risk_level=RiskLevel(entry.get("risk_level", "medium")),
            confidence_score=entry["confidence"],
            category=entry.get("category"),
        )

    @staticmethod
    def _check_separation(
        assumptions: tuple[Assumption, ...], facts: tuple[Fact, ...]
    ) -> None:
        """Reject model output that blurs the fact/assumption boundary.

        Raises:
            ValueError: If an assumption restates a fact.
        """
        fact_fingerprints = {_fingerprint(fact.statement) for fact in facts}
        for assumption in assumptions:
            if _fingerprint(assumption.statement) in fact_fingerprints:
                raise ValueError(
                    f"assumption {assumption.id} restates an observed fact"
                )

    # -- prompt construction ---------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Return the repository-provided system prompt for this execution."""
        base = self.load_prompt(self._settings.system_prompt_template)
        return f"{base}\nReturn at most {self._settings.max_assumptions} assumptions."

    @staticmethod
    def _user_prompt(facts: tuple[Fact, ...], *, subject: str) -> str:
        """Serialise the analysis input as compact JSON."""
        return json.dumps(
            {
                "subject": subject,
                "facts": [
                    {
                        "id": str(fact.id),
                        "category": fact.category,
                        "statement": fact.statement,
                        "confidence": fact.confidence_score,
                    }
                    for fact in facts
                ],
            },
            ensure_ascii=False,
        )
