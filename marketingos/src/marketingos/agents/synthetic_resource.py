"""SyntheticSourceAgent — turns research output into reusable source material.

The :class:`SyntheticSourceAgent` consumes a
:class:`~marketingos.agents.research.ResearchResult` and produces
:class:`SyntheticSourceMaterial`: deduplicated, neutrally phrased,
copyright-safe text that downstream content agents can reuse.

Scope and guarantees
--------------------
* **Purely transformational.** Every output statement is derived from an
  observed fact by deterministic normalisation (whitespace collapsing,
  neutral rephrasing, condensing, deduplication). The agent never invents
  information, never infers missing facts, never creates strategy and never
  performs marketing analysis.
* **Neutral voice.** First-person marketing phrasing ("we offer", "our
  team") is rewritten into third-person neutral phrasing ("the business
  offers", "the business's team") so the material is reusable in any
  context.
* **Copyright hygiene.** Long verbatim passages are condensed to a bounded
  excerpt at sentence boundaries, so the output never reproduces extended
  source copy word for word.
* **Structured separation.** Output is partitioned into facts (verifiable
  data points), descriptions (prose about the business), brand
  characteristics (observed messaging) and keywords (frequent content
  terms).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from marketingos.agents.base import (
    AgentConfig,
    BaseAgent,
    MemoryStore,
    PermanentAgentError,
    PromptRepository,
    ToolRegistry,
)
from marketingos.agents.research import FactCategory, ObservedFact, ResearchResult

__all__ = [
    "EmptyResearchError",
    "SyntheticSourceAgent",
    "SyntheticSourceAgentConfig",
    "SyntheticSourceMaterial",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmptyResearchError(PermanentAgentError):
    """The research result contains no observed facts to transform.

    Permanent: retrying the transformation cannot create source material
    out of an empty research result.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SyntheticSourceAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`SyntheticSourceAgent`."""

    max_keywords: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of keywords to extract.",
    )
    min_keyword_length: int = Field(
        default=3,
        ge=2,
        description="Minimum character length for a keyword candidate.",
    )
    max_statement_chars: int = Field(
        default=400,
        ge=50,
        description="Upper bound on the length of any rewritten statement; "
        "longer passages are condensed at sentence boundaries.",
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class SyntheticSourceMaterial(BaseModel):
    """Reusable, neutral source material derived from a research result.

    Sections:
        facts: Verifiable data points (contact details, offerings, social
            metrics, public references).
        descriptions: Neutral prose describing the business, derived from
            about sections and website content.
        brand_characteristics: Observed messaging traits (taglines, tone of
            voice statements), rewritten neutrally.
        keywords: Frequent content-bearing terms across all material,
            ordered by frequency.

    Traceability fields link the material back to the research execution it
    was derived from.
    """

    model_config = ConfigDict(frozen=True)

    subject: str
    facts: tuple[str, ...]
    descriptions: tuple[str, ...]
    brand_characteristics: tuple[str, ...]
    keywords: tuple[str, ...]
    source_run_id: str = Field(
        description="run_id of the ResearchAgent execution this material "
        "was derived from."
    )
    source_urls: tuple[str, ...]
    source_confidence: float = Field(ge=0.0, le=1.0)
    source_collected_at: datetime
    created_at: datetime


# ---------------------------------------------------------------------------
# Transformation tables
# ---------------------------------------------------------------------------

#: Category partition used to route observed facts into output sections.
_FACT_CATEGORIES: Final[frozenset[FactCategory]] = frozenset(
    {
        FactCategory.CONTACT,
        FactCategory.PRODUCTS_SERVICES,
        FactCategory.SOCIAL_PRESENCE,
        FactCategory.PUBLIC_INFO,
    }
)
_DESCRIPTION_CATEGORIES: Final[frozenset[FactCategory]] = frozenset(
    {FactCategory.ABOUT, FactCategory.WEBSITE_CONTENT}
)
_BRAND_CATEGORIES: Final[frozenset[FactCategory]] = frozenset(
    {FactCategory.BRAND_MESSAGING}
)

#: Ordered first-person → neutral third-person rewrites. Longer phrases come
#: first so they win over their single-word substrings. The bare "us" rule
#: is intentionally case-sensitive (lowercase only) so it never rewrites the
#: country abbreviation "US".
_NEUTRAL_REWRITES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\bwe are\b", re.IGNORECASE), "the business is"),
    (re.compile(r"\bwe're\b", re.IGNORECASE), "the business is"),
    (re.compile(r"\bwe offer\b", re.IGNORECASE), "the business offers"),
    (re.compile(r"\bwe provide\b", re.IGNORECASE), "the business provides"),
    (re.compile(r"\bwe\b", re.IGNORECASE), "the business"),
    (re.compile(r"\bour\b", re.IGNORECASE), "the business's"),
    (re.compile(r"\bours\b", re.IGNORECASE), "the business's"),
    (re.compile(r"\bus\b"), "the business"),
)

#: Words excluded from keyword extraction: English function words plus the
#: scaffolding vocabulary the ResearchAgent uses to phrase observations
#: (e.g. "website", "states", "lists"), which describes the collection
#: process rather than the business itself.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        # function words
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
        "for", "from", "has", "have", "her", "his", "in", "into", "is",
        "it", "its", "more", "most", "no", "not", "of", "on", "or", "our",
        "so", "than", "that", "the", "their", "them", "they", "this", "to",
        "was", "we", "were", "will", "with", "you", "your",
        # research-agent scaffolding vocabulary
        "about", "address", "biography", "business", "contact", "content",
        "displays", "email", "exists", "followers", "http", "https",
        "includes", "instagram", "links", "lists", "main", "name", "number",
        "offering", "page", "phone", "posts", "profile", "public", "result",
        "search", "section", "states", "tagline", "title", "titled",
        "verified", "website", "www",
    }
)

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9\-']*")
_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")
_TERMINAL_PUNCTUATION: Final[frozenset[str]] = frozenset(".!?\"'")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SyntheticSourceAgent(BaseAgent[ResearchResult, SyntheticSourceMaterial]):
    """Transforms a :class:`ResearchResult` into reusable source material.

    The transformation is deterministic and purely mechanical:

    1. **Partition** — observed facts are routed into the facts,
       descriptions or brand-characteristics section by category.
    2. **Normalise** — whitespace is collapsed, first-person marketing
       phrasing is rewritten into neutral third person, and long passages
       are condensed at sentence boundaries.
    3. **Deduplicate** — statements with identical content fingerprints
       (case- and punctuation-insensitive) are collapsed, keeping the first
       occurrence.
    4. **Extract keywords** — frequent content-bearing terms are collected
       across all normalised statements.

    Because no step generates new content, the output is guaranteed to
    contain only information present in the input research.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        config: SyntheticSourceAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            name: Logical agent name; defaults to the class name.
            config: Transformation settings (keyword limits, statement
                length bound).
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository.
        """
        settings = config or SyntheticSourceAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings

    # -- domain logic -----------------------------------------------------------

    async def run(
        self, payload: ResearchResult, *, run_id: str
    ) -> SyntheticSourceMaterial:
        """Transform the research result into synthetic source material.

        Args:
            payload: The output of a :class:`ResearchAgent` execution.
            run_id: Identifier of this execution.

        Returns:
            Structured, deduplicated, neutrally phrased source material.

        Raises:
            EmptyResearchError: If the research result contains no facts.
        """
        if not payload.facts:
            raise EmptyResearchError(
                f"Research result {payload.run_id} contains no observed "
                "facts to transform.",
                agent_name=self.name,
                run_id=run_id,
            )

        facts = self._build_section(payload.facts, _FACT_CATEGORIES)
        descriptions = self._build_section(payload.facts, _DESCRIPTION_CATEGORIES)
        brand = self._build_section(payload.facts, _BRAND_CATEGORIES)
        keywords = self._extract_keywords((*facts, *descriptions, *brand))

        self._logger.bind(
            run_id=run_id,
            event="synthetic_source.transformed",
            input_facts=len(payload.facts),
            output_facts=len(facts),
            output_descriptions=len(descriptions),
            output_brand_characteristics=len(brand),
            output_keywords=len(keywords),
        ).debug("Research transformed into source material")

        return SyntheticSourceMaterial(
            subject=payload.subject,
            facts=facts,
            descriptions=descriptions,
            brand_characteristics=brand,
            keywords=keywords,
            source_run_id=payload.run_id,
            source_urls=payload.urls_visited,
            source_confidence=payload.confidence_score,
            source_collected_at=payload.collected_at,
            created_at=datetime.now(UTC),
        )

    # -- section construction --------------------------------------------------------

    def _build_section(
        self,
        facts: tuple[ObservedFact, ...],
        categories: frozenset[FactCategory],
    ) -> tuple[str, ...]:
        """Normalise and deduplicate the statements of one output section.

        Facts are processed in input order (research output is already
        ordered by source priority), so the first occurrence of duplicated
        content wins.
        """
        deduplicated: dict[str, str] = {}
        for fact in facts:
            if fact.category not in categories:
                continue
            statement = self._normalize_statement(fact.statement)
            if not statement:
                continue
            fingerprint = self._fingerprint(statement)
            if fingerprint not in deduplicated:
                deduplicated[fingerprint] = statement
        return tuple(deduplicated.values())

    def _normalize_statement(self, text: str) -> str:
        """Rewrite one observed statement into neutral, reusable text.

        Applies, in order: whitespace collapsing, neutral third-person
        rewriting, sentence-boundary condensing, capitalisation, and
        terminal punctuation. No step adds information.
        """
        collapsed = " ".join(text.split())
        if not collapsed:
            return ""
        neutral = self._neutralize(collapsed)
        condensed = self._condense(neutral)
        capitalized = condensed[0].upper() + condensed[1:]
        if capitalized[-1] not in _TERMINAL_PUNCTUATION:
            capitalized += "."
        return capitalized

    @staticmethod
    def _neutralize(text: str) -> str:
        """Rewrite first-person marketing phrasing into neutral third person."""
        result = text
        for pattern, replacement in _NEUTRAL_REWRITES:
            result = pattern.sub(replacement, result)
        return result

    def _condense(self, text: str) -> str:
        """Bound the statement length, cutting at sentence boundaries.

        Whole sentences are kept while they fit within
        ``config.max_statement_chars``. A single over-long sentence is
        truncated at a word boundary and marked with an ellipsis, so the
        output never reproduces extended verbatim passages.
        """
        limit = self._settings.max_statement_chars
        if len(text) <= limit:
            return text
        kept: list[str] = []
        length = 0
        for sentence in _SENTENCE_BOUNDARY.split(text):
            addition = len(sentence) + (1 if kept else 0)
            if length + addition > limit:
                break
            kept.append(sentence)
            length += addition
        if kept:
            return " ".join(kept)
        cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        return text[:cut].rstrip() + "…"

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Return a case- and punctuation-insensitive identity for ``text``."""
        stripped = re.sub(r"[^a-z0-9 ]", "", text.lower())
        return " ".join(stripped.split())

    # -- keyword extraction ---------------------------------------------------------------

    def _extract_keywords(self, statements: tuple[str, ...]) -> tuple[str, ...]:
        """Collect frequent content-bearing terms across all statements.

        Tokens are lowercased, filtered against the stopword list and a
        minimum length, then ranked by frequency (ties broken
        alphabetically for deterministic output).
        """
        counts: Counter[str] = Counter()
        for statement in statements:
            for token in _TOKEN_PATTERN.findall(statement.lower()):
                cleaned = token.strip("-'")
                if (
                    len(cleaned) >= self._settings.min_keyword_length
                    and cleaned not in _STOPWORDS
                    and not cleaned.isdigit()
                ):
                    counts[cleaned] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return tuple(token for token, _ in ranked[: self._settings.max_keywords])