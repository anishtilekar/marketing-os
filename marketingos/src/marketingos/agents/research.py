"""ResearchAgent — collection of publicly available business information.

The :class:`ResearchAgent` coordinates the project's data-collection tools
(website scraper, Instagram public reader, web search) to gather factual
observations about a business identified by a website URL, an Instagram
username, and/or a business name.

Scope and guarantees
--------------------
* The agent performs **no scraping itself** — it orchestrates injected tools
  that satisfy the port protocols defined below (implemented in
  ``marketingos.tools``).
* Its output contains **only factual observations**: every
  :class:`ObservedFact` is a direct restatement of data returned by a tool,
  carries its source metadata, and is never inferred, extrapolated or
  analysed. Strategic interpretation happens in downstream agents.
* Partial failure is tolerated: if one source channel fails, its failure is
  logged and recorded in ``ResearchResult.failed_sources`` and the overall
  confidence score is reduced accordingly. Only when *every* runnable
  channel fails does the agent raise :class:`ResearchCollectionError`
  (retryable, since the dominant failure mode is transient network trouble).
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from marketingos.agents.base import (
    AgentConfig,
    BaseAgent,
    MemoryStore,
    PermanentAgentError,
    PromptRepository,
    RetryableAgentError,
    ToolRegistry,
)

__all__ = [
    "BusinessSearchPort",
    "ContactDetails",
    "FactCategory",
    "InstagramProfileSnapshot",
    "InstagramReaderPort",
    "NoResearchChannelError",
    "ObservedFact",
    "ResearchAgent",
    "ResearchAgentConfig",
    "ResearchCollectionError",
    "ResearchInput",
    "ResearchResult",
    "SearchResultSnapshot",
    "SourceMetadata",
    "SourceType",
    "WebsiteScraperPort",
    "WebsiteSnapshot",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SourceType(StrEnum):
    """The kind of public source an observation was collected from."""

    WEBSITE = "website"
    INSTAGRAM = "instagram"
    WEB_SEARCH = "web_search"


class FactCategory(StrEnum):
    """Classification of an observed fact, used by downstream agents."""

    WEBSITE_CONTENT = "website_content"
    ABOUT = "about"
    PRODUCTS_SERVICES = "products_services"
    CONTACT = "contact"
    BRAND_MESSAGING = "brand_messaging"
    SOCIAL_PRESENCE = "social_presence"
    PUBLIC_INFO = "public_info"


# ---------------------------------------------------------------------------
# Tool contracts (ports)
# ---------------------------------------------------------------------------
# The agent depends on these structural protocols, not on the concrete tool
# classes in ``marketingos.tools`` (WebsiteScraper, InstagramPublicReader,
# SearchTool). The snapshot models below define the data shape each tool
# returns; they are the boundary contract between tools and this agent.


class ContactDetails(BaseModel):
    """Contact information extracted from a website by the scraper tool."""

    model_config = ConfigDict(frozen=True)

    emails: tuple[str, ...] = ()
    phone_numbers: tuple[str, ...] = ()
    addresses: tuple[str, ...] = ()


class WebsiteSnapshot(BaseModel):
    """Factual content extracted from a business website by the scraper."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: str | None = None
    tagline: str | None = None
    about_text: str | None = None
    main_text: str | None = None
    products_services: tuple[str, ...] = ()
    contact: ContactDetails = Field(default_factory=ContactDetails)
    pages_visited: tuple[str, ...] = ()


class InstagramProfileSnapshot(BaseModel):
    """Public profile data returned by the Instagram reader tool."""

    model_config = ConfigDict(frozen=True)

    username: str
    profile_url: str
    full_name: str | None = None
    biography: str | None = None
    external_url: str | None = None
    follower_count: int | None = Field(default=None, ge=0)
    post_count: int | None = Field(default=None, ge=0)
    is_verified: bool | None = None


class SearchResultSnapshot(BaseModel):
    """A single public web-search result returned by the search tool."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str


@runtime_checkable
class WebsiteScraperPort(Protocol):
    """Port satisfied by ``marketingos.tools.WebsiteScraper``."""

    async def scrape(self, url: str) -> WebsiteSnapshot:
        """Fetch and extract the public content of ``url``."""
        ...


@runtime_checkable
class InstagramReaderPort(Protocol):
    """Port satisfied by ``marketingos.tools.InstagramPublicReader``."""

    async def fetch_profile(self, username: str) -> InstagramProfileSnapshot:
        """Fetch the public Instagram profile for ``username``."""
        ...


@runtime_checkable
class BusinessSearchPort(Protocol):
    """Port satisfied by ``marketingos.tools.SearchTool``."""

    async def search(
        self, query: str, *, max_results: int
    ) -> tuple[SearchResultSnapshot, ...]:
        """Run a public web search and return up to ``max_results`` results."""
        ...


# ---------------------------------------------------------------------------
# Input / output schemas
# ---------------------------------------------------------------------------


class ResearchInput(BaseModel):
    """Identifies the business to research.

    At least one identifier must be provided. Each identifier enables the
    corresponding collection channel (website scraping, Instagram profile
    reading, public web search).
    """

    model_config = ConfigDict(frozen=True)

    website_url: HttpUrl | None = None
    instagram_username: str | None = None
    business_name: str | None = None

    @field_validator("instagram_username")
    @classmethod
    def _normalize_username(cls, value: str | None) -> str | None:
        """Strip whitespace and a leading '@' from the username."""
        if value is None:
            return None
        cleaned = value.strip().lstrip("@")
        if not cleaned:
            raise ValueError("instagram_username must not be empty")
        return cleaned

    @field_validator("business_name")
    @classmethod
    def _normalize_business_name(cls, value: str | None) -> str | None:
        """Strip whitespace; treat an empty string as not provided."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _require_any_identifier(self) -> ResearchInput:
        """Reject inputs that identify nothing to research."""
        if not (self.website_url or self.instagram_username or self.business_name):
            raise ValueError(
                "At least one of website_url, instagram_username or "
                "business_name must be provided."
            )
        return self


class SourceMetadata(BaseModel):
    """Provenance of an observed fact: where and how it was collected."""

    model_config = ConfigDict(frozen=True)

    source_type: SourceType
    tool_name: str
    url: str | None = None
    retrieved_at: datetime


class ObservedFact(BaseModel):
    """A single factual observation collected from a public source.

    The ``statement`` is a direct restatement of tool output — never an
    inference — and always references its :class:`SourceMetadata`.
    """

    model_config = ConfigDict(frozen=True)

    category: FactCategory
    statement: str = Field(min_length=1)
    source: SourceMetadata
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchResult(BaseModel):
    """The complete, typed output of one :class:`ResearchAgent` execution."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    subject: str = Field(
        description="Human-readable identifier of the researched business."
    )
    facts: tuple[ObservedFact, ...]
    sources: tuple[SourceMetadata, ...]
    urls_visited: tuple[str, ...] = Field(
        description="URLs the tools actually fetched (search-result URLs "
        "that were only listed, not fetched, are excluded)."
    )
    failed_sources: tuple[str, ...] = Field(
        default=(),
        description="Channels that could not be collected, with the reason.",
    )
    collected_at: datetime
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean per-fact confidence weighted by the fraction of "
        "requested channels that succeeded.",
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ResearchCollectionError(RetryableAgentError):
    """Every runnable source channel failed during collection.

    Classified as retryable because the dominant cause is transient network
    or upstream-service trouble.
    """


class NoResearchChannelError(PermanentAgentError):
    """The input requested channels for which no tool was injected."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ResearchAgentConfig(AgentConfig):
    """Runtime settings specific to :class:`ResearchAgent`."""

    max_text_excerpt_chars: int = Field(
        default=2000,
        ge=100,
        description="Upper bound on the length of any single text excerpt "
        "recorded as a fact statement.",
    )
    max_search_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of web-search results to record.",
    )


# ---------------------------------------------------------------------------
# Internal aggregation structure
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ChannelHarvest:
    """Mutable accumulator for the output of one collection channel."""

    facts: list[ObservedFact] = field(default_factory=list)
    sources: list[SourceMetadata] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ResearchAgent(BaseAgent[ResearchInput, ResearchResult]):
    """Collects publicly available business information via injected tools.

    Each identifier present in the :class:`ResearchInput` activates one
    collection channel, provided the matching tool was injected:

    ==================  ==========================  =====================
    Input field         Tool port                   Source type
    ==================  ==========================  =====================
    website_url         :class:`WebsiteScraperPort`  ``WEBSITE``
    instagram_username  :class:`InstagramReaderPort` ``INSTAGRAM``
    business_name       :class:`BusinessSearchPort`  ``WEB_SEARCH``
    ==================  ==========================  =====================

    Channels run concurrently. The agent converts each tool snapshot into
    :class:`ObservedFact` records verbatim — it adds no interpretation, no
    strategy and no assumptions.
    """

    #: Baseline confidence assigned to facts by source type. First-party
    #: sources (the business's own website and profile) rank above
    #: third-party search snippets.
    SOURCE_CONFIDENCE: Final[Mapping[SourceType, float]] = {
        SourceType.WEBSITE: 0.9,
        SourceType.INSTAGRAM: 0.85,
        SourceType.WEB_SEARCH: 0.6,
    }

    def __init__(
        self,
        *,
        website_scraper: WebsiteScraperPort | None = None,
        instagram_reader: InstagramReaderPort | None = None,
        search_tool: BusinessSearchPort | None = None,
        name: str | None = None,
        config: ResearchAgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent with its collection tools.

        Args:
            website_scraper: Tool for extracting website content.
            instagram_reader: Tool for reading public Instagram profiles.
            search_tool: Tool for public web search.
            name: Logical agent name; defaults to the class name.
            config: Research-specific runtime settings.
            memory: Optional memory backend.
            tools: Optional tool registry.
            prompts: Optional prompt repository.
        """
        settings = config or ResearchAgentConfig()
        super().__init__(
            name=name, config=settings, memory=memory, tools=tools, prompts=prompts
        )
        self._settings = settings
        self._website_scraper = website_scraper
        self._instagram_reader = instagram_reader
        self._search_tool = search_tool

    # -- domain logic -----------------------------------------------------------

    async def run(self, payload: ResearchInput, *, run_id: str) -> ResearchResult:
        """Collect observations from every runnable channel concurrently.

        Args:
            payload: The business identifiers to research.
            run_id: Identifier of this execution.

        Returns:
            A :class:`ResearchResult` containing only factual observations.

        Raises:
            NoResearchChannelError: If no injected tool matches any provided
                identifier (permanent — retrying cannot help).
            ResearchCollectionError: If every runnable channel failed
                (retryable — likely transient).
        """
        channels, skipped = self._plan_channels(payload)
        if not channels:
            raise NoResearchChannelError(
                "No collection channel is runnable for the given input: "
                + "; ".join(skipped),
                agent_name=self.name,
                run_id=run_id,
            )

        labels = [label for label, _ in channels]
        outcomes = await asyncio.gather(
            *(coro for _, coro in channels), return_exceptions=True
        )

        facts: list[ObservedFact] = []
        sources: list[SourceMetadata] = []
        urls_visited: list[str] = []
        failures: list[str] = list(skipped)

        for label, outcome in zip(labels, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures.append(f"{label}: {type(outcome).__name__}: {outcome}")
                self._logger.bind(
                    run_id=run_id,
                    event="research.channel_failed",
                    channel=label,
                    error_type=type(outcome).__name__,
                ).warning("Collection channel failed")
                continue
            facts.extend(outcome.facts)
            sources.extend(outcome.sources)
            urls_visited.extend(outcome.urls)

        if not sources:
            raise ResearchCollectionError(
                "All collection channels failed: " + "; ".join(failures),
                agent_name=self.name,
                run_id=run_id,
            )

        succeeded = len(channels) - sum(
            1 for outcome in outcomes if isinstance(outcome, BaseException)
        )
        confidence = self._overall_confidence(
            facts, succeeded=succeeded, attempted=len(channels)
        )

        return ResearchResult(
            run_id=run_id,
            subject=self._subject(payload),
            facts=tuple(facts),
            sources=tuple(sources),
            urls_visited=tuple(dict.fromkeys(urls_visited)),
            failed_sources=tuple(failures),
            collected_at=self._utcnow(),
            confidence_score=confidence,
        )

    # -- channel planning ------------------------------------------------------------

    def _plan_channels(
        self, payload: ResearchInput
    ) -> tuple[
        list[tuple[str, Coroutine[None, None, _ChannelHarvest]]], list[str]
    ]:
        """Match provided identifiers against injected tools.

        Returns:
            A pair of (runnable channels as ``(label, coroutine)`` pairs,
            reasons for channels that were requested but cannot run).
        """
        channels: list[tuple[str, Coroutine[None, None, _ChannelHarvest]]] = []
        skipped: list[str] = []

        if payload.website_url is not None:
            if self._website_scraper is not None:
                channels.append(
                    ("website", self._collect_website(str(payload.website_url)))
                )
            else:
                skipped.append("website: no WebsiteScraper tool injected")

        if payload.instagram_username is not None:
            if self._instagram_reader is not None:
                channels.append(
                    ("instagram", self._collect_instagram(payload.instagram_username))
                )
            else:
                skipped.append("instagram: no InstagramPublicReader tool injected")

        if payload.business_name is not None:
            if self._search_tool is not None:
                channels.append(
                    ("web_search", self._collect_search(payload.business_name))
                )
            else:
                skipped.append("web_search: no SearchTool injected")

        return channels, skipped

    # -- channel collectors ---------------------------------------------------------------

    async def _collect_website(self, url: str) -> _ChannelHarvest:
        """Scrape ``url`` and restate its content as observed facts."""
        assert self._website_scraper is not None  # guarded by _plan_channels
        snapshot = await self._website_scraper.scrape(url)
        source = self._source(
            SourceType.WEBSITE, tool=self._website_scraper, url=snapshot.url
        )
        confidence = self.SOURCE_CONFIDENCE[SourceType.WEBSITE]
        harvest = _ChannelHarvest(
            sources=[source],
            urls=[snapshot.url, *snapshot.pages_visited],
        )

        def add(category: FactCategory, statement: str) -> None:
            harvest.facts.append(
                ObservedFact(
                    category=category,
                    statement=statement,
                    source=source,
                    confidence=confidence,
                )
            )

        if snapshot.title:
            add(
                FactCategory.WEBSITE_CONTENT,
                f'The website {snapshot.url} has the page title "{snapshot.title}".',
            )
        if snapshot.tagline:
            add(
                FactCategory.BRAND_MESSAGING,
                f'The website {snapshot.url} displays the tagline '
                f'"{snapshot.tagline}".',
            )
        if snapshot.about_text:
            add(
                FactCategory.ABOUT,
                f'The about section of {snapshot.url} states: '
                f'"{self._excerpt(snapshot.about_text)}"',
            )
        if snapshot.main_text:
            add(
                FactCategory.WEBSITE_CONTENT,
                f'The main content of {snapshot.url} includes: '
                f'"{self._excerpt(snapshot.main_text)}"',
            )
        for offering in snapshot.products_services:
            add(
                FactCategory.PRODUCTS_SERVICES,
                f'The website {snapshot.url} lists the offering "{offering}".',
            )
        for email in snapshot.contact.emails:
            add(
                FactCategory.CONTACT,
                f"The website {snapshot.url} lists the contact email {email}.",
            )
        for phone in snapshot.contact.phone_numbers:
            add(
                FactCategory.CONTACT,
                f"The website {snapshot.url} lists the phone number {phone}.",
            )
        for address in snapshot.contact.addresses:
            add(
                FactCategory.CONTACT,
                f'The website {snapshot.url} lists the address "{address}".',
            )
        return harvest

    async def _collect_instagram(self, username: str) -> _ChannelHarvest:
        """Read the public profile of ``username`` and restate it as facts."""
        assert self._instagram_reader is not None  # guarded by _plan_channels
        profile = await self._instagram_reader.fetch_profile(username)
        source = self._source(
            SourceType.INSTAGRAM, tool=self._instagram_reader, url=profile.profile_url
        )
        confidence = self.SOURCE_CONFIDENCE[SourceType.INSTAGRAM]
        harvest = _ChannelHarvest(sources=[source], urls=[profile.profile_url])
        handle = f"@{profile.username}"

        def add(statement: str) -> None:
            harvest.facts.append(
                ObservedFact(
                    category=FactCategory.SOCIAL_PRESENCE,
                    statement=statement,
                    source=source,
                    confidence=confidence,
                )
            )

        add(f"An Instagram profile {handle} exists at {profile.profile_url}.")
        if profile.full_name:
            add(
                f'The Instagram profile {handle} displays the name '
                f'"{profile.full_name}".'
            )
        if profile.biography:
            add(
                f'The Instagram biography of {handle} states: '
                f'"{self._excerpt(profile.biography)}"'
            )
        if profile.external_url:
            add(f"The Instagram profile {handle} links to {profile.external_url}.")
        if profile.follower_count is not None:
            add(
                f"The Instagram profile {handle} has "
                f"{profile.follower_count} followers."
            )
        if profile.post_count is not None:
            add(f"The Instagram profile {handle} has {profile.post_count} posts.")
        if profile.is_verified is not None:
            status = "verified" if profile.is_verified else "not verified"
            add(f"The Instagram profile {handle} is {status}.")
        return harvest

    async def _collect_search(self, business_name: str) -> _ChannelHarvest:
        """Search the public web for ``business_name`` and record snippets.

        Search-result URLs are recorded in each fact's source metadata but
        not in ``urls_visited``, because the pages themselves were not
        fetched — only listed by the search engine.
        """
        assert self._search_tool is not None  # guarded by _plan_channels
        results = await self._search_tool.search(
            business_name, max_results=self._settings.max_search_results
        )
        harvest = _ChannelHarvest()
        confidence = self.SOURCE_CONFIDENCE[SourceType.WEB_SEARCH]
        for result in results:
            source = self._source(
                SourceType.WEB_SEARCH, tool=self._search_tool, url=result.url
            )
            harvest.sources.append(source)
            harvest.facts.append(
                ObservedFact(
                    category=FactCategory.PUBLIC_INFO,
                    statement=(
                        f'A public search result titled "{result.title}" at '
                        f'{result.url} states: "{self._excerpt(result.snippet)}"'
                    ),
                    source=source,
                    confidence=confidence,
                )
            )
        return harvest

    # -- helpers -------------------------------------------------------------------------------

    def _source(
        self, source_type: SourceType, *, tool: object, url: str | None
    ) -> SourceMetadata:
        """Build source metadata for one tool retrieval."""
        return SourceMetadata(
            source_type=source_type,
            tool_name=type(tool).__name__,
            url=url,
            retrieved_at=self._utcnow(),
        )

    def _excerpt(self, text: str) -> str:
        """Normalise whitespace and cap the excerpt length at a word boundary.

        Truncation is marked with a Unicode ellipsis so downstream consumers
        can tell that the excerpt is partial.
        """
        collapsed = " ".join(text.split())
        limit = self._settings.max_text_excerpt_chars
        if len(collapsed) <= limit:
            return collapsed
        cut = collapsed.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        return collapsed[:cut].rstrip() + "…"

    @staticmethod
    def _overall_confidence(
        facts: list[ObservedFact], *, succeeded: int, attempted: int
    ) -> float:
        """Mean per-fact confidence weighted by the channel success ratio."""
        if not facts or attempted == 0:
            return 0.0
        mean_fact_confidence = sum(fact.confidence for fact in facts) / len(facts)
        return round(mean_fact_confidence * (succeeded / attempted), 4)

    @staticmethod
    def _subject(payload: ResearchInput) -> str:
        """Return the most human-readable identifier of the researched business."""
        if payload.business_name:
            return payload.business_name
        if payload.website_url:
            return str(payload.website_url)
        return f"@{payload.instagram_username}"

    @staticmethod
    def _utcnow() -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        return datetime.now(UTC)
