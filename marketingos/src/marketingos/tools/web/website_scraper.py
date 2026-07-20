"""WebsiteScraper — public business-website content extraction.

Fetches exactly one URL (no crawling) and extracts title, tagline, about
text, main text, ``h1``–``h3`` headings, JSON-LD organization identity
(name, description, official social profiles), a heuristic list of
offerings, and any ``mailto:``/``tel:`` contact links, satisfying
:class:`marketingos.agents.research.WebsiteScraperPort`. Parsing uses the
stdlib-only helper in ``tools/web/_html.py`` — no BeautifulSoup/lxml
dependency for the modest slice of a page this needs.

Headings and JSON-LD exist precisely because search engines demand them,
so they survive in the initial HTML of JS-rendered storefronts whose
``<p>`` markup is nothing but navigation labels — the pages where the
paragraph heuristics alone used to come back nearly empty. Paragraphs
shorter than a few words are treated as nav labels and excluded from
``main_text`` for the same reason.

Compliance, before any content is parsed
------------------------------------------
Two hard-refusal gates run first, per the architecture doc's "public,
unauthenticated read paths only" rule:

1. **robots.txt** — checked with the standard library's
   ``urllib.robotparser.RobotFileParser``; disallowed paths raise rather
   than being fetched.
2. **Login-wall detection** — the shared
   :func:`~marketingos.tools.web._compliance.refuse_if_login_walled`
   guard, applied to every fetched page.

Neither gate is configurable past on/off for robots.txt (useful for
tests against a fixture server with no robots.txt at all); there is no
"scrape anyway" argument.

Heuristic extraction, not perfect extraction
-----------------------------------------------
``products_services`` is every ``<li>`` on the page, deduplicated — this
will include nav/footer links on many real sites, not only an actual
offerings list. ``about_text`` is the first paragraph whose opening
words mention "about". Both are pragmatic heuristics, not a layout
classifier; ``ResearchAgent`` already treats every field here as an
unverified observation, not a judgement call this tool is trusted to
get exactly right.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Final
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from marketingos.agents.research import (
    ContactDetails,
    OrganizationFacts,
    WebsiteSnapshot,
)
from marketingos.exceptions.tool import ToolExecutionError
from marketingos.models.cost import CostCategory
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.base import Tool
from marketingos.tools.web._compliance import refuse_if_login_walled
from marketingos.tools.web._html import PageExtract, extract_page

__all__ = ["WEBSITE_SCRAPING", "WebsiteScrapeRequest", "WebsiteScraper"]

WEBSITE_SCRAPING: Final[str] = "website_scraping"
_DEFAULT_USER_AGENT: Final[str] = "MarketingOS-ResearchBot/1.0 (+public content only)"
_MAX_LIST_ITEMS: Final[int] = 20
_MAX_TEXT_CHARS: Final[int] = 5000
_MAX_HEADINGS: Final[int] = 12
_MAX_SOCIAL_PROFILES: Final[int] = 8

#: Paragraphs shorter than this many words are treated as navigation
#: labels ("Featured", "Shop Icons", ...) rather than content. JS-rendered
#: storefronts mark up their nav as ``<p>`` tags, and joining those labels
#: produced a garbage "main text" that drowned out real facts.
_MIN_PARAGRAPH_WORDS: Final[int] = 4

#: JSON-LD ``@type`` values treated as the site's own organization record.
_ORG_TYPES: Final[frozenset[str]] = frozenset(
    {"organization", "corporation", "localbusiness", "brand", "onlinestore"}
)


def _org_from_json_ld(blocks: list[dict[str, Any]]) -> OrganizationFacts | None:
    """Pull the first organization-shaped JSON-LD node into typed facts."""
    for node in blocks:
        raw_type = node.get("@type", "")
        types = raw_type if isinstance(raw_type, list) else [raw_type]
        if not any(str(t).lower() in _ORG_TYPES for t in types):
            continue
        name = node.get("name")
        description = node.get("description")
        same_as = node.get("sameAs", [])
        if isinstance(same_as, str):
            same_as = [same_as]
        profiles = tuple(
            str(u) for u in same_as if isinstance(u, str) and u.startswith("http")
        )[:_MAX_SOCIAL_PROFILES]
        if not (name or description or profiles):
            continue
        return OrganizationFacts(
            name=str(name) if isinstance(name, str) and name.strip() else None,
            description=(
                str(description)
                if isinstance(description, str) and description.strip()
                else None
            ),
            same_as=profiles,
        )
    return None


def _substantive_paragraphs(page: PageExtract) -> list[str]:
    """Drop nav-label fragments, keeping only sentence-like paragraphs."""
    return [
        p for p in page.paragraphs if len(p.split()) >= _MIN_PARAGRAPH_WORDS
    ]


class WebsiteScrapeRequest(BaseModel):
    """One page to fetch and extract."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(min_length=1)


class WebsiteScraper(Tool[WebsiteScrapeRequest, WebsiteSnapshot]):
    """Fetches one public page and extracts its factual content."""

    def __init__(
        self,
        *,
        cost_guard: CostGuard,
        user_agent: str = _DEFAULT_USER_AGENT,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
        respect_robots_txt: bool = True,
    ) -> None:
        """Initialise the scraper.

        Args:
            cost_guard: Guard enforcing the run's budget. Required: see
                ``Tool.cost_guard``.
            user_agent: Sent on every request and to ``robots.txt`` for
                the ``can_fetch`` check.
            http_client: Transport to use. Defaults to a client owned by
                this instance; tests inject one with a mock transport.
            timeout_seconds: Per-request timeout.
            respect_robots_txt: Set ``False`` only for fixture servers
                with no ``robots.txt`` at all (tests). Never disabled in
                production use.
        """
        self._cost_guard = cost_guard
        self._user_agent = user_agent
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._respect_robots_txt = respect_robots_txt
        self._logger = logger.bind(component="WebsiteScraper")

    # -- Tool identity -------------------------------------------------------

    @property
    def name(self) -> str:
        return "website-scraper"

    @property
    def capability(self) -> str:
        return WEBSITE_SCRAPING

    @property
    def provider(self) -> str:
        return "direct-fetch"

    @property
    def cost_category(self) -> CostCategory:
        return CostCategory.WEB_TOOL

    @property
    def input_schema(self) -> type[WebsiteScrapeRequest]:
        return WebsiteScrapeRequest

    @property
    def output_schema(self) -> type[WebsiteSnapshot]:
        return WebsiteSnapshot

    @property
    def cost_guard(self) -> CostGuard:
        return self._cost_guard

    # -- cost ------------------------------------------------------------

    def cost_estimate(self, payload: WebsiteScrapeRequest) -> Decimal:
        """Zero — a direct HTTP GET has no vendor bill."""
        return Decimal("0")

    # -- invocation ----------------------------------------------------------

    async def invoke(self, payload: WebsiteScrapeRequest) -> WebsiteSnapshot:
        """Fetch, compliance-gate, and extract ``payload.url``.

        Raises:
            ToolExecutionError: robots.txt disallows the fetch, the page
                reads as login-walled, the request fails, or the host is
                unreachable.
        """
        if self._respect_robots_txt:
            await self._check_robots(payload.url)

        try:
            response = await self._client.get(
                payload.url,
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"{payload.url} returned {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"Fetching {payload.url} failed: {exc}") from exc

        html = response.text
        refuse_if_login_walled(html, source=payload.url)
        page = extract_page(html)

        tagline = page.meta.get("og:description") or page.meta.get("description")
        paragraphs = _substantive_paragraphs(page)
        about = next((p for p in paragraphs if "about" in p[:80].lower()), None)
        main_text = " ".join(paragraphs)[:_MAX_TEXT_CHARS] or None

        snapshot = WebsiteSnapshot(
            url=str(response.url),
            title=page.title,
            tagline=tagline,
            about_text=about,
            main_text=main_text,
            headings=tuple(dict.fromkeys(page.headings))[:_MAX_HEADINGS],
            organization=_org_from_json_ld(page.json_ld),
            products_services=tuple(dict.fromkeys(page.list_items))[:_MAX_LIST_ITEMS],
            contact=ContactDetails(
                emails=tuple(dict.fromkeys(page.emails)),
                phone_numbers=tuple(dict.fromkeys(page.phone_numbers)),
            ),
        )
        self._logger.bind(event="website_scraper.scraped", url=snapshot.url).debug(
            "Scraped website"
        )
        return snapshot

    async def _check_robots(self, url: str) -> None:
        """Refuse if robots.txt disallows fetching ``url`` for our agent.

        An unreachable or missing robots.txt fails open (no robots.txt is
        the internet-standard default of "everything allowed"), matching
        ``RobotFileParser``'s own behaviour.
        """
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            response = await self._client.get(
                robots_url, headers={"User-Agent": self._user_agent}
            )
        except httpx.HTTPError:
            return
        if response.status_code >= 400:
            return
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        if not parser.can_fetch(self._user_agent, url):
            raise ToolExecutionError(
                f"robots.txt at {robots_url} disallows fetching {url}"
            )

    # -- WebsiteScraperPort adapter -------------------------------------------

    async def scrape(self, url: str) -> WebsiteSnapshot:
        """Fetch ``url``, satisfying ``WebsiteScraperPort``.

        Delegates to :meth:`invoke`, so the agent path is budget-enforced
        (recorded at zero cost) exactly like the tool path.
        """
        return await self.invoke(WebsiteScrapeRequest(url=url))

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
