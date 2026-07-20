"""InstagramPublicReader — public Instagram profile reading.

Reads only what Instagram serves an unauthenticated, non-JS request: the
server-rendered ``<meta property="og:*">`` tags on a profile page. There
is no official, unauthenticated public API for this data (the Graph API
requires a business account and OAuth), so this is deliberately narrow —
it does not attempt to execute JavaScript, log in, or use an unofficial
private endpoint.

Expect frequent refusals — this is correct, not a bug
---------------------------------------------------------
Instagram serves most unauthenticated profile requests either a
login-wall interstitial or an og:title/og:description pair with no
further data. The shared
:func:`~marketingos.tools.web._compliance.refuse_if_login_walled` guard
catches the former and raises; a page with neither tag raises too (there
is nothing to report). Per the architecture doc, this is a hard refusal
with no bypass — a research run should treat Instagram as an
unreliable, best-effort channel and lean on the website scraper and
search tool as the sturdier sources, exactly as ``ResearchAgent`` already
does (partial-channel failure is tolerated there, not fatal).

Best-effort count parsing
----------------------------
When Instagram does render the standard
``"{followers} Followers, {following} Following, {posts} Posts - ..."``
og:title format, follower/following/post counts (including "12.3K"/"1.2M"
style figures) are parsed out of it. Instagram has changed this string's
exact wording before and will again; a parse miss degrades to ``None``
fields rather than raising, since a name and bio with missing counts is
still useful, factual output.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Final

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marketingos.agents.research import InstagramProfileSnapshot
from marketingos.exceptions.tool import ToolExecutionError
from marketingos.models.cost import CostCategory
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.base import Tool
from marketingos.tools.web._compliance import refuse_if_login_walled
from marketingos.tools.web._html import extract_page

__all__ = ["INSTAGRAM_READING", "InstagramPublicReader", "InstagramReadRequest"]

INSTAGRAM_READING: Final[str] = "instagram_reading"
_DEFAULT_USER_AGENT: Final[str] = "MarketingOS-ResearchBot/1.0 (+public content only)"
_PROFILE_URL_TEMPLATE: Final[str] = "https://www.instagram.com/{username}/"

#: Matches Instagram's standard public og:title counts prefix, tolerant of
#: the '-' vs '•' separator and K/M/B-suffixed figures.
_COUNTS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"([\d.,]+[KMB]?)\s+Followers,\s*([\d.,]+[KMB]?)\s+Following,\s*"
    r"([\d.,]+[KMB]?)\s+Posts",
    re.IGNORECASE,
)
#: Matches the display name preceding "(@handle)" in og:title.
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"([^(]+?)\s*\(@")

_SUFFIX_MULTIPLIER: Final[dict[str, int]] = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def _parse_count(raw: str) -> int | None:
    """Parse a possibly K/M/B-suffixed figure like ``"12.3K"`` into an int."""
    text = raw.strip().upper().replace(",", "")
    multiplier = 1
    if text and text[-1] in _SUFFIX_MULTIPLIER:
        multiplier = _SUFFIX_MULTIPLIER[text[-1]]
        text = text[:-1]
    try:
        return round(float(text) * multiplier)
    except ValueError:
        return None


class InstagramReadRequest(BaseModel):
    """One public profile to read."""

    model_config = ConfigDict(frozen=True)

    username: str = Field(min_length=1)

    @field_validator("username")
    @classmethod
    def _strip_at(cls, value: str) -> str:
        """Strip whitespace and a leading '@', mirroring ResearchInput."""
        cleaned = value.strip().lstrip("@")
        if not cleaned:
            raise ValueError("username must not be empty")
        return cleaned


class InstagramPublicReader(Tool[InstagramReadRequest, InstagramProfileSnapshot]):
    """Reads a public Instagram profile's server-rendered meta tags."""

    def __init__(
        self,
        *,
        cost_guard: CostGuard,
        user_agent: str = _DEFAULT_USER_AGENT,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialise the reader.

        Args:
            cost_guard: Guard enforcing the run's budget. Required: see
                ``Tool.cost_guard``.
            user_agent: Sent on every request.
            http_client: Transport to use. Defaults to a client owned by
                this instance; tests inject one with a mock transport.
            timeout_seconds: Per-request timeout.
        """
        self._cost_guard = cost_guard
        self._user_agent = user_agent
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._logger = logger.bind(component="InstagramPublicReader")

    # -- Tool identity -------------------------------------------------------

    @property
    def name(self) -> str:
        return "instagram-public-reader"

    @property
    def capability(self) -> str:
        return INSTAGRAM_READING

    @property
    def provider(self) -> str:
        return "direct-fetch"

    @property
    def cost_category(self) -> CostCategory:
        return CostCategory.WEB_TOOL

    @property
    def input_schema(self) -> type[InstagramReadRequest]:
        return InstagramReadRequest

    @property
    def output_schema(self) -> type[InstagramProfileSnapshot]:
        return InstagramProfileSnapshot

    @property
    def cost_guard(self) -> CostGuard:
        return self._cost_guard

    # -- cost ------------------------------------------------------------

    def cost_estimate(self, payload: InstagramReadRequest) -> Decimal:
        """Zero — a direct HTTP GET has no vendor bill."""
        return Decimal("0")

    # -- invocation ----------------------------------------------------------

    async def invoke(self, payload: InstagramReadRequest) -> InstagramProfileSnapshot:
        """Fetch and parse the public profile page for ``payload.username``.

        Raises:
            ToolExecutionError: The page reads as login-walled, the
                request fails, or the response carries neither an
                og:title nor an og:description to report.
        """
        url = _PROFILE_URL_TEMPLATE.format(username=payload.username)
        try:
            response = await self._client.get(
                url, headers={"User-Agent": self._user_agent}, follow_redirects=True
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"Instagram returned {exc.response.status_code} for "
                f"@{payload.username}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                f"Fetching @{payload.username} failed: {exc}"
            ) from exc

        html = response.text
        refuse_if_login_walled(html, source=url)
        page = extract_page(html)

        og_title = page.meta.get("og:title")
        biography = page.meta.get("og:description")
        if not og_title and not biography:
            raise ToolExecutionError(
                f"No public profile data found for @{payload.username}: "
                "Instagram returned no og:title or og:description, "
                "typically an app-shell page with no server-rendered content."
            )

        followers = posts = None
        full_name = None
        if og_title:
            counts = _COUNTS_PATTERN.search(og_title)
            name_source = og_title
            if counts:
                followers = _parse_count(counts.group(1))
                # counts.group(2) is the "following" count — parsed by the
                # pattern but unused: the snapshot model has no field for it.
                posts = _parse_count(counts.group(3))
                # Search for the name only *after* the counts prefix —
                # searching the whole title would let the non-greedy
                # "anything before (@" pattern swallow the counts too.
                name_source = og_title[counts.end() :]
            name_match = _NAME_PATTERN.search(name_source)
            if name_match:
                full_name = name_match.group(1).strip(" -•\u00b7")

        snapshot = InstagramProfileSnapshot(
            username=payload.username,
            profile_url=url,
            full_name=full_name,
            biography=biography,
            follower_count=followers,
            post_count=posts,
        )
        self._logger.bind(
            event="instagram_reader.read", username=payload.username
        ).debug("Read Instagram profile")
        return snapshot

    # -- InstagramReaderPort adapter -------------------------------------------

    async def fetch_profile(self, username: str) -> InstagramProfileSnapshot:
        """Read ``username``'s public profile, satisfying ``InstagramReaderPort``.

        Delegates to :meth:`invoke`, so the agent path is budget-enforced
        (recorded at zero cost) exactly like the tool path.
        """
        return await self.invoke(InstagramReadRequest(username=username))

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
