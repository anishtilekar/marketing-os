"""Minimal, dependency-free HTML extraction shared by the web tools.

Both :mod:`~marketingos.tools.web.website_scraper` and
:mod:`~marketingos.tools.web.instagram_public_reader` need the same small
slice of a page — title, meta/Open-Graph tags, headings, paragraph text,
list items, JSON-LD structured data, and ``mailto:``/``tel:`` links — so
it's extracted once here with the standard library's
:class:`html.parser.HTMLParser` rather than adding a parsing dependency
(BeautifulSoup/lxml) neither tool otherwise needs.

Why headings and JSON-LD
------------------------
Modern storefronts are JS-rendered SPAs whose ``<p>``/``<li>`` markup is
mostly navigation labels — on such pages the old extract yielded nothing
but nav soup. Two things still ship as real content in the initial HTML,
because search engines require them: ``<h1>``–``<h3>`` headings and
``<script type="application/ld+json">`` structured data (organization
name, description, official social profiles). Capturing both multiplies
the usable facts on exactly the pages the paragraph heuristic fails on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

__all__ = ["PageExtract", "extract_page"]

_HEADING_TAGS = ("h1", "h2", "h3")


@dataclass(slots=True)
class PageExtract:
    """The slice of a page's content these tools care about."""

    title: str | None = None
    meta: dict[str, str] = field(default_factory=dict)
    headings: list[str] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    list_items: list[str] = field(default_factory=list)
    json_ld: list[dict[str, Any]] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)


class _Extractor(HTMLParser):
    """Collects title/meta/heading/paragraph/list/JSON-LD/contact content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.result = PageExtract()
        self._current_tag: str | None = None
        self._buffer: list[str] = []
        self._in_json_ld = False
        self._json_ld_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "meta":
            key = attributes.get("name") or attributes.get("property")
            content = attributes.get("content")
            if key and content:
                self.result.meta[key.lower()] = content
        elif tag == "a":
            href = attributes.get("href") or ""
            if href.startswith("mailto:"):
                self.result.emails.append(href.removeprefix("mailto:").split("?")[0])
            elif href.startswith("tel:"):
                self.result.phone_numbers.append(href.removeprefix("tel:"))
        elif tag == "script":
            script_type = (attributes.get("type") or "").strip().lower()
            if script_type == "application/ld+json":
                self._in_json_ld = True
                self._json_ld_buffer = []
        if tag in ("title", "p", "li", *_HEADING_TAGS):
            self._current_tag = tag
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_json_ld:
            self._in_json_ld = False
            self._parse_json_ld("".join(self._json_ld_buffer))
            return
        if tag != self._current_tag:
            return
        text = " ".join("".join(self._buffer).split())
        if text:
            if tag == "title":
                self.result.title = text
            elif tag == "p":
                self.result.paragraphs.append(text)
            elif tag == "li":
                self.result.list_items.append(text)
            elif tag in _HEADING_TAGS:
                self.result.headings.append(text)
        self._current_tag = None
        self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_ld_buffer.append(data)
        elif self._current_tag is not None:
            self._buffer.append(data)

    def _parse_json_ld(self, raw: str) -> None:
        """Parse one JSON-LD block, flattening top-level arrays and @graph.

        A malformed block is skipped rather than failing the whole page —
        structured data is a bonus signal, never a requirement.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, RecursionError):
            return
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                self.result.json_ld.extend(
                    node for node in graph if isinstance(node, dict)
                )
            else:
                self.result.json_ld.append(item)


def extract_page(html: str) -> PageExtract:
    """Parse ``html`` and return its extracted content slice.

    Malformed markup is tolerated the way browsers tolerate it:
    :class:`html.parser.HTMLParser` recovers from broken tags rather than
    raising, so a partially-broken page yields a partial extract instead
    of no extract.
    """
    parser = _Extractor()
    parser.feed(html)
    return parser.result
