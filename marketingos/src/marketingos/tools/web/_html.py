"""Minimal, dependency-free HTML extraction shared by the web tools.

Both :mod:`~marketingos.tools.web.website_scraper` and
:mod:`~marketingos.tools.web.instagram_public_reader` need the same small
slice of a page — title, meta/Open-Graph tags, paragraph text, list
items, and ``mailto:``/``tel:`` links — so it's extracted once here with
the standard library's :class:`html.parser.HTMLParser` rather than adding
a parsing dependency (BeautifulSoup/lxml) neither tool otherwise needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser

__all__ = ["PageExtract", "extract_page"]


@dataclass(slots=True)
class PageExtract:
    """The slice of a page's content these tools care about."""

    title: str | None = None
    meta: dict[str, str] = field(default_factory=dict)
    paragraphs: list[str] = field(default_factory=list)
    list_items: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)


class _Extractor(HTMLParser):
    """Collects title/meta/paragraph/list-item/contact-link content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.result = PageExtract()
        self._current_tag: str | None = None
        self._buffer: list[str] = []

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
        if tag in ("title", "p", "li"):
            self._current_tag = tag
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
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
        self._current_tag = None
        self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._current_tag is not None:
            self._buffer.append(data)


def extract_page(html: str) -> PageExtract:
    """Parse ``html`` and return its title/meta/paragraphs/lists/contacts.

    Malformed markup is tolerated the way browsers tolerate it:
    :class:`html.parser.HTMLParser` recovers from broken tags rather than
    raising, so a partially-broken page yields a partial extract instead
    of no extract.
    """
    parser = _Extractor()
    parser.feed(html)
    return parser.result
