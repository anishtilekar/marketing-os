"""Tests for the website scraper's extraction upgrade.

Covers the three additions that make JS-rendered storefronts yield real
facts: ``h1``–``h3`` heading capture, JSON-LD organization extraction,
and the nav-label paragraph filter. The fixture page mimics the shape of
a real SPA storefront (nav labels marked up as ``<p>``, content only in
headings + structured data), fetched through a mock httpx transport.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import httpx

from marketingos.models.cost import CostLedger
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.web._html import extract_page
from marketingos.tools.web.website_scraper import WebsiteScraper

SPA_STOREFRONT_HTML = """
<!doctype html>
<html>
<head>
  <title>Acme. Just Move. Acme IN</title>
  <meta name="description" content="Acme - Official Online Store for Athletic Gear.">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Corporation",
    "name": "Acme",
    "description": "Acme designs athletic footwear and apparel.",
    "sameAs": ["https://www.instagram.com/acme", "https://x.com/acme", "not-a-url"]
  }
  </script>
  <script type="application/ld+json">not valid json at all</script>
</head>
<body>
  <h1>Acme Official Online Store India</h1>
  <h2>DON'T PLAY BY THE BOOK</h2>
  <h2>DON'T PLAY BY THE BOOK</h2>
  <h3>Shop by Sport</h3>
  <p>Featured</p>
  <p>Trending</p>
  <p>Shop Icons</p>
  <p>Every product we ship is tested by real athletes before launch.</p>
</body>
</html>
"""


def _guard() -> CostGuard:
    return CostGuard(CostLedger(max_budget=Decimal("100")), run_id=uuid4())


def _make_scraper(html: str) -> WebsiteScraper:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=html)

    return WebsiteScraper(
        cost_guard=_guard(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


# -- extract_page ------------------------------------------------------------


def test_extract_page_captures_headings_in_document_order() -> None:
    page = extract_page(SPA_STOREFRONT_HTML)
    assert page.headings == [
        "Acme Official Online Store India",
        "DON'T PLAY BY THE BOOK",
        "DON'T PLAY BY THE BOOK",
        "Shop by Sport",
    ]


def test_extract_page_parses_json_ld_and_skips_malformed_blocks() -> None:
    page = extract_page(SPA_STOREFRONT_HTML)
    assert len(page.json_ld) == 1
    assert page.json_ld[0]["@type"] == "Corporation"
    assert page.json_ld[0]["name"] == "Acme"


def test_extract_page_flattens_graph_wrappers() -> None:
    html = (
        '<script type="application/ld+json">'
        '{"@graph": [{"@type": "Organization", "name": "Graphed"}]}'
        "</script>"
    )
    page = extract_page(html)
    assert page.json_ld == [{"@type": "Organization", "name": "Graphed"}]


def test_extract_page_still_captures_title_paragraphs_lists() -> None:
    page = extract_page("<title>T</title><p>Some para</p><li>Item one</li>")
    assert page.title == "T"
    assert page.paragraphs == ["Some para"]
    assert page.list_items == ["Item one"]


# -- WebsiteScraper snapshot -------------------------------------------------


async def test_scrape_extracts_organization_from_json_ld() -> None:
    scraper = _make_scraper(SPA_STOREFRONT_HTML)
    snapshot = await scraper.scrape("https://acme.example/")

    org = snapshot.organization
    assert org is not None
    assert org.name == "Acme"
    assert org.description == "Acme designs athletic footwear and apparel."
    # The non-URL sameAs entry is dropped.
    assert org.same_as == (
        "https://www.instagram.com/acme",
        "https://x.com/acme",
    )


async def test_scrape_deduplicates_headings() -> None:
    scraper = _make_scraper(SPA_STOREFRONT_HTML)
    snapshot = await scraper.scrape("https://acme.example/")

    assert snapshot.headings == (
        "Acme Official Online Store India",
        "DON'T PLAY BY THE BOOK",
        "Shop by Sport",
    )


async def test_scrape_filters_nav_labels_out_of_main_text() -> None:
    scraper = _make_scraper(SPA_STOREFRONT_HTML)
    snapshot = await scraper.scrape("https://acme.example/")

    assert snapshot.main_text == (
        "Every product we ship is tested by real athletes before launch."
    )
    for nav_label in ("Featured", "Trending", "Shop Icons"):
        assert nav_label not in snapshot.main_text


async def test_scrape_without_structured_data_yields_no_organization() -> None:
    scraper = _make_scraper("<title>Plain</title><p>A perfectly ordinary paragraph here.</p>")
    snapshot = await scraper.scrape("https://plain.example/")

    assert snapshot.organization is None
    assert snapshot.headings == ()
    assert snapshot.main_text == "A perfectly ordinary paragraph here."
