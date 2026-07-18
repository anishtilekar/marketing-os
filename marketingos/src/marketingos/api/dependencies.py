"""Builds the shared RunHandle and port adapters for one API run.

Every adapter returned here shares the run's single ``CostGuard``
(``RunHandle.guard``), so every tool call across the whole graph execution
is priced against the same budget ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from marketingos.agents.business_analysis import LanguageModelPort
from marketingos.agents.designer import ImageGenerationPort
from marketingos.agents.packaging import PackagingServicePort
from marketingos.agents.qa import BudgetLedgerPort
from marketingos.agents.research import (
    BusinessSearchPort,
    InstagramReaderPort,
    WebsiteScraperPort,
)
from marketingos.agents.video_director import VideoGenerationPort
from marketingos.config import load_settings
from marketingos.services.cost_guard import CostGuardBudgetLedger
from marketingos.services.packaging_service import PackagingService
from marketingos.services.run_manager import RunHandle, RunManager
from marketingos.tools.image import GeminiImageClient, PlaceholderImageClient
from marketingos.tools.llm import GeminiClient
from marketingos.tools.video import VideoAssembler
from marketingos.tools.web import InstagramPublicReader, WebsiteScraper

__all__ = ["RunAdapters", "build_run_dependencies", "run_manager"]

run_manager = RunManager()


@dataclass(slots=True)
class RunAdapters:
    """Every port adapter one run's graph needs, sharing one ``CostGuard``.

    Fields match the ``make_*_node`` factory keywords each adapter satisfies,
    so call sites can pass them straight through (e.g.
    ``make_research_node(website_scraper=adapters.website_scraper, ...)``).
    """

    llm: LanguageModelPort
    image_generator: ImageGenerationPort
    video_generator: VideoGenerationPort
    website_scraper: WebsiteScraperPort
    instagram_reader: InstagramReaderPort
    packaging_service: PackagingServicePort
    budget_ledger: BudgetLedgerPort
    search_tool: BusinessSearchPort | None = None


def build_run_dependencies(max_budget: Decimal) -> tuple[RunHandle, RunAdapters]:
    """Start a new run and construct every port adapter it needs.

    Args:
        max_budget: Spend ceiling for the run, forwarded to
            ``RunManager.start_run`` and the run's ``CostGuard``.

    Returns:
        The new run's handle, and its :class:`RunAdapters`.
    """
    settings = load_settings()
    handle = run_manager.start_run(max_budget=max_budget)
    guard = handle.guard

    adapters = RunAdapters(
        llm=GeminiClient(
            cost_guard=guard,
            model=settings.models.default_llm,
            default_max_output_tokens=settings.models.max_tokens,
        ),
        # Swapped back to GeminiImageClient once billing/quota is available.
        # image_generator=GeminiImageClient(
        #     cost_guard=guard,
        #     model=settings.models.default_image_model,
        # ),
        image_generator=PlaceholderImageClient(),
        video_generator=VideoAssembler(cost_guard=guard),
        website_scraper=WebsiteScraper(cost_guard=guard),
        instagram_reader=InstagramPublicReader(cost_guard=guard),
        packaging_service=PackagingService(
            base_dir=run_manager.run_dir(handle.run_id) / "package"
        ),
        budget_ledger=CostGuardBudgetLedger(
            guard,
            currency=settings.budget.currency,
            warning_threshold=settings.budget.warning_threshold,
        ),
    )
    return handle, adapters
