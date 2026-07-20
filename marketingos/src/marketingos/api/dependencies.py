"""Builds the shared RunHandle and port adapters for one API run.

Every adapter returned here shares the run's single ``CostGuard``
(``RunHandle.guard``), so every tool call across the whole graph execution
is priced against the same budget ceiling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

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
from marketingos.tools.factory import (
    build_image_generator,
    build_llm,
    build_video_generator,
)
from marketingos.tools.web import InstagramPublicReader, WebsiteScraper

__all__ = ["RunAdapters", "build_run_dependencies", "run_manager"]

#: Project root (the directory holding ``pyproject.toml`` and ``.env``), four
#: levels up from this file at ``<root>/src/marketingos/api/dependencies.py``.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_project_env(root: Path = _PROJECT_ROOT) -> bool:
    """Load ``<root>/.env`` into ``os.environ``; return whether a file loaded.

    An explicit path keeps this deterministic regardless of the process's
    working directory, so there is one source of truth (``marketingos/.env``)
    rather than whichever ``.env`` happens to be nearest the cwd. Non-overriding
    by design: a variable already present in the real environment (e.g. an
    explicit shell export) still wins over the file.
    """
    return load_dotenv(root / ".env", override=False)


# Load at import time — before the module-level ``RUNS_ROOT`` read below and
# before ``build_run_dependencies`` constructs any client, all of which read
# ``os.environ`` directly (the Gemini and Together clients read their API keys
# from it).
_load_project_env()

#: Overrides where run artifacts (state, eval reports, packages) are written.
#: Defaults to the existing relative path so local dev is unaffected; set
#: this to a mounted volume's path on hosts without a durable working
#: directory across deploys/restarts.
run_manager = RunManager(runs_root=Path(os.environ.get("RUNS_ROOT", "data/runs")))


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
        # Which concrete client each of these is comes from config
        # (settings.models.*_provider), resolved by the provider factory —
        # switching provider or model is a YAML edit, not a code change.
        llm=build_llm(settings.models, guard),
        image_generator=build_image_generator(settings.models, guard),
        video_generator=build_video_generator(settings.models, guard),
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
