"""Data contracts for MarketingOS domain entities.

Defines the core models used throughout the application: marketing campaign
planning, cost tracking, creative assets, business context, and run lifecycle.
Pure data models with no I/O or service logic — see individual modules for
constraints, validators, and usage examples.
"""

from __future__ import annotations

from .business_context import (
    Assumption,
    BusinessContext,
    Fact,
)
from .cost import (
    CostCategory,
    CostEntry,
    CostLedger,
    CostStatus,
    CostSummary,
)
from .creative import (
    AssetFormat,
    Creative,
    CreativeStatus,
    CreativeType,
)
from .plan import (
    ContentItem,
    ContentStatus,
    ContentType,
    Platform,
    PostItem,
    VideoItem,
    WeekPlan,
)
from .run import (
    RunRecord,
    RunSection,
    RunStatus,
)

__all__ = [
    "Assumption",
    "AssetFormat",
    "BusinessContext",
    "ContentItem",
    "ContentStatus",
    "ContentType",
    "CostCategory",
    "CostEntry",
    "CostLedger",
    "CostStatus",
    "CostSummary",
    "Creative",
    "CreativeStatus",
    "CreativeType",
    "Fact",
    "Platform",
    "PostItem",
    "RunRecord",
    "RunSection",
    "RunStatus",
    "VideoItem",
    "WeekPlan",
]
