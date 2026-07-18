"""Request and response models for the MarketingOS HTTP API."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, HttpUrl

__all__ = ["CreateRunRequest", "RunStatusResponse"]


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    website_url: HttpUrl
    business_name: str | None = None
    instagram_username: str | None = None
    budget_usd: Decimal


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    error: str | None = None
