from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


class BudgetSettings(BaseModel):
    """Budget constraints for the MarketingOS system.

    Monetary fields are ``Decimal``, matching
    :class:`~marketingos.models.cost.CostEntry` and
    :class:`~marketingos.models.cost.CostLedger`, so a ceiling read from
    ``budget.yaml`` can be handed straight to a ledger's ``max_budget``
    without a lossy float hop in between. They were previously ``float``:
    YAML numbers arrive as binary floats, in which a value such as ``0.1``
    has no exact representation, and money that accumulates through float
    drifts.
    """

    max_budget: Decimal
    warning_threshold: Decimal
    currency: str

    @field_validator("max_budget", "warning_threshold", mode="before")
    @classmethod
    def _exact_decimal(cls, value: object) -> object:
        """Convert a YAML float to Decimal through its text form.

        ``Decimal(0.1)`` inherits the float's binary error
        (``0.1000000000000000055...``); ``Decimal(str(0.1))`` is exactly
        ``Decimal("0.1")``. Doing this here keeps the correction at the
        settings-loading boundary rather than in the cost models.
        """
        if isinstance(value, float):
            return Decimal(str(value))
        return value


class WorkflowSettings(BaseModel):
    """Workflow execution behavior settings."""

    max_revisions: int
    enable_human_review: bool
    checkpoint_after_each_agent: bool


class ModelSettings(BaseModel):
    """Default LLM model configuration."""

    default_llm: str
    fallback_llm: str
    temperature: float
    max_tokens: int
    default_image_model: str
    image_quality: str


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    model: str
    temperature: float
    enabled: bool


class AgentSettings(BaseModel):
    """Configuration for every agent in the MarketingOS pipeline."""

    research: AgentConfig
    synthetic_source: AgentConfig
    business_analysis: AgentConfig
    strategist: AgentConfig
    planner: AgentConfig
    copywriter: AgentConfig
    designer: AgentConfig
    video_director: AgentConfig
    qa: AgentConfig
    packaging: AgentConfig


class Settings(BaseSettings):
    """Root settings object aggregating all configuration sections."""

    budget: BudgetSettings
    workflow: WorkflowSettings
    models: ModelSettings
    agents: AgentSettings

    model_config = {
        "frozen": True,
    }
