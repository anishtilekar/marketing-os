"""Configuration and settings for MarketingOS.

Loads application configuration from environment and YAML files,
validates settings against the :class:`Settings` schema, and raises
appropriately typed errors for missing or invalid configuration.
"""

from __future__ import annotations

from .loader import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    load_settings,
)
from .settings import (
    AgentConfig,
    AgentSettings,
    BudgetSettings,
    ModelSettings,
    Settings,
    WorkflowSettings,
)

__all__ = [
    "AgentConfig",
    "AgentSettings",
    "BudgetSettings",
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    "ModelSettings",
    "Settings",
    "WorkflowSettings",
    "load_settings",
]
