from __future__ import annotations

from .base import MarketingOSError


class ConfigurationError(MarketingOSError):
    """Raised when application configuration is invalid."""


class MissingConfigurationError(ConfigurationError):
    """Raised when a required configuration value is missing."""


class InvalidConfigurationError(ConfigurationError):
    """Raised when a configuration value is invalid."""


class ConfigurationLoadError(ConfigurationError):
    """Raised when configuration files cannot be loaded."""
