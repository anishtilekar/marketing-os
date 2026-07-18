from __future__ import annotations

from .base import MarketingOSError


class ToolError(MarketingOSError):
    """Base exception for tool-related errors."""


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool cannot be found."""


class ToolConfigurationError(ToolError):
    """Raised when a tool is improperly configured."""


class ToolTimeoutError(ToolError):
    """Raised when a tool exceeds the allowed execution time."""
