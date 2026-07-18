from __future__ import annotations

from .base import MarketingOSError


class AgentError(MarketingOSError):
    """Base exception for agent-related errors."""


class AgentExecutionError(AgentError):
    """Raised when an agent fails during execution."""


class AgentInitializationError(AgentError):
    """Raised when an agent cannot be initialized."""


class AgentOutputError(AgentError):
    """Raised when an agent produces invalid output."""


class AgentRetryLimitExceededError(AgentError):
    """Raised when an agent exceeds its maximum retry limit."""
