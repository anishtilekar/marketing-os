from __future__ import annotations

from .base import MarketingOSError


class BudgetError(MarketingOSError):
    """Base exception for budget-related errors."""


class BudgetExceededError(BudgetError):
    """Raised when an operation exceeds the available budget."""


class BudgetLimitReachedError(BudgetError):
    """Raised when the maximum budget limit has been reached."""


class InsufficientBudgetError(BudgetError):
    """Raised when there is not enough remaining budget for an operation."""


class CostTrackingError(BudgetError):
    """Raised when budget or cost tracking fails."""
