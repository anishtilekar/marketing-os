"""Exception hierarchy for MarketingOS.

All custom exceptions inherit from :class:`MarketingOSError` for consistent
exception categorization across the application. Domain-specific hierarchies
(agent, budget, config, storage, tool, validation, workflow) provide targeted
catches for callers that know which subsystem failed.
"""

from __future__ import annotations

from .agent import (
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentOutputError,
    AgentRetryLimitExceededError,
)
from .base import MarketingOSError
from .budget import (
    BudgetError,
    BudgetExceededError,
    BudgetLimitReachedError,
    CostTrackingError,
    InsufficientBudgetError,
)
from .config import (
    ConfigurationError,
    ConfigurationLoadError,
    InvalidConfigurationError,
    MissingConfigurationError,
)
from .storage import (
    DatabaseError,
    FileStorageError,
    RecordNotFoundError,
    StorageConnectionError,
    StorageError,
)
from .tool import (
    ToolConfigurationError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from .validation import (
    BusinessRuleViolation,
    ModelValidationError,
    ValidationError,
    SchemaValidationError,
)
from .workflow import (
    InvalidWorkflowStateError,
    WorkflowError,
    WorkflowExecutionError,
    WorkflowInterruptedError,
    WorkflowTimeoutError,
)

__all__ = [
    "AgentError",
    "AgentExecutionError",
    "AgentInitializationError",
    "AgentOutputError",
    "AgentRetryLimitExceededError",
    "BudgetError",
    "BudgetExceededError",
    "BudgetLimitReachedError",
    "BusinessRuleViolation",
    "ConfigurationError",
    "ConfigurationLoadError",
    "CostTrackingError",
    "DatabaseError",
    "FileStorageError",
    "InvalidConfigurationError",
    "InvalidWorkflowStateError",
    "InsufficientBudgetError",
    "MarketingOSError",
    "MissingConfigurationError",
    "ModelValidationError",
    "RecordNotFoundError",
    "SchemaValidationError",
    "StorageConnectionError",
    "StorageError",
    "ToolConfigurationError",
    "ToolError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "ValidationError",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowInterruptedError",
    "WorkflowTimeoutError",
]
