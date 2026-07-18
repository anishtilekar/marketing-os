from __future__ import annotations

from .base import MarketingOSError


class WorkflowError(MarketingOSError):
    """Base exception for workflow-related errors."""


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""


class InvalidWorkflowStateError(WorkflowError):
    """Raised when the workflow enters an invalid state."""


class WorkflowTimeoutError(WorkflowError):
    """Raised when workflow execution exceeds the allowed time."""


class WorkflowInterruptedError(WorkflowError):
    """Raised when a workflow is interrupted before completion."""
