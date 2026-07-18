from __future__ import annotations

from .base import MarketingOSError


class ValidationError(MarketingOSError):
    """Raised when domain validation fails."""


class ModelValidationError(ValidationError):
    """Raised when a Pydantic model fails validation."""


class BusinessRuleViolation(ValidationError):
    """Raised when a business rule is violated."""


class SchemaValidationError(ValidationError):
    """Raised when input or output data does not match the expected schema."""
