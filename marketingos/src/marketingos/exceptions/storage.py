from __future__ import annotations

from .base import MarketingOSError


class StorageError(MarketingOSError):
    """Base exception for storage-related errors."""


class FileStorageError(StorageError):
    """Raised when file storage operations fail."""


class DatabaseError(StorageError):
    """Raised when database operations fail."""


class RecordNotFoundError(StorageError):
    """Raised when a requested record cannot be found."""


class StorageConnectionError(StorageError):
    """Raised when a connection to the storage backend fails."""
