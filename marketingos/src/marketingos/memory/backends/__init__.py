"""Backend abstraction for the persistent Memory Store.

Concrete backends (:mod:`.sqlite_backend`, :mod:`.vector_backend`) implement
:class:`BaseMemoryBackend` so ``memory.store.MemoryStore`` can depend on
behaviour, not on a specific persistence technology — SQLite today, a vector
store for embedding-based recall tomorrow, without touching caller code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from ..schemas import (
    BrandVoiceProfile,
    MemoryQuery,
    QAFailurePattern,
    SyntheticSourceTemplate,
)

__all__ = [
    "BackendConnectionError",
    "BaseMemoryBackend",
    "DuplicateRecordError",
    "MemoryBackendError",
    "MemoryRecord",
    "RecordNotFoundError",
]

MemoryRecord = Union[BrandVoiceProfile, SyntheticSourceTemplate, QAFailurePattern]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MemoryBackendError(Exception):
    """Base exception for all memory backend failures."""


class RecordNotFoundError(MemoryBackendError):
    """Raised when a ``record_id`` does not exist in the backend."""

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        super().__init__(f"No memory record found with record_id={record_id!r}")


class DuplicateRecordError(MemoryBackendError):
    """Raised when saving a ``record_id`` that already exists.

    Records are treated as immutable-by-id; callers wanting to update one
    should delete and re-save, keeping writes explicit and auditable.
    """

    def __init__(self, record_id: str) -> None:
        self.record_id = record_id
        super().__init__(f"A memory record with record_id={record_id!r} already exists")


class BackendConnectionError(MemoryBackendError):
    """Raised when the backend cannot be reached or initialized."""


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class BaseMemoryBackend(ABC):
    """Storage-agnostic contract for Memory Store persistence.

    Implementations must be safe for concurrent async use. All methods are
    async so a network-backed vector store (e.g. a hosted embedding index)
    is a drop-in replacement for a local SQLite file.
    """

    @abstractmethod
    async def save(self, record: MemoryRecord) -> None:
        """Persist ``record``.

        Raises:
            DuplicateRecordError: If ``record.record_id`` already exists.
            BackendConnectionError: If the backend cannot be written to.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, record_id: str) -> MemoryRecord:
        """Return the record stored under ``record_id``.

        Raises:
            RecordNotFoundError: If no such record exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        """Return records matching ``query``, most-relevant first.

        SQLite backends filter on ``record_type``/``customer_id`` and, at
        best, substring-match ``text_query``. Vector backends additionally
        rank by embedding similarity when ``text_query`` is set.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, record_id: str) -> None:
        """Remove the record stored under ``record_id``.

        Raises:
            RecordNotFoundError: If no such record exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def exists(self, record_id: str) -> bool:
        """Return whether ``record_id`` is present in the backend."""
        raise NotImplementedError