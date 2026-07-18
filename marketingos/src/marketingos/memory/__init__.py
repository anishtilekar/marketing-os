"""Persistent, cross-run agent memory (brand voice, source templates, QA patterns).

Public API: construct a backend, wrap it in :class:`MemoryStore`, inject the
store into agents/services via the ``MemoryStore`` protocol in
:mod:`marketingos.agents.base`.

    from marketingos.memory import MemoryStore
    from marketingos.memory.backends.sqlite_backend import SQLiteMemoryBackend

    store = MemoryStore(SQLiteMemoryBackend(db_path="marketingos_memory.db"))
"""

from __future__ import annotations

from .backends import (
    BackendConnectionError,
    BaseMemoryBackend,
    DuplicateRecordError,
    MemoryBackendError,
    MemoryRecord,
    RecordNotFoundError,
)
from .schemas import (
    BrandVoiceProfile,
    MemoryQuery,
    MemoryRecordType,
    QAFailurePattern,
    SyntheticSourceTemplate,
)
from .store import MemoryStore

__all__ = [
    "BackendConnectionError",
    "BaseMemoryBackend",
    "BrandVoiceProfile",
    "DuplicateRecordError",
    "MemoryBackendError",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryRecordType",
    "MemoryStore",
    "QAFailurePattern",
    "RecordNotFoundError",
    "SyntheticSourceTemplate",
]
