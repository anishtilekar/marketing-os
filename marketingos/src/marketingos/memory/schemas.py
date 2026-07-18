"""Data contracts for the persistent, cross-run Memory Store.

Defines the record types agents read/write via ``memory.store.MemoryStore``
and its pluggable backends (``backends/sqlite_backend.py``,
``backends/vector_backend.py``). Pure data only — no I/O, no backend logic.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


__all__ = [
    "BrandVoiceProfile",
    "MemoryQuery",
    "MemoryRecordType",
    "QAFailurePattern",
    "SyntheticSourceTemplate",
]


class MemoryRecordType(StrEnum):
    """Discriminates the kinds of records the store persists."""

    BRAND_VOICE = "brand_voice"
    SYNTHETIC_SOURCE_TEMPLATE = "synthetic_source_template"
    QA_FAILURE_PATTERN = "qa_failure_pattern"


class _BaseMemoryRecord(BaseModel):
    """Fields shared by every persisted memory record."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(default_factory=_new_id)
    customer_id: str = Field(description="Owning customer/tenant identifier.")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrandVoiceProfile(_BaseMemoryRecord):
    """Learned tone/voice characteristics for a repeat customer."""

    record_type: MemoryRecordType = MemoryRecordType.BRAND_VOICE
    tone_descriptors: tuple[str, ...] = Field(
        default_factory=tuple, description="e.g. 'playful', 'formal'."
    )
    vocabulary_preferences: tuple[str, ...] = Field(default_factory=tuple)
    avoided_terms: tuple[str, ...] = Field(default_factory=tuple)
    sample_captions: tuple[str, ...] = Field(
        default_factory=tuple, description="Approved past captions used as few-shot exemplars."
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_run_ids: tuple[str, ...] = Field(default_factory=tuple)


class SyntheticSourceTemplate(_BaseMemoryRecord):
    """Reusable, copyright-safe source material for recall across runs."""

    record_type: MemoryRecordType = MemoryRecordType.SYNTHETIC_SOURCE_TEMPLATE
    subject: str
    facts: tuple[str, ...] = Field(default_factory=tuple)
    descriptions: tuple[str, ...] = Field(default_factory=tuple)
    brand_characteristics: tuple[str, ...] = Field(default_factory=tuple)
    keywords: tuple[str, ...] = Field(default_factory=tuple)
    embedding: tuple[float, ...] | None = Field(
        default=None, description="Optional vector for embedding-based recall."
    )
    reuse_count: int = Field(default=0, ge=0)
    source_run_id: str


class QAFailurePattern(_BaseMemoryRecord):
    """A recurring QA-critique pattern worth feeding back into prompts."""

    record_type: MemoryRecordType = MemoryRecordType.QA_FAILURE_PATTERN
    agent_name: str = Field(description="Producing agent whose output failed QA.")
    failure_category: str = Field(description="e.g. 'off_brand_tone', 'wrong_count'.")
    description: str
    example_run_ids: tuple[str, ...] = Field(default_factory=tuple)
    occurrence_count: int = Field(default=1, ge=1)


class MemoryQuery(BaseModel):
    """Filter/search parameters accepted by store and backend lookups."""

    model_config = ConfigDict(frozen=True)

    record_type: MemoryRecordType | None = None
    customer_id: str | None = None
    text_query: str | None = Field(
        default=None, description="Free-text or embedding-recall query, backend-dependent."
    )
    limit: int = Field(default=10, ge=1, le=100)