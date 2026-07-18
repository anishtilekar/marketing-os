"""Embedding-based recall backend for the Memory Store.

:class:`VectorMemoryBackend` does not implement its own persistence — it
wraps another :class:`.BaseMemoryBackend` (typically
:class:`.sqlite_backend.SQLiteMemoryBackend`) for durable storage and adds
an in-process cosine-similarity index on top, so callers get "find a similar
past business context"-style semantic recall via ``MemoryQuery.text_query``
without duplicating CRUD logic or trusting an in-memory-only store with
data durability.

Embedding generation is injected via :class:`EmbeddingFunction` so this
module has no dependency on any specific embedding provider.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from ..schemas import (
    BrandVoiceProfile,
    MemoryQuery,
    SyntheticSourceTemplate,
)
from . import BaseMemoryBackend, MemoryRecord, RecordNotFoundError

__all__ = ["EmbeddingFunction", "VectorMemoryBackend"]


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Structural contract for pluggable embedding providers."""

    async def __call__(self, text: str) -> tuple[float, ...]:
        """Return the embedding vector for ``text``."""
        ...


def _extract_text(record: MemoryRecord) -> str:
    """Return the natural-language representation of ``record`` to embed."""
    if isinstance(record, SyntheticSourceTemplate):
        parts = (record.subject, *record.facts, *record.descriptions, *record.brand_characteristics)
    elif isinstance(record, BrandVoiceProfile):
        parts = (*record.tone_descriptors, *record.vocabulary_preferences, *record.sample_captions)
    else:
        parts = (record.failure_category, record.description)
    return " ".join(p for p in parts if p)


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorMemoryBackend(BaseMemoryBackend):
    """Adds embedding-based semantic recall on top of a delegate backend.

    Parameters
    ----------
    delegate:
        The backend of record for durable CRUD (e.g. ``SQLiteMemoryBackend``).
    embed:
        Async callable producing an embedding vector for a text string.
    """

    def __init__(self, delegate: BaseMemoryBackend, embed: EmbeddingFunction) -> None:
        self._delegate = delegate
        self._embed = embed
        self._index: dict[str, tuple[float, ...]] = {}

    # -- BaseMemoryBackend ----------------------------------------------------

    async def save(self, record: MemoryRecord) -> None:
        await self._delegate.save(record)
        text = _extract_text(record)
        if text:
            self._index[record.record_id] = await self._embed(text)

    async def get(self, record_id: str) -> MemoryRecord:
        return await self._delegate.get(record_id)

    async def delete(self, record_id: str) -> None:
        await self._delegate.delete(record_id)
        self._index.pop(record_id, None)

    async def exists(self, record_id: str) -> bool:
        return await self._delegate.exists(record_id)

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        if not query.text_query or not self._index:
            return await self._delegate.query(query)
        return await self._semantic_query(query)

    # -- semantic recall --------------------------------------------------------

    async def _semantic_query(self, query: MemoryQuery) -> list[MemoryRecord]:
        query_embedding = await self._embed(query.text_query)  # type: ignore[arg-type]

        candidate_ids = list(self._index)
        candidates: list[MemoryRecord] = []
        for record_id in candidate_ids:
            try:
                record = await self._delegate.get(record_id)
            except RecordNotFoundError:
                self._index.pop(record_id, None)
                continue
            if query.record_type is not None and record.record_type != query.record_type:
                continue
            if query.customer_id is not None and record.customer_id != query.customer_id:
                continue
            candidates.append(record)

        scored = sorted(
            candidates,
            key=lambda r: _cosine_similarity(query_embedding, self._index[r.record_id]),
            reverse=True,
        )
        return scored[: query.limit]
