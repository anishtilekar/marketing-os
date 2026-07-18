"""Public facade for the persistent, cross-run Memory Store.

``MemoryStore`` is the only thing agents/services depend on (per
``AgentConfig``/DI conventions in ``agents.base``); it hides which
:class:`.backends.BaseMemoryBackend` is actually doing the persisting
(SQLite today, vector-augmented tomorrow) behind typed, record-specific
convenience methods.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .backends import BaseMemoryBackend, MemoryRecord
from .schemas import (
    BrandVoiceProfile,
    MemoryQuery,
    MemoryRecordType,
    QAFailurePattern,
    SyntheticSourceTemplate,
)

__all__ = ["MemoryStore"]


def _now() -> datetime:
    return datetime.now(UTC)


class MemoryStore:
    """Typed read/write access to cross-run agent memory.

    Wraps a :class:`.backends.BaseMemoryBackend` (constructor-injected, so
    the backend is swappable without touching call sites) and exposes both
    generic CRUD and record-specific helpers for the three memory kinds:
    brand voice, synthetic source templates, and QA-failure patterns.
    """

    def __init__(self, backend: BaseMemoryBackend) -> None:
        self._backend = backend

    # -- generic CRUD -----------------------------------------------------------

    async def save(self, record: MemoryRecord) -> None:
        await self._backend.save(record)

    async def get(self, record_id: str) -> MemoryRecord:
        return await self._backend.get(record_id)

    async def delete(self, record_id: str) -> None:
        await self._backend.delete(record_id)

    async def exists(self, record_id: str) -> bool:
        return await self._backend.exists(record_id)

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        return await self._backend.query(query)

    # -- brand voice --------------------------------------------------------------

    async def get_brand_voice(self, customer_id: str) -> BrandVoiceProfile | None:
        """Return the current brand-voice profile for ``customer_id``, if any."""
        results = await self.query(
            MemoryQuery(
                record_type=MemoryRecordType.BRAND_VOICE,
                customer_id=customer_id,
                limit=1,
            )
        )
        return results[0] if results else None  # type: ignore[return-value]

    async def learn_brand_voice(self, customer_id: str, **updates: Any) -> BrandVoiceProfile:
        """Create or refine the brand-voice profile for ``customer_id``.

        Merges ``updates`` (any :class:`BrandVoiceProfile` field) onto the
        existing profile if one exists, otherwise creates a new one. Records
        are frozen, so refining replaces the stored record under a fresh
        write rather than mutating in place.
        """
        existing = await self.get_brand_voice(customer_id)
        if existing is None:
            profile = BrandVoiceProfile(customer_id=customer_id, **updates)
        else:
            await self._backend.delete(existing.record_id)
            profile = existing.model_copy(update={**updates, "updated_at": _now()})
        await self._backend.save(profile)
        return profile

    # -- synthetic source templates -----------------------------------------------

    async def record_source_template(self, template: SyntheticSourceTemplate) -> None:
        """Persist a reusable synthetic source template for future recall."""
        await self._backend.save(template)

    async def find_similar_source_templates(
        self,
        subject_query: str,
        *,
        customer_id: str | None = None,
        limit: int = 5,
    ) -> list[SyntheticSourceTemplate]:
        """Return templates most relevant to ``subject_query``.

        Ranking quality depends on the injected backend: a plain
        :class:`.backends.sqlite_backend.SQLiteMemoryBackend` substring-
        matches; a :class:`.backends.vector_backend.VectorMemoryBackend`
        ranks by embedding similarity.
        """
        results = await self.query(
            MemoryQuery(
                record_type=MemoryRecordType.SYNTHETIC_SOURCE_TEMPLATE,
                customer_id=customer_id,
                text_query=subject_query,
                limit=limit,
            )
        )
        return results  # type: ignore[return-value]

    # -- QA failure patterns --------------------------------------------------------

    async def record_qa_failure(
        self,
        *,
        agent_name: str,
        failure_category: str,
        description: str,
        run_id: str | None = None,
        customer_id: str = "_global",
    ) -> QAFailurePattern:
        """Record (or bump the count of) a recurring QA-failure pattern.

        Patterns are deduplicated on ``(agent_name, failure_category,
        customer_id)`` so repeated occurrences increment ``occurrence_count``
        instead of creating unbounded duplicate rows.
        """
        candidates = await self.query(
            MemoryQuery(
                record_type=MemoryRecordType.QA_FAILURE_PATTERN,
                customer_id=customer_id,
                text_query=failure_category,
                limit=50,
            )
        )
        existing = next(
            (
                c
                for c in candidates
                if isinstance(c, QAFailurePattern)
                and c.agent_name == agent_name
                and c.failure_category == failure_category
            ),
            None,
        )
        if existing is None:
            pattern = QAFailurePattern(
                customer_id=customer_id,
                agent_name=agent_name,
                failure_category=failure_category,
                description=description,
                example_run_ids=(run_id,) if run_id else (),
            )
        else:
            await self._backend.delete(existing.record_id)
            example_run_ids = existing.example_run_ids
            if run_id and run_id not in example_run_ids:
                example_run_ids = (*example_run_ids, run_id)
            pattern = existing.model_copy(
                update={
                    "occurrence_count": existing.occurrence_count + 1,
                    "example_run_ids": example_run_ids,
                    "updated_at": _now(),
                }
            )
        await self._backend.save(pattern)
        return pattern