"""SQLite-backed implementation of :class:`BaseMemoryBackend`.

Suitable as the default, single-node Memory Store backend: durable,
zero-ops, no separate database server. Records are stored as JSON blobs
keyed by ``record_id`` with ``record_type``/``customer_id`` columns for
cheap filtering; ``text_query`` filtering is a simple substring match over
the serialized payload (a :class:`.vector_backend.VectorMemoryBackend` is
the upgrade path for real semantic recall).
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from ..schemas import (
    BrandVoiceProfile,
    MemoryQuery,
    MemoryRecordType,
    QAFailurePattern,
    SyntheticSourceTemplate,
)
from . import (
    BackendConnectionError,
    BaseMemoryBackend,
    DuplicateRecordError,
    MemoryRecord,
    RecordNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["SQLiteMemoryBackend"]

_SCHEMA_BY_TYPE: dict[MemoryRecordType, type[MemoryRecord]] = {
    MemoryRecordType.BRAND_VOICE: BrandVoiceProfile,
    MemoryRecordType.SYNTHETIC_SOURCE_TEMPLATE: SyntheticSourceTemplate,
    MemoryRecordType.QA_FAILURE_PATTERN: QAFailurePattern,
}


class SQLiteMemoryBackend(BaseMemoryBackend):
    """Durable Memory Store backend backed by a local SQLite file.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file. Parent directories are
        created automatically. Defaults to ``"marketingos_memory.db"`` in
        the current working directory.

    Notes
    -----
    Uses ``check_same_thread=False`` with an internal lock, and offloads
    blocking sqlite3 calls to a thread via ``asyncio.to_thread`` so this
    backend behaves correctly under an async event loop.
    """

    _SCHEMA_VERSION: ClassVar[int] = 1

    def __init__(self, db_path: str | Path = "marketingos_memory.db") -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._closed = False

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error as exc:
            raise BackendConnectionError(
                f"Failed to open SQLite database at {self._db_path}: {exc}"
            ) from exc
        self._initialize_schema()

    # -- schema / connection ------------------------------------------------

    def _initialize_schema(self) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    record_id    TEXT PRIMARY KEY,
                    record_type  TEXT NOT NULL,
                    customer_id  TEXT NOT NULL,
                    payload      TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_customer "
                "ON memory_records(customer_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_type "
                "ON memory_records(record_type);"
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN;")
                yield cursor
                self._connection.commit()
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise BackendConnectionError(f"SQLite transaction failed: {exc}") from exc
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise BackendConnectionError("This SQLiteMemoryBackend has been closed.")

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        record_type = MemoryRecordType(row["record_type"])
        model_cls = _SCHEMA_BY_TYPE[record_type]
        return model_cls.model_validate_json(row["payload"])

    # -- BaseMemoryBackend ----------------------------------------------------

    async def save(self, record: MemoryRecord) -> None:
        await asyncio.to_thread(self._save_sync, record)

    def _save_sync(self, record: MemoryRecord) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT 1 FROM memory_records WHERE record_id = ?;",
                (record.record_id,),
            )
            if cursor.fetchone() is not None:
                raise DuplicateRecordError(record.record_id)
            cursor.execute(
                """
                INSERT INTO memory_records
                    (record_id, record_type, customer_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    record.record_id,
                    record.record_type.value,
                    record.customer_id,
                    record.model_dump_json(),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )

    async def get(self, record_id: str) -> MemoryRecord:
        return await asyncio.to_thread(self._get_sync, record_id)

    def _get_sync(self, record_id: str) -> MemoryRecord:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM memory_records WHERE record_id = ?;", (record_id,)
            )
            row = cursor.fetchone()
        if row is None:
            raise RecordNotFoundError(record_id)
        return self._row_to_record(row)

    async def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._query_sync, query)

    def _query_sync(self, query: MemoryQuery) -> list[MemoryRecord]:
        self._ensure_open()
        clauses: list[str] = []
        params: list[str] = []
        if query.record_type is not None:
            clauses.append("record_type = ?")
            params.append(query.record_type.value)
        if query.customer_id is not None:
            clauses.append("customer_id = ?")
            params.append(query.customer_id)
        if query.text_query:
            clauses.append("payload LIKE ?")
            params.append(f"%{query.text_query}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM memory_records {where} ORDER BY updated_at DESC LIMIT ?;"
        params.append(str(query.limit))
        with self._lock:
            cursor = self._connection.execute(sql, params)
            rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def delete(self, record_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, record_id)

    def _delete_sync(self, record_id: str) -> None:
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM memory_records WHERE record_id = ?;", (record_id,))
            if cursor.rowcount == 0:
                raise RecordNotFoundError(record_id)

    async def exists(self, record_id: str) -> bool:
        return await asyncio.to_thread(self._exists_sync, record_id)

    def _exists_sync(self, record_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(
                "SELECT 1 FROM memory_records WHERE record_id = ?;", (record_id,)
            )
            return cursor.fetchone() is not None

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying SQLite connection."""
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteMemoryBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()