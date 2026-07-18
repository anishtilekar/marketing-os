"""
checkpointer.py
===============

Checkpoint Persistence for MarketingOS
---------------------------------------

This module provides a self-contained, backend-agnostic subsystem for
persisting and retrieving LangGraph-style workflow checkpoints within
MarketingOS. It is designed so long-running, interruptible workflows
(human approval pauses, multi-stage content pipelines, crash recovery)
can save their execution state and resume it later, without this module
knowing anything about *what* that state represents.

Scope
-----
This module is responsible ONLY for checkpoint *persistence and
retrieval*:

- Defining a storage-agnostic checkpoint data model (:class:`Checkpoint`).
- Defining a clean abstraction (:class:`BaseCheckpointManager`) that any
  storage backend can implement.
- Providing two concrete backends out of the box: an in-memory
  implementation (:class:`MemoryCheckpointManager`, ideal for tests) and a
  SQLite-backed implementation (:class:`SQLiteCheckpointManager`, ideal for
  single-node production deployments).
- Providing a pluggable serialization layer (:class:`StateSerializer` and
  friends) so the on-disk/on-wire representation of workflow state can
  evolve (JSON today, Pickle/MsgPack tomorrow) without touching manager
  code or application code.

This module explicitly does NOT contain:

- LangGraph graph construction, nodes, edges, or execution logic.
- API routes or UI logic.
- Business logic, approval logic, cost management, or agent execution.
- Any definition of ``MarketingState`` -- checkpoints store arbitrary,
  caller-supplied, serializable state.

Integration with LangGraph
---------------------------
LangGraph workflows periodically need to persist their entire state so
that execution can be resumed later -- for example, after a human
approval ``interrupt()``, or after a process crash/restart. A typical
integration pattern (implemented by orchestration code elsewhere in
MarketingOS, NOT by this module) looks like::

    manager = SQLiteCheckpointManager(db_path="marketingos.db")

    # After a node finishes, before pausing for human approval:
    checkpoint = Checkpoint(
        checkpoint_id=str(uuid4()),
        run_id=run_id,
        workflow_id="content_pipeline_v1",
        node_name="strategy_generation",
        state=graph_state.model_dump(),  # arbitrary serializable state
        version=1,
    )
    manager.save(checkpoint)

    # ... later, on resume ...
    latest = manager.latest(run_id)
    if latest is not None:
        graph_state = MarketingState.model_validate(latest.state)

This module does not import or depend on LangGraph. It is deliberately
generic: any workflow engine that needs "save state now, load it back
later, keyed by an ID" can use it.

Architecture
------------
This module follows a small hexagonal/clean-architecture shape:

- :class:`Checkpoint` -- the data model. Pure, storage-agnostic.
- :class:`StateSerializer` (ABC) -- converts arbitrary Python state to/from
  a storable representation. :class:`JSONStateSerializer` is the default
  implementation; ``PickleStateSerializer`` or ``MsgPackStateSerializer``
  could be added later without touching manager code.
- :class:`BaseCheckpointManager` (ABC) -- the storage-facing contract.
  Concrete backends (:class:`MemoryCheckpointManager`,
  :class:`SQLiteCheckpointManager`, and future
  ``PostgreSQLCheckpointManager`` / ``RedisCheckpointManager`` /
  ``S3CheckpointManager`` implementations) satisfy this contract.

Application code should depend on ``BaseCheckpointManager`` (dependency
injection friendly), not on a specific backend, so backends can be swapped
by changing only the object that gets constructed and injected.

Thread Safety
-------------
Both concrete managers guard mutable state with a ``threading.RLock``.
:class:`SQLiteCheckpointManager` additionally opens SQLite connections
with ``check_same_thread=False`` and serializes all access through the
same lock, since SQLite's own locking is coarse-grained and does not by
itself guarantee safe concurrent writes from a single process using
multiple connections.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Checkpoint",
    "StateSerializer",
    "JSONStateSerializer",
    "BaseCheckpointManager",
    "MemoryCheckpointManager",
    "SQLiteCheckpointManager",
    "CheckpointError",
    "CheckpointNotFoundError",
    "DuplicateCheckpointError",
    "CheckpointSerializationError",
    "CheckpointStorageError",
    "InvalidCheckpointDataError",
    "build_checkpointer",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware ``datetime``.

    Centralizing "now" generation gives a single seam for testing and
    guarantees all timestamps are timezone-aware.
    """
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CheckpointError(Exception):
    """Base exception for all checkpoint-related errors.

    Catching this exception type is sufficient to handle any error raised
    by this module's public API.
    """


class CheckpointNotFoundError(CheckpointError):
    """Raised when an operation references a ``checkpoint_id`` that does
    not exist in the backing store."""

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        super().__init__(f"No checkpoint found with checkpoint_id={checkpoint_id!r}")


class DuplicateCheckpointError(CheckpointError):
    """Raised when attempting to save a checkpoint whose ``checkpoint_id``
    already exists in the backing store.

    Checkpoints are treated as immutable, append-only records identified by
    ``checkpoint_id``; callers that want to update an existing checkpoint's
    metadata should use :meth:`BaseCheckpointManager.update_metadata`
    instead of re-saving under the same ID.
    """

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        super().__init__(
            f"A checkpoint with checkpoint_id={checkpoint_id!r} already exists. "
            "Checkpoints are immutable; use a new checkpoint_id or "
            "update_metadata() to modify metadata in place."
        )


class CheckpointSerializationError(CheckpointError):
    """Raised when workflow state cannot be serialized for storage, or
    cannot be deserialized after retrieval (e.g. corrupted data)."""


class CheckpointStorageError(CheckpointError):
    """Raised when the underlying storage backend (e.g. SQLite) encounters
    an error not covered by a more specific exception, such as a disk I/O
    failure or a connection error."""


class InvalidCheckpointDataError(CheckpointError):
    """Raised when checkpoint data retrieved from storage fails validation
    against the :class:`Checkpoint` model, indicating corruption or a
    schema mismatch."""


# ---------------------------------------------------------------------------
# Checkpoint model
# ---------------------------------------------------------------------------

class Checkpoint(BaseModel):
    """A single, storage-agnostic snapshot of workflow execution state.

    A ``Checkpoint`` captures everything needed to resume a workflow run
    from a particular point: which run it belongs to, which node produced
    it, and the arbitrary state payload itself. This model does not know
    or care what shape ``state`` takes -- that is entirely up to the
    workflow engine and application code.

    Attributes
    ----------
    checkpoint_id:
        Globally unique identifier for this specific checkpoint. Callers
        are responsible for generating unique IDs (e.g. via ``uuid4()``);
        this module does not generate IDs itself so that callers retain
        full control over ID schemes (sequential, UUID, content-hash,
        etc.).
    run_id:
        Identifier for the overall workflow *run* this checkpoint belongs
        to. A single run typically produces many checkpoints over its
        lifetime (one per node, or per pause point).
    workflow_id:
        Identifier for the *workflow definition* being executed (e.g.
        ``"content_pipeline_v1"``), as distinct from a specific run of it.
        Useful for querying/filtering checkpoints across runs of the same
        workflow type.
    state:
        Arbitrary, serializable workflow state. Typically a ``dict``
        (e.g. from ``SomeStateModel.model_dump()``) but any structure
        supported by the configured :class:`StateSerializer` is valid.
    metadata:
        Free-form dictionary for auxiliary information that is not part of
        the workflow state itself (e.g. execution duration, triggering
        event, environment tags). Mutable independently of ``state`` via
        :meth:`BaseCheckpointManager.update_metadata`.
    node_name:
        Name of the workflow node that produced this checkpoint, if
        applicable (e.g. ``"strategy_generation"``). ``None`` for
        checkpoints not tied to a specific node (e.g. a run-start marker).
    timestamp:
        Logical timestamp representing when the checkpointed state was
        produced (as opposed to ``created_at``, which is when the
        checkpoint record itself was persisted -- these are usually the
        same but may differ, e.g. for replayed/backfilled checkpoints).
    version:
        Schema/format version of the checkpoint payload. Allows the
        checkpoint format to evolve over time while remaining able to
        detect and handle older versions during deserialization.
    parent_checkpoint:
        The ``checkpoint_id`` of the checkpoint this one logically follows,
        if any. Enables reconstructing a checkpoint lineage/history for a
        run (e.g. for execution replay or audit trails).
    created_at:
        UTC timestamp when this checkpoint record was persisted by a
        :class:`BaseCheckpointManager`. Distinct from ``timestamp`` (see
        above).

    Notes
    -----
    Checkpoints are treated as immutable once saved, with one exception:
    ``metadata`` may be updated in place via
    :meth:`BaseCheckpointManager.update_metadata` without creating a new
    checkpoint record, since metadata is explicitly modeled as auxiliary
    and not part of the authoritative execution state.
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(..., min_length=1, description="Globally unique ID for this checkpoint.")
    run_id: str = Field(..., min_length=1, description="ID of the workflow run this checkpoint belongs to.")
    workflow_id: str = Field(..., min_length=1, description="ID of the workflow definition being executed.")
    state: Any = Field(..., description="Arbitrary, serializable workflow state payload.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form auxiliary metadata, mutable independently of state."
    )
    node_name: str | None = Field(
        default=None, description="Name of the node that produced this checkpoint, if applicable."
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Logical time the checkpointed state was produced."
    )
    version: int = Field(default=1, ge=1, description="Schema/format version of the checkpoint payload.")
    parent_checkpoint: str | None = Field(
        default=None, description="checkpoint_id of the logical predecessor checkpoint, if any."
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="UTC timestamp when this checkpoint record was persisted."
    )


# ---------------------------------------------------------------------------
# Serialization layer
# ---------------------------------------------------------------------------

class StateSerializer(ABC):
    """Abstract contract for converting between arbitrary workflow state
    and a storable string representation.

    Concrete storage backends depend on this abstraction rather than on a
    specific serialization format, so the wire/on-disk format can change
    (JSON -> Pickle -> MsgPack, or a versioned combination thereof) without
    touching :class:`BaseCheckpointManager` implementations.
    """

    @abstractmethod
    def serialize(self, state: Any) -> str:
        """Convert ``state`` into a string suitable for storage.

        Raises
        ------
        CheckpointSerializationError
            If ``state`` cannot be serialized.
        """
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, payload: str) -> Any:
        """Convert a previously serialized string back into Python state.

        Raises
        ------
        CheckpointSerializationError
            If ``payload`` cannot be deserialized (e.g. corrupted data).
        """
        raise NotImplementedError


class JSONStateSerializer(StateSerializer):
    """Default :class:`StateSerializer` implementation using ``json``.

    Suitable for workflow state composed of JSON-compatible primitives
    (dicts, lists, strings, numbers, booleans, ``None``) -- which is the
    expected shape for state derived from ``BaseModel.model_dump(mode="json")``
    calls elsewhere in MarketingOS.

    Notes
    -----
    A future ``PickleStateSerializer`` or ``MsgPackStateSerializer`` can be
    added by implementing :class:`StateSerializer` and passing an instance
    to the desired :class:`BaseCheckpointManager` subclass's constructor --
    no changes to manager code are required.
    """

    def serialize(self, state: Any) -> str:
        try:
            return json.dumps(state, default=str)
        except (TypeError, ValueError) as exc:
            raise CheckpointSerializationError(f"Failed to serialize state to JSON: {exc}") from exc

    def deserialize(self, payload: str) -> Any:
        try:
            return json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise CheckpointSerializationError(f"Failed to deserialize JSON checkpoint payload: {exc}") from exc


# ---------------------------------------------------------------------------
# BaseCheckpointManager
# ---------------------------------------------------------------------------

class BaseCheckpointManager(ABC):
    """Abstract contract for checkpoint persistence backends.

    Application and orchestration code should depend on this abstract type
    rather than any specific implementation, enabling dependency injection
    and easy swapping of backends (e.g. :class:`MemoryCheckpointManager` in
    tests, :class:`SQLiteCheckpointManager` in a single-node deployment, a
    future ``PostgreSQLCheckpointManager`` in a multi-node deployment).

    Implementing a New Backend
    ---------------------------
    To add a new backend (e.g. ``RedisCheckpointManager``,
    ``S3CheckpointManager``), subclass this class and implement every
    abstract method below. Concrete backends are expected to raise the
    exceptions defined in this module (:class:`CheckpointNotFoundError`,
    :class:`DuplicateCheckpointError`, etc.) rather than backend-specific
    exceptions, so calling code can handle errors uniformly regardless of
    which backend is in use.
    """

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None:
        """Persist ``checkpoint``.

        Raises
        ------
        DuplicateCheckpointError
            If a checkpoint with the same ``checkpoint_id`` already exists.
        CheckpointSerializationError
            If ``checkpoint.state`` cannot be serialized.
        CheckpointStorageError
            If the underlying storage backend fails.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint_id: str) -> Checkpoint:
        """Retrieve the checkpoint identified by ``checkpoint_id``.

        Raises
        ------
        CheckpointNotFoundError
            If no checkpoint with that ID exists.
        CheckpointSerializationError
            If the stored state cannot be deserialized.
        InvalidCheckpointDataError
            If the stored record fails validation as a :class:`Checkpoint`.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, checkpoint_id: str) -> None:
        """Delete the checkpoint identified by ``checkpoint_id``.

        Raises
        ------
        CheckpointNotFoundError
            If no checkpoint with that ID exists.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, checkpoint_id: str) -> bool:
        """Return ``True`` if a checkpoint with ``checkpoint_id`` exists."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Delete every checkpoint managed by this instance.

        Intended primarily for test teardown and for deliberately resetting
        all persisted execution state.
        """
        raise NotImplementedError

    @abstractmethod
    def list_checkpoints(
        self,
        *,
        run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> list[Checkpoint]:
        """List checkpoints, optionally filtered by ``run_id`` and/or
        ``workflow_id``.

        Parameters
        ----------
        run_id:
            If provided, only checkpoints belonging to this run are
            returned.
        workflow_id:
            If provided, only checkpoints belonging to this workflow
            definition are returned.

        Returns
        -------
        list[Checkpoint]
            Matching checkpoints, ordered oldest-to-newest by
            ``created_at``.
        """
        raise NotImplementedError

    # -- convenience methods built on the abstract primitives above --------

    def latest(self, run_id: str) -> Checkpoint | None:
        """Return the most recently created checkpoint for ``run_id``, or
        ``None`` if the run has no checkpoints.

        This is the primary method orchestration code should use to resume
        a workflow run: it hides the "sort by recency" concern behind a
        single call.
        """
        checkpoints = self.list_checkpoints(run_id=run_id)
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda cp: cp.created_at)

    def count(
        self,
        *,
        run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> int:
        """Return the number of checkpoints matching the given filters.

        Equivalent to ``len(self.list_checkpoints(...))`` but named
        separately so backends may override it with a more efficient
        ``COUNT(*)``-style implementation if desired.
        """
        return len(self.list_checkpoints(run_id=run_id, workflow_id=workflow_id))

    @abstractmethod
    def update_metadata(
        self,
        checkpoint_id: str,
        updates: dict[str, Any],
        *,
        replace: bool = False,
    ) -> Checkpoint:
        """Update the ``metadata`` dict of an existing checkpoint in place.

        Parameters
        ----------
        checkpoint_id:
            The checkpoint to update.
        updates:
            Key/value pairs to merge into (or replace) the checkpoint's
            metadata.
        replace:
            If ``True``, ``updates`` entirely replaces the existing
            metadata dict. If ``False`` (default), ``updates`` is shallow
            merged into the existing metadata.

        Returns
        -------
        Checkpoint
            The updated checkpoint.

        Raises
        ------
        CheckpointNotFoundError
            If no checkpoint with that ID exists.

        Notes
        -----
        This is the one sanctioned way to mutate a persisted checkpoint;
        ``state`` itself is never mutated in place -- a new checkpoint
        should be saved instead when execution state changes.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MemoryCheckpointManager
# ---------------------------------------------------------------------------

class MemoryCheckpointManager(BaseCheckpointManager):
    """In-memory :class:`BaseCheckpointManager` implementation.

    Stores checkpoints in a plain ``dict`` keyed by ``checkpoint_id``. No
    data is persisted beyond the lifetime of the process/instance.

    Intended primarily for:

    - Unit and integration tests, where a real database is undesirable.
    - Local development and prototyping.
    - Any scenario where checkpoint durability across process restarts is
      not required.

    Thread Safety
    --------------
    All operations acquire an internal ``threading.RLock``, making this
    class safe to share across threads within a single process. It offers
    no protection across separate processes (unlike
    :class:`SQLiteCheckpointManager`, which is backed by a file every
    process can open).
    """

    def __init__(self, serializer: StateSerializer | None = None) -> None:
        """Initialize an empty in-memory checkpoint store.

        Parameters
        ----------
        serializer:
            Optional :class:`StateSerializer` used to validate that stored
            state is serializable (for parity with persistent backends,
            which must actually serialize state to store it). Defaults to
            :class:`JSONStateSerializer`. State is kept in memory as-is
            (not actually serialized to a string) for speed, but is passed
            through ``serialize``/``deserialize`` once to fail fast on
            non-serializable input, matching the behavior a caller would
            see from a real persistent backend.
        """
        self._serializer = serializer or JSONStateSerializer()
        self._store: dict[str, Checkpoint] = {}
        self._lock = threading.RLock()

    def save(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            if checkpoint.checkpoint_id in self._store:
                raise DuplicateCheckpointError(checkpoint.checkpoint_id)
            # Round-trip through the serializer to validate serializability
            # and to ensure behavior parity with persistent backends.
            serialized = self._serializer.serialize(checkpoint.state)
            validated_state = self._serializer.deserialize(serialized)
            stored = checkpoint.model_copy(update={"state": validated_state})
            self._store[checkpoint.checkpoint_id] = stored

    def load(self, checkpoint_id: str) -> Checkpoint:
        with self._lock:
            checkpoint = self._store.get(checkpoint_id)
            if checkpoint is None:
                raise CheckpointNotFoundError(checkpoint_id)
            return checkpoint.model_copy(deep=True)

    def delete(self, checkpoint_id: str) -> None:
        with self._lock:
            if checkpoint_id not in self._store:
                raise CheckpointNotFoundError(checkpoint_id)
            del self._store[checkpoint_id]

    def exists(self, checkpoint_id: str) -> bool:
        with self._lock:
            return checkpoint_id in self._store

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def list_checkpoints(
        self,
        *,
        run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> list[Checkpoint]:
        with self._lock:
            results = [
                cp.model_copy(deep=True)
                for cp in self._store.values()
                if (run_id is None or cp.run_id == run_id)
                and (workflow_id is None or cp.workflow_id == workflow_id)
            ]
            return sorted(results, key=lambda cp: cp.created_at)

    def update_metadata(
        self,
        checkpoint_id: str,
        updates: dict[str, Any],
        *,
        replace: bool = False,
    ) -> Checkpoint:
        with self._lock:
            checkpoint = self._store.get(checkpoint_id)
            if checkpoint is None:
                raise CheckpointNotFoundError(checkpoint_id)
            new_metadata = dict(updates) if replace else {**checkpoint.metadata, **updates}
            updated = checkpoint.model_copy(update={"metadata": new_metadata})
            self._store[checkpoint_id] = updated
            return updated.model_copy(deep=True)


# ---------------------------------------------------------------------------
# SQLiteCheckpointManager
# ---------------------------------------------------------------------------

class SQLiteCheckpointManager(BaseCheckpointManager):
    """SQLite-backed :class:`BaseCheckpointManager` implementation.

    Suitable for single-node production deployments requiring durable
    checkpoint storage without the operational overhead of a separate
    database server. The database file and schema are created
    automatically on first use -- application code never needs to run
    manual migrations or setup steps.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file. Parent directories
        are created automatically if they do not exist. Defaults to
        ``"marketingos_checkpoints.db"`` in the current working directory.
    serializer:
        :class:`StateSerializer` used to convert workflow state to/from a
        storable string. Defaults to :class:`JSONStateSerializer`.

    Notes
    -----
    - Uses ``check_same_thread=False`` combined with an internal
      ``threading.RLock`` to allow safe use from multiple threads within a
      single process, since a single ``sqlite3.Connection`` is not
      inherently thread-safe for concurrent writes.
    - Uses ``PRAGMA journal_mode=WAL`` to improve concurrent read
      performance and crash resilience.
    - Callers should call :meth:`close` when finished (or use this class
      as a context manager) to release the underlying connection cleanly.

    Usage Example
    -------------
    ::

        with SQLiteCheckpointManager(db_path="data/checkpoints.db") as mgr:
            mgr.save(checkpoint)
            latest = mgr.latest(run_id="run-123")
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str | Path = "marketingos_checkpoints.db",
        serializer: StateSerializer | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._serializer = serializer or JSONStateSerializer()
        self._lock = threading.RLock()
        self._closed = False

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                isolation_level=None,  # autocommit; we manage transactions explicitly
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL;")
            self._connection.execute("PRAGMA foreign_keys=ON;")
        except sqlite3.Error as exc:
            raise CheckpointStorageError(f"Failed to open SQLite database at {self._db_path}: {exc}") from exc

        self._initialize_schema()

    # -- setup / connection management --------------------------------

    def _initialize_schema(self) -> None:
        """Create the checkpoints table and indexes if they do not already
        exist. Safe to call repeatedly (idempotent)."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id       TEXT PRIMARY KEY,
                    run_id              TEXT NOT NULL,
                    workflow_id         TEXT NOT NULL,
                    node_name           TEXT,
                    serialized_state    TEXT NOT NULL,
                    metadata            TEXT NOT NULL,
                    version             INTEGER NOT NULL,
                    parent_checkpoint   TEXT,
                    timestamp           TEXT NOT NULL,
                    created_at          TEXT NOT NULL
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow_id ON checkpoints(workflow_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at);"
            )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager providing a cursor within an explicit
        transaction, committing on success and rolling back on error.

        Centralizes transaction handling so every method gets consistent,
        safe commit/rollback behavior without repeating boilerplate.
        """
        self._ensure_open()
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN;")
                yield cursor
                self._connection.commit()
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise CheckpointStorageError(f"SQLite transaction failed: {exc}") from exc
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    def _ensure_open(self) -> None:
        """Raise if this manager has already been closed."""
        if self._closed:
            raise CheckpointStorageError(
                "This SQLiteCheckpointManager has been closed and can no longer be used."
            )

    # -- row <-> model conversion ----------------------------------------

    def _row_to_checkpoint(self, row: sqlite3.Row) -> Checkpoint:
        """Convert a raw SQLite row into a validated :class:`Checkpoint`.

        Raises
        ------
        CheckpointSerializationError
            If the stored state payload cannot be deserialized.
        InvalidCheckpointDataError
            If the row's data fails :class:`Checkpoint` validation.
        """
        try:
            state = self._serializer.deserialize(row["serialized_state"])
            metadata = json.loads(row["metadata"])
        except CheckpointSerializationError:
            raise
        except (TypeError, ValueError) as exc:
            raise CheckpointSerializationError(f"Failed to deserialize checkpoint metadata: {exc}") from exc

        try:
            return Checkpoint(
                checkpoint_id=row["checkpoint_id"],
                run_id=row["run_id"],
                workflow_id=row["workflow_id"],
                node_name=row["node_name"],
                state=state,
                metadata=metadata,
                version=row["version"],
                parent_checkpoint=row["parent_checkpoint"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
        except (ValueError, TypeError) as exc:
            raise InvalidCheckpointDataError(
                f"Stored checkpoint row for checkpoint_id={row['checkpoint_id']!r} is invalid: {exc}"
            ) from exc

    # -- BaseCheckpointManager implementation -----------------------------

    def save(self, checkpoint: Checkpoint) -> None:
        serialized_state = self._serializer.serialize(checkpoint.state)
        serialized_metadata = json.dumps(checkpoint.metadata, default=str)

        with self._transaction() as cursor:
            cursor.execute("SELECT 1 FROM checkpoints WHERE checkpoint_id = ?;", (checkpoint.checkpoint_id,))
            if cursor.fetchone() is not None:
                raise DuplicateCheckpointError(checkpoint.checkpoint_id)

            cursor.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id, run_id, workflow_id, node_name,
                    serialized_state, metadata, version, parent_checkpoint,
                    timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.run_id,
                    checkpoint.workflow_id,
                    checkpoint.node_name,
                    serialized_state,
                    serialized_metadata,
                    checkpoint.version,
                    checkpoint.parent_checkpoint,
                    checkpoint.timestamp.isoformat(),
                    checkpoint.created_at.isoformat(),
                ),
            )

    def load(self, checkpoint_id: str) -> Checkpoint:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?;", (checkpoint_id,)
            )
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            raise CheckpointNotFoundError(checkpoint_id)
        return self._row_to_checkpoint(row)

    def delete(self, checkpoint_id: str) -> None:
        with self._transaction() as cursor:
            cursor.execute("SELECT 1 FROM checkpoints WHERE checkpoint_id = ?;", (checkpoint_id,))
            if cursor.fetchone() is None:
                raise CheckpointNotFoundError(checkpoint_id)
            cursor.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?;", (checkpoint_id,))

    def exists(self, checkpoint_id: str) -> bool:
        self._ensure_open()
        with self._lock:
            cursor = self._connection.execute(
                "SELECT 1 FROM checkpoints WHERE checkpoint_id = ?;", (checkpoint_id,)
            )
            row = cursor.fetchone()
            cursor.close()
        return row is not None

    def clear(self) -> None:
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM checkpoints;")

    def list_checkpoints(
        self,
        *,
        run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> list[Checkpoint]:
        self._ensure_open()
        query = "SELECT * FROM checkpoints WHERE 1=1"
        params: list[Any] = []
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if workflow_id is not None:
            query += " AND workflow_id = ?"
            params.append(workflow_id)
        query += " ORDER BY created_at ASC;"

        with self._lock:
            cursor = self._connection.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
        return [self._row_to_checkpoint(row) for row in rows]

    def count(
        self,
        *,
        run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> int:
        """Return the number of checkpoints matching the given filters.

        Overrides the base class's default implementation with a native
        ``COUNT(*)`` query for efficiency, avoiding materializing every
        matching row into a :class:`Checkpoint` object.
        """
        self._ensure_open()
        query = "SELECT COUNT(*) FROM checkpoints WHERE 1=1"
        params: list[Any] = []
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if workflow_id is not None:
            query += " AND workflow_id = ?"
            params.append(workflow_id)

        with self._lock:
            cursor = self._connection.execute(query, params)
            (total,) = cursor.fetchone()
            cursor.close()
        return int(total)

    def update_metadata(
        self,
        checkpoint_id: str,
        updates: dict[str, Any],
        *,
        replace: bool = False,
    ) -> Checkpoint:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT metadata FROM checkpoints WHERE checkpoint_id = ?;", (checkpoint_id,)
            )
            row = cursor.fetchone()
            if row is None:
                raise CheckpointNotFoundError(checkpoint_id)

            current_metadata = json.loads(row["metadata"])
            new_metadata = dict(updates) if replace else {**current_metadata, **updates}
            cursor.execute(
                "UPDATE checkpoints SET metadata = ? WHERE checkpoint_id = ?;",
                (json.dumps(new_metadata, default=str), checkpoint_id),
            )

        return self.load(checkpoint_id)

    # -- SQLite-specific maintenance methods -------------------------------

    def vacuum(self) -> None:
        """Run SQLite's ``VACUUM`` command to reclaim disk space and
        defragment the database file.

        Notes
        -----
        ``VACUUM`` requires that no transaction is active and can be
        relatively slow on large databases; it is intended for periodic
        maintenance (e.g. a scheduled job), not routine use after every
        write.
        """
        self._ensure_open()
        with self._lock:
            try:
                self._connection.execute("VACUUM;")
            except sqlite3.Error as exc:
                raise CheckpointStorageError(f"VACUUM failed: {exc}") from exc

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Safe to call multiple times. After calling this, the manager
        instance can no longer be used -- any further calls raise
        :class:`CheckpointStorageError`.
        """
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> SQLiteCheckpointManager:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if close() was never called explicitly.
        # Swallow all errors: __del__ runs during interpreter teardown,
        # where raising is unsafe and unhelpful.
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LangGraph checkpointer factory
# ---------------------------------------------------------------------------
#
# Everything above this point is a storage-agnostic checkpoint model that
# knows nothing about LangGraph, by design. ``graph.py``, however, compiles
# its ``StateGraph`` with a LangGraph-native ``BaseCheckpointSaver`` (passed
# to ``StateGraph.compile(checkpointer=...)``), which is a distinct
# interface from :class:`BaseCheckpointManager` above. ``build_checkpointer``
# is the adapter point that supplies that LangGraph-native saver, keeping
# the choice of default checkpointer backend in this module.

def build_checkpointer() -> BaseCheckpointSaver:
    """Build the default LangGraph checkpoint saver for compiled graphs.

    Returns
    -------
    BaseCheckpointSaver
        An in-memory :class:`~langgraph.checkpoint.memory.MemorySaver`.
        Sufficient for a single-process workflow run; swap in a durable
        saver (e.g. a Postgres- or SQLite-backed one) here if cross-process
        or cross-restart resumption is required.
    """
    return MemorySaver()
