"""Discover, validate, and resolve prompt versions on disk.

This module owns one concern: given an agent, find the prompt *versions* that
exist and decide which one a caller means. It answers "what versions are there?"
and "which version is 'latest' / 'v3'?" — nothing more.

It deliberately does **not** read ``system.jinja`` / ``user.jinja``, parse
``metadata.yaml``, render templates, cache prompt *contents*, or know how the
registry orchestrates things. Those belong to
:class:`~marketingos.prompts.loader.PromptLoader`, the renderer, and the
registry respectively. The resolver's output — a
:class:`~marketingos.prompts.models.PromptVersion` carrying the resolved
directory path — is exactly what the loader needs and nothing it doesn't.

Layout it understands::

    <base_path>/
        <agent>/
            v1/
            v2/
            v10/

Version directories are named ``v<number>`` (``v1``, ``v2``, ``v10``, ``v25``).
Directories that do not match — ``version1``, ``temp``, ``old``, ``v1_backup`` —
are simply not versions and are ignored during discovery. Ordering is always
*numeric*: ``v2`` precedes ``v10``, never the other way around, and ``latest``
resolves to the highest number.

Dependency direction (this module sits in the middle, and only looks *down*)::

    PromptRegistry -> PromptVersionResolver -> PromptLoader -> PromptRenderer

Extension seams (kept deliberately small so the public API can stay fixed):

* **More aliases** (``stable``, ``experimental``, ...): extend
  :meth:`PromptVersionResolver._resolve_alias`.
* **Richer version grammars** (prerelease ``v2-beta``, semantic versioning):
  change :attr:`PromptVersionResolver._VERSION_PATTERN` and
  :meth:`PromptVersionResolver._extract_version_number` (the sort key).
* **Other sources** (tenant-specific trees, remote registries): override
  :meth:`PromptVersionResolver._scan_versions`.

None of these require touching :meth:`list_versions`, :meth:`latest`,
:meth:`resolve`, or :meth:`exists`.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from loguru import logger

from marketingos.prompts.exceptions import (
    PromptDirectoryError,
    PromptNotFoundError,
    PromptValidationError,
    PromptVersionNotFoundError,
)
from marketingos.prompts.models import PromptVersion

__all__ = ["PromptVersionResolver"]


class PromptVersionResolver:
    """Discover and resolve the prompt versions available for an agent.

    A resolver is bound to a base directory (the root holding the per-agent
    subtrees) and answers version queries against it. Discovered version lists
    are cached per agent by default; the cache holds *version metadata*
    (directory names and paths), never prompt contents, which remain the
    loader's concern.

    Thread safety:
        A resolver is safe to share across threads. Its base path and compiled
        pattern are immutable; the per-agent cache is guarded by a lock.
        Directory scans are idempotent, so at worst two threads scan the same
        agent once each with a last-write-wins store — no corruption, no lost
        updates.

    Args:
        base_path: Root directory containing ``<agent>/<version>/`` subtrees.
            It need not exist at construction; existence is checked per query.
        use_cache: When ``True`` (default), discovered version lists are cached
            per agent. Prompt trees are normally immutable deploy artifacts, so
            caching is safe and avoids repeated directory scans; call
            :meth:`clear_cache` if versions change on disk at runtime. Set to
            ``False`` for a fully stateless resolver (e.g. in tests).
    """

    #: The one built-in alias. Kept as a constant so future aliases can join it.
    LATEST_ALIAS: str = "latest"

    #: Version directory grammar, compiled once at class definition time.
    _VERSION_PATTERN: re.Pattern[str] = re.compile(r"^v(\d+)$")

    def __init__(self, base_path: Path | str, *, use_cache: bool = True) -> None:
        self._base_path: Path = Path(base_path)
        self._use_cache: bool = use_cache
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[PromptVersion, ...]] = {}
        self._logger = logger.bind(component="PromptVersionResolver")

    # -- public API ---------------------------------------------------------

    @property
    def base_path(self) -> Path:
        """The root directory this resolver discovers versions under."""
        return self._base_path

    def list_versions(self, agent: str) -> list[PromptVersion]:
        """Return every valid version for ``agent``, sorted numerically ascending.

        Args:
            agent: The agent whose versions to list.

        Returns:
            Versions ordered ``v1, v2, ..., v10, v11`` (numeric, not
            lexicographic). Empty if the agent directory exists but holds no
            validly named version directories.

        Raises:
            PromptValidationError: If ``agent`` is empty or not a single path
                segment.
            PromptDirectoryError: If the base directory does not exist.
            PromptNotFoundError: If the agent directory does not exist.
        """
        self._validate_agent(agent)
        return list(self._discover_versions(agent))

    def latest(self, agent: str) -> PromptVersion:
        """Return the highest-numbered version for ``agent``.

        Raises:
            PromptValidationError: If ``agent`` is invalid.
            PromptDirectoryError: If the base directory does not exist.
            PromptNotFoundError: If the agent directory does not exist.
            PromptVersionNotFoundError: If the agent has no valid versions.
        """
        self._validate_agent(agent)
        versions = self._discover_versions(agent)
        if not versions:
            raise PromptVersionNotFoundError(
                agent,
                self.LATEST_ALIAS,
                available=(),
                message=f"No prompt versions found for agent {agent!r}.",
            )
        return versions[-1]

    def resolve(self, agent: str, version: str = "latest") -> PromptVersion:
        """Resolve ``version`` (an explicit name or an alias) to a concrete one.

        ``version`` may be the alias ``"latest"`` (the default) or an explicit
        version name such as ``"v2"``. Explicit names are format-validated
        before lookup, so a well-formed-but-absent version and a malformed
        request produce different, specific errors.

        Args:
            agent: The agent to resolve within.
            version: An alias or explicit version name. Defaults to ``"latest"``.

        Returns:
            The resolved :class:`PromptVersion`.

        Raises:
            PromptValidationError: If ``agent`` or ``version`` is empty, or if an
                explicit ``version`` does not match the ``v<number>`` grammar.
            PromptDirectoryError: If the base directory does not exist.
            PromptNotFoundError: If the agent directory does not exist.
            PromptVersionNotFoundError: If the requested version does not exist.
        """
        self._validate_agent(agent)
        if not isinstance(version, str) or not version.strip():
            raise PromptValidationError(detail="version must be a non-empty string.")
        request = version.strip()

        resolved = self._resolve_alias(agent, request)
        if resolved is not None:
            return resolved

        # Explicit version: validate grammar first, then look it up.
        self._validate_version_name(request)
        versions = self._discover_versions(agent)
        for candidate in versions:
            if candidate.version == request:
                return candidate

        raise PromptVersionNotFoundError(
            agent,
            request,
            available=tuple(candidate.version for candidate in versions),
        )

    def exists(self, agent: str, version: str) -> bool:
        """Report whether ``version`` resolves for ``agent``, without raising.

        Accepts the same values as :meth:`resolve` (an explicit name or the
        ``latest`` alias). Returns ``False`` for a missing agent, a missing
        version, or a malformed version string; only a structurally unsafe
        ``agent`` (empty or containing a path separator) raises, since that is a
        caller error rather than a miss.
        """
        self._validate_agent(agent)
        if not isinstance(version, str) or not version.strip():
            return False
        request = version.strip()

        if not self._base_path.is_dir() or not (self._base_path / agent).is_dir():
            return False

        versions = self._discover_versions(agent)
        if self._is_latest_alias(request):
            return len(versions) > 0
        if not self._is_valid_version_name(request):
            return False
        return any(candidate.version == request for candidate in versions)

    def clear_cache(self) -> None:
        """Discard cached version lists so the next query re-scans the disk."""
        with self._lock:
            self._cache.clear()

    # -- discovery ----------------------------------------------------------

    def _discover_versions(self, agent: str) -> tuple[PromptVersion, ...]:
        """Return the sorted versions for ``agent``, using the cache if enabled.

        The filesystem scan runs outside the lock so I/O never serialises
        callers; only the tiny cache read/write is guarded. Assumes ``agent``
        has already been validated by the public method that called in.
        """
        if self._use_cache:
            with self._lock:
                cached = self._cache.get(agent)
            if cached is not None:
                return cached

        versions = self._scan_versions(agent)

        if self._use_cache:
            with self._lock:
                self._cache[agent] = versions
        return versions

    def _scan_versions(self, agent: str) -> tuple[PromptVersion, ...]:
        """Scan the agent directory and build sorted :class:`PromptVersion` objects.

        This is the seam a future subclass would override to source versions
        from somewhere other than the local base directory (a tenant-specific
        tree, a remote registry). It must return versions sorted numerically
        ascending.
        """
        base = self._base_path
        if not base.is_dir():
            raise PromptDirectoryError(
                path=base,
                detail="Prompt templates base directory does not exist.",
            )

        agent_dir = base / agent
        if not agent_dir.is_dir():
            raise PromptNotFoundError(agent, available=self._list_subdir_names(base))

        discovered = [
            PromptVersion(agent=agent, version=child.name, path=child)
            for child in agent_dir.iterdir()
            if child.is_dir() and self._is_valid_version_name(child.name)
        ]
        ordered = self._sort_versions(discovered)

        self._logger.bind(
            event="versioning.discovered",
            agent=agent,
            count=len(ordered),
        ).debug("Discovered prompt versions")
        return ordered

    # -- validation & ordering ---------------------------------------------

    def _sort_versions(
        self, versions: list[PromptVersion]
    ) -> tuple[PromptVersion, ...]:
        """Sort versions by their numeric component, ascending.

        Numeric ordering is the whole point: lexicographic sorting would place
        ``v10`` before ``v2``.
        """
        return tuple(
            sorted(versions, key=lambda version: self._extract_version_number(version.version))
        )

    def _extract_version_number(self, version: str) -> int:
        """Extract the integer component of a ``v<number>`` name.

        Raises:
            PromptValidationError: If ``version`` does not match the grammar.
                Only reachable via a malformed explicit request; discovered
                names are pre-filtered.
        """
        match = self._VERSION_PATTERN.match(version)
        if match is None:
            raise PromptValidationError(
                detail=f"Invalid version name {version!r}; expected 'v<number>'."
            )
        return int(match.group(1))

    @classmethod
    def _is_valid_version_name(cls, name: str) -> bool:
        """Return whether ``name`` matches the version grammar (predicate form)."""
        return cls._VERSION_PATTERN.match(name) is not None

    def _validate_version_name(self, version: str) -> None:
        """Raise if ``version`` is not a well-formed explicit version name.

        Raises:
            PromptValidationError: If ``version`` does not match ``v<number>``.
        """
        if not self._is_valid_version_name(version):
            raise PromptValidationError(
                detail=(
                    f"Invalid version name {version!r}; expected the form "
                    f"'v<number>' (e.g. 'v1', 'v10')."
                )
            )

    # -- alias resolution ---------------------------------------------------

    def _resolve_alias(self, agent: str, version: str) -> PromptVersion | None:
        """Resolve a symbolic alias to a concrete version, or ``None``.

        Returns ``None`` when ``version`` is not a known alias, signalling the
        caller to treat it as an explicit version name. New aliases are added
        here without changing the public API.
        """
        if self._is_latest_alias(version):
            return self.latest(agent)
        return None

    @classmethod
    def _is_latest_alias(cls, version: str) -> bool:
        """Return whether ``version`` is the ``latest`` alias (case-insensitive)."""
        return version.strip().lower() == cls.LATEST_ALIAS

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _validate_agent(agent: str) -> None:
        """Reject an empty or multi-segment agent name (guards path traversal)."""
        if not isinstance(agent, str) or not agent.strip():
            raise PromptValidationError(detail="agent must be a non-empty string.")
        if "/" in agent or "\\" in agent or "\x00" in agent or agent in {".", ".."}:
            raise PromptValidationError(
                detail=f"agent must be a single path segment, got {agent!r}."
            )

    @staticmethod
    def _list_subdir_names(path: Path) -> tuple[str, ...]:
        """Sorted names of immediate subdirectories, for error hints."""
        try:
            return tuple(sorted(child.name for child in path.iterdir() if child.is_dir()))
        except OSError:  # pragma: no cover - hint best-effort only
            return ()