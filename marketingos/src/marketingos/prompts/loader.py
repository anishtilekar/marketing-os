"""Load prompt packages from the filesystem into typed models.

This module has a single responsibility: read a prompt package off disk and
return a fully populated :class:`~marketingos.prompts.models.PromptTemplate`.
It is the only place in the prompts package that touches the filesystem.

It deliberately does **not** render templates, resolve which version to load,
orchestrate anything, or know that a renderer or registry exists. Given an
already-chosen ``(agent, version)`` it reads the package; deciding *which*
version to read is the version resolver's job, and turning a template into text
is the renderer's.

Expected on-disk layout::

    <base_path>/
        <agent>/
            <version>/
                system.jinja      (optional)
                user.jinja        (optional)
                metadata.yaml     (optional)

A package must contain at least one of ``system.jinja`` / ``user.jinja`` — that
mirrors the invariant on :class:`PromptTemplate`. ``metadata.yaml`` is optional;
when absent the template carries an empty
:class:`~marketingos.prompts.models.PromptMetadata`, and when present it must be
valid YAML that parses into that model.

All failures surface as the package's own exceptions
(:mod:`marketingos.prompts.exceptions`), never raw ``OSError`` / ``yaml`` /
``pydantic`` errors, with the original preserved via chaining.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from marketingos.prompts.exceptions import (
    PromptDirectoryError,
    PromptMetadataError,
    PromptNotFoundError,
    PromptTemplateNotFoundError,
    PromptValidationError,
    PromptVersionNotFoundError,
)
from marketingos.prompts.models import PromptAsset, PromptMetadata, PromptTemplate

__all__ = ["PromptLoader"]


class PromptLoader:
    """Load prompt packages from a base directory into typed models.

    A loader is bound to one base directory (the root that holds the per-agent
    subtrees) and reads packages beneath it on demand. File contents are cached
    per instance so repeated loads of the same package do not re-hit the disk;
    call :meth:`clear_cache` after changing files on disk.

    Thread safety:
        A loader is safe to share across threads. Its base path is immutable
        after construction, and the only shared mutable state is the content
        cache, whose bookkeeping is synchronised internally by
        :func:`functools.lru_cache`. File reads are idempotent, so a benign
        concurrent double-read is the worst case; no additional locking is
        needed.

    Args:
        base_path: Root directory containing the ``<agent>/<version>/`` subtrees.
            It is not required to exist at construction time; existence is
            checked when a package is loaded.
        cache_size: Maximum number of distinct files whose contents are cached.
    """

    #: Canonical filename of the system prompt template within a version dir.
    SYSTEM_FILENAME: str = "system.jinja"
    #: Canonical filename of the user prompt template within a version dir.
    USER_FILENAME: str = "user.jinja"
    #: Canonical filename of the metadata sidecar within a version dir.
    METADATA_FILENAME: str = "metadata.yaml"

    def __init__(self, base_path: Path | str, *, cache_size: int = 256) -> None:
        self._base_path: Path = Path(base_path)
        self._logger = logger.bind(component="PromptLoader")
        # Per-instance content cache. Wrapping a *static* reader keeps ``self``
        # out of the closure (no reference cycle) while giving every loader its
        # own isolated, clearable cache — which also keeps unit tests hermetic.
        self._read_cached: Callable[[Path], str] = lru_cache(maxsize=cache_size)(
            self._read_text_from_disk
        )

    # -- public API ---------------------------------------------------------

    @property
    def base_path(self) -> Path:
        """The root directory this loader reads packages from."""
        return self._base_path

    def load(self, agent: str, version: str) -> PromptTemplate:
        """Load the complete prompt package for ``agent`` and ``version``.

        Args:
            agent: The agent whose prompt package to load.
            version: The version identifier to load.

        Returns:
            A fully populated :class:`PromptTemplate` with whichever of the
            system/user assets are present and the parsed metadata.

        Raises:
            PromptValidationError: If ``agent`` or ``version`` is empty or is not
                a single path segment, or if the read assets do not form a valid
                package.
            PromptDirectoryError: If the base directory is missing or a file
                cannot be read.
            PromptNotFoundError: If the agent directory does not exist.
            PromptVersionNotFoundError: If the version directory does not exist.
            PromptTemplateNotFoundError: If neither template file is present.
            PromptMetadataError: If ``metadata.yaml`` is present but invalid.
        """
        self._validate_segment(agent, kind="agent")
        self._validate_segment(version, kind="version")

        version_dir = self._validate_structure(agent, version)

        system = self._load_system(version_dir)
        user = self._load_user(version_dir)
        metadata = self._load_metadata(version_dir)

        try:
            template = PromptTemplate(system=system, user=user, metadata=metadata)
        except ValidationError as exc:
            # Structure validation already guarantees at least one asset, so this
            # is a defensive guard against any future model-level constraint.
            raise PromptValidationError(
                name=f"{agent}/{version}",
                detail="Loaded assets did not form a valid prompt package.",
                cause=exc,
            ) from exc

        self._logger.bind(
            event="loader.loaded",
            agent=agent,
            version=version,
            has_system=system is not None,
            has_user=user is not None,
        ).debug("Loaded prompt package")
        return template

    def exists(self, agent: str, version: str) -> bool:
        """Report whether a loadable package exists for ``agent``/``version``.

        A package "exists" when its version directory is present and contains at
        least one template file. This never raises for a merely-absent package;
        it does raise :class:`PromptValidationError` for a structurally invalid
        ``agent``/``version`` (an empty value or one containing a path
        separator), since that indicates a caller error rather than a miss.
        """
        self._validate_segment(agent, kind="agent")
        self._validate_segment(version, kind="version")

        version_dir = self._version_dir(agent, version)
        if not version_dir.is_dir():
            return False
        return self._has_any_template(version_dir)

    def clear_cache(self) -> None:
        """Discard all cached file contents.

        Call this after prompt files change on disk; subsequent loads will read
        fresh content.
        """
        self._read_cached.cache_clear()  # type: ignore[attr-defined]

    # -- structure validation ----------------------------------------------

    def _validate_structure(self, agent: str, version: str) -> Path:
        """Validate the directory layout and return the version directory.

        Checks, in order, that the base directory, the agent directory, and the
        version directory all exist, and that the version directory holds at
        least one template file. Each failure raises the most specific matching
        exception, with a list of available alternatives where useful.
        """
        base = self._base_path
        if not base.is_dir():
            raise PromptDirectoryError(
                path=base,
                detail="Prompt templates base directory does not exist.",
            )

        agent_dir = base / agent
        if not agent_dir.is_dir():
            raise PromptNotFoundError(agent, available=self._list_dirs(base))

        version_dir = agent_dir / version
        if not version_dir.is_dir():
            raise PromptVersionNotFoundError(
                agent, version, available=self._list_dirs(agent_dir)
            )

        if not self._has_any_template(version_dir):
            raise PromptTemplateNotFoundError(
                agent,
                version,
                available=self._list_files(version_dir),
                message=(
                    f"Prompt package {agent!r}/{version!r} contains neither "
                    f"{self.SYSTEM_FILENAME!r} nor {self.USER_FILENAME!r}."
                ),
            )

        return version_dir

    # -- asset loading ------------------------------------------------------

    def _load_system(self, version_dir: Path) -> PromptAsset | None:
        """Load the system template asset, or ``None`` if it is absent."""
        return self._load_asset(version_dir, self.SYSTEM_FILENAME)

    def _load_user(self, version_dir: Path) -> PromptAsset | None:
        """Load the user template asset, or ``None`` if it is absent."""
        return self._load_asset(version_dir, self.USER_FILENAME)

    def _load_asset(self, version_dir: Path, filename: str) -> PromptAsset | None:
        """Read a single template file into a :class:`PromptAsset`.

        Returns ``None`` when the file is not present, so callers can treat an
        optional asset uniformly.
        """
        path = version_dir / filename
        if not path.is_file():
            return None

        content = self._read_cached(path)
        try:
            return PromptAsset(filename=filename, content=content)
        except ValidationError as exc:  # pragma: no cover - defensive
            raise PromptValidationError(
                name=filename,
                detail="Template asset failed model validation.",
                cause=exc,
            ) from exc

    def _load_metadata(self, version_dir: Path) -> PromptMetadata:
        """Parse ``metadata.yaml`` into :class:`PromptMetadata`.

        Returns an empty :class:`PromptMetadata` when the file is absent. When
        present, the file must be valid YAML describing a mapping that conforms
        to the model; anything else raises :class:`PromptMetadataError`.
        """
        path = version_dir / self.METADATA_FILENAME
        if not path.is_file():
            return PromptMetadata()

        raw = self._read_cached(path)
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise PromptMetadataError(
                path=path,
                detail=f"{self.METADATA_FILENAME} is not valid YAML.",
                cause=exc,
            ) from exc

        if parsed is None:
            parsed = {}
        if not isinstance(parsed, Mapping):
            raise PromptMetadataError(
                path=path,
                detail=(
                    f"{self.METADATA_FILENAME} must contain a mapping, got "
                    f"{type(parsed).__name__}."
                ),
            )

        try:
            return PromptMetadata.model_validate(dict(parsed))
        except ValidationError as exc:
            raise PromptMetadataError(
                path=path,
                detail=f"{self.METADATA_FILENAME} does not match the metadata schema.",
                cause=exc,
            ) from exc

    # -- filesystem primitives ---------------------------------------------

    @staticmethod
    def _read_text_from_disk(path: Path) -> str:
        """Read a file's UTF-8 text, translating I/O errors into prompt errors.

        This is the single point of disk reads and the function wrapped by the
        per-instance cache. It is static so the cache holds no reference to the
        loader instance. It never caches failures — a raised exception is not
        stored — so a transient error will be retried on the next call.
        """
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:  # pragma: no cover - race after is_file()
            raise PromptDirectoryError(
                path=path,
                detail="File disappeared before it could be read.",
                cause=exc,
            ) from exc
        except UnicodeDecodeError as exc:
            raise PromptDirectoryError(
                path=path,
                detail="File is not valid UTF-8 text.",
                cause=exc,
            ) from exc
        except OSError as exc:
            raise PromptDirectoryError(
                path=path,
                detail="Failed to read prompt file.",
                cause=exc,
            ) from exc

    def _version_dir(self, agent: str, version: str) -> Path:
        """Resolve the directory that would hold ``agent``/``version``."""
        return self._base_path / agent / version

    def _has_any_template(self, version_dir: Path) -> bool:
        """Return whether the version directory holds at least one template."""
        return (version_dir / self.SYSTEM_FILENAME).is_file() or (
            version_dir / self.USER_FILENAME
        ).is_file()

    @staticmethod
    def _list_dirs(path: Path) -> tuple[str, ...]:
        """Sorted names of immediate subdirectories, for error hints."""
        try:
            return tuple(sorted(child.name for child in path.iterdir() if child.is_dir()))
        except OSError:  # pragma: no cover - hint best-effort only
            return ()

    @staticmethod
    def _list_files(path: Path) -> tuple[str, ...]:
        """Sorted names of immediate files, for error hints."""
        try:
            return tuple(sorted(child.name for child in path.iterdir() if child.is_file()))
        except OSError:  # pragma: no cover - hint best-effort only
            return ()

    @staticmethod
    def _validate_segment(value: str, *, kind: str) -> None:
        """Reject empty or multi-segment ``agent`` / ``version`` values.

        This guards against path traversal and malformed references: a segment
        must be a single, non-blank path component with no separators and no
        ``.`` / ``..`` navigation.
        """
        if not isinstance(value, str) or not value.strip():
            raise PromptValidationError(
                detail=f"{kind} must be a non-empty string."
            )
        if (
            "/" in value
            or "\\" in value
            or "\x00" in value
            or value in {".", ".."}
        ):
            raise PromptValidationError(
                detail=f"{kind} must be a single path segment, got {value!r}."
            )