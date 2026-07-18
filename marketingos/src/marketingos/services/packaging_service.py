"""Filesystem and compression backend for campaign packaging.

Implements the :class:`~marketingos.agents.packaging.PackagingServicePort`
Protocol that ``PackagingAgent`` depends on: copying/writing files into the
run structure, computing checksums, and producing the final archive. The
agent decides *what* goes where (paths, ordering, manifest composition);
this module does the actual I/O, so it has no knowledge of campaigns,
manifests, or QA — only files and bytes.

All blocking filesystem work runs via ``asyncio.to_thread`` so the async
Protocol methods never block the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import shutil
import zipfile
from pathlib import Path

from loguru import logger

from marketingos.agents.packaging import Checksum, PackageArchiveRef, StagedFile
from marketingos.exceptions.tool import ToolExecutionError

__all__ = ["PackagingService"]

_DEFAULT_MEDIA_TYPE = "application/octet-stream"
_CHUNK_SIZE = 1024 * 1024


def _sha256_file(path: Path) -> str:
    """Compute the sha256 hex digest of a file, streaming in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guess_media_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or _DEFAULT_MEDIA_TYPE


class PackagingService:
    """Local-filesystem implementation of ``PackagingServicePort``.

    All paths the agent passes (``target_path``, ``root_path``) are treated
    as relative to ``base_dir`` — matching the agent's convention of
    already prefixing paths with ``{root_prefix}/{run_id}``.
    """

    def __init__(self, *, base_dir: Path = Path(".")) -> None:
        """Initialise the service.

        Args:
            base_dir: Filesystem root that ``target_path``/``root_path``
                strings are resolved against.
        """
        self._base_dir = base_dir
        self._logger = logger.bind(component="PackagingService")

    # -- PackagingServicePort ---------------------------------------------

    async def stage_asset(self, *, source_uri: str, target_path: str) -> StagedFile:
        """Copy an existing asset (image/video) into the run structure.

        Args:
            source_uri: Location of the source file. A ``file://`` prefix
                is stripped if present; otherwise treated as a local path.
            target_path: Destination path, relative to ``base_dir``.

        Returns:
            The staged file's record, with a checksum computed post-copy.

        Raises:
            ToolExecutionError: If the source file does not exist or the
                copy fails.
        """
        source = Path(source_uri.removeprefix("file://"))
        destination = self._base_dir / target_path

        def _copy() -> None:
            if not source.is_file():
                raise ToolExecutionError(f"Source asset not found: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        try:
            await asyncio.to_thread(_copy)
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to stage asset {source_uri!r} to {target_path!r}: {exc}"
            ) from exc

        return await self._index(destination, target_path)

    async def stage_text(
        self, *, content: str, target_path: str, media_type: str
    ) -> StagedFile:
        """Write generated text (manifest, metadata, README) into the run.

        Args:
            content: Text content to write, UTF-8 encoded.
            target_path: Destination path, relative to ``base_dir``.
            media_type: Media type recorded on the resulting ``StagedFile``.

        Returns:
            The staged file's record.

        Raises:
            ToolExecutionError: If the write fails.
        """
        destination = self._base_dir / target_path

        def _write() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to stage text to {target_path!r}: {exc}"
            ) from exc

        return await self._index(destination, target_path, media_type=media_type)

    async def finalize(self, *, root_path: str) -> PackageArchiveRef:
        """Compress the completed run structure into a single archive.

        The archive is written to ``{root_path}/package/campaign.zip`` and
        excludes itself from its own contents. Entry names inside the zip
        are paths relative to ``root_path``, so extracting it reproduces
        the run's directory layout.

        Args:
            root_path: Run root to compress, relative to ``base_dir``.

        Returns:
            A reference to the archive, with its checksum and size.

        Raises:
            ToolExecutionError: If the run directory is missing or empty,
                or compression fails.
        """
        root = self._base_dir / root_path
        archive_target = f"{root_path}/package/campaign.zip"
        archive_path = self._base_dir / archive_target

        def _zip() -> None:
            if not root.is_dir():
                raise ToolExecutionError(f"Run root not found: {root}")
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            files = [p for p in sorted(root.rglob("*")) if p.is_file() and p != archive_path]
            if not files:
                raise ToolExecutionError(f"No files to archive under {root}")
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in files:
                    zf.write(file_path, arcname=str(file_path.relative_to(root)))

        try:
            await asyncio.to_thread(_zip)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ToolExecutionError(
                f"Failed to finalise archive for {root_path!r}: {exc}"
            ) from exc

        staged = await self._index(archive_path, archive_target, media_type="application/zip")
        self._logger.bind(
            event="packaging_service.finalized",
            root_path=root_path,
            archive=archive_target,
            size_bytes=staged.size_bytes,
        ).info("Archive finalised")
        return PackageArchiveRef(
            uri=str(archive_path),
            size_bytes=staged.size_bytes,
            checksum=staged.checksum,
        )

    # -- internals ----------------------------------------------------------

    async def _index(
        self, path: Path, target_path: str, *, media_type: str | None = None
    ) -> StagedFile:
        """Build a ``StagedFile`` record for an already-written file."""
        def _stat_and_hash() -> tuple[int, str]:
            return path.stat().st_size, _sha256_file(path)

        size_bytes, digest = await asyncio.to_thread(_stat_and_hash)
        return StagedFile(
            path=target_path,
            size_bytes=size_bytes,
            checksum=Checksum(value=digest),
            media_type=media_type or _guess_media_type(target_path),
        )