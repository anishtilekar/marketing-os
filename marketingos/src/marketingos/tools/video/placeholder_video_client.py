"""Local, zero-cost placeholder video generator.

Stands in for :class:`~marketingos.tools.video.video_assembler.VideoAssembler`
when a full MoviePy/ffmpeg render is unnecessary weight — e.g. a
frontend-focused development loop where the actual video content doesn't
matter yet. Makes no external calls and does no rendering: it copies a
small, static stub MP4 shipped alongside this module. Satisfies the same
:class:`marketingos.agents.video_director.VideoGenerationPort` protocol as
``VideoAssembler``, so ``VideoDirectorAgent`` requires no changes to use it.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Final
from uuid import uuid4

from marketingos.agents.video_director import GeneratedVideoRef, VideoDirection

__all__ = ["PlaceholderVideoClient"]

_DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/cache/generated_videos")
_STUB_PATH: Final[Path] = Path(__file__).parent / "assets" / "placeholder.mp4"


class PlaceholderVideoClient:
    """Copies a static stub MP4 instead of rendering one.

    No network calls, no ``CostGuard`` dependency, no MoviePy/ffmpeg
    invocation — every call succeeds instantly. A drop-in substitute for
    ``VideoAssembler`` wherever a ``VideoGenerationPort`` is expected (e.g.
    ``VideoDirectorAgent(video_generator=...)``).
    """

    def __init__(self, *, output_dir: Path = _DEFAULT_OUTPUT_DIR) -> None:
        """Initialise the client.

        Args:
            output_dir: Local directory placeholder videos are written to.
                Created on first use if it doesn't already exist. Same
                default as ``VideoAssembler``, for consistency.
        """
        self._output_dir = output_dir

    async def render(self, *, direction: VideoDirection) -> GeneratedVideoRef:
        """Copy the stub video, satisfying ``VideoGenerationPort``.

        Args:
            direction: The planned script/storyboard. Only
                ``total_duration_seconds`` is used here — reported as the
                asset's duration so QA's planned-vs-actual drift check
                (which never inspects real video content) sees a perfect
                match rather than comparing against the stub file's own
                (unrelated) runtime.

        Returns:
            The copied stub's reference, with ``duration_seconds`` set to
            the direction's planned duration.
        """
        asset_id = f"vid_{uuid4().hex}"
        path = self._output_dir / f"{asset_id}.mp4"
        await asyncio.to_thread(self._copy_stub, path)

        return GeneratedVideoRef(
            asset_id=asset_id,
            uri=str(path),
            duration_seconds=direction.total_duration_seconds,
            media_type="video/mp4",
        )

    def _copy_stub(self, path: Path) -> None:
        """Copy the bundled stub MP4 to ``path``."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_STUB_PATH, path)
