"""Programmatic video assembly for MarketingOS.

Renders a :class:`~marketingos.agents.video_director.VideoDirection` into
an MP4 by animating the campaign's own approved still images (Ken-Burns
style pan/zoom) with burned-in, precisely-timed subtitles — no
third-party rendering API and no generative text-to-video. This is the
architecture doc's explicit, structural cost/reliability decision, not a
stopgap: generative video is slow, expensive per second, and
non-deterministic, whereas MoviePy/OpenCV assembly over pre-approved
stills is fast, free, and reproducible.

No narration audio
-------------------
No text-to-speech tool exists in this tool inventory yet, so
``direction.voice_over_text`` is not synthesized into audio here — the
rendered video is silent with burned-in subtitles, which is a normal,
common pattern for short-form social video (Reels/TikTok/Shorts are
routinely watched muted). Wiring a TTS tool in later is a natural
extension of this render step, not a redesign of it.

Image-to-shot assignment
--------------------------
``VideoDirection.asset_references`` is an unordered pool of approved
asset ids the direction says it draws on — the schema has no per-shot
image assignment. This assembler cycles through the resolved images
round-robin across the flattened shot list, one image per shot for that
shot's ``duration_seconds``, applying a subtle continuous zoom so a
single still doesn't look like a static frame. A shot beyond the last
resolved image wraps back to the first. A direction with no
resolvable images at all falls back to a solid background colour rather
than failing the render outright.

Not API-metered
-----------------
Still a :class:`~marketingos.tools.base.Tool` — registered, logged, and
passed through the same cost-guard/ledger seam as every other tool — but
``cost_estimate`` always returns ``Decimal("0")``: there is no third-party
bill for local compute. Recording a zero-cost entry (rather than skipping
the guard) is what keeps the ledger a complete account of every tool
call, paid or not.
"""

from __future__ import annotations

import asyncio
import io
from decimal import Decimal
from pathlib import Path
from typing import Final, cast
from uuid import uuid4

import numpy as np
from loguru import logger
from moviepy import (
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    concatenate_videoclips,
)
from PIL import Image

from marketingos.agents.video_director import (
    GeneratedVideoRef,
    Shot,
    SubtitleLine,
    VideoDirection,
)
from marketingos.exceptions.tool import ToolExecutionError
from marketingos.models.cost import CostCategory
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.base import Tool
from marketingos.tools.image.compositor import Compositor, ImageMediaType

__all__ = ["VIDEO_GENERATION", "VideoAssembler"]

#: Capability key under which this tool registers.
VIDEO_GENERATION: Final[str] = "video_generation"

_DEFAULT_ASSET_DIR: Final[Path] = Path("data/cache/generated_images")
_DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/cache/generated_videos")

#: Portrait, matching short-form platforms (Reels/TikTok/Shorts).
_DEFAULT_RESOLUTION: Final[tuple[int, int]] = (1080, 1920)

#: Neutral dark background used when a shot has no resolvable image.
_DEFAULT_BACKGROUND: Final[tuple[int, int, int]] = (17, 24, 39)

#: Total zoom applied linearly across a shot's duration (1.0 -> 1 + this).
_KEN_BURNS_ZOOM: Final[float] = 0.08

_SUBTITLE_BOTTOM_MARGIN: Final[int] = 96


class VideoAssembler(Tool[VideoDirection, GeneratedVideoRef]):
    """Assembles a :class:`VideoDirection` into a silent, subtitled MP4."""

    def __init__(
        self,
        *,
        cost_guard: CostGuard,
        asset_dir: Path = _DEFAULT_ASSET_DIR,
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
        resolution: tuple[int, int] = _DEFAULT_RESOLUTION,
        fps: int = 30,
        background_color: tuple[int, int, int] = _DEFAULT_BACKGROUND,
        subtitle_font_size: int = 56,
        compositor: Compositor | None = None,
    ) -> None:
        """Initialise the assembler.

        Args:
            cost_guard: Guard enforcing the run's budget. Required: see
                ``Tool.cost_guard``. Every render still records a
                ``Decimal("0")`` ledger entry.
            asset_dir: Directory searched for each ``asset_references``
                id. Defaults to the same directory
                ``GeminiImageClient`` writes to, so no wiring is needed
                to hand rendered images to this tool. Resolution is by
                filename stem: ``{asset_dir}/{asset_id}.*``.
            output_dir: Local directory rendered videos are written to.
                Created on first use if it doesn't already exist.
            resolution: Output ``(width, height)`` in pixels.
            fps: Output frame rate.
            background_color: Fallback background for shots with no
                resolvable image.
            subtitle_font_size: Burned-in subtitle text size, in pixels.
            compositor: Reused to fit each source image to ``resolution``
                exactly before animating it — the same cover-crop used
                for still creatives, so a portrait short and a square
                feed post can share one source image. Defaults to a
                fresh ``Compositor``.
        """
        self._cost_guard = cost_guard
        self._asset_dir = asset_dir
        self._output_dir = output_dir
        self._resolution = resolution
        self._fps = fps
        self._background_color = background_color
        self._subtitle_font_size = subtitle_font_size
        self._compositor = compositor or Compositor()
        self._logger = logger.bind(component="VideoAssembler")

    # -- Tool identity -------------------------------------------------------

    @property
    def name(self) -> str:
        return "moviepy-assembler"

    @property
    def capability(self) -> str:
        return VIDEO_GENERATION

    @property
    def provider(self) -> str:
        """Local compute, not a third-party vendor."""
        return "local"

    @property
    def cost_category(self) -> CostCategory:
        return CostCategory.VIDEO_GENERATION

    @property
    def input_schema(self) -> type[VideoDirection]:
        return VideoDirection

    @property
    def output_schema(self) -> type[GeneratedVideoRef]:
        return GeneratedVideoRef

    @property
    def cost_guard(self) -> CostGuard:
        return self._cost_guard

    # -- cost ------------------------------------------------------------

    def cost_estimate(self, payload: VideoDirection) -> Decimal:
        """Always zero — local MoviePy/OpenCV rendering, no vendor bill."""
        return Decimal("0")

    # cost_actual: inherited default (reuses cost_estimate) — correct,
    # there is no post-hoc usage to reconcile for a free local render.

    # -- invocation ----------------------------------------------------------

    async def invoke(self, payload: VideoDirection) -> GeneratedVideoRef:
        """Render ``payload`` to an MP4 and return its asset reference.

        Budget enforcement is applied automatically by
        ``Tool.__init_subclass__`` (a no-op here, since the cost is
        always zero, but the call is still priced and recorded).

        Raises:
            ToolExecutionError: If rendering fails — a malformed shot
                list, an unwritable output path, or an ffmpeg failure.
        """
        try:
            path = await asyncio.to_thread(self._render, payload)
        except ToolExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - MoviePy/ffmpeg raise several types
            raise ToolExecutionError(
                f"Rendering video for item {payload.item_id!r} failed: {exc}"
            ) from exc

        asset_id = f"vid_{uuid4().hex}"
        final_path = self._output_dir / f"{asset_id}.mp4"
        await asyncio.to_thread(path.replace, final_path)

        self._logger.bind(
            event="video_assembler.rendered",
            asset_id=asset_id,
            item_id=payload.item_id,
            duration_seconds=payload.total_duration_seconds,
            shots=len(payload.shot_list),
            subtitles=len(payload.subtitles),
        ).debug("Rendered video")

        return GeneratedVideoRef(
            asset_id=asset_id,
            uri=str(final_path),
            duration_seconds=payload.total_duration_seconds,
            media_type="video/mp4",
        )

    # -- VideoGenerationPort adapter --------------------------------------------

    async def render(self, *, direction: VideoDirection) -> GeneratedVideoRef:
        """Render ``direction``, satisfying ``VideoGenerationPort``.

        This is the shape ``VideoDirectorAgent`` already depends on
        (``marketingos.agents.video_director.VideoGenerationPort``), so a
        ``VideoAssembler`` can be injected as
        ``VideoDirectorAgent(video_generator=...)`` with no change to any
        agent. Delegates to :meth:`invoke`, so the agent path is
        budget-enforced (recorded, even at zero cost) exactly like the
        tool path.
        """
        return await self.invoke(direction)

    # -- rendering (synchronous; run off the event loop via to_thread) ---------

    def _render(self, direction: VideoDirection) -> Path:
        """Build and export the composite video. Returns a temp output path."""
        images = self._resolve_images(direction.asset_references)
        shot_clips = [
            self._shot_clip(shot, images[index % len(images)] if images else None)
            for index, shot in enumerate(direction.shot_list)
        ]
        visual_track = concatenate_videoclips(shot_clips, method="compose")

        subtitle_clips = [
            self._subtitle_clip(line) for line in direction.subtitles
        ]
        composite = CompositeVideoClip(
            [visual_track, *subtitle_clips], size=self._resolution
        ).with_duration(direction.total_duration_seconds)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self._output_dir / f".tmp-{uuid4().hex}.mp4"
        composite.write_videofile(
            str(temp_path),
            fps=self._fps,
            codec="libx264",
            audio=False,
            logger=None,
        )
        return temp_path

    def _resolve_images(self, asset_references: tuple[str, ...]) -> list[np.ndarray]:
        """Load and cover-fit every resolvable referenced asset.

        Ids that don't resolve to a file in ``asset_dir`` are silently
        skipped rather than failing the render — an unresolvable
        reference is treated the same as no reference at all, so a
        rendering-time asset-store hiccup doesn't take down the whole
        video.
        """
        width, height = self._resolution
        frames: list[np.ndarray] = []
        for asset_id in asset_references:
            match = next(iter(sorted(self._asset_dir.glob(f"{asset_id}.*"))), None)
            if match is None:
                continue
            normalized = self._compositor.normalize(
                match.read_bytes(),
                width=width,
                height=height,
                media_type=ImageMediaType.PNG,
            )
            with Image.open(io.BytesIO(normalized)) as image:
                frames.append(np.array(image.convert("RGB")))
        return frames

    def _shot_clip(self, shot: Shot, frame: np.ndarray | None) -> CompositeVideoClip:
        """Build one shot's clip: a still frame with a subtle continuous zoom.

        ``resized`` grows the frame anchored at its top-left corner, so a
        plain ``cropped(x_center=..., y_center=...)`` on the *un-composited*
        clip would use a window fixed at render time and drift toward that
        corner as the zoom progresses instead of staying centered.
        Compositing onto a fixed-size canvas with ``with_position("center")``
        re-centers the growing frame on every frame instead, at its actual
        per-frame size, and lets the canvas edges clip the overflow.
        """
        if frame is None:
            base = ColorClip(
                size=self._resolution, color=self._background_color
            ).with_duration(shot.duration_seconds)
        else:
            base = ImageClip(img=frame, duration=shot.duration_seconds)

        duration = shot.duration_seconds
        zoomed = cast(
            VideoClip, base.resized(lambda t: 1 + _KEN_BURNS_ZOOM * (t / duration))
        )
        return CompositeVideoClip(
            [zoomed.with_position("center")], size=self._resolution
        )

    def _subtitle_clip(self, line: SubtitleLine) -> TextClip:
        """Build one timed, bottom-anchored subtitle clip."""
        width, _height = self._resolution
        text_clip = TextClip(
            text=line.text,
            font_size=self._subtitle_font_size,
            color="white",
            stroke_color="black",
            stroke_width=2,
            size=(round(width * 0.9), None),
            method="caption",
            text_align="center",
        )
        return (
            text_clip.with_position(("center", "bottom"))
            .with_start(line.start_seconds)
            .with_duration(line.end_seconds - line.start_seconds)
        )