"""Local, zero-cost placeholder image generator.

Stands in for :class:`~marketingos.tools.image.image_gen_client.GeminiImageClient`
during demo/pilot runs where real Gemini image quota/billing is unavailable.
Makes no external API calls: it renders a deterministic placeholder locally
with Pillow — a gradient background derived from a hash of the prompt, with
the prompt text drawn on top — instead of generating real pixels. Satisfies
the same :class:`marketingos.agents.designer.ImageGenerationPort` protocol as
``GeminiImageClient``, so ``DesignerAgent`` requires no changes to use it.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import textwrap
from pathlib import Path
from typing import Final
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from marketingos.agents.designer import GeneratedImageRef

__all__ = ["PlaceholderImageClient"]

_DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/cache/generated_images")

#: Small, deterministic palette of (top, bottom) gradient endpoint colors.
#: The prompt's hash picks an entry, so a given prompt always renders the
#: same colors within a run, while different prompts look visually distinct.
_PALETTE: Final[tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]] = (
    ((0x1E, 0x3A, 0x8A), (0x60, 0xA5, 0xFA)),  # blue
    ((0x7C, 0x2D, 0x12), (0xFB, 0x92, 0x3C)),  # orange
    ((0x14, 0x53, 0x2D), (0x4A, 0xDE, 0x80)),  # green
    ((0x58, 0x1C, 0x87), (0xC0, 0x84, 0xFC)),  # purple
    ((0x83, 0x18, 0x1B), (0xF8, 0x71, 0x71)),  # red
    ((0x11, 0x4B, 0x5A), (0x22, 0xD3, 0xEE)),  # teal
)

#: Only the first ~80 characters of the prompt are rendered as overlay text.
_MAX_PROMPT_CHARS: Final[int] = 80

#: Character-count wrap width for the overlay text, tuned for the default
#: bitmap font at typical creative dimensions (1080px+).
_WRAP_CHARS: Final[int] = 24


class PlaceholderImageClient:
    """Renders a local placeholder image instead of calling Gemini.

    No network calls, no billable usage, no ``CostGuard`` dependency —
    every call succeeds instantly and deterministically. A drop-in
    substitute for ``GeminiImageClient`` wherever an ``ImageGenerationPort``
    is expected (e.g. ``DesignerAgent(image_generator=...)``).
    """

    def __init__(self, *, output_dir: Path = _DEFAULT_OUTPUT_DIR) -> None:
        """Initialise the client.

        Args:
            output_dir: Local directory generated images are written to.
                Created on first use if it doesn't already exist. Same
                default as ``GeminiImageClient``, for consistency.
        """
        self._output_dir = output_dir

    async def generate(
        self, *, prompt: str, negative_prompt: str, width: int, height: int
    ) -> GeneratedImageRef:
        """Render one placeholder image, satisfying ``ImageGenerationPort``.

        Args:
            prompt: The generation prompt. Used both to derive the
                background gradient (via a hash) and as the text rendered
                on the image (first ~80 characters, wrapped and centered).
            negative_prompt: Accepted for interface compatibility with
                ``ImageGenerationPort``; ignored, since a local placeholder
                render has nothing to steer away from.
            width: Exact output width in pixels.
            height: Exact output height in pixels.

        Returns:
            The rendered asset's reference, at exactly the requested
            dimensions, matching ``GeminiImageClient``'s exact-dimensions
            guarantee.
        """
        asset_id = f"img_{uuid4().hex}"
        path = self._output_dir / f"{asset_id}.png"
        await asyncio.to_thread(self._render_and_save, prompt, path, width, height)

        return GeneratedImageRef(
            asset_id=asset_id,
            uri=str(path),
            width=width,
            height=height,
            media_type="image/png",
        )

    def _render_and_save(self, prompt: str, path: Path, width: int, height: int) -> None:
        """Render the placeholder and write it to ``path`` as PNG."""
        image = self._render(prompt, width=width, height=height)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        path.write_bytes(buffer.getvalue())

    def _render(self, prompt: str, *, width: int, height: int) -> Image.Image:
        """Draw the gradient background and centered wrapped prompt text."""
        top, bottom = self._pick_colors(prompt)
        image = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(image)
        for y in range(height):
            t = y / max(height - 1, 1)
            color = tuple(
                round(top[channel] + (bottom[channel] - top[channel]) * t)
                for channel in range(3)
            )
            draw.line([(0, y), (width, y)], fill=color)

        text = textwrap.fill(prompt[:_MAX_PROMPT_CHARS], width=_WRAP_CHARS)
        font = ImageFont.load_default()
        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = (
            (width - text_width) / 2 - bbox[0],
            (height - text_height) / 2 - bbox[1],
        )
        draw.multiline_text(
            position,
            text,
            font=font,
            fill=(255, 255, 255),
            align="center",
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        return image

    @staticmethod
    def _pick_colors(
        prompt: str,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Derive a stable palette entry from a hash of ``prompt``."""
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        index = digest[0] % len(_PALETTE)
        return _PALETTE[index]
