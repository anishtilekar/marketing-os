"""Local image post-processing for MarketingOS.

Generative image providers rarely return pixels at exactly the caller's
requested aspect ratio, and MarketingOS's post/video creatives must land
at fixed dimensions (see the ``CreativeTemplate`` records in
``agents/designer.py``: 1080x1080, 1080x1350, ...). :class:`Compositor`
is the local, zero-cost step between "whatever the provider returned" and
"exactly what the campaign needs" — resized without distortion, cropped
to fit, and encoded in the requested format.

Scope (v1)
----------
This is deliberately narrow for now: it only normalizes dimensions and
format. Brand overlays (logo, color wash, text) described for this module
in the architecture doc are a natural next step once ``DesignerAgent``
threads a ``BrandStyle`` through to the image tool, but that's out of
scope here — adding them later means adding methods to this class, not
changing :meth:`Compositor.normalize`'s signature or behaviour.

Not a Tool
----------
Unlike the API-backed tools in this package, :class:`Compositor` performs
no external I/O and costs nothing, so it is a plain utility, not a
:class:`~marketingos.tools.base.Tool` subclass. It has no capability key
and never touches a :class:`~marketingos.services.cost_guard.CostGuard`.
``GeminiImageClient`` (``tools/image/image_gen_client.py``) owns one
instance internally and calls it as a normal post-processing step inside
its own already-guarded ``invoke``.

Dependency note
----------------
This module requires ``Pillow``, which is not yet in ``pyproject.toml``'s
dependency list — add it (``pillow>=11.0.0`` is a reasonable floor) before
this import resolves.
"""

from __future__ import annotations

import io
from enum import StrEnum
from typing import Final

from PIL import Image

from marketingos.exceptions.tool import ToolConfigurationError, ToolExecutionError

__all__ = ["Compositor", "ImageMediaType"]


class ImageMediaType(StrEnum):
    """Output encodings :class:`Compositor` can produce.

    Values are the MIME types
    ``marketingos.agents.designer.GeneratedImageRef.media_type`` already
    expects, so a caller can pass the result straight through with no
    translation.
    """

    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"


#: Maps each supported MIME type to its Pillow format string and whether
#: that format supports an alpha channel. JPEG does not: compositing onto
#: white before dropping alpha avoids the black-background artifact Pillow
#: would otherwise silently produce for a transparent source.
_PIL_FORMAT: Final[dict[ImageMediaType, tuple[str, bool]]] = {
    ImageMediaType.PNG: ("PNG", True),
    ImageMediaType.JPEG: ("JPEG", False),
    ImageMediaType.WEBP: ("WEBP", True),
}

_MIN_QUALITY: Final[int] = 1
_MAX_QUALITY: Final[int] = 100
_DEFAULT_QUALITY: Final[int] = 92


class Compositor:
    """Resizes, crops and re-encodes generated images to an exact spec.

    Stateless and safe to share across concurrent calls — every method
    operates purely on its arguments, no instance state is mutated.
    """

    def normalize(
        self,
        image_bytes: bytes,
        *,
        width: int,
        height: int,
        media_type: ImageMediaType = ImageMediaType.PNG,
        quality: int = _DEFAULT_QUALITY,
    ) -> bytes:
        """Resize, center-crop and re-encode ``image_bytes`` to spec.

        Uses a "cover" resize: the source is scaled up just enough that
        both dimensions meet the target, then center-cropped to it
        exactly. This fills the frame with no letterboxing and no
        distortion, at the cost of trimming the source's edges — the
        standard behaviour for fixed-aspect social creative slots.

        Args:
            image_bytes: Raw encoded image bytes from the generation
                provider, in any format Pillow can decode.
            width: Exact target width in pixels.
            height: Exact target height in pixels.
            media_type: Output encoding. See :class:`ImageMediaType`.
            quality: JPEG/WEBP encoder quality, 1-100. Ignored for PNG,
                which is always lossless.

        Returns:
            The re-encoded image bytes, exactly ``width`` x ``height``.

        Raises:
            ToolConfigurationError: If ``width``, ``height`` or ``quality``
                is out of range.
            ToolExecutionError: If ``image_bytes`` cannot be decoded as an
                image.
        """
        if width <= 0 or height <= 0:
            raise ToolConfigurationError(
                f"width and height must be positive; got {width}x{height}."
            )
        if not _MIN_QUALITY <= quality <= _MAX_QUALITY:
            raise ToolConfigurationError(
                f"quality must be {_MIN_QUALITY}-{_MAX_QUALITY}; got {quality}."
            )

        source = self._decode(image_bytes)
        fitted = self._cover_resize(source, width=width, height=height)
        return self._encode(fitted, media_type=media_type, quality=quality)

    # -- decode / encode -----------------------------------------------------

    @staticmethod
    def _decode(image_bytes: bytes) -> Image.Image:
        """Decode raw bytes into a Pillow image.

        Raises:
            ToolExecutionError: If the bytes are not a decodable image.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception as exc:  # noqa: BLE001 - Pillow raises several types
            raise ToolExecutionError(
                f"Could not decode generated image bytes: {exc}"
            ) from exc
        return image

    @staticmethod
    def _encode(
        image: Image.Image, *, media_type: ImageMediaType, quality: int
    ) -> bytes:
        """Encode ``image`` as ``media_type``, flattening alpha for JPEG."""
        pil_format, supports_alpha = _PIL_FORMAT[media_type]
        if not supports_alpha and image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            rgba = image.convert("RGBA")
            background.paste(rgba, mask=rgba.split()[-1])
            image = background
        elif image.mode == "P":
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")

        buffer = io.BytesIO()
        save_kwargs: dict[str, object] = {}
        if pil_format in ("JPEG", "WEBP"):
            save_kwargs["quality"] = quality
        image.save(buffer, format=pil_format, **save_kwargs)
        return buffer.getvalue()

    # -- resize ----------------------------------------------------------------

    @staticmethod
    def _cover_resize(image: Image.Image, *, width: int, height: int) -> Image.Image:
        """Scale ``image`` to cover ``width`` x ``height``, then center-crop."""
        source_ratio = image.width / image.height
        target_ratio = width / height

        if source_ratio > target_ratio:
            # Source is relatively wider than the target: match target
            # height, let width overhang for a horizontal center-crop.
            scaled_height = height
            scaled_width = max(width, round(height * source_ratio))
        else:
            scaled_width = width
            scaled_height = max(height, round(width / source_ratio))

        resized = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        left = (scaled_width - width) // 2
        top = (scaled_height - height) // 2
        return resized.crop((left, top, left + width, top + height))