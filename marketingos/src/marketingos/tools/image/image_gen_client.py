"""Gemini image-generation tool.

Wraps Google's Generative Language REST API's image-output path (Gemini
2.5 Flash Image, marketed as "Nano Banana") as a
:class:`~marketingos.tools.base.Tool` registered under the
``image_generation`` capability, satisfying
:class:`marketingos.agents.designer.ImageGenerationPort`.

Same transport shape as
:class:`~marketingos.tools.llm.gemini_client.GeminiClient` — one
``generateContent`` call, ``httpx`` transport, injectable client for
tests — but the response carries inline image bytes instead of text, and
pricing is flat-per-image (Google's own billing model for this endpoint:
500 free images/day at 1024x1024) rather than token-metered.

Post-processing
----------------
Gemini rarely returns pixels at exactly the requested aspect ratio. Every
generated image is piped through
:class:`~marketingos.tools.image.compositor.Compositor` before this tool
returns, so the caller's ``width``/``height`` are always exact — the
agent-facing contract never leaks provider quirks.

No native negative-prompt parameter
-------------------------------------
Unlike Stable-Diffusion-style APIs, Gemini's image endpoint has no
discrete negative-prompt field. ``negative_prompt`` is folded into the
sent prompt text as an explicit "avoid" instruction instead — a
best-effort steering hint, not a hard constraint the way it would be on
a diffusion API.

Storage (dev-local)
--------------------
The normalized bytes are written to ``output_dir`` (default
``data/cache/generated_images/``) under a generated asset id, and the
returned ``GeneratedImageRef.uri`` is that file path — matching the
architecture doc's "local disk (dev)" decision. The packaging service
(``PackagingServicePort.stage_asset``) later copies from this uri into
the final numbered run structure; this tool never writes there directly.

Cross-layer import note
-------------------------
``GeneratedImageRef`` is defined in ``marketingos.agents.designer``, not
in ``models/``, because that's where the ``ImageGenerationPort`` this
tool must satisfy already lives. Importing it here technically crosses
the "tools must not depend on agents" line in the architecture doc's
dependency table; it's a one-way, acyclic import (``designer.py`` imports
nothing from ``tools``) and there's no other way to return an object
DesignerAgent will accept. The clean long-term fix is relocating these
port DTOs into ``models/``; noted as follow-up, not blocking here.
"""

from __future__ import annotations

import asyncio
import base64
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from marketingos.agents.designer import GeneratedImageRef
from marketingos.exceptions.tool import ToolConfigurationError, ToolExecutionError
from marketingos.models.cost import CostCategory
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.base import Tool
from marketingos.tools.image.compositor import Compositor, ImageMediaType

__all__ = [
    "DEFAULT_MODEL",
    "GEMINI_API_KEY_ENV",
    "GeminiImageClient",
    "IMAGE_GENERATION",
    "ImageGenerationRequest",
]

#: Capability key under which this tool registers.
IMAGE_GENERATION: Final[str] = "image_generation"

#: "Nano Banana" — Gemini 2.5 Flash Image.
DEFAULT_MODEL: Final[str] = "gemini-2.5-flash-image"

#: Same environment variable as the text client — one Google API key
#: covers both endpoints.
GEMINI_API_KEY_ENV: Final[str] = "GEMINI_API_KEY"

_API_BASE: Final[str] = "https://generativelanguage.googleapis.com/v1beta/models"

_DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/cache/generated_images")

_EXTENSION_BY_MEDIA_TYPE: Final[dict[ImageMediaType, str]] = {
    ImageMediaType.PNG: ".png",
    ImageMediaType.JPEG: ".jpg",
    ImageMediaType.WEBP: ".webp",
}


class ImageGenerationRequest(BaseModel):
    """One image-generation request."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    negative_prompt: str = Field(default="", max_length=2000)
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    media_type: ImageMediaType = Field(default=ImageMediaType.PNG)


class GeminiImageClient(Tool[ImageGenerationRequest, GeneratedImageRef]):
    """Image generation via Gemini 2.5 Flash Image, priced per image.

    Free tier is expressed as a *rate of zero*, not a hardcoded zero
    cost, mirroring ``GeminiClient``: the moment a paid rate is
    configured, this tool starts charging correctly with no code change.
    """

    def __init__(
        self,
        *,
        cost_guard: CostGuard,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cost_per_image: Decimal = Decimal("0"),
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
        compositor: Compositor | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Initialise the client.

        Args:
            cost_guard: Guard enforcing the run's budget. Required: see
                ``Tool.cost_guard``.
            api_key: API key. Defaults to the ``GEMINI_API_KEY``
                environment variable — the same key used by
                ``GeminiClient``.
            model: Model id to call.
            cost_per_image: Flat price per generated image. Zero on the
                free tier (500 images/day at 1024x1024).
            output_dir: Local directory generated images are written to.
                Created on first use if it doesn't already exist.
            compositor: Post-processor normalizing output to the exact
                requested dimensions/format. Defaults to a fresh
                ``Compositor`` (stateless, so sharing one instance across
                tools is also fine).
            http_client: Transport to use. Defaults to a client owned by
                this instance; tests inject one with a mock transport.
            timeout_seconds: Per-request timeout. Generous default —
                image generation is slower than text completion.

        Raises:
            ToolConfigurationError: If no API key is available.
        """
        resolved_key = (
            api_key if api_key is not None else os.environ.get(GEMINI_API_KEY_ENV)
        )
        if not resolved_key:
            raise ToolConfigurationError(
                f"No Gemini API key: pass api_key= or set {GEMINI_API_KEY_ENV} "
                "(see .env.example)."
            )
        self._api_key = resolved_key
        self._model = model
        self._cost_per_image = cost_per_image
        self._output_dir = output_dir
        self._compositor = compositor or Compositor()
        self._cost_guard = cost_guard
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._logger = logger.bind(component="GeminiImageClient", model=model)

    # -- Tool identity -------------------------------------------------------

    @property
    def name(self) -> str:
        """The model id, recorded as ``CostEntry.tool_name``."""
        return self._model

    @property
    def capability(self) -> str:
        """This tool provides ``image_generation``."""
        return IMAGE_GENERATION

    @property
    def provider(self) -> str:
        """Recorded as ``CostEntry.provider``."""
        return "google"

    @property
    def cost_category(self) -> CostCategory:
        """Spend here is image generation."""
        return CostCategory.IMAGE_GENERATION

    @property
    def input_schema(self) -> type[ImageGenerationRequest]:
        """The request model."""
        return ImageGenerationRequest

    @property
    def output_schema(self) -> type[GeneratedImageRef]:
        """The response model."""
        return GeneratedImageRef

    @property
    def cost_guard(self) -> CostGuard:
        """The guard consulted by :meth:`invoke` on every call."""
        return self._cost_guard

    # -- cost ----------------------------------------------------------------

    def cost_estimate(self, payload: ImageGenerationRequest) -> Decimal:
        """Flat per-image price.

        Unlike text generation, Gemini bills this endpoint per image, not
        per token, so pricing doesn't depend on the request's contents —
        the estimate is exact, not an approximation.
        """
        return self._cost_per_image

    # cost_actual: inherited default (reuses cost_estimate) is correct
    # here — a flat-rate endpoint has no post-hoc usage to reconcile
    # against, unlike GeminiClient's token-metered text calls.

    # -- invocation ----------------------------------------------------------

    async def invoke(self, payload: ImageGenerationRequest) -> GeneratedImageRef:
        """Generate one image, normalize it to spec, and persist it locally.

        Budget enforcement is applied automatically by
        :meth:`marketingos.tools.base.Tool.__init_subclass__`, which wraps
        this method with the cost guard, so the request is priced and
        authorised before it is sent and recorded after it succeeds.

        Args:
            payload: The request to send.

        Returns:
            The normalized asset's reference, at exactly the requested
            dimensions and format.

        Raises:
            InsufficientBudgetError: If the call would exceed the budget.
                Raised by the decorator, before any request is sent.
            ToolExecutionError: If the request fails, the response
                carries no image data, or the bytes can't be
                decoded/re-encoded.
        """
        body: dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": self._compose_prompt(payload)}]}
            ],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }
        url = f"{_API_BASE}/{self._model}:generateContent"
        try:
            response = await self._client.post(
                url, json=body, headers={"x-goog-api-key": self._api_key}
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"Gemini returned {exc.response.status_code} for model "
                f"{self._model!r}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                f"Gemini image request failed for model {self._model!r}: {exc}"
            ) from exc

        raw_bytes = self._extract_image_bytes(data)
        normalized = await asyncio.to_thread(
            self._compositor.normalize,
            raw_bytes,
            width=payload.width,
            height=payload.height,
            media_type=payload.media_type,
        )
        asset_id = f"img_{uuid4().hex}"
        path = await self._persist(
            asset_id, normalized, media_type=payload.media_type
        )

        self._logger.bind(
            event="gemini_image.generated",
            asset_id=asset_id,
            width=payload.width,
            height=payload.height,
            output_bytes=len(normalized),
        ).debug("Generated image")

        return GeneratedImageRef(
            asset_id=asset_id,
            uri=str(path),
            width=payload.width,
            height=payload.height,
            media_type=payload.media_type.value,
        )

    @staticmethod
    def _compose_prompt(payload: ImageGenerationRequest) -> str:
        """Fold the negative prompt into the sent text; see module docstring."""
        if not payload.negative_prompt.strip():
            return payload.prompt
        return f"{payload.prompt}\n\nAvoid: {payload.negative_prompt}."

    @staticmethod
    def _extract_image_bytes(data: dict[str, Any]) -> bytes:
        """Extract and base64-decode the first inline image in the response.

        Raises:
            ToolExecutionError: If no candidate or image part is present,
                or the inline data isn't valid base64.
        """
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ToolExecutionError(
                f"Gemini response contained no candidates: {str(data)[:500]}"
            ) from exc

        for part in parts:
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                try:
                    return base64.b64decode(inline["data"])
                except (ValueError, TypeError) as exc:
                    raise ToolExecutionError(
                        f"Gemini returned undecodable inline image data: {exc}"
                    ) from exc

        raise ToolExecutionError(
            f"Gemini response contained no image data: {str(data)[:500]}"
        )

    async def _persist(
        self, asset_id: str, image_bytes: bytes, *, media_type: ImageMediaType
    ) -> Path:
        """Write ``image_bytes`` to ``output_dir`` and return the file path."""
        extension = _EXTENSION_BY_MEDIA_TYPE[media_type]
        path = self._output_dir / f"{asset_id}{extension}"

        def _write() -> None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_bytes)

        await asyncio.to_thread(_write)
        return path

    # -- ImageGenerationPort adapter -------------------------------------------

    async def generate(
        self, *, prompt: str, negative_prompt: str, width: int, height: int
    ) -> GeneratedImageRef:
        """Render one image, satisfying ``ImageGenerationPort``.

        This is the shape the agent already depends on
        (``marketingos.agents.designer.ImageGenerationPort``), so a
        ``GeminiImageClient`` can be injected as
        ``DesignerAgent(image_generator=...)`` with no change to any
        agent. Delegates to :meth:`invoke`, so the agent path is
        budget-enforced exactly like the tool path.

        Args:
            prompt: The generation prompt.
            negative_prompt: Content to steer away from (folded into the
                prompt text; see module docstring).
            width: Exact target width in pixels.
            height: Exact target height in pixels.

        Returns:
            The normalized asset's reference.

        Raises:
            InsufficientBudgetError: If the call would exceed the budget.
            ToolExecutionError: If the request fails.
        """
        return await self.invoke(
            ImageGenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
            )
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()