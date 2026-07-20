"""FLUX Schnell image-generation tool (Together AI).

Wraps Together AI's image-generations REST endpoint serving Black Forest
Labs' FLUX.1 [schnell] (the free ``black-forest-labs/FLUX.1-schnell-Free``
model) as a :class:`~marketingos.tools.base.Tool` registered under the
``image_generation`` capability, satisfying
:class:`marketingos.agents.designer.ImageGenerationPort`.

Deliberately the same shape as
:class:`~marketingos.tools.image.image_gen_client.GeminiImageClient` — one
POST, ``httpx`` transport, injectable client for tests, output piped
through :class:`~marketingos.tools.image.compositor.Compositor` for exact
dimensions — so it drops into ``DesignerAgent(image_generator=...)`` with no
agent change and switching between it and Gemini is a config edit (see
:mod:`marketingos.tools.factory`), never a code change.

Differences from the Gemini image path:

* **Auth.** Together uses a bearer token from ``TOGETHER_API_KEY``, a
  different key from Gemini's ``GEMINI_API_KEY``.
* **No negative-prompt field.** Like Gemini's image endpoint, FLUX has no
  discrete negative prompt, so ``negative_prompt`` is folded into the sent
  prompt text as an explicit "avoid" instruction.
* **Bounded steps.** The free FLUX schnell endpoint caps sampling steps at
  four; :data:`_MAX_STEPS` clamps the configured value.
* **Generation size.** Together requires explicit ``width``/``height``.
  They are rounded to the provider's 16-pixel grid and clamped to a
  supported range for the request; the exact caller-requested dimensions
  are still guaranteed afterwards by the ``Compositor`` pass, exactly as in
  the Gemini path.
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

from marketingos.agents.designer import GeneratedImageRef
from marketingos.exceptions.tool import ToolConfigurationError, ToolExecutionError
from marketingos.models.cost import CostCategory
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.base import Tool
from marketingos.tools.image.compositor import Compositor, ImageMediaType
from marketingos.tools.image.image_gen_client import (
    IMAGE_GENERATION,
    ImageGenerationRequest,
)

__all__ = [
    "DEFAULT_MODEL",
    "FluxSchnellClient",
    "TOGETHER_API_KEY_ENV",
]

#: Free FLUX.1 [schnell] model id on Together AI.
DEFAULT_MODEL: Final[str] = "black-forest-labs/FLUX.1-schnell-Free"

#: Environment variable holding the Together AI API key.
TOGETHER_API_KEY_ENV: Final[str] = "TOGETHER_API_KEY"

_API_URL: Final[str] = "https://api.together.xyz/v1/images/generations"

_DEFAULT_OUTPUT_DIR: Final[Path] = Path("data/cache/generated_images")

#: The free schnell endpoint caps sampling steps at four.
_MAX_STEPS: Final[int] = 4
_DEFAULT_STEPS: Final[int] = 4

#: Provider size constraints: multiples of 16, within this pixel range.
_SIZE_GRID: Final[int] = 16
_MIN_SIZE: Final[int] = 256
_MAX_SIZE: Final[int] = 1440

_EXTENSION_BY_MEDIA_TYPE: Final[dict[ImageMediaType, str]] = {
    ImageMediaType.PNG: ".png",
    ImageMediaType.JPEG: ".jpg",
    ImageMediaType.WEBP: ".webp",
}


def _to_generation_size(value: int) -> int:
    """Round ``value`` to the provider's 16px grid within its bounds."""
    snapped = round(value / _SIZE_GRID) * _SIZE_GRID
    return max(_MIN_SIZE, min(_MAX_SIZE, snapped))


class FluxSchnellClient(Tool[ImageGenerationRequest, GeneratedImageRef]):
    """Image generation via FLUX.1 [schnell] on Together AI, priced per image.

    Free tier is expressed as a *rate of zero*, not a hardcoded zero cost,
    mirroring ``GeminiImageClient``: the moment a paid rate is configured,
    this tool starts charging correctly with no code change.
    """

    def __init__(
        self,
        *,
        cost_guard: CostGuard,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cost_per_image: Decimal = Decimal("0"),
        steps: int = _DEFAULT_STEPS,
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
        compositor: Compositor | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Initialise the client.

        Args:
            cost_guard: Guard enforcing the run's budget. Required: see
                ``Tool.cost_guard``.
            api_key: API key. Defaults to the ``TOGETHER_API_KEY``
                environment variable (see ``.env.example``).
            model: Model id to call.
            cost_per_image: Flat price per generated image. Zero on the
                free tier.
            steps: Sampling steps, clamped to :data:`_MAX_STEPS`.
            output_dir: Local directory generated images are written to.
                Created on first use if it doesn't already exist. Same
                default as ``GeminiImageClient``, so the video assembler
                resolves FLUX-generated stills with no extra wiring.
            compositor: Post-processor normalizing output to the exact
                requested dimensions/format. Defaults to a fresh
                ``Compositor``.
            http_client: Transport to use. Defaults to a client owned by
                this instance; tests inject one with a mock transport.
            timeout_seconds: Per-request timeout for the default client.

        Raises:
            ToolConfigurationError: If no API key is available.
        """
        resolved_key = (
            api_key if api_key is not None else os.environ.get(TOGETHER_API_KEY_ENV)
        )
        if not resolved_key:
            raise ToolConfigurationError(
                f"No Together API key: pass api_key= or set "
                f"{TOGETHER_API_KEY_ENV} (see .env.example)."
            )
        self._api_key = resolved_key
        self._model = model
        self._cost_per_image = cost_per_image
        self._steps = max(1, min(_MAX_STEPS, steps))
        self._output_dir = output_dir
        self._compositor = compositor or Compositor()
        self._cost_guard = cost_guard
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._logger = logger.bind(component="FluxSchnellClient", model=model)

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
        return "together"

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
        """Flat per-image price; free tier is a rate of zero."""
        return self._cost_per_image

    # cost_actual: inherited default (reuses cost_estimate) is correct — a
    # flat-rate endpoint has no post-hoc usage to reconcile against.

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
            ToolExecutionError: If the request fails, the response carries
                no image data, or the bytes can't be decoded/re-encoded.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "prompt": self._compose_prompt(payload),
            "width": _to_generation_size(payload.width),
            "height": _to_generation_size(payload.height),
            "steps": self._steps,
            "n": 1,
            "response_format": "b64_json",
        }
        try:
            response = await self._client.post(
                _API_URL,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"Together returned {exc.response.status_code} for model "
                f"{self._model!r}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError(
                f"Together image request failed for model {self._model!r}: {exc}"
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
        path = await self._persist(asset_id, normalized, media_type=payload.media_type)

        self._logger.bind(
            event="flux_schnell.generated",
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
        """Extract and base64-decode the first image in the response.

        Raises:
            ToolExecutionError: If no image entry is present, or the inline
                data isn't valid base64.
        """
        try:
            entry = data["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ToolExecutionError(
                f"Together response contained no image data: {str(data)[:500]}"
            ) from exc

        encoded = entry.get("b64_json") if isinstance(entry, dict) else None
        if not encoded:
            raise ToolExecutionError(
                f"Together response carried no b64_json image: {str(data)[:500]}"
            )
        try:
            return base64.b64decode(encoded)
        except (ValueError, TypeError) as exc:
            raise ToolExecutionError(
                f"Together returned undecodable inline image data: {exc}"
            ) from exc

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

        Delegates to :meth:`invoke`, so the agent path is budget-enforced
        exactly like the tool path.

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
