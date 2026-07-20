"""Config-driven provider factory for MarketingOS model capabilities.

This is the dispatch layer that turns a *provider name in configuration*
into a *concrete client instance*, so switching providers (or models) is a
YAML edit rather than a code change. Agents already depend only on the
capability ports (:class:`~marketingos.agents.business_analysis.LanguageModelPort`,
:class:`~marketingos.agents.designer.ImageGenerationPort`,
:class:`~marketingos.agents.video_director.VideoGenerationPort`); this module
is the single place that decides *which* implementation of each port a run
gets, driven by :class:`~marketingos.config.settings.ModelSettings`.

Adding a provider
-----------------
Write a client satisfying the capability port (usually a
:class:`~marketingos.tools.base.Tool` subclass), then register a one-line
builder in the matching ``_*_PROVIDERS`` table below and name it in
``models.yaml``. No agent, node, or dependency-wiring code changes — that is
the whole point of the abstraction.

Why a table of builder callables rather than a class map: the concrete
clients have genuinely different constructor signatures (``GeminiClient``
takes token cost rates and an output ceiling; ``PlaceholderImageClient``
takes no cost guard at all), so each builder adapts ``ModelSettings`` +
``CostGuard`` to its client's specific shape.
"""

from __future__ import annotations

from collections.abc import Callable

from marketingos.agents.business_analysis import LanguageModelPort
from marketingos.agents.designer import ImageGenerationPort
from marketingos.agents.video_director import VideoGenerationPort
from marketingos.config.settings import ModelSettings
from marketingos.exceptions.tool import ToolConfigurationError
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.image import (
    DEFAULT_MODEL as GEMINI_IMAGE_DEFAULT_MODEL,
)
from marketingos.tools.image import (
    FLUX_DEFAULT_MODEL,
    FluxSchnellClient,
    GeminiImageClient,
    PlaceholderImageClient,
)
from marketingos.tools.llm import GeminiClient
from marketingos.tools.video import PlaceholderVideoClient, VideoAssembler

__all__ = [
    "build_image_generator",
    "build_llm",
    "build_video_generator",
]

LLMBuilder = Callable[[ModelSettings, CostGuard], LanguageModelPort]
ImageBuilder = Callable[[ModelSettings, CostGuard], ImageGenerationPort]
VideoBuilder = Callable[[ModelSettings, CostGuard], VideoGenerationPort]


# -- LLM providers -----------------------------------------------------------


def _build_gemini_llm(models: ModelSettings, guard: CostGuard) -> LanguageModelPort:
    return GeminiClient(
        cost_guard=guard,
        model=models.default_llm,
        default_max_output_tokens=models.max_tokens,
        input_cost_per_1k_tokens=models.llm_input_cost_per_1k,
        output_cost_per_1k_tokens=models.llm_output_cost_per_1k,
    )


_LLM_PROVIDERS: dict[str, LLMBuilder] = {
    "gemini": _build_gemini_llm,
}


# -- image providers ---------------------------------------------------------


def _build_gemini_image(models: ModelSettings, guard: CostGuard) -> ImageGenerationPort:
    # default_image_model is an optional override: when unset, each provider
    # falls back to its own default model id, so flipping image_provider
    # alone switches providers without also editing the model id.
    return GeminiImageClient(
        cost_guard=guard,
        model=models.default_image_model or GEMINI_IMAGE_DEFAULT_MODEL,
        cost_per_image=models.image_cost_per_image,
    )


def _build_flux_image(models: ModelSettings, guard: CostGuard) -> ImageGenerationPort:
    return FluxSchnellClient(
        cost_guard=guard,
        model=models.default_image_model or FLUX_DEFAULT_MODEL,
        cost_per_image=models.image_cost_per_image,
    )


def _build_placeholder_image(
    models: ModelSettings, guard: CostGuard
) -> ImageGenerationPort:
    # Local render: no provider API, no cost guard, no model id.
    return PlaceholderImageClient()


_IMAGE_PROVIDERS: dict[str, ImageBuilder] = {
    "gemini": _build_gemini_image,
    "flux_schnell": _build_flux_image,
    "placeholder": _build_placeholder_image,
}


# -- video providers ---------------------------------------------------------


def _build_video_assembler(
    models: ModelSettings, guard: CostGuard
) -> VideoGenerationPort:
    return VideoAssembler(cost_guard=guard)


def _build_placeholder_video(
    models: ModelSettings, guard: CostGuard
) -> VideoGenerationPort:
    # Local stub copy: no rendering, no cost guard, no model id.
    return PlaceholderVideoClient()


_VIDEO_PROVIDERS: dict[str, VideoBuilder] = {
    "local_assembler": _build_video_assembler,
    "placeholder": _build_placeholder_video,
}


# -- public builders ---------------------------------------------------------


def _resolve[T](
    table: dict[str, Callable[[ModelSettings, CostGuard], T]],
    provider: str,
    *,
    capability: str,
    models: ModelSettings,
    guard: CostGuard,
) -> T:
    """Look up ``provider`` in ``table`` and build, or raise a config error."""
    builder = table.get(provider)
    if builder is None:
        known = ", ".join(sorted(table)) or "<none>"
        raise ToolConfigurationError(
            f"Unknown {capability} provider {provider!r}. "
            f"Known providers: {known}."
        )
    return builder(models, guard)


def build_llm(models: ModelSettings, guard: CostGuard) -> LanguageModelPort:
    """Build the text-generation client named by ``models.llm_provider``."""
    return _resolve(
        _LLM_PROVIDERS,
        models.llm_provider,
        capability="LLM",
        models=models,
        guard=guard,
    )


def build_image_generator(
    models: ModelSettings, guard: CostGuard
) -> ImageGenerationPort:
    """Build the image client named by ``models.image_provider``."""
    return _resolve(
        _IMAGE_PROVIDERS,
        models.image_provider,
        capability="image",
        models=models,
        guard=guard,
    )


def build_video_generator(
    models: ModelSettings, guard: CostGuard
) -> VideoGenerationPort:
    """Build the video client named by ``models.video_provider``."""
    return _resolve(
        _VIDEO_PROVIDERS,
        models.video_provider,
        capability="video",
        models=models,
        guard=guard,
    )
