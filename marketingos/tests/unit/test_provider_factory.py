"""Tests for the config-driven provider factory.

These assert the property the whole abstraction exists for: which concrete
client a run gets is decided by the ``*_provider`` config value alone, so
switching provider is a configuration change with no code edit.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from marketingos.config.settings import ModelSettings
from marketingos.exceptions.tool import ToolConfigurationError
from marketingos.models.cost import CostLedger
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.factory import (
    build_image_generator,
    build_llm,
    build_video_generator,
)


def _guard() -> CostGuard:
    return CostGuard(CostLedger(max_budget=Decimal("100")), run_id=uuid4())


def _models(**overrides: object) -> ModelSettings:
    base: dict[str, object] = dict(
        default_llm="gemini-flash-latest",
        fallback_llm="gemini-2.0-flash-001",
        temperature=0.5,
        max_tokens=100,
        image_quality="standard",
    )
    base.update(overrides)
    return ModelSettings(**base)  # type: ignore[arg-type]


# -- LLM ---------------------------------------------------------------------


def test_build_llm_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = build_llm(_models(llm_provider="gemini"), _guard())
    assert type(llm).__name__ == "GeminiClient"


def test_build_llm_unknown_provider_raises() -> None:
    with pytest.raises(ToolConfigurationError) as excinfo:
        build_llm(_models(llm_provider="openai"), _guard())
    # Error names the unknown provider and lists the known ones.
    assert "openai" in str(excinfo.value)
    assert "gemini" in str(excinfo.value)


# -- image -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("placeholder", "PlaceholderImageClient"),
        ("gemini", "GeminiImageClient"),
        ("flux_schnell", "FluxSchnellClient"),
    ],
)
def test_build_image_by_provider(
    provider: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    image = build_image_generator(_models(image_provider=provider), _guard())
    assert type(image).__name__ == expected


def test_build_image_unknown_provider_raises() -> None:
    with pytest.raises(ToolConfigurationError) as excinfo:
        build_image_generator(_models(image_provider="dalle"), _guard())
    assert "dalle" in str(excinfo.value)
    assert "flux_schnell" in str(excinfo.value)


# -- video -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("local_assembler", "VideoAssembler"),
        ("placeholder", "PlaceholderVideoClient"),
    ],
)
def test_build_video_by_provider(provider: str, expected: str) -> None:
    video = build_video_generator(_models(video_provider=provider), _guard())
    assert type(video).__name__ == expected


def test_build_video_unknown_provider_raises() -> None:
    with pytest.raises(ToolConfigurationError) as excinfo:
        build_video_generator(_models(video_provider="runway"), _guard())
    assert "runway" in str(excinfo.value)
    assert "placeholder" in str(excinfo.value)


# -- the core demonstration --------------------------------------------------


def test_switching_image_provider_is_config_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flipping only image_provider yields a different client — no code edit."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    produced = {
        provider: type(
            build_image_generator(_models(image_provider=provider), _guard())
        ).__name__
        for provider in ("placeholder", "gemini", "flux_schnell")
    }
    assert produced == {
        "placeholder": "PlaceholderImageClient",
        "gemini": "GeminiImageClient",
        "flux_schnell": "FluxSchnellClient",
    }
