"""Tests for the Together AI FLUX Schnell image client.

Uses an injected mock httpx transport (the same pattern as the Gemini client
tests in ``test_cost_guard.py``) so nothing touches the network.
"""

from __future__ import annotations

import base64
import io
import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from PIL import Image

from marketingos.exceptions.tool import ToolConfigurationError
from marketingos.models.cost import CostLedger
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.image.flux_schnell_client import (
    DEFAULT_MODEL,
    FluxSchnellClient,
)


def _png_b64(width: int = 64, height: int = 64) -> str:
    """Return a valid base64-encoded PNG the Compositor can open."""
    image = Image.new("RGB", (width, height), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FluxRecorder:
    """Mock transport capturing the request and returning a b64 PNG."""

    def __init__(self) -> None:
        self.request_body: dict[str, object] = {}
        self.authorization: str | None = None
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.request_body = json.loads(request.content)
        self.authorization = request.headers.get("Authorization")
        return httpx.Response(200, json={"data": [{"b64_json": _png_b64()}]})


def _guard() -> CostGuard:
    return CostGuard(CostLedger(max_budget=Decimal("100")), run_id=uuid4())


def _make_client(tmp_path: Path, recorder: FluxRecorder) -> FluxSchnellClient:
    return FluxSchnellClient(
        cost_guard=_guard(),
        api_key="test-key",
        output_dir=tmp_path,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    )


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(ToolConfigurationError):
        FluxSchnellClient(cost_guard=_guard())


async def test_generate_returns_exact_dimensions(tmp_path: Path) -> None:
    recorder = FluxRecorder()
    client = _make_client(tmp_path, recorder)

    ref = await client.generate(
        prompt="a cat on a skateboard",
        negative_prompt="dogs",
        width=1080,
        height=1350,
    )

    # The Compositor guarantees the caller's exact dimensions regardless of
    # what the provider returned.
    assert ref.width == 1080
    assert ref.height == 1350
    assert ref.media_type == "image/png"
    assert Path(ref.uri).exists()
    assert recorder.calls == 1


async def test_generate_sends_expected_request(tmp_path: Path) -> None:
    recorder = FluxRecorder()
    client = _make_client(tmp_path, recorder)

    await client.generate(
        prompt="a cat", negative_prompt="dogs", width=1024, height=1024
    )

    assert recorder.authorization == "Bearer test-key"
    body = recorder.request_body
    assert body["model"] == DEFAULT_MODEL
    assert body["response_format"] == "b64_json"
    assert body["n"] == 1
    # Free schnell tier caps steps at 4.
    assert isinstance(body["steps"], int) and body["steps"] <= 4
    # No native negative-prompt field: it's folded into the prompt text.
    assert "Avoid: dogs" in str(body["prompt"])


async def test_generate_raises_on_missing_image_data(tmp_path: Path) -> None:
    def _no_image(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = FluxSchnellClient(
        cost_guard=_guard(),
        api_key="test-key",
        output_dir=tmp_path,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_no_image)),
    )

    from marketingos.exceptions.tool import ToolExecutionError

    with pytest.raises(ToolExecutionError):
        await client.generate(
            prompt="x", negative_prompt="", width=512, height=512
        )
