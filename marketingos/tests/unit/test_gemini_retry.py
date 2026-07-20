"""Tests for GeminiClient's transient-failure retry behaviour.

The whole point is that a free-tier run (which fires ~8 calls in quick
succession and trips the 5-per-minute rate limit) completes instead of
failing. Every test here uses a mocked httpx transport and a sleep-spy —
no network call is ever made and no real time is spent waiting, so the
suite stays fast and costs nothing.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import httpx
import pytest

from marketingos.exceptions.tool import ToolExecutionError
from marketingos.models.cost import CostLedger
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.llm import gemini_client
from marketingos.tools.llm.gemini_client import GeminiClient

_SUCCESS_BODY = {
    "candidates": [{"content": {"parts": [{"text": "Generated copy."}]}}],
    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7},
    "modelVersion": "gemini-flash-latest",
}


def _rate_limit_body(retry_delay: str | None = "1s") -> dict:
    details = []
    if retry_delay is not None:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": retry_delay,
            }
        )
    return {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota.",
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }


class _SleepSpy:
    """Async stand-in for ``asyncio.sleep`` that records requested delays."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _guard() -> CostGuard:
    return CostGuard(CostLedger(max_budget=Decimal("100")), run_id=uuid4())


def _client(responses: list[httpx.Response]) -> tuple[GeminiClient, dict]:
    """A GeminiClient whose transport replays ``responses`` in order.

    Returns the client plus a mutable ``calls`` dict tracking how many POSTs
    the transport actually saw.
    """
    state = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(state["count"], len(responses) - 1)
        state["count"] += 1
        return responses[index]

    client = GeminiClient(
        cost_guard=_guard(),
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_base_delay_seconds=2.0,
        retry_max_delay_seconds=30.0,
    )
    return client, state


async def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SleepSpy()
    monkeypatch.setattr(gemini_client.asyncio, "sleep", spy)

    client, calls = _client(
        [
            httpx.Response(429, json=_rate_limit_body()),
            httpx.Response(429, json=_rate_limit_body()),
            httpx.Response(200, json=_SUCCESS_BODY),
        ]
    )

    text = await client.complete(system_prompt="sys", user_prompt="hi")

    assert text == "Generated copy."
    assert calls["count"] == 3  # two failures + one success
    assert len(spy.delays) == 2  # slept once before each retry


async def test_honours_server_suggested_retry_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SleepSpy()
    monkeypatch.setattr(gemini_client.asyncio, "sleep", spy)

    client, _ = _client(
        [
            httpx.Response(429, json=_rate_limit_body(retry_delay="5s")),
            httpx.Response(200, json=_SUCCESS_BODY),
        ]
    )

    await client.complete(system_prompt="sys", user_prompt="hi")

    # Server asked for 5s; the wait honours it (plus <=0.5s jitter), rather
    # than using the 2s exponential-backoff base.
    assert 5.0 <= spy.delays[0] <= 5.5


async def test_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SleepSpy()
    monkeypatch.setattr(gemini_client.asyncio, "sleep", spy)

    client = GeminiClient(
        cost_guard=_guard(),
        api_key="test-key",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(429, json=_rate_limit_body())
            )
        ),
        max_retries=2,
    )

    with pytest.raises(ToolExecutionError) as excinfo:
        await client.complete(system_prompt="sys", user_prompt="hi")

    assert "429" in str(excinfo.value)
    assert len(spy.delays) == 2  # retried exactly max_retries times


async def test_non_retryable_status_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SleepSpy()
    monkeypatch.setattr(gemini_client.asyncio, "sleep", spy)

    client, calls = _client(
        [httpx.Response(400, json={"error": {"message": "bad request"}})]
    )

    with pytest.raises(ToolExecutionError) as excinfo:
        await client.complete(system_prompt="sys", user_prompt="hi")

    assert "400" in str(excinfo.value)
    assert calls["count"] == 1  # no retry
    assert spy.delays == []  # never slept


async def test_success_on_first_try_never_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SleepSpy()
    monkeypatch.setattr(gemini_client.asyncio, "sleep", spy)

    client, calls = _client([httpx.Response(200, json=_SUCCESS_BODY)])

    text = await client.complete(system_prompt="sys", user_prompt="hi")

    assert text == "Generated copy."
    assert calls["count"] == 1
    assert spy.delays == []


def test_suggested_retry_delay_reads_header_and_body() -> None:
    header_resp = httpx.Response(429, headers={"retry-after": "12"}, json={})
    assert gemini_client._suggested_retry_delay(header_resp) == 12.0

    body_resp = httpx.Response(429, json=_rate_limit_body(retry_delay="8s"))
    assert gemini_client._suggested_retry_delay(body_resp) == 8.0

    none_resp = httpx.Response(429, json=_rate_limit_body(retry_delay=None))
    assert gemini_client._suggested_retry_delay(none_resp) is None
