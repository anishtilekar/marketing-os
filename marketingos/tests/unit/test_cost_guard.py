from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel

from marketingos.exceptions.budget import CostTrackingError, InsufficientBudgetError
from marketingos.exceptions.tool import ToolNotFoundError
from marketingos.models.cost import CostCategory, CostLedger, CostStatus
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.base import Tool
from marketingos.tools.llm.gemini_client import (
    TEXT_GENERATION,
    GeminiClient,
    GeminiRequest,
)
from marketingos.tools.registry import ToolRegistry

# Paid rates, so cost_estimate() returns a real non-zero figure and the guard
# has something falsifiable to check. The free tier is exercised separately.
PAID_INPUT_RATE = Decimal("10")
PAID_OUTPUT_RATE = Decimal("30")

SYSTEM_PROMPT = "You are a rigorous business analyst."
USER_PROMPT = "Analyse this business."


class CallRecorder:
    """A mock Gemini transport that records whether it was ever called."""

    def __init__(self, *, status: int = 200) -> None:
        self.calls = 0
        self._status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(
            self._status,
            json={
                "candidates": [{"content": {"parts": [{"text": "generated text"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        )


def make_client(
    *,
    budget: Decimal,
    recorder: CallRecorder | None = None,
    input_rate: Decimal = PAID_INPUT_RATE,
    output_rate: Decimal = PAID_OUTPUT_RATE,
) -> tuple[GeminiClient, CostGuard, CallRecorder]:
    """Build a Gemini client wired to a mock transport and a fresh ledger."""
    recorder = recorder or CallRecorder()
    ledger = CostLedger(max_budget=budget)
    guard = CostGuard(ledger, run_id=uuid4())
    client = GeminiClient(
        cost_guard=guard,
        api_key="test-key",
        input_cost_per_1k_tokens=input_rate,
        output_cost_per_1k_tokens=output_rate,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    )
    return client, guard, recorder


def make_request(**overrides) -> GeminiRequest:
    defaults = dict(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        max_output_tokens=1000,
    )
    defaults.update(overrides)
    return GeminiRequest(**defaults)


# ---------------------------------------------------------------------------
# Blocking: the estimate exceeds the budget
# ---------------------------------------------------------------------------


async def test_guard_blocks_call_that_would_exceed_a_near_zero_budget():
    """A near-zero budget must refuse a call the tool prices above it."""
    client, _guard, recorder = make_client(budget=Decimal("0.0001"))
    request = make_request()

    assert client.cost_estimate(request) > Decimal("0.0001")
    with pytest.raises(InsufficientBudgetError):
        await client.invoke(request)


async def test_blocked_call_never_reaches_the_provider():
    """The abort happens before the request is sent, not after."""
    client, _guard, recorder = make_client(budget=Decimal("0.0001"))

    with pytest.raises(InsufficientBudgetError):
        await client.invoke(make_request())

    assert recorder.calls == 0


async def test_blocked_call_records_nothing_on_the_ledger():
    """A refused call must not consume budget."""
    client, guard, _recorder = make_client(budget=Decimal("0.0001"))

    with pytest.raises(InsufficientBudgetError):
        await client.invoke(make_request())

    assert guard.ledger.entries == []
    assert guard.spent == Decimal("0")
    assert guard.remaining == Decimal("0.0001")


# ---------------------------------------------------------------------------
# Allowing: the budget is sufficient
# ---------------------------------------------------------------------------


async def test_guard_allows_call_when_budget_is_sufficient():
    """A sufficient budget lets the call through and returns the result."""
    client, _guard, recorder = make_client(budget=Decimal("100"))

    result = await client.invoke(make_request())

    assert result.text == "generated text"
    assert recorder.calls == 1


async def test_successful_call_records_actual_cost_on_the_ledger():
    """The real cost, from the provider's reported usage, lands on the ledger."""
    client, guard, _recorder = make_client(budget=Decimal("100"))

    await client.invoke(make_request())

    assert len(guard.ledger.entries) == 1
    entry = guard.ledger.entries[0]
    assert entry.status is CostStatus.COMPLETED
    assert entry.category is CostCategory.LLM_GENERATION
    assert entry.provider == "google"
    assert entry.tool_name == "gemini-2.5-flash"
    # 10 input tokens @ 10/1k + 5 output tokens @ 30/1k = 0.1 + 0.15
    assert entry.actual_cost == Decimal("0.25")
    assert guard.spent == Decimal("0.25")
    assert guard.remaining == Decimal("99.75")


async def test_budget_depletes_across_calls_until_the_ceiling_blocks():
    """Recorded spend accumulates, and the ceiling eventually refuses a call.

    The budget is sized so exactly one call fits: the first request's
    estimate clears it, but once that call's actual cost is recorded the
    remainder no longer covers an identical second request.
    """
    client, guard, recorder = make_client(budget=Decimal("30.2"))
    request = make_request()
    estimate = client.cost_estimate(request)

    await client.invoke(request)
    assert guard.spent == Decimal("0.25")
    assert guard.remaining < estimate  # the next identical call cannot fit

    with pytest.raises(InsufficientBudgetError):
        await client.invoke(request)

    assert recorder.calls == 1


# ---------------------------------------------------------------------------
# Free tier: zero rates, real arithmetic
# ---------------------------------------------------------------------------


async def test_free_tier_estimates_zero_but_still_records_an_entry():
    """Zero rates cost nothing yet still produce a real, checked figure."""
    client, guard, recorder = make_client(
        budget=Decimal("100"), input_rate=Decimal("0"), output_rate=Decimal("0")
    )

    assert client.cost_estimate(make_request()) == Decimal("0")
    await client.invoke(make_request())

    assert recorder.calls == 1
    assert guard.spent == Decimal("0")
    assert len(guard.ledger.entries) == 1


def test_cost_estimate_scales_with_requested_output():
    """The estimate is derived, not hardcoded: more output costs more."""
    client, _guard, _recorder = make_client(budget=Decimal("100"))

    small = client.cost_estimate(make_request(max_output_tokens=100))
    large = client.cost_estimate(make_request(max_output_tokens=10_000))

    assert large > small > Decimal("0")


# ---------------------------------------------------------------------------
# The agent path (LanguageModelPort.complete) is guarded too
# ---------------------------------------------------------------------------


async def test_complete_is_budget_enforced_like_invoke():
    """complete() delegates to invoke(), so agents cannot bypass the ceiling."""
    client, _guard, recorder = make_client(budget=Decimal("0.0001"))

    with pytest.raises(InsufficientBudgetError):
        await client.complete(system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT)

    assert recorder.calls == 0


async def test_complete_returns_text_when_budget_allows():
    """The adapter returns the generated text agents expect."""
    client, guard, _recorder = make_client(budget=Decimal("100"))

    text = await client.complete(
        system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT
    )

    assert text == "generated text"
    assert len(guard.ledger.entries) == 1


# ---------------------------------------------------------------------------
# Fail-closed: an unguarded tool refuses to call the provider
# ---------------------------------------------------------------------------


async def test_tool_without_a_guard_refuses_to_call_the_provider():
    """An unpriced call fails closed rather than proceeding for free."""
    client, _guard, recorder = make_client(budget=Decimal("100"))
    # Simulate a tool that reaches invoke() with no guard attached.
    client._cost_guard = None  # noqa: SLF001 - asserting the fail-closed path

    with pytest.raises(CostTrackingError):
        await client.invoke(make_request())

    assert recorder.calls == 0


# ---------------------------------------------------------------------------
# Structural enforcement: a subclass cannot opt out of the guard
# ---------------------------------------------------------------------------


class UnguardedToolInput(BaseModel):
    """Input for the throwaway tool below."""

    cost: Decimal


class UnguardedToolOutput(BaseModel):
    """Output for the throwaway tool below."""

    called: bool


class NaiveTool(Tool[UnguardedToolInput, UnguardedToolOutput]):
    """A tool whose author forgot the guard — note: no @cost_guarded here.

    This is the regression the structural fix exists to prevent: before
    Tool.__init_subclass__ auto-wrapped invoke(), a subclass like this one
    would reach its provider without ever consulting the budget.
    """

    def __init__(self, cost_guard: CostGuard | None) -> None:
        self.cost_guard = cost_guard
        self.calls = 0

    @property
    def name(self) -> str:
        return "naive-tool"

    @property
    def capability(self) -> str:
        return "naive_capability"

    @property
    def provider(self) -> str:
        return "test-provider"

    @property
    def input_schema(self) -> type[UnguardedToolInput]:
        return UnguardedToolInput

    @property
    def output_schema(self) -> type[UnguardedToolOutput]:
        return UnguardedToolOutput

    def cost_estimate(self, payload: UnguardedToolInput) -> Decimal:
        return payload.cost

    async def invoke(self, payload: UnguardedToolInput) -> UnguardedToolOutput:
        """Deliberately undecorated: the base class must guard this anyway."""
        self.calls += 1
        return UnguardedToolOutput(called=True)


def test_undecorated_subclass_is_wrapped_automatically():
    """The ABC wraps invoke at class-creation time, not by convention."""
    assert getattr(NaiveTool.invoke, "__cost_guarded__", False) is True


async def test_guard_fires_on_a_subclass_that_never_applied_the_decorator():
    """An author who forgets @cost_guarded still cannot overspend."""
    ledger = CostLedger(max_budget=Decimal("1"))
    guard = CostGuard(ledger, run_id=uuid4())
    tool = NaiveTool(guard)

    with pytest.raises(InsufficientBudgetError):
        await tool.invoke(UnguardedToolInput(cost=Decimal("5")))

    assert tool.calls == 0, "the provider was reached despite an over-budget call"


async def test_undecorated_subclass_still_records_cost_when_allowed():
    """The auto-applied guard records spend, not just blocks it."""
    ledger = CostLedger(max_budget=Decimal("10"))
    guard = CostGuard(ledger, run_id=uuid4())
    tool = NaiveTool(guard)

    result = await tool.invoke(UnguardedToolInput(cost=Decimal("4")))

    assert result.called is True
    assert tool.calls == 1
    assert guard.spent == Decimal("4")
    assert guard.remaining == Decimal("6")


async def test_undecorated_subclass_without_a_guard_fails_closed():
    """No guard attached means no call, even for a hand-rolled tool."""
    tool = NaiveTool(None)

    with pytest.raises(CostTrackingError):
        await tool.invoke(UnguardedToolInput(cost=Decimal("1")))

    assert tool.calls == 0


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_gemini_registers_under_the_text_generation_capability():
    """Capability-keyed lookup resolves Gemini for text generation."""
    client, _guard, _recorder = make_client(budget=Decimal("100"))
    registry = ToolRegistry()
    registry.register(client)

    assert TEXT_GENERATION in registry
    assert registry.get(TEXT_GENERATION) is client
    assert registry.capabilities() == [TEXT_GENERATION]


def test_missing_capability_raises_tool_not_found_and_not_key_error():
    """An unregistered capability is a permanent fault, not a KeyError.

    Agents map ``except (KeyError, TypeError, ValueError)`` from model-output
    parsing onto *retryable* errors. ToolNotFoundError must stay outside that
    hierarchy so a misconfigured registry fails loudly instead of being
    retried as a transient glitch.
    """
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError) as excinfo:
        registry.get("text_generation")

    assert not isinstance(excinfo.value, KeyError)
    assert "text_generation" in str(excinfo.value)
