"""Gemini text-generation tool.

Wraps Google's Generative Language REST API as a
:class:`~marketingos.tools.base.Tool` registered under the
``text_generation`` capability, defaulting to ``gemini-flash-latest`` — the
model every agent entry in ``config/agents.yaml`` already names.

Two call shapes, one guarded path
---------------------------------
:meth:`GeminiClient.invoke` is the tool-shaped entry point. :meth:`complete`
is a thin adapter satisfying the ``LanguageModelPort`` protocol that
``BusinessAnalysisAgent``, ``StrategistAgent`` and friends already depend on
(``complete(*, system_prompt, user_prompt) -> str``). ``complete()``
delegates to ``invoke()`` rather than calling the API itself, so the agent
path inherits the same budget enforcement — there is no unguarded route to
the provider.

Transport is :mod:`httpx` (already a project dependency) rather than a
provider SDK, keeping the dependency surface unchanged. The client is
injectable so tests can supply a mock transport and never touch the network.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
from decimal import Decimal
from typing import Any, Final

import httpx
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from marketingos.exceptions.tool import ToolConfigurationError, ToolExecutionError
from marketingos.models.cost import CostCategory
from marketingos.services.cost_guard import CostGuard
from marketingos.tools.base import Tool

__all__ = [
    "DEFAULT_MODEL",
    "GEMINI_API_KEY_ENV",
    "GeminiClient",
    "GeminiRequest",
    "GeminiResponse",
    "TEXT_GENERATION",
]

#: Capability key under which this tool registers.
TEXT_GENERATION: Final[str] = "text_generation"

#: Model used by default; matches ``default_llm`` in ``config/models.yaml``.
#: A text model, deliberately: this is the text-generation client, so it must
#: not default to an image model (e.g. ``gemini-2.5-flash-image``).
DEFAULT_MODEL: Final[str] = "gemini-flash-latest"

#: Environment variable holding the API key.
GEMINI_API_KEY_ENV: Final[str] = "GEMINI_API_KEY"

_API_BASE: Final[str] = "https://generativelanguage.googleapis.com/v1beta/models"

#: Rough characters-per-token ratio used to price a request before sending it.
#: Only ever used for the pre-flight estimate; the recorded actual cost uses
#: the provider's reported token counts.
_CHARS_PER_TOKEN: Final[int] = 4

_TOKENS_PER_UNIT: Final[Decimal] = Decimal("1000")

#: HTTP statuses that are transient and worth retrying rather than failing
#: the whole run: 429 (rate limit — the free tier's 5-requests/minute cap,
#: which one campaign trips because it fires ~8 calls back to back), 500
#: (transient provider error), 503 (temporarily overloaded/unavailable).
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 503})


class GeminiRequest(BaseModel):
    """One text-generation request."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, gt=0)


class GeminiResponse(BaseModel):
    """One text-generation result, with the provider's reported usage."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class GeminiClient(Tool[GeminiRequest, GeminiResponse]):
    """Text generation via the Gemini API, priced per token.

    Free tier is expressed as *rates of zero*, not as a hardcoded zero cost:
    :meth:`cost_estimate` performs the full token arithmetic either way, so
    the cost guard has a real figure to check and the tool starts charging
    correctly the moment a paid rate is configured.
    """

    def __init__(
        self,
        *,
        cost_guard: CostGuard,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        input_cost_per_1k_tokens: Decimal = Decimal("0"),
        output_cost_per_1k_tokens: Decimal = Decimal("0"),
        default_max_output_tokens: int = 2048,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 6,
        retry_base_delay_seconds: float = 2.0,
        retry_max_delay_seconds: float = 30.0,
    ) -> None:
        """Initialise the client.

        Args:
            cost_guard: Guard enforcing the run's budget. Required: without
                it :meth:`invoke` fails closed, so a client cannot be built
                that reaches the provider unpriced.
            api_key: API key. Defaults to the ``GEMINI_API_KEY`` environment
                variable (see ``.env.example``).
            model: Model id to call.
            input_cost_per_1k_tokens: Price per 1000 input tokens. Zero on
                the free tier.
            output_cost_per_1k_tokens: Price per 1000 output tokens. Zero on
                the free tier.
            http_client: Transport to use. Defaults to a client owned by this
                instance; tests inject one with a mock transport.
            timeout_seconds: Per-request timeout for the default client.
            max_retries: How many times to retry a transient failure (429 /
                500 / 503) before giving up. The default clears a free-tier
                rate-limit window comfortably; set to ``0`` to disable retry.
            retry_base_delay_seconds: Base for exponential backoff, used only
                when the provider does not supply its own retry delay.
            retry_max_delay_seconds: Ceiling on any single backoff wait.

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
        self._input_rate = input_cost_per_1k_tokens
        self._output_rate = output_cost_per_1k_tokens
        self._default_max_output_tokens = default_max_output_tokens
        self._cost_guard = cost_guard
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._max_retries = max(0, max_retries)
        self._retry_base_delay = retry_base_delay_seconds
        self._retry_max_delay = retry_max_delay_seconds
        self._logger = logger.bind(component="GeminiClient", model=model)

    # -- Tool identity -------------------------------------------------------

    @property
    def name(self) -> str:
        """The model id, recorded as ``CostEntry.tool_name``."""
        return self._model

    @property
    def capability(self) -> str:
        """This tool provides ``text_generation``."""
        return TEXT_GENERATION

    @property
    def provider(self) -> str:
        """Recorded as ``CostEntry.provider``."""
        return "google"

    @property
    def cost_category(self) -> CostCategory:
        """Spend here is language-model generation."""
        return CostCategory.LLM_GENERATION

    @property
    def input_schema(self) -> type[GeminiRequest]:
        """The request model."""
        return GeminiRequest

    @property
    def output_schema(self) -> type[GeminiResponse]:
        """The response model."""
        return GeminiResponse

    @property
    def cost_guard(self) -> CostGuard:
        """The guard consulted by :meth:`invoke` on every call."""
        return self._cost_guard

    # -- cost ----------------------------------------------------------------

    def cost_estimate(self, payload: GeminiRequest) -> Decimal:
        """Price a request from its prompt size and output ceiling.

        Input tokens are approximated from prompt length; output tokens are
        priced at ``max_output_tokens``, the worst case, so the guard never
        under-estimates a call it is about to authorise.

        Args:
            payload: The request to price.

        Returns:
            The estimated cost. Zero while both rates are zero (free tier),
            but derived from real token counts rather than hardcoded.
        """
        input_tokens = self._estimate_input_tokens(payload)
        return self._price(input_tokens, payload.max_output_tokens)

    def cost_actual(self, payload: GeminiRequest, result: GeminiResponse) -> Decimal:
        """Price a completed call from the provider's reported usage.

        Falls back to the estimate's token arithmetic when the response
        carries no usage metadata.

        Args:
            payload: The request that was sent.
            result: The response received.

        Returns:
            The actual cost incurred.
        """
        input_tokens = (
            result.input_tokens
            if result.input_tokens is not None
            else self._estimate_input_tokens(payload)
        )
        output_tokens = (
            result.output_tokens
            if result.output_tokens is not None
            else _estimate_tokens(result.text)
        )
        return self._price(input_tokens, output_tokens)

    def _price(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Apply the configured per-1k rates to a token pair."""
        input_units = Decimal(input_tokens) / _TOKENS_PER_UNIT
        output_units = Decimal(output_tokens) / _TOKENS_PER_UNIT
        return input_units * self._input_rate + output_units * self._output_rate

    @staticmethod
    def _estimate_input_tokens(payload: GeminiRequest) -> int:
        """Approximate the request's input tokens from its prompt lengths."""
        return _estimate_tokens(payload.system_prompt) + _estimate_tokens(
            payload.user_prompt
        )

    # -- invocation ----------------------------------------------------------

    async def invoke(self, payload: GeminiRequest) -> GeminiResponse:
        """Generate text for one request, retrying transient failures.

        A transient failure (429 rate limit, 500, or 503) is retried with
        backoff — honouring the provider's own suggested retry delay when it
        supplies one — up to ``max_retries`` times, because a single campaign
        fires ~8 calls in quick succession and the free tier's 5-per-minute
        cap would otherwise fail the whole run. Non-transient errors, and a
        transient error that outlasts every retry, raise as before. Budget
        enforcement is applied automatically by
        :meth:`marketingos.tools.base.Tool.__init_subclass__`, which wraps
        this method with the cost guard, so the request is priced and
        authorised once before the first attempt and recorded after success;
        retries of the same logical call are not re-charged.

        Args:
            payload: The request to send.

        Returns:
            The generated text with the provider's reported token usage.

        Raises:
            InsufficientBudgetError: If the call would exceed the budget.
                Raised by the decorator, before any request is sent.
            ToolExecutionError: If the request fails after exhausting retries,
                or the response is not shaped as expected.
        """
        body: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": payload.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": payload.user_prompt}]}],
            "generationConfig": {
                "temperature": payload.temperature,
                "maxOutputTokens": payload.max_output_tokens,
            },
        }
        url = f"{_API_BASE}/{self._model}:generateContent"
        data = await self._post_with_retry(url, body)

        result = self._parse(data)
        self._logger.bind(
            event="gemini.generated",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            output_length=len(result.text),
        ).debug("Generated text")
        return result

    async def _post_with_retry(
        self, url: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST ``body`` to ``url``, retrying transient failures with backoff.

        Returns the decoded JSON on success. Raises :class:`ToolExecutionError`
        for a non-retryable status, or for a transient one that survives every
        retry.
        """
        headers = {"x-goog-api-key": self._api_key}
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(url, json=body, headers=headers)
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _RETRYABLE_STATUSES or attempt == self._max_retries:
                    raise ToolExecutionError(
                        f"Gemini returned {status} for model "
                        f"{self._model!r}: {exc.response.text[:500]}"
                    ) from exc
                delay = self._retry_delay(exc.response, attempt)
                self._logger.bind(
                    event="gemini.retry",
                    status=status,
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                ).warning("Transient Gemini error; backing off before retry")
                await asyncio.sleep(delay)
            except httpx.HTTPError as exc:
                # Transport-level failures (timeouts, connection resets) are
                # transient too; retry them on the same schedule.
                if attempt == self._max_retries:
                    raise ToolExecutionError(
                        f"Gemini request failed for model {self._model!r}: {exc}"
                    ) from exc
                delay = self._backoff(attempt)
                self._logger.bind(
                    event="gemini.retry",
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                ).warning("Transient Gemini transport error; backing off")
                await asyncio.sleep(delay)
        # The loop always returns or raises within max_retries+1 iterations;
        # this satisfies the type checker that a value is always produced.
        raise ToolExecutionError(  # pragma: no cover
            f"Gemini request failed for model {self._model!r}: retries exhausted"
        )

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Seconds to wait before the next attempt.

        Prefers the delay the provider itself asks for (``Retry-After`` header
        or the ``RetryInfo.retryDelay`` in the error body), falling back to
        exponential backoff. The result is always capped at
        ``retry_max_delay_seconds``.
        """
        suggested = _suggested_retry_delay(response)
        delay = suggested if suggested is not None else self._backoff(attempt)
        jitter = random.uniform(0.0, 0.5)
        return min(delay, self._retry_max_delay) + jitter

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff for ``attempt`` (0-indexed), capped."""
        return float(min(self._retry_base_delay * (2**attempt), self._retry_max_delay))

    def _parse(self, data: dict[str, Any]) -> GeminiResponse:
        """Extract text and usage from a ``generateContent`` response.

        Raises:
            ToolExecutionError: If no text candidate is present.
        """
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ToolExecutionError(
                f"Gemini response contained no text candidate: {str(data)[:500]}"
            ) from exc
        if not text:
            raise ToolExecutionError("Gemini returned an empty completion.")

        usage = data.get("usageMetadata") or {}
        return GeminiResponse(
            text=text,
            model=data.get("modelVersion") or self._model,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
        )

    # -- LanguageModelPort adapter -------------------------------------------

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's completion, satisfying ``LanguageModelPort``.

        This is the shape the agents already depend on
        (``marketingos.agents.business_analysis.LanguageModelPort``), so a
        ``GeminiClient`` can be injected as ``BusinessAnalysisAgent(llm=...)``
        with no change to any agent. It delegates to :meth:`invoke`, so the
        agent path is budget-enforced exactly like the tool path.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user prompt.

        Returns:
            The generated text.

        Raises:
            InsufficientBudgetError: If the call would exceed the budget.
            ToolExecutionError: If the request fails.
        """
        result = await self.invoke(
            GeminiRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=self._default_max_output_tokens,
            )
        )
        return result.text

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _estimate_tokens(text: str) -> int:
    """Approximate a token count from character length."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _suggested_retry_delay(response: httpx.Response) -> float | None:
    """The retry delay the provider asks for, in seconds, if any.

    Checks the standard ``Retry-After`` header first, then Google's
    structured ``RetryInfo`` detail in the error body (``retryDelay: "9s"``).
    Returns ``None`` when neither is present or parseable, leaving the caller
    to fall back to its own backoff.
    """
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        details = response.json().get("error", {}).get("details", [])
    except (ValueError, AttributeError, TypeError):
        return None
    if not isinstance(details, list):
        return None
    for detail in details:
        raw = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(raw, str) and raw.endswith("s"):
            try:
                return float(raw[:-1])
            except ValueError:
                continue
    return None
