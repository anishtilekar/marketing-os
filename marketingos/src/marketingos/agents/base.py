"""Foundational abstractions for MarketingOS agents.

This module provides the building blocks shared by every agent in the
``marketingos.agents`` package:

* The :class:`AgentError` exception hierarchy — the only exception types
  agents are allowed to raise across their public boundary.
* :class:`AgentConfig` — an immutable, constructor-injected settings object.
* Dependency-injection protocols (:class:`MemoryStore`,
  :class:`ToolRegistry`, :class:`PromptRepository`) so agents depend on
  *behaviour*, never on concrete infrastructure classes.
* :class:`BaseAgent` — the abstract execution shell that owns run
  identification, lifecycle hooks, retry handling, execution timing, cost
  estimation and structured logging, while delegating all domain work to the
  abstract :meth:`BaseAgent.run` method.

Architectural contract
----------------------
Agents are **stateless**: every collaborator arrives through the constructor
and every execution is fully described by its input payload and ``run_id``.
The LangGraph orchestration layer invokes each agent independently through
``await agent.execute(payload)``; agents never call one another directly.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from loguru import Logger

__all__ = [
    "AgentConfig",
    "AgentConfigurationError",
    "AgentError",
    "BaseAgent",
    "MemoryStore",
    "PermanentAgentError",
    "PromptRepository",
    "RetryableAgentError",
    "ToolRegistry",
]


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Base class for every error raised by a MarketingOS agent.

    All exceptions crossing an agent's public boundary are instances of this
    class. Unexpected exceptions raised inside :meth:`BaseAgent.run` are
    converted through :meth:`BaseAgent.handle_error` — they are never
    swallowed and never leak as raw built-in exceptions.

    Attributes:
        message: Human-readable description of the failure.
        agent_name: Name of the agent that raised the error, when known.
        run_id: Identifier of the execution during which the error occurred,
            when known.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.agent_name = agent_name
        self.run_id = run_id

    def __str__(self) -> str:
        context = ", ".join(
            part
            for part in (
                f"agent={self.agent_name}" if self.agent_name else "",
                f"run_id={self.run_id}" if self.run_id else "",
            )
            if part
        )
        return f"{self.message} [{context}]" if context else self.message


class RetryableAgentError(AgentError):
    """A transient failure that is safe to retry.

    Raise (or map to) this class for failures caused by temporary conditions
    such as network timeouts, connection resets, rate limiting, or upstream
    service unavailability. :meth:`BaseAgent.execute` retries these errors
    with exponential backoff up to ``AgentConfig.max_retries`` times.
    """


class PermanentAgentError(AgentError):
    """A non-recoverable failure that must propagate immediately.

    Raise (or map to) this class for failures that retrying cannot fix:
    invalid input, missing dependencies, contract violations, or programming
    errors. :meth:`BaseAgent.execute` never retries these errors.
    """


class AgentConfigurationError(PermanentAgentError):
    """The agent was constructed or used with an invalid configuration."""


# ---------------------------------------------------------------------------
# Dependency-injection protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Structural contract for the optional agent memory backend.

    Satisfied by the concrete stores in ``marketingos.memory``. Agents depend
    only on this protocol, keeping the memory implementation swappable
    (Redis, Postgres, in-process, ...) without touching agent code.
    """

    async def get(self, key: str) -> Any:
        """Return the value stored under ``key``, or ``None`` if absent."""
        ...

    async def set(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key``."""
        ...


@runtime_checkable
class ToolRegistry(Protocol):
    """Structural contract for the optional tool registry.

    Satisfied by :class:`marketingos.tools.registry.ToolRegistry`, which
    resolves tools by *capability* (``"text_generation"``) rather than by
    vendor, for agents that resolve tools dynamically rather than receiving
    them as dedicated constructor arguments. ``name`` is therefore a
    capability key.
    """

    def get(self, name: str) -> Any:
        """Return the tool providing capability ``name``.

        Raises:
            ToolNotFoundError: If no tool provides ``name``. This is
                :class:`marketingos.exceptions.tool.ToolNotFoundError`, a
                ``MarketingOSError`` — deliberately *not* a ``KeyError``.
                Agents wrap model-output parsing in
                ``except (KeyError, TypeError, ValueError)`` and map it to a
                *retryable* error; an unregistered tool is a permanent
                configuration fault, and must not be swallowed by those
                handlers and retried.
        """
        ...

    def __contains__(self, name: str) -> bool:
        """Return whether a tool provides capability ``name``."""
        ...


@runtime_checkable
class PromptRepository(Protocol):
    """Structural contract for the optional prompt loader.

    Satisfied by the loader in ``marketingos.prompts``. Renders a named
    prompt template with the supplied variables.
    """

    def render(self, template_name: str, /, **variables: Any) -> str:
        """Return the rendered prompt for ``template_name``.

        Raises:
            KeyError: If the template does not exist.
        """
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Immutable runtime settings shared by all agents.

    Instances are frozen so a config object can be shared safely between
    agents and across concurrent executions. Agent-specific settings are
    added by subclassing (see ``ResearchAgentConfig`` for an example).
    """

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for retryable failures "
        "(0 disables retrying).",
    )
    retry_initial_delay_seconds: float = Field(
        default=0.5,
        gt=0.0,
        description="Backoff delay before the first retry.",
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Multiplier applied to the delay after each retry.",
    )
    retry_max_delay_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Upper bound for the backoff delay.",
    )
    base_cost_per_run_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Flat estimated cost of one execution, used by the "
        "default cost estimator. Agents with model- or token-based costs "
        "override BaseAgent.estimate_cost() instead.",
    )


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent[InputT: BaseModel, OutputT: BaseModel](ABC):
    """Abstract execution shell for all MarketingOS agents.

    Subclasses implement a single method — :meth:`run` — containing the
    agent's domain logic. Everything operational lives here:

    * **Entrypoint** — callers (the LangGraph workflow) use
      ``await agent.execute(payload)``; :meth:`run` is never called directly
      from outside.
    * **Retries** — :class:`RetryableAgentError` triggers exponential
      backoff up to ``config.max_retries``; :class:`PermanentAgentError`
      propagates immediately.
    * **Error normalisation** — unexpected exceptions are mapped through
      :meth:`handle_error` into the agent error hierarchy with the original
      exception chained as ``__cause__``; nothing is ever swallowed.
    * **Lifecycle hooks** — :meth:`before_run` / :meth:`after_run` let
      subclasses validate, enrich or post-process without touching the
      execution machinery.
    * **Observability** — every execution emits structured Loguru records
      carrying the agent name, run id, execution time, success flag, retry
      count, error type and cost estimate.

    Type parameters:
        InputT: Pydantic model accepted by the agent.
        OutputT: Pydantic model produced by the agent.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        config: AgentConfig | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        prompts: PromptRepository | None = None,
    ) -> None:
        """Initialise the agent with injected collaborators.

        Args:
            name: Logical agent name used in logs; defaults to the class name.
            config: Runtime settings; defaults to :class:`AgentConfig` with
                its documented defaults.
            memory: Optional memory backend for agents that persist context.
            tools: Optional tool registry for dynamic tool resolution.
            prompts: Optional prompt repository for template rendering.
        """
        self._name = name or type(self).__name__
        self._config = config or AgentConfig()
        self._memory = memory
        self._tools = tools
        self._prompts = prompts
        self._logger: Logger = logger.bind(agent=self._name)

    # -- read-only accessors --------------------------------------------------

    @property
    def name(self) -> str:
        """Logical name of this agent, used in logs and error context."""
        return self._name

    @property
    def config(self) -> AgentConfig:
        """The immutable runtime configuration of this agent."""
        return self._config

    @property
    def memory(self) -> MemoryStore | None:
        """The injected memory backend, if any."""
        return self._memory

    @property
    def tools(self) -> ToolRegistry | None:
        """The injected tool registry, if any."""
        return self._tools

    @property
    def prompts(self) -> PromptRepository | None:
        """The injected prompt repository, if any."""
        return self._prompts

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"

    # -- prompt support --------------------------------------------------------

    def load_prompt(self, template_name: str, /, **variables: Any) -> str:
        """Render a named prompt template through the injected repository.

        Args:
            template_name: Name of the template to render.
            **variables: Substitution variables for the template.

        Returns:
            The rendered prompt text.

        Raises:
            AgentConfigurationError: If no prompt repository was injected or
                the template cannot be rendered.
        """
        if self._prompts is None:
            raise AgentConfigurationError(
                f"Agent requested prompt {template_name!r} but no "
                "PromptRepository was injected.",
                agent_name=self._name,
            )
        try:
            return self._prompts.render(template_name, **variables)
        except Exception as exc:
            raise AgentConfigurationError(
                f"Failed to render prompt template {template_name!r}: {exc}",
                agent_name=self._name,
            ) from exc

    # -- abstract domain logic ---------------------------------------------------

    @abstractmethod
    async def run(self, payload: InputT, *, run_id: str) -> OutputT:
        """Perform the agent's domain work for a single execution.

        Implementations must be side-effect free with respect to the agent
        instance (agents are stateless) and should raise
        :class:`RetryableAgentError` for transient failures and
        :class:`PermanentAgentError` for non-recoverable ones. Any other
        exception is normalised by :meth:`handle_error`.

        Args:
            payload: Validated, typed input for this execution.
            run_id: Unique identifier of this execution, for logging and
                result traceability.

        Returns:
            The typed result of the execution.
        """

    # -- public entrypoint ----------------------------------------------------------

    async def execute(self, payload: InputT) -> OutputT:
        """Execute the agent with retries, timing, and structured logging.

        This is the only public execution entrypoint. It generates a run id,
        invokes the lifecycle hooks around :meth:`run`, retries transient
        failures with exponential backoff, and logs the outcome.

        Args:
            payload: Validated, typed input for this execution.

        Returns:
            The typed result produced by :meth:`run` (possibly transformed
            by :meth:`after_run`).

        Raises:
            RetryableAgentError: If a transient failure persists after
                ``config.max_retries`` retries.
            PermanentAgentError: If a non-recoverable failure occurs.
        """
        run_id = uuid.uuid4().hex
        started = time.perf_counter()
        retry_count = 0
        self.log_start(payload, run_id=run_id)

        while True:
            try:
                prepared = await self.before_run(payload, run_id=run_id)
                result = await self.run(prepared, run_id=run_id)
                result = await self.after_run(prepared, result, run_id=run_id)
            except Exception as exc:  # noqa: BLE001 - normalised below, never swallowed
                error = self._as_agent_error(exc, run_id=run_id)
                if (
                    isinstance(error, RetryableAgentError)
                    and retry_count < self._config.max_retries
                ):
                    retry_count += 1
                    delay = self._retry_delay(retry_count)
                    self._logger.bind(
                        run_id=run_id,
                        event="agent.retry",
                        retry_count=retry_count,
                        max_retries=self._config.max_retries,
                        delay_seconds=round(delay, 3),
                        error_type=type(error).__name__,
                        error_message=error.message,
                    ).warning("Transient failure; retrying")
                    await asyncio.sleep(delay)
                    continue
                elapsed = time.perf_counter() - started
                self.log_failure(
                    error,
                    run_id=run_id,
                    elapsed_seconds=elapsed,
                    retry_count=retry_count,
                )
                if error is exc:
                    raise
                raise error from exc

            elapsed = time.perf_counter() - started
            cost = self.estimate_cost(payload, result)
            self.log_success(
                result,
                run_id=run_id,
                elapsed_seconds=elapsed,
                retry_count=retry_count,
                cost_usd=cost,
            )
            return result

    # -- lifecycle hooks --------------------------------------------------------------

    async def before_run(self, payload: InputT, *, run_id: str) -> InputT:
        """Hook invoked before :meth:`run` on every attempt.

        Subclasses may override to validate or enrich the payload; the
        returned object is what :meth:`run` receives. The default emits a
        debug record and returns the payload unchanged.
        """
        self._logger.bind(
            run_id=run_id,
            event="agent.before_run",
            input_model=type(payload).__name__,
        ).debug("Entering run")
        return payload

    async def after_run(
        self, payload: InputT, result: OutputT, *, run_id: str
    ) -> OutputT:
        """Hook invoked after a successful :meth:`run`.

        Subclasses may override to post-process or persist the result; the
        returned object is what :meth:`execute` returns to the caller. The
        default emits a debug record and returns the result unchanged.
        """
        self._logger.bind(
            run_id=run_id,
            event="agent.after_run",
            output_model=type(result).__name__,
        ).debug("Run produced a result")
        return result

    # -- error handling ------------------------------------------------------------------

    def handle_error(self, error: Exception, *, run_id: str) -> AgentError:
        """Map an unexpected exception into the agent error hierarchy.

        Known-transient built-ins (timeouts and connection failures) become
        :class:`RetryableAgentError`; everything else becomes
        :class:`PermanentAgentError`. Subclasses may override to refine the
        mapping for their tool stack. The original exception is chained by
        :meth:`execute`, so no information is lost.

        Args:
            error: The unexpected exception raised inside the run cycle.
            run_id: Identifier of the failing execution.

        Returns:
            The corresponding :class:`AgentError` instance.
        """
        message = f"Unexpected {type(error).__name__}: {error}"
        if isinstance(error, (TimeoutError, ConnectionError)):
            return RetryableAgentError(
                message, agent_name=self._name, run_id=run_id
            )
        return PermanentAgentError(message, agent_name=self._name, run_id=run_id)

    def _as_agent_error(self, error: Exception, *, run_id: str) -> AgentError:
        """Return ``error`` as an :class:`AgentError`, converting if needed."""
        if isinstance(error, AgentError):
            error.agent_name = error.agent_name or self._name
            error.run_id = error.run_id or run_id
            return error
        return self.handle_error(error, run_id=run_id)

    def _retry_delay(self, attempt: int) -> float:
        """Return the capped exponential backoff delay for ``attempt`` (1-based)."""
        cfg = self._config
        delay = cfg.retry_initial_delay_seconds * (
            cfg.retry_backoff_factor ** (attempt - 1)
        )
        return min(delay, cfg.retry_max_delay_seconds)

    # -- cost estimation --------------------------------------------------------------------

    def estimate_cost(self, payload: InputT, result: OutputT) -> float:
        """Estimate the monetary cost (USD) of one completed execution.

        The default returns the flat ``config.base_cost_per_run_usd``.
        Agents that call metered services (LLMs, paid APIs) override this
        with token- or call-based accounting. The estimate is included in
        the structured success log for cost observability.
        """
        return self._config.base_cost_per_run_usd

    # -- structured logging ----------------------------------------------------------------------

    def log_start(self, payload: InputT, *, run_id: str) -> None:
        """Emit the structured record marking the start of an execution."""
        self._logger.bind(
            run_id=run_id,
            event="agent.start",
            input_model=type(payload).__name__,
        ).info("Agent execution started")

    def log_success(
        self,
        result: OutputT,
        *,
        run_id: str,
        elapsed_seconds: float,
        retry_count: int,
        cost_usd: float,
    ) -> None:
        """Emit the structured record marking a successful execution."""
        self._logger.bind(
            run_id=run_id,
            event="agent.success",
            success=True,
            output_model=type(result).__name__,
            execution_time_seconds=round(elapsed_seconds, 6),
            retry_count=retry_count,
            cost_estimate_usd=round(cost_usd, 6),
        ).info("Agent execution succeeded")

    def log_failure(
        self,
        error: AgentError,
        *,
        run_id: str,
        elapsed_seconds: float,
        retry_count: int,
    ) -> None:
        """Emit the structured record marking a failed execution."""
        self._logger.bind(
            run_id=run_id,
            event="agent.failure",
            success=False,
            error_type=type(error).__name__,
            error_message=error.message,
            execution_time_seconds=round(elapsed_seconds, 6),
            retry_count=retry_count,
        ).error("Agent execution failed")
