"""Capability-keyed tool registry for MarketingOS.

:class:`ToolRegistry` is the concrete implementation of the structural
``ToolRegistry`` Protocol declared in
:mod:`marketingos.agents.base` — ``get(name) -> Any`` /
``__contains__(name) -> bool`` — so any instance of this class can be
injected into an agent's ``tools=`` constructor argument.

Capability-keyed, not vendor-keyed
-----------------------------------
Tools register under their own :attr:`~marketingos.tools.base.Tool.capability`
(``"text_generation"``, ``"image_generation"``, ``"video_generation"``,
``"website_scraping"``, ...), and agents that resolve tools dynamically ask
for that capability rather than a vendor name. Swapping DALL·E for
Stability, or one search provider for another, is therefore a registration
change at wiring time — zero agent code touched, which is the whole point
of the capability layer described in the architecture doc's Tool
Abstraction Layer section.

At most one tool occupies a given capability at a time, which is what
makes :meth:`get` unambiguous; see :meth:`register` for how to
intentionally replace one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from marketingos.exceptions.tool import ToolConfigurationError, ToolNotFoundError
from marketingos.tools.base import Tool

if TYPE_CHECKING:
    from loguru import Logger

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """Resolves tools by capability.

    A thin, synchronous, in-process mapping — no persistence, no remote
    discovery. One registry is typically built once per process (or per
    run, if different runs need different provider configurations) and
    shared by every agent that declares a ``tools`` dependency.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}
        self._logger: Logger = logger.bind(component="ToolRegistry")

    # -- construction convenience --------------------------------------------

    @classmethod
    def from_tools(cls, *tools: Tool[Any, Any]) -> ToolRegistry:
        """Build a registry from already-constructed tools in one call.

        Args:
            *tools: Tool instances to register, each under its own
                ``capability``.

        Returns:
            A populated :class:`ToolRegistry`.

        Raises:
            ToolConfigurationError: If two of the given tools share a
                capability — see :meth:`register`.
        """
        registry = cls()
        for tool in tools:
            registry.register(tool)
        return registry

    # -- mutation --------------------------------------------------------------

    def register(self, tool: Tool[Any, Any], *, replace: bool = False) -> None:
        """Register ``tool`` under its own :attr:`Tool.capability`.

        Args:
            tool: The tool instance to register.
            replace: If ``False`` (default) and ``tool.capability`` is
                already occupied by a *different* tool, raises rather than
                silently shadowing it — a duplicate registration is far
                more likely to be a wiring mistake than an intentional
                swap. Set ``True`` to replace deliberately (provider
                swaps, tests). Re-registering the same instance is always
                a no-op, regardless of ``replace``.

        Raises:
            ToolConfigurationError: If the capability is already
                registered to a different tool and ``replace`` is
                ``False``.
        """
        capability = tool.capability
        existing = self._tools.get(capability)
        if existing is not None and existing is not tool and not replace:
            raise ToolConfigurationError(
                f"Capability {capability!r} is already registered to "
                f"{existing.name!r} (provider={existing.provider!r}); pass "
                f"replace=True to swap it for {tool.name!r} "
                f"(provider={tool.provider!r})."
            )
        self._tools[capability] = tool
        self._logger.bind(
            event="tool_registry.registered",
            capability=capability,
            tool=tool.name,
            provider=tool.provider,
            replaced=existing is not None,
        ).info("Tool registered")

    def unregister(self, capability: str) -> None:
        """Remove the tool registered for ``capability``, if any.

        Never raises: unregistering an absent capability is a no-op, so
        callers (tests, hot-reload paths) don't need to guard the call
        with a membership check first.
        """
        removed = self._tools.pop(capability, None)
        if removed is not None:
            self._logger.bind(
                event="tool_registry.unregistered",
                capability=capability,
                tool=removed.name,
            ).info("Tool unregistered")

    # -- lookup (ToolRegistry Protocol) -----------------------------------------

    def get(self, name: str) -> Tool[Any, Any]:
        """Return the tool providing capability ``name``.

        Args:
            name: The capability key to resolve (e.g. ``"image_generation"``).

        Returns:
            The registered :class:`~marketingos.tools.base.Tool`.

        Raises:
            ToolNotFoundError: If no tool provides ``name``. Deliberately
                not a ``KeyError`` — see the ``ToolRegistry`` Protocol
                docstring in :mod:`marketingos.agents.base`: agents wrap
                model-output parsing in
                ``except (KeyError, TypeError, ValueError)`` and treat that
                as a retryable error, whereas an unregistered capability is
                a permanent configuration fault that must propagate as
                such.
        """
        try:
            return self._tools[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise ToolNotFoundError(
                f"No tool registered for capability {name!r}. "
                f"Known capabilities: {known}"
            ) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(capabilities={sorted(self._tools)!r})"

    def capabilities(self) -> tuple[str, ...]:
        """Return every registered capability key, sorted."""
        return tuple(sorted(self._tools))