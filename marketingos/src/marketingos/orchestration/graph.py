"""LangGraph ``StateGraph`` assembly for the MarketingOS orchestration workflow.

This module defines :class:`GraphBuilder`, the sole component responsible
for wiring together a typed ``langgraph.graph.StateGraph`` over the shared
:class:`~orchestration.nodes.state.MarketingState`. It registers nodes,
connects them via conditional routing, attaches a checkpointer, and inserts
human-in-the-loop approval interrupts.

This module contains **no** agent logic, **no** business logic, and **no**
API logic. Node callables, routing functions, checkpoint savers, and
approval-gate node factories are all supplied by their respective modules
(``nodes package``, ``edges.py``, ``checkpointer.py``, ``approval_gates.py``)
and injected into :class:`GraphBuilder` rather than hardcoded here. This
keeps ``graph.py`` a pure composition root, satisfying the single
responsibility and dependency-inversion principles: this module depends on
narrow callable/type contracts, never on concrete node implementations.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .approval_gates import build_approval_gate_nodes, list_approval_gate_names
from .checkpointer import build_checkpointer
from .edges import build_conditional_edges
from .state import MarketingState

type NodeAction = (
    Callable[[MarketingState], MarketingState | dict[str, Any]]
    | Callable[[MarketingState], Awaitable[MarketingState | dict[str, Any]]]
)
"""A LangGraph node callable: accepts the shared state and returns either a
full :class:`MarketingState` or a partial state update ``dict``, synchronously
or asynchronously."""

type RouterAction = (
    Callable[[MarketingState], str] | Callable[[MarketingState], Awaitable[str]]
)
"""A LangGraph conditional-edge router callable: accepts the shared state and
returns the name of the next node (or ``END``), synchronously or
asynchronously."""


class GraphAssemblyError(RuntimeError):
    """Raised when a :class:`GraphBuilder` is asked to build or compile an
    invalid or incomplete graph definition.

    Examples of invalid definitions include: no nodes registered, no entry
    point set, an edge or interrupt referencing an unregistered node name,
    or ``compile()`` being called before any nodes have been added.
    """


@dataclass(frozen=True, slots=True)
class NodeDefinition:
    """A single registered LangGraph node.

    Attributes:
        name: Unique name of the node within the graph.
        action: The callable implementing the node's behavior. Supplied by
            the caller; this module never implements node behavior itself.
    """

    name: str
    action: NodeAction


@dataclass(frozen=True, slots=True)
class ConditionalEdgeDefinition:
    """A single registered conditional (branching) edge.

    Attributes:
        source: Name of the node the edge originates from.
        router: Callable that inspects the state and returns the key of the
            next node to visit.
        path_map: Optional explicit mapping from router return values to
            destination node names. When omitted, LangGraph treats the
            router's return value as the destination node name directly.
    """

    source: str
    router: RouterAction
    path_map: Mapping[str, str] | None = None


@dataclass(slots=True)
class GraphAssembly:
    """Immutable-in-spirit snapshot of everything needed to assemble a graph.

    This is an internal bookkeeping structure used by :class:`GraphBuilder`
    to accumulate registrations before ``build()`` is called. It has no
    behavior of its own beyond storage.

    Attributes:
        nodes: Registered nodes keyed by name.
        static_edges: Unconditional ``(source, target)`` edges.
        conditional_edges: Registered conditional edge definitions.
        entry_point: Name of the node the graph starts execution from.
        interrupt_before: Node names LangGraph should pause execution before,
            used to implement human approval gates.
        interrupt_after: Node names LangGraph should pause execution after.
    """

    nodes: dict[str, NodeDefinition] = field(default_factory=dict)
    static_edges: list[tuple[str, str]] = field(default_factory=list)
    conditional_edges: list[ConditionalEdgeDefinition] = field(default_factory=list)
    entry_point: str | None = None
    interrupt_before: list[str] = field(default_factory=list)
    interrupt_after: list[str] = field(default_factory=list)


class GraphBuilder:
    """Assembles a typed LangGraph ``StateGraph`` over :class:`MarketingState`.

    ``GraphBuilder`` is a pure composition root: it owns no agent logic and
    no business logic. Every node, router, checkpointer, and approval gate
    it wires together is supplied through dependency injection, either via
    the fluent ``add_*``/``with_*`` methods or via the ``use_default_*``
    convenience methods that source their dependencies from
    :mod:`orchestration.nodes.edges`, :mod:`orchestration.nodes.checkpointer`,
    and :mod:`orchestration.nodes.approval_gates`.

    The builder is stateful and fluent: registration methods return ``self``
    so calls can be chained. ``build()`` produces an uncompiled
    ``StateGraph``; ``compile()`` produces a runnable ``CompiledStateGraph``.

    Example:
        Typical usage from the composition root that owns concrete node
        implementations::

            builder = (
                GraphBuilder(entry_point="research")
                .add_nodes(workflow_nodes)
                .use_default_approval_gates()
                .use_default_conditional_edges()
                .use_default_checkpointer()
            )
            app = builder.compile()
    """

    def __init__(self, entry_point: str | None = None) -> None:
        """Initialize an empty builder.

        Args:
            entry_point: Optional name of the node execution should start
                from. May also be set later via :meth:`set_entry_point`.
        """
        self._assembly = GraphAssembly(entry_point=entry_point)
        self._checkpointer: BaseCheckpointSaver | None = None
        self._graph: StateGraph | None = None
        self._compiled: CompiledStateGraph | None = None

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------

    def add_node(self, name: str, action: NodeAction, *, replace: bool = False) -> Self:
        """Register a single node.

        Args:
            name: Unique name for the node.
            action: Callable implementing the node's behavior.
            replace: If ``True``, silently overwrite an existing node
                registered under the same name. If ``False`` (default), a
                duplicate registration raises :class:`GraphAssemblyError`.

        Returns:
            ``self``, to support fluent chaining.

        Raises:
            GraphAssemblyError: If ``name`` is already registered and
                ``replace`` is ``False``.
        """
        if not replace and name in self._assembly.nodes:
            raise GraphAssemblyError(f"Node '{name}' is already registered.")
        self._assembly.nodes[name] = NodeDefinition(name=name, action=action)
        self._invalidate_build_cache()
        return self

    def add_nodes(self, nodes: Mapping[str, NodeAction], *, replace: bool = False) -> Self:
        """Register multiple nodes at once.

        Args:
            nodes: Mapping of node name to node callable.
            replace: Forwarded to :meth:`add_node` for each entry.

        Returns:
            ``self``, to support fluent chaining.
        """
        for name, action in nodes.items():
            self.add_node(name, action, replace=replace)
        return self

    # ------------------------------------------------------------------
    # Edge registration
    # ------------------------------------------------------------------

    def add_edge(self, source: str, target: str) -> Self:
        """Register a static, unconditional edge between two nodes.

        Either endpoint may be ``langgraph.graph.START`` or
        ``langgraph.graph.END``.

        Args:
            source: Name of the origin node.
            target: Name of the destination node.

        Returns:
            ``self``, to support fluent chaining.
        """
        self._assembly.static_edges.append((source, target))
        self._invalidate_build_cache()
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterAction,
        path_map: Mapping[str, str] | None = None,
    ) -> Self:
        """Register a conditional (branching) edge originating from a node.

        Args:
            source: Name of the origin node.
            router: Callable that inspects :class:`MarketingState` and
                returns the routing key for the next node.
            path_map: Optional mapping from routing keys to destination node
                names, forwarded to LangGraph.

        Returns:
            ``self``, to support fluent chaining.
        """
        self._assembly.conditional_edges.append(
            ConditionalEdgeDefinition(source=source, router=router, path_map=path_map)
        )
        self._invalidate_build_cache()
        return self

    def use_default_conditional_edges(self) -> Self:
        """Register the conditional edges defined in :mod:`edges.py`.

        Delegates to ``edges.build_conditional_edges()``, which is expected
        to return an iterable of ``(source, router, path_map)`` tuples
        describing every branching decision in the workflow. This keeps all
        routing policy in ``edges.py`` while ``graph.py`` remains a pure
        assembler.

        Returns:
            ``self``, to support fluent chaining.
        """
        for source, router, path_map in build_conditional_edges():
            self.add_conditional_edges(source, router, path_map)
        return self

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def set_entry_point(self, name: str) -> Self:
        """Set the node execution begins from.

        Args:
            name: Name of the entry node.

        Returns:
            ``self``, to support fluent chaining.
        """
        self._assembly.entry_point = name
        self._invalidate_build_cache()
        return self

    # ------------------------------------------------------------------
    # Approval gates (human-in-the-loop)
    # ------------------------------------------------------------------

    def with_approval_gates(
        self,
        gate_nodes: Mapping[str, NodeAction],
        gate_names: Sequence[str],
    ) -> Self:
        """Register approval-gate nodes and pause execution before each one.

        Approval gates are implemented using LangGraph's ``interrupt_before``
        mechanism: execution is checkpointed and suspended immediately before
        each named gate node runs, allowing an external actor (an API
        endpoint, typically) to inspect state and call
        :meth:`~orchestration.nodes.state.MarketingState.approve_stage`
        before resuming the graph.

        Args:
            gate_nodes: Mapping of gate node name to the node callable that
                implements the gate (e.g. persisting the pending-approval
                state, notifying reviewers).
            gate_names: Names of nodes execution should halt before. This is
                typically, but not necessarily, the same set of names as the
                keys of ``gate_nodes``.

        Returns:
            ``self``, to support fluent chaining.
        """
        self.add_nodes(gate_nodes, replace=True)
        for gate_name in gate_names:
            if gate_name not in self._assembly.interrupt_before:
                self._assembly.interrupt_before.append(gate_name)
        self._invalidate_build_cache()
        return self

    def use_default_approval_gates(self) -> Self:
        """Register the approval-gate nodes defined in :mod:`approval_gates.py`.

        Delegates to ``approval_gates.build_approval_gate_nodes()`` for the
        node implementations and ``approval_gates.list_approval_gate_names()``
        for the set of node names to interrupt before, keeping all
        human-approval policy in ``approval_gates.py``.

        Returns:
            ``self``, to support fluent chaining.
        """
        return self.with_approval_gates(
            gate_nodes=build_approval_gate_nodes(),
            gate_names=list_approval_gate_names(),
        )

    def interrupt_after(self, *node_names: str) -> Self:
        """Register additional nodes to pause execution after.

        Args:
            *node_names: Names of nodes execution should halt after
                completing.

        Returns:
            ``self``, to support fluent chaining.
        """
        for name in node_names:
            if name not in self._assembly.interrupt_after:
                self._assembly.interrupt_after.append(name)
        self._invalidate_build_cache()
        return self

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def with_checkpointer(self, checkpointer: BaseCheckpointSaver) -> Self:
        """Attach an explicit checkpoint saver to be used on ``compile()``.

        Args:
            checkpointer: The checkpoint saver instance to use.

        Returns:
            ``self``, to support fluent chaining.
        """
        self._checkpointer = checkpointer
        self._compiled = None
        return self

    def use_default_checkpointer(self) -> Self:
        """Attach the checkpoint saver produced by ``checkpointer.py``.

        Delegates to ``checkpointer.build_checkpointer()``, keeping storage
        backend selection and configuration in ``checkpointer.py``.

        Returns:
            ``self``, to support fluent chaining.
        """
        return self.with_checkpointer(build_checkpointer())

    # ------------------------------------------------------------------
    # Build / compile / visualize
    # ------------------------------------------------------------------

    def build(self) -> StateGraph:
        """Assemble and return the uncompiled, typed ``StateGraph``.

        Validates that at least one node and an entry point have been
        registered, and that every edge and interrupt refers to a node
        that actually exists (or to ``START``/``END``), before constructing
        the LangGraph ``StateGraph`` instance.

        Returns:
            The assembled, uncompiled ``StateGraph`` typed over
            :class:`MarketingState`.

        Raises:
            GraphAssemblyError: If the current registrations are invalid or
                incomplete.
        """
        self._validate()

        graph: StateGraph = StateGraph(MarketingState)

        for node in self._assembly.nodes.values():
            # ``NodeAction`` (a plain ``Callable[...]`` alias) can't structurally
            # satisfy langgraph's ``StateNode`` protocols, which accept ``state``
            # by keyword as well as position -- a type-checker-only mismatch, not
            # a real one, since any ``NodeAction`` is a valid LangGraph node.
            graph.add_node(node.name, cast(Any, node.action))

        assert self._assembly.entry_point is not None  # enforced by _validate()
        graph.set_entry_point(self._assembly.entry_point)

        for source, target in self._assembly.static_edges:
            graph.add_edge(source, target)

        for edge in self._assembly.conditional_edges:
            path_map = cast("dict[Hashable, str] | None", edge.path_map)
            graph.add_conditional_edges(edge.source, edge.router, path_map)

        self._graph = graph
        return graph

    def compile(self, **compile_kwargs: Any) -> CompiledStateGraph:
        """Build (if necessary) and compile the graph into a runnable app.

        Args:
            **compile_kwargs: Additional keyword arguments forwarded verbatim
                to ``StateGraph.compile()``, allowing callers to pass through
                LangGraph options this builder does not explicitly model
                (for example ``debug=True``).

        Returns:
            The compiled, runnable ``CompiledStateGraph``.

        Raises:
            GraphAssemblyError: If the current registrations are invalid or
                incomplete.
        """
        graph = self._graph if self._graph is not None else self.build()

        compiled = graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=self._assembly.interrupt_before or None,
            interrupt_after=self._assembly.interrupt_after or None,
            **compile_kwargs,
        )
        self._compiled = compiled
        return compiled

    def visualize(self, output_path: str | Path | None = None) -> str:
        """Render the compiled graph's structure as Mermaid diagram source.

        Compiles the graph first if it has not been compiled yet. This is a
        diagnostic convenience for engineers inspecting workflow shape; it
        is not required for the graph to run.

        Args:
            output_path: If provided, the Mermaid source is also written to
                this file path.

        Returns:
            The Mermaid diagram source as a string.

        Raises:
            GraphAssemblyError: If the current registrations are invalid or
                incomplete.
        """
        compiled = self._compiled if self._compiled is not None else self.compile()
        mermaid_source: str = compiled.get_graph().draw_mermaid()

        if output_path is not None:
            Path(output_path).write_text(mermaid_source, encoding="utf-8")

        return mermaid_source

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def with_default_wiring(
        cls,
        nodes: Mapping[str, NodeAction],
        entry_point: str,
    ) -> GraphBuilder:
        """Construct a builder pre-wired with the project's default policies.

        Registers the supplied workflow nodes, then attaches the default
        approval gates (``approval_gates.py``), conditional routing
        (``edges.py``), and checkpointer (``checkpointer.py``). Any of these
        defaults can still be overridden afterward via the instance's
        fluent methods before calling :meth:`build` or :meth:`compile`.

        Args:
            nodes: Mapping of node name to node callable for the
                domain-specific workflow steps (research, strategy,
                planning, creative generation, etc.). These implementations
                live outside this module and are injected here.
            entry_point: Name of the node execution should start from.

        Returns:
            A fully wired :class:`GraphBuilder`, ready to :meth:`compile`.
        """
        return (
            cls(entry_point=entry_point)
            .add_nodes(nodes)
            .use_default_approval_gates()
            .use_default_conditional_edges()
            .use_default_checkpointer()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invalidate_build_cache(self) -> None:
        """Discard any previously built/compiled graph after a registration change."""
        self._graph = None
        self._compiled = None

    def _validate(self) -> None:
        """Validate the current registrations before assembling a graph.

        Raises:
            GraphAssemblyError: If no nodes are registered, no entry point
                is set, or any edge/interrupt references a node name that
                was never registered.
        """
        if not self._assembly.nodes:
            raise GraphAssemblyError("At least one node must be registered before building.")

        if self._assembly.entry_point is None:
            raise GraphAssemblyError("An entry point must be set before building.")

        if self._assembly.entry_point not in self._assembly.nodes:
            raise GraphAssemblyError(
                f"Entry point '{self._assembly.entry_point}' is not a registered node."
            )

        known_names = set(self._assembly.nodes) | {START, END}

        for source, target in self._assembly.static_edges:
            if source not in known_names:
                raise GraphAssemblyError(f"Edge source '{source}' is not a registered node.")
            if target not in known_names:
                raise GraphAssemblyError(f"Edge target '{target}' is not a registered node.")

        for edge in self._assembly.conditional_edges:
            if edge.source not in known_names:
                raise GraphAssemblyError(
                    f"Conditional edge source '{edge.source}' is not a registered node."
                )
            if edge.path_map is not None:
                unknown_targets = set(edge.path_map.values()) - known_names
                if unknown_targets:
                    raise GraphAssemblyError(
                        f"Conditional edge from '{edge.source}' targets unregistered "
                        f"node(s): {sorted(unknown_targets)}."
                    )

        for name in (*self._assembly.interrupt_before, *self._assembly.interrupt_after):
            if name not in self._assembly.nodes:
                raise GraphAssemblyError(f"Interrupt target '{name}' is not a registered node.")
