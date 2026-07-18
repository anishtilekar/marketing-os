"""Conditional-edge routing policy for the MarketingOS LangGraph workflow.

This module is responsible **only** for deciding, given the current
:class:`~orchestration.nodes.state.MarketingState`, which node LangGraph
should visit next. It contains no node implementations, no business logic,
and no API logic — every routing function is a pure, synchronous, easily
unit-testable predicate over state.

Each routing function returns the literal name of the next node to execute
(a :class:`NodeName` value, or ``langgraph.graph.END``), which is the
LangGraph pattern that requires no accompanying ``path_map``: the router's
return value *is* the destination node. Node names and revision/retry
thresholds are expressed as typed constants rather than magic strings so
that a single source of truth can be shared with the rest of the
orchestration package.

:class:`EdgeRouter` packages these functions into a configurable,
dependency-injectable object (thresholds and rejection targets are
constructor arguments, not hardcoded), and :func:`build_conditional_edges`
assembles the default ``(source, router, path_map)`` wiring consumed by
``graph.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from langgraph.graph import END

from .state import ApprovalStage, ApprovalStatus, MarketingState, WorkflowStatus

type RouterAction = Callable[[MarketingState], str]
"""A LangGraph conditional-edge router: accepts the shared state and
returns the name of the next node to execute (or ``END``)."""


class NodeName(StrEnum):
    """Canonical node-name constants referenced by routing decisions.

    These values are the contract between ``edges.py`` and the rest of the
    orchestration package (node registration in the ``nodes`` package,
    approval-gate wiring in ``approval_gates.py``, and graph assembly in
    ``graph.py``). If the concrete node names used elsewhere in the project
    differ, update this enum rather than embedding string literals in
    routing logic.
    """

    RESEARCH = "research"
    BUSINESS_CONTEXT = "business_context"
    STRATEGY = "strategy"
    PLANNING = "planning"
    CREATIVE_GENERATION = "creative_generation"
    QA_REVIEW = "qa_review"
    BUDGET_CHECK = "budget_check"
    BUDGET_EXCEEDED_HANDLER = "budget_exceeded_handler"
    EXECUTION_GUARD = "execution_guard"
    ERROR_HANDLER = "error_handler"
    BUSINESS_REVIEW_GATE = "business_review"
    PLANNING_REVIEW_GATE = "planning_review"
    CREATIVE_REVIEW_GATE = "creative_review"
    FINAL_APPROVAL_GATE = "final_approval"
    COMPLETION_CHECK = "completion_check"
    WORKFLOW_COMPLETE = "workflow_complete"


DEFAULT_MAX_RETRIES: int = 3
"""Default number of times a failing node may be retried before the
workflow is diverted to the error handler."""

DEFAULT_MAX_QA_REVISIONS: int = 3
"""Default number of creative-revision cycles QA may request before the
workflow is diverted to the error handler."""


@dataclass(frozen=True, slots=True)
class ApprovalRoutingTargets:
    """Per-stage destinations used when a human approval gate is rejected.

    Attributes:
        rejection_targets: Mapping from :class:`ApprovalStage` to the node
            that should re-run when a reviewer rejects that stage.
    """

    rejection_targets: Mapping[ApprovalStage, str] = field(
        default_factory=lambda: {
            ApprovalStage.BUSINESS_REVIEW: NodeName.BUSINESS_CONTEXT,
            ApprovalStage.PLANNING_REVIEW: NodeName.PLANNING,
            ApprovalStage.CREATIVE_REVIEW: NodeName.CREATIVE_GENERATION,
            ApprovalStage.FINAL_APPROVAL: NodeName.CREATIVE_GENERATION,
        }
    )

    def target_for(self, stage: ApprovalStage) -> str:
        """Return the rejection target node for the given approval stage.

        Args:
            stage: The approval stage that was rejected.

        Returns:
            The node name execution should return to.

        Raises:
            KeyError: If no rejection target is configured for ``stage``.
        """
        return self.rejection_targets[stage]


class EdgeRouter:
    """Configurable collection of LangGraph conditional-edge routers.

    All thresholds and rejection targets are supplied at construction time
    rather than hardcoded, so the same routing logic can be reused across
    environments (e.g. a lower ``max_retries`` in a staging workflow) and
    exercised in isolation in unit tests without constructing a full graph.

    Attributes:
        max_retries: Maximum retry attempts for a failing node before
            escalating to the error handler.
        max_qa_revisions: Maximum QA revision cycles before escalating to
            the error handler.
        approval_targets: Per-stage rejection routing targets.
    """

    def __init__(
        self,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_qa_revisions: int = DEFAULT_MAX_QA_REVISIONS,
        approval_targets: ApprovalRoutingTargets | None = None,
    ) -> None:
        """Initialize the router with configurable thresholds and targets.

        Args:
            max_retries: Maximum retry attempts for a failing node.
            max_qa_revisions: Maximum QA revision cycles.
            approval_targets: Per-stage rejection targets. Defaults to
                :class:`ApprovalRoutingTargets`'s standard mapping.
        """
        self.max_retries = max_retries
        self.max_qa_revisions = max_qa_revisions
        self.approval_targets = approval_targets or ApprovalRoutingTargets()

    # ------------------------------------------------------------------
    # QA passed / QA failed
    # ------------------------------------------------------------------

    def route_after_qa(self, state: MarketingState) -> str:
        """Route after the QA review node based on QA outcome.

        Rule:
            - If QA has been approved, proceed to the creative review
              approval gate.
            - If QA failed but the revision budget is not exhausted, send
              the work back to creative generation for another pass.
            - If QA failed and the revision budget is exhausted, escalate
              to the error handler rather than looping indefinitely.

        Args:
            state: Current workflow state.

        Returns:
            The name of the next node to execute.
        """
        if state.qa.approval_status is ApprovalStatus.APPROVED:
            return NodeName.CREATIVE_REVIEW_GATE

        if state.qa.revision_count < self.max_qa_revisions:
            return NodeName.CREATIVE_GENERATION

        return NodeName.ERROR_HANDLER

    # ------------------------------------------------------------------
    # Budget exceeded
    # ------------------------------------------------------------------

    def route_after_budget_check(self, state: MarketingState) -> str:
        """Route after the budget check node based on remaining budget.

        Rule:
            - If actual spend has met or exceeded the total allocated
              budget, divert to the budget-exceeded handler instead of
              continuing to spend against the plan.
            - Otherwise, proceed to planning.

        Args:
            state: Current workflow state.

        Returns:
            The name of the next node to execute.
        """
        if state.remaining_budget() <= 0:
            return NodeName.BUDGET_EXCEEDED_HANDLER

        return NodeName.PLANNING

    # ------------------------------------------------------------------
    # Retry required / fatal error
    # ------------------------------------------------------------------

    def route_after_execution(self, state: MarketingState) -> str:
        """Route after the execution guard, handling retries and fatal errors.

        Rule:
            - A fatal error, or an overall ``FAILED`` workflow status, always
              routes to the error handler regardless of retry budget.
            - If the current node has recorded a failure and retries remain,
              route back to that same node to retry it.
            - If the current node has failed and retries are exhausted,
              route to the error handler.
            - Otherwise, execution is healthy: proceed to whichever node
              the last completed node designated as ``next_node``, falling
              back to the completion check if none was set.

        Args:
            state: Current workflow state.

        Returns:
            The name of the next node to execute.
        """
        if state.fatal_errors or state.status is WorkflowStatus.FAILED:
            return NodeName.ERROR_HANDLER

        if state.failed_nodes:
            if state.retry_count < self.max_retries:
                return state.current_node or NodeName.ERROR_HANDLER
            return NodeName.ERROR_HANDLER

        return state.next_node or NodeName.COMPLETION_CHECK

    # ------------------------------------------------------------------
    # Fatal error (standalone guard)
    # ------------------------------------------------------------------

    def route_fatal_error(self, state: MarketingState) -> str:
        """Route based solely on the presence of a fatal error.

        This is a narrower guard than :meth:`route_after_execution`, useful
        as a pre-flight check immediately after any node that can raise an
        unrecoverable failure but has no retry semantics of its own (for
        example, an external API integration node).

        Rule:
            - Any fatal error, or an overall ``FAILED`` status, routes to
              the error handler.
            - Otherwise, proceed to whichever node was designated next.

        Args:
            state: Current workflow state.

        Returns:
            The name of the next node to execute.
        """
        if state.fatal_errors or state.status is WorkflowStatus.FAILED:
            return NodeName.ERROR_HANDLER

        return state.next_node or NodeName.COMPLETION_CHECK

    # ------------------------------------------------------------------
    # Human approval required
    # ------------------------------------------------------------------

    def route_after_approval(self, state: MarketingState, stage: ApprovalStage) -> str:
        """Route after a human approval gate for the given stage.

        Rule:
            - ``APPROVED`` proceeds to whichever node was designated next.
            - ``REJECTED`` routes back to the stage-specific rework target
              configured in ``approval_targets``.
            - ``PENDING`` (or ``SKIPPED`` treated defensively as pending)
              routes back to the gate node itself; in practice LangGraph's
              ``interrupt_before`` on the gate node prevents this branch
              from being reached until a decision has been recorded, but
              the explicit fallback keeps this function total and safe to
              call directly in tests.

        Args:
            state: Current workflow state.
            stage: The approval stage whose outcome is being routed.

        Returns:
            The name of the next node to execute.
        """
        outcome = state.approvals.status_for(stage)

        if outcome is ApprovalStatus.APPROVED:
            return state.next_node or NodeName.COMPLETION_CHECK

        if outcome is ApprovalStatus.REJECTED:
            return self.approval_targets.target_for(stage)

        return _gate_node_for(stage)

    def approval_router_for(self, stage: ApprovalStage) -> RouterAction:
        """Build a single-argument router bound to a specific approval stage.

        LangGraph conditional-edge routers must accept exactly the state as
        their sole argument, so this factory closes over ``stage`` to adapt
        :meth:`route_after_approval` to that signature.

        Args:
            stage: The approval stage this router will evaluate.

        Returns:
            A ``RouterAction`` suitable for ``StateGraph.add_conditional_edges``.
        """

        def _router(state: MarketingState) -> str:
            return self.route_after_approval(state, stage)

        return _router

    # ------------------------------------------------------------------
    # Workflow completed
    # ------------------------------------------------------------------

    def route_workflow_completion(self, state: MarketingState) -> str:
        """Route based on whether every stage of the workflow has finished.

        Rule:
            - Any fatal error routes to the error handler, taking priority
              over completion.
            - The workflow is considered complete only once the final
              approval gate has been approved and QA has been approved;
              in that case, route to the terminal workflow-complete node.
            - Otherwise, proceed to whichever node was designated next.

        Args:
            state: Current workflow state.

        Returns:
            The name of the next node to execute, or the terminal
            workflow-complete node.
        """
        if state.fatal_errors or state.status is WorkflowStatus.FAILED:
            return NodeName.ERROR_HANDLER

        final_approved = state.approvals.final_approval is ApprovalStatus.APPROVED
        qa_approved = state.qa.approval_status is ApprovalStatus.APPROVED

        if final_approved and qa_approved:
            return NodeName.WORKFLOW_COMPLETE

        return state.next_node or NodeName.COMPLETION_CHECK

    def route_terminal(self, state: MarketingState) -> str:
        """Route the terminal workflow-complete node to LangGraph's ``END``.

        Args:
            state: Current workflow state.

        Returns:
            ``langgraph.graph.END``, unconditionally.
        """
        return END


def _gate_node_for(stage: ApprovalStage) -> str:
    """Map an approval stage to the node name of its approval gate.

    Args:
        stage: The approval stage to resolve.

    Returns:
        The corresponding gate node name.
    """
    mapping: Mapping[ApprovalStage, str] = {
        ApprovalStage.BUSINESS_REVIEW: NodeName.BUSINESS_REVIEW_GATE,
        ApprovalStage.PLANNING_REVIEW: NodeName.PLANNING_REVIEW_GATE,
        ApprovalStage.CREATIVE_REVIEW: NodeName.CREATIVE_REVIEW_GATE,
        ApprovalStage.FINAL_APPROVAL: NodeName.FINAL_APPROVAL_GATE,
    }
    return mapping[stage]


def build_conditional_edges(
    router: EdgeRouter | None = None,
) -> Sequence[tuple[str, RouterAction, Mapping[str, str] | None]]:
    """Assemble the default conditional-edge wiring for the workflow graph.

    Every router returns a literal node name (see module docstring), so no
    ``path_map`` is required for any of these edges; the third tuple element
    is included only to satisfy the general edge-registration contract
    expected by ``graph.py``.

    Args:
        router: The :class:`EdgeRouter` instance to bind edges to. Defaults
            to a router constructed with standard thresholds, which is
            sufficient for production use; pass a custom instance to
            override retry/revision limits or rejection targets.

    Returns:
        A sequence of ``(source_node, router_callable, path_map)`` tuples
        describing every conditional edge in the workflow.
    """
    active_router = router or EdgeRouter()

    return (
        (NodeName.QA_REVIEW, active_router.route_after_qa, None),
        (NodeName.BUDGET_CHECK, active_router.route_after_budget_check, None),
        (NodeName.EXECUTION_GUARD, active_router.route_after_execution, None),
        (
            NodeName.BUSINESS_REVIEW_GATE,
            active_router.approval_router_for(ApprovalStage.BUSINESS_REVIEW),
            None,
        ),
        (
            NodeName.PLANNING_REVIEW_GATE,
            active_router.approval_router_for(ApprovalStage.PLANNING_REVIEW),
            None,
        ),
        (
            NodeName.CREATIVE_REVIEW_GATE,
            active_router.approval_router_for(ApprovalStage.CREATIVE_REVIEW),
            None,
        ),
        (
            NodeName.FINAL_APPROVAL_GATE,
            active_router.approval_router_for(ApprovalStage.FINAL_APPROVAL),
            None,
        ),
        (NodeName.COMPLETION_CHECK, active_router.route_workflow_completion, None),
        (NodeName.WORKFLOW_COMPLETE, active_router.route_terminal, None),
    )