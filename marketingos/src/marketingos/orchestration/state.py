"""Shared LangGraph execution state for the MarketingOS orchestration workflow.

This module defines :class:`MarketingState`, the single source of truth for
one workflow execution. Every node in the LangGraph workflow receives an
instance of this state, mutates it through the provided helper methods, and
returns it. The state is checkpointed after every node execution.

This module intentionally contains only state definitions, helper models,
and state-manipulation methods. Graph construction, node implementations,
conditional routing, API logic, database logic, and tool implementations
live elsewhere in the ``orchestration`` package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketingos.agents.business_analysis import BusinessAnalysisAgent
from marketingos.agents.copywriter import CaptionPackage
from marketingos.agents.designer import CreativePackage
from marketingos.agents.packaging import CampaignPackage
from marketingos.agents.planner import WeekPlan as PlannerWeekPlan
from marketingos.agents.qa import QAReport
from marketingos.agents.research import ResearchResult
from marketingos.agents.strategist import Strategy
from marketingos.agents.synthetic_resource import SyntheticSourceMaterial
from marketingos.agents.video_director import VideoPackage
from marketingos.models.business_context import BusinessContext
from marketingos.models.cost import CostLedger
from marketingos.models.creative import PostCreative, VideoCreative
from marketingos.models.plan import WeekPlan
from marketingos.models.strategy import StrategyOutput


def _utc_now() -> datetime:
    """Return the current UTC timestamp.

    Returns:
        The current time as a timezone-aware ``datetime`` in UTC.
    """
    return datetime.now(timezone.utc)


class WorkflowStatus(StrEnum):
    """Overall lifecycle status of a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeExecutionStatus(StrEnum):
    """Lifecycle status of a single node execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ApprovalStatus(StrEnum):
    """Status of a human-in-the-loop approval gate."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ApprovalStage(StrEnum):
    """Named human approval checkpoints within the workflow."""

    BUSINESS_REVIEW = "business_review"
    PLANNING_REVIEW = "planning_review"
    CREATIVE_REVIEW = "creative_review"
    FINAL_APPROVAL = "final_approval"


class MessageRole(StrEnum):
    """Author role for an entry in the workflow message history."""

    SYSTEM = "system"
    AGENT = "agent"
    USER = "user"


class ErrorSeverity(StrEnum):
    """Severity classification for a recorded error."""

    WARNING = "warning"
    RECOVERABLE = "recoverable"
    FATAL = "fatal"


class MessageRecord(BaseModel):
    """A single append-only entry in the workflow message history.

    Attributes:
        id: Unique identifier for this message.
        role: Author role of the message (system, agent, or user).
        content: Free-text content of the message.
        node: Name of the node that produced the message, if any.
        created_at: Timestamp at which the message was recorded.
        metadata: Arbitrary structured metadata associated with the message.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4, description="Unique message identifier.")
    role: MessageRole = Field(description="Author role of the message.")
    content: str = Field(min_length=1, description="Free-text message content.")
    node: str | None = Field(
        default=None, description="Name of the node that produced this message."
    )
    created_at: datetime = Field(
        default_factory=_utc_now, description="UTC timestamp the message was recorded."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary structured metadata."
    )


class ErrorRecord(BaseModel):
    """A single recorded error, warning, or fatal failure.

    Attributes:
        id: Unique identifier for this error record.
        severity: Severity classification of the error.
        message: Human-readable description of the error.
        node: Name of the node that raised the error, if any.
        details: Arbitrary structured context about the error.
        created_at: Timestamp at which the error was recorded.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4, description="Unique error record identifier.")
    severity: ErrorSeverity = Field(description="Severity classification of the error.")
    message: str = Field(min_length=1, description="Human-readable error description.")
    node: str | None = Field(
        default=None, description="Name of the node that raised this error."
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary structured error context."
    )
    created_at: datetime = Field(
        default_factory=_utc_now, description="UTC timestamp the error was recorded."
    )


class NodeExecution(BaseModel):
    """A record describing one execution attempt of a single node.

    Attributes:
        node: Name of the executed node.
        status: Current status of this execution attempt.
        started_at: Timestamp the execution attempt started.
        completed_at: Timestamp the execution attempt finished, if finished.
        attempt: Ordinal attempt number for this node, starting at 1.
        error: Error record associated with this attempt, if it failed.
    """

    node: str = Field(min_length=1, description="Name of the executed node.")
    status: NodeExecutionStatus = Field(
        default=NodeExecutionStatus.PENDING, description="Status of this execution attempt."
    )
    started_at: datetime = Field(
        default_factory=_utc_now, description="UTC timestamp the attempt started."
    )
    completed_at: datetime | None = Field(
        default=None, description="UTC timestamp the attempt finished, if finished."
    )
    attempt: int = Field(default=1, ge=1, description="Ordinal attempt number for this node.")
    error: ErrorRecord | None = Field(
        default=None, description="Error record associated with this attempt, if failed."
    )


class QAState(BaseModel):
    """Quality-assurance tracking for the workflow.

    Attributes:
        reports: Structured QA report payloads produced by QA nodes.
        revision_count: Number of revision cycles triggered by QA.
        validation_failures: Human-readable validation failure descriptions.
        approval_status: Current QA approval status.
    """

    reports: list[dict[str, Any]] = Field(
        default_factory=list, description="Structured QA report payloads."
    )
    revision_count: int = Field(
        default=0, ge=0, description="Number of revision cycles triggered by QA."
    )
    validation_failures: list[str] = Field(
        default_factory=list, description="Human-readable validation failure descriptions."
    )
    approval_status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Current QA approval status."
    )


class BudgetState(BaseModel):
    """Budget and spend tracking for the workflow.

    Attributes:
        cost_ledger: The authoritative cost ledger for this execution.
        estimated_spend: Total estimated spend across the plan.
        actual_spend: Total actual spend recorded so far.
        total_budget: Total budget allocated to this workflow execution.
    """

    cost_ledger: CostLedger = Field(description="Authoritative cost ledger for this execution.")
    estimated_spend: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Total estimated spend."
    )
    actual_spend: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Total actual spend recorded so far."
    )
    total_budget: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Total budget allocated."
    )


class ApprovalState(BaseModel):
    """Human approval status for each gated stage of the workflow.

    Attributes:
        business_review: Approval status for the business context review.
        planning_review: Approval status for the weekly plan review.
        creative_review: Approval status for the creative output review.
        final_approval: Approval status for the final release gate.
        history: Chronological record of approval decisions.
    """

    business_review: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Approval status for business review."
    )
    planning_review: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Approval status for planning review."
    )
    creative_review: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Approval status for creative review."
    )
    final_approval: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Approval status for the final gate."
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="Chronological record of approval decisions."
    )

    def status_for(self, stage: ApprovalStage) -> ApprovalStatus:
        """Return the current approval status for the given stage.

        Args:
            stage: The approval stage to look up.

        Returns:
            The current :class:`ApprovalStatus` for that stage.
        """
        return getattr(self, stage.value)


class MarketingState(BaseModel):
    """Complete shared execution state for one MarketingOS workflow run.

    This is the single source of truth passed between every node in the
    LangGraph workflow. It is checkpointed after each node execution and
    should be treated as the canonical record of workflow progress.

    Attributes:
        run_id: Unique identifier for this workflow execution.
        workflow_id: Identifier of the workflow definition being executed.
        status: Overall lifecycle status of the execution.
        current_node: Name of the node currently executing, if any.
        previous_node: Name of the most recently completed node, if any.
        next_node: Name of the node scheduled to execute next, if any.
        created_at: Timestamp the execution was created.
        updated_at: Timestamp the state was last modified.
        source_pack: Extensible container for raw research source material.
        research_notes: Free-form notes captured during research.
        scraped_references: Reference URLs or documents gathered during research.
        business_context: Structured business context for this execution.
        strategy: Strategy output for this execution.
        weekly_plan: Structured weekly content plan.
        post_creatives: Generated post creative outputs.
        video_creatives: Generated video creative outputs.
        qa: Quality-assurance tracking state.
        budget: Budget and spend tracking state.
        approvals: Human approval tracking state.
        current_agent: Name of the agent currently active, if any.
        completed_nodes: Names of nodes that have completed successfully.
        failed_nodes: Names of nodes that have failed.
        retry_count: Number of retries performed for the current node.
        execution_history: Chronological record of node execution attempts.
        messages: Append-only history of system, agent, and user messages.
        warnings: Recorded non-blocking warnings.
        recoverable_errors: Recorded errors that did not halt execution.
        fatal_errors: Recorded errors that halted execution.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # Run Information
    run_id: UUID = Field(default_factory=uuid4, description="Unique execution identifier.")
    workflow_id: str = Field(min_length=1, description="Identifier of the workflow definition.")
    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING, description="Overall lifecycle status."
    )
    current_node: str | None = Field(default=None, description="Currently executing node.")
    previous_node: str | None = Field(default=None, description="Most recently completed node.")
    next_node: str | None = Field(default=None, description="Node scheduled to execute next.")
    created_at: datetime = Field(
        default_factory=_utc_now, description="UTC timestamp the execution was created."
    )
    updated_at: datetime = Field(
        default_factory=_utc_now, description="UTC timestamp the state was last modified."
    )

    # Research
    source_pack: dict[str, Any] = Field(
        default_factory=dict, description="Extensible container for raw research source material."
    )
    research_notes: list[str] = Field(
        default_factory=list, description="Free-form notes captured during research."
    )
    scraped_references: list[str] = Field(
        default_factory=list, description="Reference URLs or documents gathered during research."
    )
    research_result: ResearchResult | None = Field(
        default=None, description="ResearchAgent output: factual observations about the business."
    )
    synthetic_source: SyntheticSourceMaterial | None = Field(
        default=None, description="SyntheticSourceAgent output: synthetic source material."
    )

    # Business Context
    business_context: BusinessContext | None = Field(
        default=None, description="Structured business context for this execution."
    )
    business_analysis: BusinessContext | None = Field(
        default=None, description="BusinessAnalysisAgent output: analyzed business context."
    )

    # Strategy
    strategy: StrategyOutput | None = Field(
        default=None, description="Strategy output for this execution."
    )
    strategy_output: Strategy | None = Field(
        default=None, description="StrategistAgent output: marketing strategy."
    )

    # Weekly Plan
    weekly_plan: WeekPlan | None = Field(
        default=None, description="Structured weekly content plan."
    )
    week_plan: PlannerWeekPlan | None = Field(
        default=None, description="PlannerAgent output: weekly content plan."
    )

    # Creative Outputs
    post_creatives: list[PostCreative] = Field(
        default_factory=list, description="Generated post creative outputs."
    )
    video_creatives: list[VideoCreative] = Field(
        default_factory=list, description="Generated video creative outputs."
    )
    captions: CaptionPackage | None = Field(
        default=None, description="CopywriterAgent output: captions and copy."
    )
    creatives: CreativePackage | None = Field(
        default=None, description="DesignerAgent output: creative package."
    )
    videos: VideoPackage | None = Field(
        default=None, description="VideoDirectorAgent output: video package."
    )

    # QA
    qa: QAState = Field(default_factory=QAState, description="Quality-assurance tracking state.")
    qa_report: QAReport | None = Field(
        default=None, description="QAAgent output: quality assurance report."
    )

    # Budget
    budget: BudgetState = Field(description="Budget and spend tracking state.")

    # Human Approval
    approvals: ApprovalState = Field(
        default_factory=ApprovalState, description="Human approval tracking state."
    )
    campaign_package: CampaignPackage | None = Field(
        default=None, description="PackagingAgent output: final campaign package."
    )

    # Execution
    current_agent: str | None = Field(default=None, description="Currently active agent name.")
    completed_nodes: list[str] = Field(
        default_factory=list, description="Names of nodes that have completed successfully."
    )
    failed_nodes: list[str] = Field(
        default_factory=list, description="Names of nodes that have failed."
    )
    retry_count: int = Field(
        default=0, ge=0, description="Number of retries performed for the current node."
    )
    execution_history: list[NodeExecution] = Field(
        default_factory=list, description="Chronological record of node execution attempts."
    )

    # Messages
    messages: list[MessageRecord] = Field(
        default_factory=list, description="Append-only history of workflow messages."
    )

    # Errors
    warnings: list[ErrorRecord] = Field(
        default_factory=list, description="Recorded non-blocking warnings."
    )
    recoverable_errors: list[ErrorRecord] = Field(
        default_factory=list, description="Recorded errors that did not halt execution."
    )
    fatal_errors: list[ErrorRecord] = Field(
        default_factory=list, description="Recorded errors that halted execution."
    )

    @field_validator("completed_nodes")
    @classmethod
    def _validate_completed_nodes_unique(cls, value: list[str]) -> list[str]:
        """Ensure completed node names contain no duplicates.

        Args:
            value: The proposed list of completed node names.

        Returns:
            The validated list, unchanged.

        Raises:
            ValueError: If duplicate node names are present.
        """
        if len(value) != len(set(value)):
            raise ValueError("completed_nodes must not contain duplicate node names")
        return value

    @field_validator("failed_nodes")
    @classmethod
    def _validate_failed_nodes_unique(cls, value: list[str]) -> list[str]:
        """Ensure failed node names contain no duplicates.

        Args:
            value: The proposed list of failed node names.

        Returns:
            The validated list, unchanged.

        Raises:
            ValueError: If duplicate node names are present.
        """
        if len(value) != len(set(value)):
            raise ValueError("failed_nodes must not contain duplicate node names")
        return value

    @model_validator(mode="after")
    def _touch_updated_at_on_construction(self) -> Self:
        """Ensure ``updated_at`` is never earlier than ``created_at``.

        Returns:
            The validated model instance.
        """
        if self.updated_at < self.created_at:
            self.updated_at = self.created_at
        return self

    def _touch(self) -> None:
        """Refresh the ``updated_at`` timestamp to the current UTC time."""
        self.updated_at = _utc_now()

    def add_message(
        self,
        role: MessageRole,
        content: str,
        node: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageRecord:
        """Append a new entry to the message history.

        Args:
            role: Author role of the message.
            content: Free-text content of the message.
            node: Name of the node producing the message, if any.
            metadata: Arbitrary structured metadata to attach.

        Returns:
            The :class:`MessageRecord` that was appended.
        """
        record = MessageRecord(
            role=role,
            content=content,
            node=node or self.current_node,
            metadata=metadata or {},
        )
        self.messages.append(record)
        self._touch()
        return record

    def record_error(
        self,
        severity: ErrorSeverity,
        message: str,
        node: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ErrorRecord:
        """Record an error, warning, or fatal failure against the state.

        The record is appended to the list matching its severity and, for
        fatal errors, the overall workflow status is updated to ``FAILED``.

        Args:
            severity: Severity classification of the error.
            message: Human-readable description of the error.
            node: Name of the node that raised the error, if any.
            details: Arbitrary structured context about the error.

        Returns:
            The :class:`ErrorRecord` that was appended.
        """
        record = ErrorRecord(
            severity=severity,
            message=message,
            node=node or self.current_node,
            details=details or {},
        )
        if severity is ErrorSeverity.WARNING:
            self.warnings.append(record)
        elif severity is ErrorSeverity.RECOVERABLE:
            self.recoverable_errors.append(record)
        else:
            self.fatal_errors.append(record)
            self.status = WorkflowStatus.FAILED
        self._touch()
        return record

    def mark_node_completed(self, node_name: str) -> None:
        """Mark a node as successfully completed.

        Updates ``completed_nodes``, clears the node from ``failed_nodes``
        if present, appends a completed :class:`NodeExecution` record, and
        advances ``previous_node``/``current_node``.

        Args:
            node_name: Name of the node that completed successfully.
        """
        if node_name not in self.completed_nodes:
            self.completed_nodes.append(node_name)
        if node_name in self.failed_nodes:
            self.failed_nodes.remove(node_name)
        self.execution_history.append(
            NodeExecution(
                node=node_name,
                status=NodeExecutionStatus.COMPLETED,
                completed_at=_utc_now(),
                attempt=self.retry_count + 1,
            )
        )
        self.previous_node = node_name
        if self.current_node == node_name:
            self.current_node = None
        self._touch()

    def mark_node_failed(self, node_name: str, error: ErrorRecord | None = None) -> None:
        """Mark a node as failed.

        Updates ``failed_nodes``, appends a failed :class:`NodeExecution`
        record, and optionally attaches an associated error.

        Args:
            node_name: Name of the node that failed.
            error: Error record describing the failure, if available.
        """
        if node_name not in self.failed_nodes:
            self.failed_nodes.append(node_name)
        self.execution_history.append(
            NodeExecution(
                node=node_name,
                status=NodeExecutionStatus.FAILED,
                completed_at=_utc_now(),
                attempt=self.retry_count + 1,
                error=error,
            )
        )
        self._touch()

    def increment_retry(self) -> int:
        """Increment the retry counter for the current node.

        Returns:
            The updated retry count.
        """
        self.retry_count += 1
        self._touch()
        return self.retry_count

    def approve_stage(self, stage: ApprovalStage, approved: bool, actor: str | None = None) -> None:
        """Record a human approval decision for a workflow stage.

        Args:
            stage: The approval stage being decided.
            approved: Whether the stage was approved (``True``) or rejected
                (``False``).
            actor: Identifier of the person or system making the decision.
        """
        new_status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        setattr(self.approvals, stage.value, new_status)
        self.approvals.history.append(
            {
                "stage": stage.value,
                "status": new_status.value,
                "actor": actor,
                "decided_at": _utc_now().isoformat(),
            }
        )
        self._touch()

    def remaining_budget(self) -> Decimal:
        """Compute the remaining budget for this workflow execution.

        Returns:
            The total budget minus actual spend recorded so far.
        """
        return self.budget.total_budget - self.budget.actual_spend

    def reset_execution(self) -> None:
        """Reset execution-tracking fields to their initial state.

        Clears completed/failed nodes, execution history, retry count, and
        current agent, and resets the overall status to ``PENDING``. Research,
        strategy, plan, creative, QA, budget, approval, message, and error
        state are left untouched.
        """
        self.status = WorkflowStatus.PENDING
        self.current_node = None
        self.previous_node = None
        self.next_node = None
        self.current_agent = None
        self.completed_nodes = []
        self.failed_nodes = []
        self.retry_count = 0
        self.execution_history = []
        self._touch()