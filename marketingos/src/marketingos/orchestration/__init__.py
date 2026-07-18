"""Orchestration layer for MarketingOS LangGraph workflows.

This package assembles autonomous agents into a typed LangGraph workflow
that executes a marketing campaign from business research through final
packaging. It manages state transitions, approval gates, checkpointing,
and conditional routing between pipeline stages.
"""

from __future__ import annotations

from .approval_gates import (
    ApprovalAlreadyFinalizedError,
    ApprovalDecision,
    ApprovalGate,
    ApprovalGateError,
    ApprovalManager,
    DuplicateApprovalError,
    InvalidApprovalTransitionError,
    UnknownApprovalStageError,
    build_approval_gate_nodes,
    list_approval_gate_names,
)
from .checkpointer import (
    BaseCheckpointManager,
    Checkpoint,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointSerializationError,
    CheckpointStorageError,
    DuplicateCheckpointError,
    InvalidCheckpointDataError,
    JSONStateSerializer,
    MemoryCheckpointManager,
    SQLiteCheckpointManager,
    StateSerializer,
    build_checkpointer,
)
from .edges import (
    ApprovalRoutingTargets,
    EdgeRouter,
    NodeName,
)
from .graph import (
    ConditionalEdgeDefinition,
    GraphAssembly,
    GraphAssemblyError,
    GraphBuilder,
    NodeDefinition,
)
from .state import (
    ApprovalStage,
    ApprovalState,
    ApprovalStatus,
    BudgetState,
    ErrorRecord,
    ErrorSeverity,
    MarketingState,
    MessageRecord,
    MessageRole,
    NodeExecution,
    NodeExecutionStatus,
    QAState,
    WorkflowStatus,
)

__all__ = [
    "ApprovalAlreadyFinalizedError",
    "ApprovalDecision",
    "ApprovalGate",
    "ApprovalGateError",
    "ApprovalManager",
    "ApprovalRoutingTargets",
    "ApprovalStage",
    "ApprovalState",
    "ApprovalStatus",
    "BaseCheckpointManager",
    "BudgetState",
    "Checkpoint",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointSerializationError",
    "CheckpointStorageError",
    "ConditionalEdgeDefinition",
    "DuplicateApprovalError",
    "DuplicateCheckpointError",
    "EdgeRouter",
    "ErrorRecord",
    "ErrorSeverity",
    "GraphAssembly",
    "GraphAssemblyError",
    "GraphBuilder",
    "InvalidApprovalTransitionError",
    "InvalidCheckpointDataError",
    "JSONStateSerializer",
    "MarketingState",
    "MemoryCheckpointManager",
    "MessageRecord",
    "MessageRole",
    "NodeDefinition",
    "NodeExecution",
    "NodeExecutionStatus",
    "NodeName",
    "QAState",
    "SQLiteCheckpointManager",
    "StateSerializer",
    "UnknownApprovalStageError",
    "WorkflowStatus",
    "build_approval_gate_nodes",
    "build_checkpointer",
    "list_approval_gate_names",
]
