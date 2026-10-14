"""Structured contracts shared by agents, the orchestrator, storage, and interfaces."""

from forge.core.contracts.base import Contract
from forge.core.contracts.decisions import TERMINAL_ACTIONS, Decision, OrchestratorAction
from forge.core.contracts.execution import (
    ChangeType,
    ExecutionResult,
    ExecutionStatus,
    FileChange,
    TestRun,
)
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import (
    BLOCKING_SEVERITIES,
    CriterionVerdict,
    FindingType,
    QAFinding,
    QAReport,
    QAScope,
    QAStatus,
    Severity,
)
from forge.core.contracts.requirements import (
    AcceptanceCriterion,
    FunctionalRequirement,
    NFRCategory,
    NonFunctionalRequirement,
    Priority,
    RequirementContract,
    VerificationMethod,
)
from forge.core.contracts.state import (
    TERMINAL_STATUSES,
    Checkpoint,
    ErrorRecord,
    RepositorySnapshot,
    UsageTotals,
    WorkflowState,
    WorkflowStatus,
)

__all__ = [
    "BLOCKING_SEVERITIES",
    "TERMINAL_ACTIONS",
    "TERMINAL_STATUSES",
    "AcceptanceCriterion",
    "ChangeType",
    "Checkpoint",
    "Contract",
    "CriterionVerdict",
    "Decision",
    "ErrorRecord",
    "ExecutionResult",
    "ExecutionStatus",
    "FileChange",
    "FindingType",
    "FunctionalRequirement",
    "NFRCategory",
    "NonFunctionalRequirement",
    "OrchestratorAction",
    "Plan",
    "Priority",
    "QAFinding",
    "QAReport",
    "QAScope",
    "QAStatus",
    "RepositorySnapshot",
    "RequirementContract",
    "Severity",
    "Task",
    "TaskStatus",
    "TestRun",
    "UsageTotals",
    "VerificationMethod",
    "WorkflowState",
    "WorkflowStatus",
]
