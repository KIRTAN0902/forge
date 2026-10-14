"""Shared workflow state.

Agents are mostly stateless workers; the orchestrator owns this object and it is the
single source of workflow truth. It is persisted after every step, so a workflow can be
resumed from storage.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from forge.core.contracts.base import Contract
from forge.core.contracts.decisions import Decision
from forge.core.contracts.execution import ExecutionResult, TestRun
from forge.core.contracts.plan import Plan, Task
from forge.core.contracts.qa import QAReport
from forge.core.contracts.requirements import RequirementContract
from forge.core.ids import new_id
from forge.core.time import utcnow


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
)


class RepositorySnapshot(Contract):
    root: str
    vcs: str | None = None
    branch: str | None = None
    head: str | None = None
    dirty: bool = False
    summary: str = ""


class Checkpoint(Contract):
    id: str = Field(default_factory=lambda: new_id("ckpt"))
    ref: str = Field(description="VCS reference that restores this checkpoint.")
    label: str = ""
    iteration: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class ErrorRecord(Contract):
    source: str
    message: str
    signature: str | None = Field(
        default=None, description="Normalized fingerprint used to detect repeated failures."
    )
    iteration: int = 0
    at: datetime = Field(default_factory=utcnow)


class UsageTotals(Contract):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0


class WorkflowState(Contract):
    workflow_id: str = Field(default_factory=lambda: new_id("wf"))
    user_request: str = Field(min_length=1)
    status: WorkflowStatus = WorkflowStatus.PENDING
    iteration: int = Field(default=0, ge=0)

    requirements: RequirementContract | None = None
    repository_snapshot: RepositorySnapshot | None = None
    architecture: str | None = Field(default=None, description="Planner's architecture notes.")
    plan: Plan | None = None
    current_task_id: str | None = None

    execution_results: list[ExecutionResult] = Field(default_factory=list)
    test_results: list[TestRun] = Field(default_factory=list)
    qa_reports: list[QAReport] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)

    usage: UsageTotals = Field(default_factory=UsageTotals)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def tasks(self) -> list[Task]:
        return self.plan.tasks if self.plan else []

    @property
    def current_task(self) -> Task | None:
        if self.plan is None or self.current_task_id is None:
            return None
        return self.plan.task(self.current_task_id)

    @property
    def latest_qa(self) -> QAReport | None:
        return self.qa_reports[-1] if self.qa_reports else None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def touch(self) -> None:
        self.updated_at = utcnow()
