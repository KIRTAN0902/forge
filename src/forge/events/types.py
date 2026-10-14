from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator
from pydantic_core import to_jsonable_python

from forge.core.contracts.base import Contract
from forge.core.ids import new_id
from forge.core.time import utcnow


class EventType(StrEnum):
    # Workflow lifecycle
    WORKFLOW_STARTED = "WorkflowStarted"
    WORKFLOW_RESUMED = "WorkflowResumed"
    WORKFLOW_COMPLETED = "WorkflowCompleted"
    WORKFLOW_BLOCKED = "WorkflowBlocked"
    WORKFLOW_FAILED = "WorkflowFailed"
    # Requirements and planning
    REPOSITORY_ANALYZED = "RepositoryAnalyzed"
    REQUIREMENTS_DEFINED = "RequirementsDefined"
    PLAN_CREATED = "PlanCreated"
    PLAN_APPROVED = "PlanApproved"
    PLAN_REVISED = "PlanRevised"
    # Orchestration
    DECISION_MADE = "DecisionMade"
    # Execution
    TASK_STARTED = "TaskStarted"
    TASK_COMPLETED = "TaskCompleted"
    TASK_FAILED = "TaskFailed"
    TOOL_CALLED = "ToolCalled"
    TOOL_COMPLETED = "ToolCompleted"
    FILE_MODIFIED = "FileModified"
    TEST_STARTED = "TestStarted"
    TEST_COMPLETED = "TestCompleted"
    # QA
    QA_STARTED = "QAStarted"
    QA_FINDING_CREATED = "QAFindingCreated"
    QA_COMPLETED = "QACompleted"
    REWORK_REQUESTED = "ReworkRequested"
    # Safety and recovery
    POLICY_EVALUATED = "PolicyEvaluated"
    APPROVAL_REQUESTED = "ApprovalRequested"
    APPROVAL_RESOLVED = "ApprovalResolved"
    CHECKPOINT_CREATED = "CheckpointCreated"
    ROLLBACK_PERFORMED = "RollbackPerformed"
    # Models
    MODEL_CALLED = "ModelCalled"


class Event(Contract):
    id: str = Field(default_factory=lambda: new_id("evt"))
    type: EventType
    workflow_id: str | None = None
    source: str = "engine"
    payload: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=utcnow)

    @field_validator("payload", mode="before")
    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        # Events are persisted and streamed to future UIs, so payloads must be JSON-safe.
        return to_jsonable_python(value)
