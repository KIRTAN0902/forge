"""What the Executor reports back: exactly what changed, what ran, and why."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from forge.core.contracts.base import Contract


class ChangeType(StrEnum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class FileChange(Contract):
    path: str
    change_type: ChangeType
    reason: str = ""


class TestRun(Contract):
    __test__ = False  # not a pytest class

    command: str
    exit_code: int
    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    duration_s: float | None = None
    output_tail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"


class ExecutionResult(Contract):
    task_id: str
    status: ExecutionStatus
    summary: str
    changes: list[FileChange] = Field(default_factory=list)
    tests: list[TestRun] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    iterations: int = Field(default=0, ge=0)
