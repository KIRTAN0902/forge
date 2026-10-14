from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.events.types import Event, EventType


class WorkflowSummary(BaseModel):
    workflow_id: str
    user_request: str
    status: WorkflowStatus
    iteration: int
    created_at: datetime
    updated_at: datetime


@runtime_checkable
class Store(Protocol):
    """Persistence for workflow state and the event log."""

    async def save_workflow(self, state: WorkflowState) -> None: ...

    async def load_workflow(self, workflow_id: str) -> WorkflowState: ...

    async def list_workflows(
        self, *, limit: int = 20, status: WorkflowStatus | None = None
    ) -> list[WorkflowSummary]: ...

    async def append_event(self, event: Event) -> None: ...

    async def list_events(
        self,
        workflow_id: str | None = None,
        *,
        types: Iterable[EventType] | None = None,
        limit: int | None = None,
    ) -> list[Event]: ...

    async def close(self) -> None: ...
