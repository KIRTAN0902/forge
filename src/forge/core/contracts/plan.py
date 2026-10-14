"""The Planner's primary output: a dependency-ordered task graph."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from forge.core.contracts.base import Contract


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


_FINISHED = {TaskStatus.DONE, TaskStatus.SKIPPED}


class Task(Contract):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Acceptance criterion ids this task contributes to."
    )
    files_hint: list[str] = Field(
        default_factory=list, description="Files the planner expects to be touched."
    )
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)


class Plan(Contract):
    goal: str = Field(min_length=1)
    approach: str = ""
    tasks: list[Task] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Acceptance criterion ids the plan claims to cover."
    )

    @model_validator(mode="after")
    def _check_graph(self) -> Plan:
        ids = [t.id for t in self.tasks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate task ids: {dupes}")
        known = set(ids)
        for task in self.tasks:
            missing = [d for d in task.dependencies if d not in known]
            if missing:
                raise ValueError(f"task {task.id} depends on unknown tasks: {missing}")
            if task.id in task.dependencies:
                raise ValueError(f"task {task.id} depends on itself")
        self.topological_order()  # raises on cycles
        return self

    def task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def topological_order(self) -> list[Task]:
        """Tasks ordered so dependencies come first; stable w.r.t. declaration order."""
        by_id = {t.id: t for t in self.tasks}
        order: list[Task] = []
        state: dict[str, int] = {}  # 1 = visiting, 2 = done

        def visit(task_id: str, path: list[str]) -> None:
            mark = state.get(task_id)
            if mark == 2:
                return
            if mark == 1:
                cycle = " -> ".join([*path[path.index(task_id) :], task_id])
                raise ValueError(f"dependency cycle: {cycle}")
            state[task_id] = 1
            for dep in by_id[task_id].dependencies:
                visit(dep, [*path, task_id])
            state[task_id] = 2
            order.append(by_id[task_id])

        for task in self.tasks:
            visit(task.id, [])
        return order

    def ready_tasks(self) -> list[Task]:
        """Pending tasks whose dependencies are all finished."""
        finished = {t.id for t in self.tasks if t.status in _FINISHED}
        return [
            t
            for t in self.topological_order()
            if t.status == TaskStatus.PENDING and all(d in finished for d in t.dependencies)
        ]

    @property
    def is_complete(self) -> bool:
        return all(t.status in _FINISHED for t in self.tasks)
