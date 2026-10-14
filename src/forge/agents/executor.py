"""Executor specialist agent (spec §5).

Responsible for taking an approved task, reading and modifying files, running tests,
debugging failures, and returning a structured ExecutionResult.
"""

from __future__ import annotations

from collections.abc import Sequence

from forge.agents.base import BaseAgent
from forge.agents.prompts import EXECUTOR_SYSTEM_PROMPT
from forge.context.builder import BuiltContext
from forge.core.contracts.execution import (
    ChangeType,
    ExecutionResult,
    ExecutionStatus,
    FileChange,
    TestRun,
)
from forge.core.contracts.plan import Plan, Task
from forge.core.contracts.qa import QAFinding
from forge.core.contracts.requirements import RequirementContract
from forge.events.bus import EventBus
from forge.events.types import EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import Message
from forge.tools.base import ToolResult
from forge.tools.defaults import ROLE_TOOLSETS
from forge.tools.runtime import ToolRuntime


class ExecutorAgent(BaseAgent):
    """Senior implementation engineer."""

    def __init__(
        self,
        adapter: ModelAdapter,
        runtime: ToolRuntime,
        *,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> None:
        super().__init__(
            role="executor",
            adapter=adapter,
            runtime=runtime,
            events=events,
            workflow_id=workflow_id,
        )

    async def execute(
        self,
        task: Task,
        *,
        context: BuiltContext | None = None,
        plan: Plan | None = None,
        requirements: RequirementContract | None = None,
        findings: Sequence[QAFinding] = (),
        errors: Sequence[str] = (),
        max_steps: int = 15,
    ) -> ExecutionResult:
        """Execute a single task using the tool runtime and return structured results."""
        await self._emit(
            EventType.TASK_STARTED,
            task_id=task.id,
            description=task.description,
            dependencies=task.dependencies,
            acceptance_criteria=task.acceptance_criteria,
        )

        messages = self._build_initial_messages(
            task=task,
            context=context,
            findings=findings,
            errors=errors,
        )

        outcome = await self._run_tool_loop(
            messages,
            allowed_tools=ROLE_TOOLSETS["executor"],
            max_steps=max_steps,
        )

        changes = self._extract_changes(outcome.results)
        test_runs = self._extract_test_runs(outcome.results)

        status = self._determine_status(outcome.steps, max_steps, test_runs, outcome.results)
        summary = outcome.final_text or f"Executed task {task.id} in {outcome.steps} steps."

        result = ExecutionResult(
            task_id=task.id,
            status=status,
            summary=summary,
            changes=changes,
            tests=test_runs,
            iterations=outcome.steps,
        )

        event_type = (
            EventType.TASK_COMPLETED
            if status == ExecutionStatus.COMPLETED
            else EventType.TASK_FAILED
        )
        await self._emit(
            event_type,
            task_id=task.id,
            status=status.value,
            summary=summary,
            changes_count=len(changes),
            tests_count=len(test_runs),
        )

        return result

    def _build_initial_messages(
        self,
        *,
        task: Task,
        context: BuiltContext | None,
        findings: Sequence[QAFinding],
        errors: Sequence[str],
    ) -> list[Message]:
        messages: list[Message] = [
            Message.system(EXECUTOR_SYSTEM_PROMPT),
        ]
        if context is not None:
            messages.append(context.to_message())

        parts = [
            f"You are assigned to implement Task {task.id}:\n"
            f"Description: {task.description}\n"
            f"Acceptance Criteria: {', '.join(task.acceptance_criteria) or 'None specified'}\n"
            f"Expected Files: {', '.join(task.files_hint) or 'None specified'}"
        ]

        if findings:
            finding_lines = [
                f"- [{f.severity}] {f.description} ({', '.join(f.affected_files)})"
                for f in findings
            ]
            parts.append(
                "Previous QA findings that must be addressed:\n" + "\n".join(finding_lines)
            )

        if errors:
            parts.append(
                "Errors encountered in previous attempts:\n"
                + "\n".join(f"- {err}" for err in errors[-5:])
            )

        parts.append(
            "Instructions:\n"
            "1. Read necessary files to understand existing implementations.\n"
            "2. Make precise edits using `edit_file` or `write_file`.\n"
            "3. Run tests using `run_tests` to verify your changes.\n"
            "4. When all criteria are met and tests pass, report your conclusion."
        )

        messages.append(Message.user("\n\n".join(parts)))
        return messages

    def _extract_changes(self, results: list[ToolResult]) -> list[FileChange]:
        """Collect file changes performed by filesystem tools."""
        changes_by_path: dict[str, FileChange] = {}
        for r in results:
            if not r.ok:
                continue
            for path in r.changed_paths:
                change_type = ChangeType.MODIFIED
                if r.tool == "delete_file":
                    change_type = ChangeType.DELETED
                elif r.tool == "write_file":
                    change_type = (
                        ChangeType.MODIFIED if path in changes_by_path else ChangeType.ADDED
                    )
                changes_by_path[path] = FileChange(
                    path=path,
                    change_type=change_type,
                    reason=f"Changed via {r.tool}",
                )
        return list(changes_by_path.values())

    def _extract_test_runs(self, results: list[ToolResult]) -> list[TestRun]:
        """Collect test runs from run_tests tool calls."""
        runs: list[TestRun] = []
        for r in results:
            if r.tool != "run_tests":
                continue
            data = r.data or {}
            runs.append(
                TestRun(
                    command=str(data.get("command", "pytest")),
                    exit_code=int(data.get("exit_code", 0 if r.ok else 1)),
                    passed=data.get("passed"),
                    failed=data.get("failed"),
                    errors=data.get("errors"),
                    duration_s=r.duration_s,
                    output_tail=r.content[-500:],
                )
            )
        return runs

    def _determine_status(
        self,
        steps: int,
        max_steps: int,
        test_runs: list[TestRun],
        results: list[ToolResult],
    ) -> ExecutionStatus:
        """Determine whether the execution completed, failed, or is partial."""
        if test_runs:
            last_run = test_runs[-1]
            if not last_run.succeeded or (last_run.failed and last_run.failed > 0):
                return ExecutionStatus.FAILED

        if steps >= max_steps:
            return ExecutionStatus.PARTIAL

        if results and all(not r.ok for r in results):
            return ExecutionStatus.FAILED

        return ExecutionStatus.COMPLETED
