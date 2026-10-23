from __future__ import annotations

from pathlib import Path

from forge.agents import ExecutorAgent
from forge.core.contracts.execution import ChangeType, ExecutionStatus
from forge.core.contracts.plan import Task
from forge.events import Event, EventBus, EventType
from forge.models import (
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response

from ...conftest import PYTHON, RuntimeFactory

TEST_CMD = f'"{PYTHON}" -m pytest -q -p no:cacheprovider'


def _make_adapter(provider: ScriptedProvider, events: EventBus | None = None) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model="executor-test")
    return ModelAdapter(
        profile_name="executor",
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role="executor",
        events=events,
        workflow_id="wf_exec",
    )


class TestExecutorAgent:
    async def test_executes_task_modifies_file_and_runs_tests(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        provider = ScriptedProvider(
            [
                tool_call_response("read_file", {"path": "src/calc/ops.py"}),
                tool_call_response(
                    "edit_file",
                    {
                        "path": "src/calc/ops.py",
                        "old_string": "return a - b  # BUG: should add",
                        "new_string": "return a + b",
                    },
                ),
                tool_call_response("run_tests", {"command": TEST_CMD}),
                "Task completed successfully. All tests pass.",
            ]
        )
        adapter = _make_adapter(provider, events=events_bus)
        executor = ExecutorAgent(adapter=adapter, runtime=runtime, events=events_bus)

        task = Task(
            id="TASK-1",
            description="Fix add() operator in ops.py",
            acceptance_criteria=["AC-001"],
            files_hint=["src/calc/ops.py"],
        )

        result = await executor.execute(task, max_steps=5)

        assert result.status == ExecutionStatus.COMPLETED
        assert result.task_id == "TASK-1"
        assert len(result.changes) == 1
        assert result.changes[0].path == "src/calc/ops.py"
        assert result.changes[0].change_type == ChangeType.MODIFIED
        assert len(result.tests) == 1
        assert result.tests[0].succeeded
        assert "return a + b" in (sample_repo / "src/calc/ops.py").read_text()

        event_types = [e.type for e in emitted]
        assert EventType.TASK_STARTED in event_types
        assert EventType.TASK_COMPLETED in event_types

    async def test_fails_when_tests_fail(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        # Do not fix the bug, just run the failing tests
        provider = ScriptedProvider(
            [
                tool_call_response("run_tests", {"command": TEST_CMD}),
                "Tests failed as expected.",
            ]
        )
        adapter = _make_adapter(provider, events=events_bus)
        executor = ExecutorAgent(adapter=adapter, runtime=runtime, events=events_bus)

        task = Task(
            id="TASK-FAIL",
            description="Test that should fail",
        )

        result = await executor.execute(task, max_steps=3)

        assert result.status == ExecutionStatus.FAILED
        assert len(result.tests) == 1
        assert not result.tests[0].succeeded
        event_types = [e.type for e in emitted]
        assert EventType.TASK_STARTED in event_types
        assert EventType.TASK_FAILED in event_types

    async def test_executes_qwen_xml_tool_call(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        provider = ScriptedProvider(
            [
                "<tool_call>\n"
                "<function=write_file>\n"
                "<parameter=path>notes/hello.txt</parameter>\n"
                "<parameter=content>hello from XML\n</parameter>\n"
                "</function>\n"
                "</tool_call>",
                "Created notes/hello.txt.",
            ]
        )
        executor = ExecutorAgent(adapter=_make_adapter(provider), runtime=runtime)

        result = await executor.execute(
            Task(id="TASK-XML", description="Create a file using the XML tool protocol")
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert (sample_repo / "notes" / "hello.txt").read_text(encoding="utf-8") == (
            "hello from XML\n"
        )

    async def test_partial_when_exceeding_max_steps(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        # Model keeps calling read_file repeatedly beyond max_steps
        provider = ScriptedProvider(
            [
                tool_call_response("read_file", {"path": "src/calc/ops.py"}),
                tool_call_response("read_file", {"path": "src/calc/ops.py"}),
                tool_call_response("read_file", {"path": "src/calc/ops.py"}),
            ]
        )
        adapter = _make_adapter(provider)
        executor = ExecutorAgent(adapter=adapter, runtime=runtime)

        task = Task(id="TASK-LOOP", description="Loop task")
        result = await executor.execute(task, max_steps=2)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.iterations == 2
        assert "2-step tool limit" in result.summary
