"""Phase 4 exit condition: each specialist agent independently performs its role.

1. PlannerAgent: analyzes repo, investigates code with read-only tools, and outputs a valid
   RequirementContract and topological Plan.
2. ExecutorAgent: receives an approved task, explores, edits code, runs tests, and produces a
   structured ExecutionResult proving the change works.
3. QAAgent: independently inspects changes, runs tests, detects security/tampering issues,
   and produces a structured QAReport proving acceptance criteria.
"""

from __future__ import annotations

from pathlib import Path

from forge.agents import ExecutorAgent, PlannerAgent, QAAgent
from forge.core.contracts.execution import ChangeType, ExecutionStatus
from forge.core.contracts.plan import Plan, Task
from forge.core.contracts.qa import CriterionVerdict, QAReport, QAStatus
from forge.core.contracts.requirements import (
    AcceptanceCriterion,
    FunctionalRequirement,
    RequirementContract,
    VerificationMethod,
)
from forge.events import Event, EventBus, EventType
from forge.models import (
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
    ModelRegistry,
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response
from forge.policies import StaticApprover

from ..conftest import PYTHON, RuntimeFactory

TEST_CMD = f'"{PYTHON}" -m pytest -q -p no:cacheprovider'


def _adapter(role: str, provider: ScriptedProvider, events: EventBus | None = None) -> ModelAdapter:
    registry = ModelRegistry(
        profiles={
            "m": ModelProfile(
                provider="p",
                model="m",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            )
        }
    )
    registry.register_provider("p", provider)
    return ModelAdapter(
        profile_name="m",
        profile=registry.profiles["m"],
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role=role,
        events=events,
        workflow_id="wf_exit",
    )


async def test_specialist_agents_independently_perform_roles(
    sample_repo: Path, make_runtime: RuntimeFactory
) -> None:
    events: list[Event] = []
    runtime = make_runtime(sample_repo, approver=StaticApprover(True), events=events)
    event_bus = runtime.events

    # -------------------------------------------------------------------------
    # 1. PLANNER AGENT INDEPENDENT EXECUTION
    # -------------------------------------------------------------------------
    reqs_contract = RequirementContract(
        summary="Fix addition operator in calc ops",
        functional=[FunctionalRequirement(id="FR-001", description="add(a, b) must return a + b")],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                description="test_add passes in pytest",
                verification=VerificationMethod.AUTOMATED_TEST,
                requirement_ids=["FR-001"],
            )
        ],
    )
    plan_contract = Plan(
        goal="Fix addition bug in src/calc/ops.py",
        approach="Change '-' to '+' in add function",
        tasks=[
            Task(
                id="TASK-1",
                description="Fix subtraction bug in add() in src/calc/ops.py",
                dependencies=[],
                acceptance_criteria=["AC-001"],
                files_hint=["src/calc/ops.py"],
            )
        ],
        acceptance_criteria=["AC-001"],
        risks=["Regression in existing ops"],
    )

    planner_provider = ScriptedProvider(
        [
            tool_call_response("read_file", {"path": "src/calc/ops.py"}),
            "Investigated ops.py. Ready to formulate plan.",
            reqs_contract.model_dump_json(),
            plan_contract.model_dump_json(),
        ]
    )
    planner_adapter = _adapter("planner", planner_provider, events=event_bus)
    planner = PlannerAgent(planner_adapter, runtime=runtime, events=event_bus)

    plan_result = await planner.plan(
        "The tests fail. Fix the addition operator.",
        investigate=True,
        max_investigation_steps=3,
    )

    assert plan_result.requirements.summary == reqs_contract.summary
    assert len(plan_result.plan.tasks) == 1
    task = plan_result.plan.tasks[0]
    assert task.id == "TASK-1"

    # Verify planner events
    emitted_types = [e.type for e in events]
    assert EventType.REQUIREMENTS_DEFINED in emitted_types
    assert EventType.PLAN_CREATED in emitted_types

    # -------------------------------------------------------------------------
    # 2. EXECUTOR AGENT INDEPENDENT EXECUTION
    # -------------------------------------------------------------------------
    executor_provider = ScriptedProvider(
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
            "Implemented fix. Tests now pass.",
        ]
    )
    executor_adapter = _adapter("executor", executor_provider, events=event_bus)
    executor = ExecutorAgent(executor_adapter, runtime=runtime, events=event_bus)

    exec_result = await executor.execute(
        task=task,
        requirements=plan_result.requirements,
        plan=plan_result.plan,
        max_steps=5,
    )

    assert exec_result.status == ExecutionStatus.COMPLETED
    assert exec_result.task_id == "TASK-1"
    assert len(exec_result.changes) == 1
    assert exec_result.changes[0].path == "src/calc/ops.py"
    assert exec_result.changes[0].change_type == ChangeType.MODIFIED
    assert len(exec_result.tests) == 1
    assert exec_result.tests[0].succeeded
    assert "return a + b" in (sample_repo / "src/calc/ops.py").read_text()

    # Verify executor events
    emitted_types = [e.type for e in events]
    assert EventType.TASK_STARTED in emitted_types
    assert EventType.TASK_COMPLETED in emitted_types

    # -------------------------------------------------------------------------
    # 3. QA AGENT INDEPENDENT EXECUTION
    # -------------------------------------------------------------------------
    qa_report_payload = QAReport(
        status=QAStatus.PASS,
        summary="All tests pass and acceptance criteria AC-001 verified.",
        criteria=[
            CriterionVerdict(
                criterion_id="AC-001",
                satisfied=True,
                evidence="Pytest suite passed with 0 failures.",
            )
        ],
        findings=[],
    )
    qa_provider = ScriptedProvider(
        [
            tool_call_response("run_tests", {"command": TEST_CMD}),
            "Verification successful.",
            qa_report_payload.model_dump_json(),
        ]
    )
    qa_adapter = _adapter("qa", qa_provider, events=event_bus)
    qa = QAAgent(qa_adapter, runtime=runtime, events=event_bus)

    qa_report = await qa.review(
        requirements=plan_result.requirements,
        plan=plan_result.plan,
        changed_files=["src/calc/ops.py"],
        execution_results=[exec_result],
        max_steps=3,
    )

    assert qa_report.status == QAStatus.PASS
    assert len(qa_report.findings) == 0
    assert len(qa_report.criteria) == 1
    assert qa_report.criteria[0].satisfied is True

    # Verify QA events
    emitted_types = [e.type for e in events]
    assert EventType.QA_STARTED in emitted_types
    assert EventType.QA_COMPLETED in emitted_types
