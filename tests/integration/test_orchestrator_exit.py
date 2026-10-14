"""Phase 5 exit condition: a full workflow can run without hardcoding every scenario.

The LLM Orchestrator controls the full multi-agent cycle:
1. Orchestrator decides PLAN -> Planner formulates requirements and task graph.
2. Orchestrator decides EXECUTE -> Executor modifies ops.py and verifies with tests.
3. Orchestrator decides FULL_QA -> QA independently audits the changes and passes criteria.
4. Orchestrator decides COMPLETE -> Workflow concludes in COMPLETED status.
"""

from __future__ import annotations

from pathlib import Path

from forge.agents import ExecutorAgent, PlannerAgent, QAAgent
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import CriterionVerdict, QAReport, QAStatus
from forge.core.contracts.requirements import (
    AcceptanceCriterion,
    FunctionalRequirement,
    RequirementContract,
    VerificationMethod,
)
from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.events import Event, EventBus, EventType
from forge.models import (
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
    ModelRegistry,
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response
from forge.policies import StaticApprover
from forge.storage.sqlite import SQLiteStore
from forge.workflow import AgentRouter, Orchestrator

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
        workflow_id="wf_full",
    )


async def test_full_workflow_driven_by_orchestrator(
    sample_repo: Path, make_runtime: RuntimeFactory, store: SQLiteStore
) -> None:
    events: list[Event] = []
    runtime = make_runtime(sample_repo, approver=StaticApprover(True), events=events)
    event_bus = runtime.events

    # 1. Orchestrator decisions stream: PLAN -> EXECUTE -> FULL_QA -> COMPLETE
    orch_provider = ScriptedProvider(
        [
            Decision(
                action=OrchestratorAction.PLAN,
                rationale="Analyze requirements and build task plan.",
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.EXECUTE,
                rationale="Implement task TASK-1.",
                task_ids=["TASK-1"],
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.FULL_QA,
                rationale="Verify all criteria and run regression tests.",
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.COMPLETE,
                rationale="All tasks done and QA passed.",
            ).model_dump_json(),
        ]
    )

    # 2. Planner returns requirements and plan
    reqs_payload = RequirementContract(
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
    plan_payload = Plan(
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
    )
    planner_provider = ScriptedProvider(
        [reqs_payload.model_dump_json(), plan_payload.model_dump_json()]
    )

    # 3. Executor edits file and runs tests
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
            "Fixed add function and tests pass.",
        ]
    )

    # 4. QA verifies criteria and tests
    qa_payload = QAReport(
        status=QAStatus.PASS,
        summary="Verified criteria AC-001; pytest passed with 0 errors.",
        criteria=[
            CriterionVerdict(
                criterion_id="AC-001",
                satisfied=True,
                evidence="pytest tests/test_ops.py passed.",
            )
        ],
        findings=[],
    )
    qa_provider = ScriptedProvider(
        [
            tool_call_response("run_tests", {"command": TEST_CMD}),
            "QA tests passed cleanly.",
            qa_payload.model_dump_json(),
        ]
    )

    planner = PlannerAgent(_adapter("planner", planner_provider, event_bus), runtime=runtime)
    executor = ExecutorAgent(_adapter("executor", executor_provider, event_bus), runtime=runtime)
    qa = QAAgent(_adapter("qa", qa_provider, event_bus), runtime=runtime)

    router = AgentRouter(
        planner=planner,
        executor=executor,
        qa=qa,
        runtime=runtime,
        events=event_bus,
    )

    orchestrator = Orchestrator(
        adapter=_adapter("orchestrator", orch_provider, event_bus),
        router=router,
        store=store,
        events=event_bus,
        max_iterations=10,
    )

    state = WorkflowState(user_request="The tests fail. Fix the addition operator in ops.py.")
    final_state = await orchestrator.run(state)

    # Assert final status and completion
    assert final_state.status == WorkflowStatus.COMPLETED
    assert final_state.iteration == 4
    assert len(final_state.decisions) == 4
    assert [d.action for d in final_state.decisions] == [
        OrchestratorAction.PLAN,
        OrchestratorAction.EXECUTE,
        OrchestratorAction.FULL_QA,
        OrchestratorAction.COMPLETE,
    ]

    # Task and QA state
    assert final_state.plan is not None
    assert final_state.plan.task("TASK-1").status == TaskStatus.DONE
    assert final_state.latest_qa is not None
    assert final_state.latest_qa.status == QAStatus.PASS

    # Code on disk
    assert "return a + b" in (sample_repo / "src/calc/ops.py").read_text()

    # SQLite persistence
    persisted = await store.load_workflow(final_state.workflow_id)
    assert persisted.status == WorkflowStatus.COMPLETED
    assert persisted.iteration == 4

    # Events lifecycle
    emitted_types = [e.type for e in events]
    assert EventType.WORKFLOW_STARTED in emitted_types
    assert EventType.DECISION_MADE in emitted_types
    assert EventType.WORKFLOW_COMPLETED in emitted_types
