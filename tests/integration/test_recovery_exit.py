"""Phase 6 exit condition: Forge can autonomously complete a task and recover from QA failures.

Demonstrates the autonomous recovery loop:
1. Orchestrator decides PLAN -> Planner creates requirement contract and task plan.
2. Orchestrator decides EXECUTE -> Checkpoint is taken, Executor makes initial modification.
3. Orchestrator decides FULL_QA -> QA identifies a defect, returning QAStatus.FAIL with
   blocking findings and rework instructions (emitting ReworkRequested).
4. Orchestrator decides REEXECUTE -> Executor receives QA findings and fixes the defect.
5. Orchestrator decides FULL_QA -> QA verifies the fix and passes all acceptance criteria.
6. Orchestrator decides COMPLETE -> Workflow completes successfully.
"""

from __future__ import annotations

from pathlib import Path

from forge.agents import ExecutorAgent, PlannerAgent, QAAgent
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import (
    CriterionVerdict,
    FindingType,
    QAFinding,
    QAReport,
    QAStatus,
    Severity,
)
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
        workflow_id="wf_recovery",
    )


async def test_autonomous_recovery_from_qa_failure(
    sample_repo: Path, make_runtime: RuntimeFactory, store: SQLiteStore
) -> None:
    events: list[Event] = []
    runtime = make_runtime(sample_repo, approver=StaticApprover(True), events=events)
    event_bus = runtime.events

    # 1. Orchestrator scripted decision trajectory:
    # Turn 1: PLAN
    # Turn 2: EXECUTE (initial attempt)
    # Turn 3: FULL_QA (identifies bug)
    # Turn 4: REEXECUTE (rework based on findings)
    # Turn 5: FULL_QA (second review, passes)
    # Turn 6: COMPLETE
    orch_provider = ScriptedProvider(
        [
            Decision(
                action=OrchestratorAction.PLAN,
                rationale="Analyze requirements and build task plan.",
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.EXECUTE,
                rationale="Implement division enhancement in TASK-1.",
                task_ids=["TASK-1"],
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.FULL_QA,
                rationale="Run QA audit on initial implementation.",
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.REEXECUTE,
                rationale="Address blocking QA finding QA-BUG-1.",
                task_ids=["TASK-1"],
                finding_ids=["QA-BUG-1"],
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.FULL_QA,
                rationale="Re-run QA audit to verify rework.",
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.COMPLETE,
                rationale="All criteria verified, QA passed with zero findings.",
            ).model_dump_json(),
        ]
    )

    # 2. Planner returns requirements and plan
    reqs_payload = RequirementContract(
        summary="Enhance divide function in calc ops",
        functional=[
            FunctionalRequirement(id="FR-001", description="divide(a, b) must handle floats")
        ],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                description="divide returns float division",
                verification=VerificationMethod.AUTOMATED_TEST,
                requirement_ids=["FR-001"],
            )
        ],
    )
    plan_payload = Plan(
        goal="Ensure divide function handles decimal inputs and checks",
        approach="Update divide in src/calc/ops.py",
        tasks=[
            Task(
                id="TASK-1",
                description="Refactor divide in ops.py",
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

    # 3. Executor provider:
    # First execution attempt: fixes add() and edits divide() but introduces a precision flaw
    # Second execution attempt (rework): edits divide() to fix the defect
    executor_provider = ScriptedProvider(
        [
            # --- First attempt (Turn 2) ---
            tool_call_response("read_file", {"path": "src/calc/ops.py"}),
            tool_call_response(
                "edit_file",
                {
                    "path": "src/calc/ops.py",
                    "old_string": (
                        "def add(a: int, b: int) -> int:\n"
                        "    return a - b  # BUG: should add\n\n\n"
                        "def divide(a: float, b: float) -> float:\n"
                        "    if b == 0:\n"
                        "        raise ZeroDivisionError('division by zero')\n"
                        "    return a / b"
                    ),
                    "new_string": (
                        "def add(a: int, b: int) -> int:\n"
                        "    return a + b\n\n\n"
                        "def divide(a: float, b: float) -> float:\n"
                        "    if b == 0:\n"
                        "        raise ZeroDivisionError('division by zero')\n"
                        "    return round(a / b, 2)  # Flaw: premature rounding"
                    ),
                },
            ),
            tool_call_response("run_tests", {"command": TEST_CMD}),
            "Initial implementation complete with rounding.",
            # --- Second attempt / rework (Turn 4) ---
            tool_call_response("read_file", {"path": "src/calc/ops.py"}),
            tool_call_response(
                "edit_file",
                {
                    "path": "src/calc/ops.py",
                    "old_string": "    return round(a / b, 2)  # Flaw: premature rounding",
                    "new_string": "    return float(a / b)",
                },
            ),
            tool_call_response("run_tests", {"command": TEST_CMD}),
            "Rework complete: removed premature rounding and ensured float division.",
        ]
    )

    # 4. QA provider:
    # First review (Turn 3): fails with blocking finding QA-BUG-1
    # Second review (Turn 5): passes all criteria
    qa_fail_report = QAReport(
        status=QAStatus.FAIL,
        summary="Premature rounding detected in divide implementation.",
        criteria=[
            CriterionVerdict(
                criterion_id="AC-001",
                satisfied=False,
                evidence="Premature rounding loses precision on floating-point division.",
            )
        ],
        findings=[
            QAFinding(
                id="QA-BUG-1",
                severity=Severity.P1,
                type=FindingType.FUNCTIONAL_BUG,
                description="Premature rounding truncates precision to 2 decimal places.",
                affected_files=["src/calc/ops.py"],
                required_rework=["Remove round(...) call and preserve float precision."],
            )
        ],
    )
    qa_pass_report = QAReport(
        status=QAStatus.PASS,
        summary="All acceptance criteria verified; regression tests pass cleanly.",
        criteria=[
            CriterionVerdict(
                criterion_id="AC-001",
                satisfied=True,
                evidence="Full float precision preserved without premature rounding.",
            )
        ],
        findings=[],
    )
    qa_provider = ScriptedProvider(
        [
            # --- Turn 3 QA ---
            tool_call_response("run_tests", {"command": TEST_CMD}),
            "Identified precision truncation bug in divide.",
            qa_fail_report.model_dump_json(),
            # --- Turn 5 QA ---
            tool_call_response("run_tests", {"command": TEST_CMD}),
            "Verified float division fix cleanly.",
            qa_pass_report.model_dump_json(),
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

    state = WorkflowState(user_request="Refactor the divide operation in calc ops.")
    final_state = await orchestrator.run(state)

    # 1. Assert workflow completion
    assert final_state.status == WorkflowStatus.COMPLETED
    assert final_state.iteration == 6
    assert len(final_state.decisions) == 6
    assert [d.action for d in final_state.decisions] == [
        OrchestratorAction.PLAN,
        OrchestratorAction.EXECUTE,
        OrchestratorAction.FULL_QA,
        OrchestratorAction.REEXECUTE,
        OrchestratorAction.FULL_QA,
        OrchestratorAction.COMPLETE,
    ]
    assert final_state.plan is not None
    assert final_state.plan.task("TASK-1").status == TaskStatus.DONE

    # 2. Assert QA state recovery
    assert len(final_state.qa_reports) == 2
    assert final_state.qa_reports[0].status == QAStatus.FAIL
    assert len(final_state.qa_reports[0].blocking_findings) == 1
    assert final_state.latest_qa is not None
    assert final_state.latest_qa.status == QAStatus.PASS
    assert len(final_state.latest_qa.findings) == 0

    # 3. Assert checkpoints were captured before execution
    assert len(final_state.checkpoints) >= 2
    checkpoint_labels = [c.label for c in final_state.checkpoints]
    assert any("task_TASK-1_iter_2" in label for label in checkpoint_labels)

    # 4. Assert disk state contains fixed code
    ops_code = (sample_repo / "src" / "calc" / "ops.py").read_text(encoding="utf-8")
    assert "return float(a / b)" in ops_code
    assert "round(" not in ops_code

    # 5. Assert SQLite persistence
    persisted = await store.load_workflow(final_state.workflow_id)
    assert persisted.status == WorkflowStatus.COMPLETED
    assert persisted.iteration == 6
    assert len(persisted.qa_reports) == 2

    # 6. Assert recovery events emitted
    emitted_types = [e.type for e in events]
    assert EventType.WORKFLOW_STARTED in emitted_types
    assert EventType.CHECKPOINT_CREATED in emitted_types
    assert EventType.REWORK_REQUESTED in emitted_types
    assert EventType.DECISION_MADE in emitted_types
    assert EventType.WORKFLOW_COMPLETED in emitted_types
