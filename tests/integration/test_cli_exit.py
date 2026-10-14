"""Phase 7 exit condition: complete CLI workflow from terminal to completion.

Demonstrates the full end-to-end user experience through the Forge CLI:
1. `forge init`: initialize project configuration and directory.
2. `forge status`: inspect unstarted repository state.
3. `forge run "..." -y`: orchestrate multi-agent workflow via CLI with live event rendering.
4. `forge status`: view detailed task graph, iteration count, and QA verification.
5. `forge history`: view workflow list and detailed event timeline.
6. `forge diff`: inspect git diff of changes made by the workflow.
7. `forge rollback`: roll back working directory to initial checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from forge.cli.app import app
from forge.cli.workflow import build_workflow_harness
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import CriterionVerdict, QAReport, QAStatus
from forge.core.contracts.requirements import (
    AcceptanceCriterion,
    FunctionalRequirement,
    RequirementContract,
    VerificationMethod,
)
from forge.models import (
    ModelCapabilities,
    ModelProfile,
    ModelRegistry,
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response
from forge.storage.sqlite import SQLiteStore

from ..conftest import PYTHON

TEST_CMD = f'"{PYTHON}" -m pytest -q -p no:cacheprovider'


def _make_registry(
    orch_provider: ScriptedProvider,
    planner_provider: ScriptedProvider,
    executor_provider: ScriptedProvider,
    qa_provider: ScriptedProvider,
) -> ModelRegistry:
    registry = ModelRegistry(
        profiles={
            "orch": ModelProfile(
                provider="p_orch",
                model="m",
                capabilities=ModelCapabilities(structured_output=True),
                detect_capabilities=False,
            ),
            "planner": ModelProfile(
                provider="p_plan",
                model="m",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            ),
            "executor": ModelProfile(
                provider="p_exec",
                model="m",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            ),
            "qa": ModelProfile(
                provider="p_qa",
                model="m",
                capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
                detect_capabilities=False,
            ),
        },
        roles={
            "orchestrator": "orch",
            "planner": "planner",
            "executor": "executor",
            "qa": "qa",
        },
    )
    registry.register_provider("p_orch", orch_provider)
    registry.register_provider("p_plan", planner_provider)
    registry.register_provider("p_exec", executor_provider)
    registry.register_provider("p_qa", qa_provider)
    return registry


def test_cli_end_to_end_exit_condition(sample_repo: Path) -> None:
    runner = CliRunner()

    # Step 1: Initialize project via CLI
    init_res = runner.invoke(app, ["init", str(sample_repo)])
    assert init_res.exit_code == 0
    assert (sample_repo / "forge.yaml").is_file()
    assert (sample_repo / ".forge").is_dir()

    # Step 2: Check status before running anything
    status_empty = runner.invoke(app, ["status", "--path", str(sample_repo)])
    assert status_empty.exit_code == 0
    assert "No workflows found" in status_empty.stdout

    # Step 3: Configure multi-agent scripted trajectory for execution
    orch_provider = ScriptedProvider(
        [
            Decision(
                action=OrchestratorAction.PLAN,
                rationale="Analyze requirements and build task plan.",
            ).model_dump_json(),
            Decision(
                action=OrchestratorAction.EXECUTE,
                rationale="Implement task TASK-1 in calc ops.",
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

    reqs_payload = RequirementContract(
        summary="Fix add operator bug in calc ops",
        functional=[FunctionalRequirement(id="FR-001", description="add(a, b) returns a + b")],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                description="add(2, 3) evaluates to 5 without regression",
                verification=VerificationMethod.AUTOMATED_TEST,
            )
        ],
    )
    plan_payload = Plan(
        goal="Fix addition bug in ops.py",
        tasks=[
            Task(
                id="TASK-1",
                description="Fix add return value to a + b in ops.py",
                status=TaskStatus.PENDING,
            )
        ],
        acceptance_criteria=["AC-001"],
    )
    planner_provider = ScriptedProvider(
        [reqs_payload.model_dump_json(), plan_payload.model_dump_json()]
    )

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
            "Fixed addition implementation; unit tests pass.",
        ]
    )

    qa_payload = QAReport(
        status=QAStatus.PASS,
        summary="Verified criteria AC-001; pytest passed cleanly.",
        criteria=[
            CriterionVerdict(
                criterion_id="AC-001",
                satisfied=True,
                evidence="pytest tests/test_ops.py passed with 0 failures.",
            )
        ],
        findings=[],
    )
    qa_provider = ScriptedProvider(
        [
            tool_call_response("run_tests", {"command": TEST_CMD}),
            "QA verification passed.",
            qa_payload.model_dump_json(),
        ]
    )

    registry = _make_registry(orch_provider, planner_provider, executor_provider, qa_provider)

    # Patch harness construction to inject our scripted model registry
    with patch("forge.cli.app.build_workflow_harness") as mock_harness:

        async def _create_harness(*args: object, **kwargs: object) -> object:
            kwargs["registry"] = registry
            return await build_workflow_harness(*args, **kwargs)  # type: ignore[arg-type]

        mock_harness.side_effect = _create_harness

        # Execute workflow through CLI
        run_res = runner.invoke(
            app,
            [
                "run",
                "Fix addition bug in calculator",
                "--path",
                str(sample_repo),
                "-y",
            ],
        )
        assert run_res.exit_code == 0, run_res.stdout
        assert "Workflow Started" in run_res.stdout
        assert "Plan Created" in run_res.stdout
        assert "Task Completed: TASK-1" in run_res.stdout
        assert "QA Audit Completed: PASS" in run_res.stdout
        assert "Workflow Completed Successfully" in run_res.stdout

    # Verify ops.py was modified
    ops_code = (sample_repo / "src" / "calc" / "ops.py").read_text(encoding="utf-8")
    assert "return a + b" in ops_code

    # Step 4: Verify detailed workflow status via CLI
    status_res = runner.invoke(app, ["status", "--path", str(sample_repo)])
    assert status_res.exit_code == 0
    assert "Task Graph" in status_res.stdout
    assert "TASK-1" in status_res.stdout
    assert "Fix add return" in status_res.stdout
    assert "QA Verification" in status_res.stdout

    # Extract workflow ID from the store
    import asyncio

    async def _get_wid() -> str:
        s = await SQLiteStore.open(sample_repo / ".forge" / "forge.db")
        try:
            wfs = await s.list_workflows(limit=1)
            return wfs[0].workflow_id
        finally:
            await s.close()

    wid = asyncio.run(_get_wid())

    # Step 5: Verify history commands
    # List workflows
    hist_res = runner.invoke(app, ["history", "--path", str(sample_repo)])
    assert hist_res.exit_code == 0
    assert wid in hist_res.stdout
    assert "COMPLETED" in hist_res.stdout

    # Timeline of events
    timeline_res = runner.invoke(app, ["history", wid, "--path", str(sample_repo)])
    assert timeline_res.exit_code == 0
    assert "WorkflowStarted" in timeline_res.stdout
    assert "PlanCreated" in timeline_res.stdout
    assert "TaskCompleted" in timeline_res.stdout
    assert "QACompleted" in timeline_res.stdout
    assert "WorkflowCompleted" in timeline_res.stdout

    # Step 6: Verify diff command
    diff_res = runner.invoke(app, ["diff", wid, "--path", str(sample_repo)])
    assert diff_res.exit_code == 0
    assert "return a + b" in diff_res.stdout

    # Step 7: Verify rollback command restores original tree
    rollback_res = runner.invoke(app, ["rollback", wid, "--path", str(sample_repo)])
    assert rollback_res.exit_code == 0
    assert "Restored checkpoint" in rollback_res.stdout

    reverted_code = (sample_repo / "src" / "calc" / "ops.py").read_text(encoding="utf-8")
    assert "return a - b  # BUG: should add" in reverted_code
