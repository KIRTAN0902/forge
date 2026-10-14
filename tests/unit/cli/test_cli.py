"""Unit tests for Forge CLI commands (Phase 7, spec §18-19, §22)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from forge.cli.app import app
from forge.config import ForgeConfig
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import QAReport, QAStatus
from forge.core.contracts.requirements import (
    AcceptanceCriterion,
    RequirementContract,
    VerificationMethod,
)
from forge.core.contracts.state import Checkpoint, WorkflowState, WorkflowStatus
from forge.events.types import Event, EventType
from forge.models import (
    ModelCapabilities,
    ModelProfile,
    ModelRegistry,
)
from forge.models.providers.scripted import ScriptedProvider
from forge.storage.sqlite import SQLiteStore
from forge.tools import create_environment
from forge.workflow.recovery import CheckpointManager


def _scripted_registry(
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


class TestCliCommands:
    def test_version_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "forge" in result.stdout

    def test_status_empty_project(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["status", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert "No workflows found" in result.stdout

    async def test_status_with_saved_workflow(self, tmp_path: Path) -> None:
        store = await SQLiteStore.open(tmp_path / ".forge" / "forge.db")
        plan = Plan(
            goal="Test CLI status",
            tasks=[Task(id="T-1", description="Implement auth", status=TaskStatus.DONE)],
        )
        state = WorkflowState(
            user_request="Build auth service",
            status=WorkflowStatus.RUNNING,
            iteration=2,
            plan=plan,
        )
        await store.save_workflow(state)
        await store.close()

        runner = CliRunner()
        result = runner.invoke(app, ["status", "--path", str(tmp_path)])
        assert result.exit_code == 0
        assert state.workflow_id in result.stdout
        assert "RUNNING" in result.stdout
        assert "T-1" in result.stdout

    async def test_history_list_and_events(self, tmp_path: Path) -> None:
        store = await SQLiteStore.open(tmp_path / ".forge" / "forge.db")
        state = WorkflowState(
            user_request="Refactor math ops",
            status=WorkflowStatus.COMPLETED,
            iteration=4,
        )
        await store.save_workflow(state)
        await store.append_event(
            Event(
                type=EventType.WORKFLOW_STARTED,
                workflow_id=state.workflow_id,
                payload={"user_request": state.user_request},
            )
        )
        await store.append_event(
            Event(
                type=EventType.WORKFLOW_COMPLETED,
                workflow_id=state.workflow_id,
                payload={"status": "completed"},
            )
        )
        await store.close()

        runner = CliRunner()
        # 1. history without workflow_id -> lists workflows
        res_list = runner.invoke(app, ["history", "--path", str(tmp_path)])
        assert res_list.exit_code == 0
        assert state.workflow_id in res_list.stdout
        assert "COMPLETED" in res_list.stdout

        # 2. history with workflow_id -> lists event timeline
        res_events = runner.invoke(app, ["history", state.workflow_id, "--path", str(tmp_path)])
        assert res_events.exit_code == 0
        assert "WorkflowStarted" in res_events.stdout
        assert "WorkflowCompleted" in res_events.stdout

    async def test_diff_command(self, sample_repo: Path) -> None:
        store = await SQLiteStore.open(sample_repo / ".forge" / "forge.db")
        state = WorkflowState(
            user_request="Inspect diff",
            status=WorkflowStatus.RUNNING,
            checkpoints=[Checkpoint(ref="HEAD", label="initial")],
        )
        await store.save_workflow(state)
        await store.close()

        # Mutate a file in sample_repo
        ops_file = sample_repo / "src" / "calc" / "ops.py"
        ops_file.write_text("# Modified ops\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(app, ["diff", state.workflow_id, "--path", str(sample_repo)])
        assert result.exit_code == 0
        assert "Modified ops" in result.stdout

    def test_run_command_end_to_end(self, sample_repo: Path) -> None:
        runner = CliRunner()

        # Orchestrator decision trajectory: PLAN -> COMPLETE
        orch_provider = ScriptedProvider(
            [
                Decision(
                    action=OrchestratorAction.PLAN,
                    rationale="Plan requirements",
                ).model_dump_json(),
                Decision(
                    action=OrchestratorAction.FULL_QA,
                    rationale="Audit plan and execution",
                ).model_dump_json(),
                Decision(
                    action=OrchestratorAction.COMPLETE,
                    rationale="Done",
                ).model_dump_json(),
            ]
        )
        reqs = RequirementContract(
            summary="Test CLI requirements",
            acceptance_criteria=[
                AcceptanceCriterion(
                    id="AC-001",
                    description="Valid criterion",
                    verification=VerificationMethod.AUTOMATED_TEST,
                )
            ],
        )
        plan = Plan(
            goal="CLI plan",
            tasks=[Task(id="TASK-1", description="CLI task", status=TaskStatus.DONE)],
        )
        planner_provider = ScriptedProvider([reqs.model_dump_json(), plan.model_dump_json()])
        executor_provider = ScriptedProvider([])

        qa_report = QAReport(status=QAStatus.PASS, summary="QA verified", criteria=[])
        # QA is called when COMPLETE is proposed with done tasks
        qa_provider = ScriptedProvider(
            ["Investigation complete: all criteria satisfied.", qa_report.model_dump_json()]
        )

        registry = _scripted_registry(
            orch_provider, planner_provider, executor_provider, qa_provider
        )

        with patch("forge.cli.app.build_workflow_harness") as mock_harness:
            from forge.cli.workflow import build_workflow_harness

            async def _create_harness(*args, **kwargs):
                kwargs["registry"] = registry
                return await build_workflow_harness(*args, **kwargs)

            mock_harness.side_effect = _create_harness

            result = runner.invoke(
                app,
                [
                    "run",
                    "Build something via CLI",
                    "--path",
                    str(sample_repo),
                    "-y",
                ],
            )
            assert result.exit_code == 0, result.stdout
            assert "Workflow Started" in result.stdout
            assert "Workflow Completed Successfully" in result.stdout

    async def test_approve_command(self, sample_repo: Path) -> None:
        store = await SQLiteStore.open(sample_repo / ".forge" / "forge.db")
        state = WorkflowState(
            user_request="Refactor database",
            status=WorkflowStatus.AWAITING_APPROVAL,
        )
        await store.save_workflow(state)
        await store.close()

        runner = CliRunner()
        # Approve without resume
        result = runner.invoke(
            app,
            [
                "approve",
                state.workflow_id,
                "--no-resume",
                "--feedback",
                "Approved by admin",
                "--path",
                str(sample_repo),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Approved workflow" in result.stdout

        # Verify state is updated in store
        check_store = await SQLiteStore.open(sample_repo / ".forge" / "forge.db")
        loaded = await check_store.load_workflow(state.workflow_id)
        assert loaded.status == WorkflowStatus.RUNNING
        assert "[Human Guidance]: Approved by admin" in loaded.user_request
        await check_store.close()

    async def test_rollback_command(self, sample_repo: Path) -> None:
        mgr = CheckpointManager()
        runtime_env = create_environment(ForgeConfig(), root=sample_repo)

        # Create checkpoint
        readme = sample_repo / "README.md"
        readme.write_text("# Stable Baseline\n", encoding="utf-8")
        ckpt = await mgr.create_checkpoint(runtime_env, label="stable", iteration=1)
        assert ckpt is not None

        # Mutate file
        readme.write_text("# Corrupted Content\n", encoding="utf-8")

        # Save workflow state with checkpoint
        store = await SQLiteStore.open(sample_repo / ".forge" / "forge.db")
        state = WorkflowState(
            user_request="Update docs",
            status=WorkflowStatus.RUNNING,
            checkpoints=[ckpt],
        )
        await store.save_workflow(state)
        await store.close()
        await runtime_env.aclose()

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "rollback",
                state.workflow_id,
                "--path",
                str(sample_repo),
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "Restored checkpoint" in result.stdout
        assert readme.read_text(encoding="utf-8") == "# Stable Baseline\n"
