from __future__ import annotations

from pathlib import Path

from forge.core.contracts.state import WorkflowState
from forge.events.bus import EventBus
from forge.events.types import Event, EventType
from forge.models import ModelAdapter, ModelCapabilities, ModelProfile
from forge.models.providers.scripted import ScriptedProvider
from forge.workflow.orchestrator import Orchestrator
from forge.workflow.recovery import CheckpointManager
from forge.workflow.router import AgentRouter

from ...conftest import RuntimeFactory


def _make_adapter(role: str) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model="test-model")
    return ModelAdapter(
        profile_name=role,
        profile=profile,
        provider=ScriptedProvider([]),
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role=role,
        workflow_id="wf_rec",
    )


class TestRecovery:
    async def test_checkpoint_create_and_rollback(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        mgr = CheckpointManager()

        readme = sample_repo / "README.md"
        readme.write_text("# Checkpoint Version 1\n", encoding="utf-8")

        ckpt = await mgr.create_checkpoint(runtime.env, label="step_1", iteration=1)
        assert ckpt is not None
        assert ckpt.label == "step_1"
        assert ckpt.iteration == 1

        # Modify the file further
        readme.write_text("# Broken Modification\n", encoding="utf-8")
        assert readme.read_text(encoding="utf-8") == "# Broken Modification\n"

        # Roll back to checkpoint
        result = await mgr.rollback(runtime.env, ckpt.id)
        assert result is not None
        assert readme.read_text(encoding="utf-8") == "# Checkpoint Version 1\n"

    async def test_rollback_to_latest(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        mgr = CheckpointManager()

        target = sample_repo / "src" / "calc" / "ops.py"

        # Checkpoint 1
        target.write_text("# Vers 1\n", encoding="utf-8")
        ckpt1 = await mgr.create_checkpoint(runtime.env, label="v1", iteration=1)
        assert ckpt1 is not None

        # Checkpoint 2
        target.write_text("# Vers 2\n", encoding="utf-8")
        ckpt2 = await mgr.create_checkpoint(runtime.env, label="v2", iteration=2)
        assert ckpt2 is not None

        # Dirty modification
        target.write_text("# Corrupted\n", encoding="utf-8")

        # Roll back to latest (ckpt2)
        res = await mgr.rollback_to_latest(runtime.env, [ckpt1, ckpt2])
        assert res is not None
        assert target.read_text(encoding="utf-8") == "# Vers 2\n"

    async def test_rollback_empty_list_returns_none(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        mgr = CheckpointManager()
        res = await mgr.rollback_to_latest(runtime.env, [])
        assert res is None

    async def test_orchestrator_rollback_method(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        from forge.agents.executor import ExecutorAgent
        from forge.agents.planner import PlannerAgent
        from forge.agents.qa import QAAgent

        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(emitted.append)

        runtime = make_runtime(sample_repo)
        planner = PlannerAgent(_make_adapter("planner"), runtime=runtime)
        executor = ExecutorAgent(_make_adapter("executor"), runtime=runtime)
        qa = QAAgent(_make_adapter("qa"), runtime=runtime)

        router = AgentRouter(
            planner=planner,
            executor=executor,
            qa=qa,
            runtime=runtime,
            events=events_bus,
        )
        orchestrator = Orchestrator(
            adapter=_make_adapter("orchestrator"),
            router=router,
            events=events_bus,
        )

        state = WorkflowState(user_request="Refactor math")
        readme = sample_repo / "README.md"
        readme.write_text("Saved checkpoint\n", encoding="utf-8")

        ckpt = await orchestrator.router.checkpoint_mgr.create_checkpoint(
            runtime.env, "test_ckpt", 1
        )
        assert ckpt is not None
        state.checkpoints.append(ckpt)

        readme.write_text("Mangled text\n", encoding="utf-8")

        # Trigger rollback via Orchestrator
        restore = await orchestrator.rollback(state)
        assert restore is not None
        assert readme.read_text(encoding="utf-8") == "Saved checkpoint\n"

        types = [e.type for e in emitted]
        assert EventType.ROLLBACK_PERFORMED in types
