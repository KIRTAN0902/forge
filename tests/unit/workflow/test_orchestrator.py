from __future__ import annotations

from pathlib import Path

from forge.agents import ExecutorAgent, PlannerAgent, QAAgent
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task
from forge.core.contracts.requirements import AcceptanceCriterion, RequirementContract
from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.events import Event, EventBus, EventType
from forge.models import (
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
)
from forge.models.providers.scripted import ScriptedProvider
from forge.storage.sqlite import SQLiteStore
from forge.workflow import AgentRouter, Orchestrator

from ...conftest import RuntimeFactory


def _make_adapter(role: str, provider: ScriptedProvider) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model="test-model")
    return ModelAdapter(
        profile_name=role,
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role=role,
        workflow_id="wf_orch",
    )


class TestOrchestrator:
    async def test_decide_emits_event_and_records_decision(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        decision_payload = Decision(
            action=OrchestratorAction.PLAN,
            rationale="Initial planning step",
        )
        provider = ScriptedProvider([decision_payload.model_dump_json()])
        adapter = _make_adapter("orchestrator", provider)

        planner = PlannerAgent(_make_adapter("planner", ScriptedProvider([])), runtime=runtime)
        executor = ExecutorAgent(_make_adapter("executor", ScriptedProvider([])), runtime=runtime)
        qa = QAAgent(_make_adapter("qa", ScriptedProvider([])), runtime=runtime)

        router = AgentRouter(
            planner=planner,
            executor=executor,
            qa=qa,
            runtime=runtime,
            events=events_bus,
        )
        orchestrator = Orchestrator(
            adapter=adapter,
            router=router,
            events=events_bus,
        )

        state = WorkflowState(user_request="Implement feature")
        dec = await orchestrator.decide(state)

        assert dec.action == OrchestratorAction.PLAN
        assert len(state.decisions) == 1
        assert state.iteration == 1

        types = [e.type for e in emitted]
        assert EventType.DECISION_MADE in types

    async def test_step_persists_to_sqlite_store(
        self, sample_repo: Path, make_runtime: RuntimeFactory, store: SQLiteStore
    ) -> None:
        runtime = make_runtime(sample_repo)

        # Step: Orchestrator decides PLAN, router runs planner
        reqs = RequirementContract(
            summary="Requirements",
            acceptance_criteria=[AcceptanceCriterion(id="AC-001", description="Criterion")],
        )
        plan = Plan(
            goal="Goal",
            tasks=[Task(id="TASK-1", description="Task 1")],
            acceptance_criteria=["AC-001"],
        )

        orch_provider = ScriptedProvider(
            [Decision(action=OrchestratorAction.PLAN, rationale="Plan first").model_dump_json()]
        )
        plan_provider = ScriptedProvider([reqs.model_dump_json(), plan.model_dump_json()])

        planner = PlannerAgent(_make_adapter("planner", plan_provider), runtime=runtime)
        executor = ExecutorAgent(_make_adapter("executor", ScriptedProvider([])), runtime=runtime)
        qa = QAAgent(_make_adapter("qa", ScriptedProvider([])), runtime=runtime)

        router = AgentRouter(
            planner=planner,
            executor=executor,
            qa=qa,
            runtime=runtime,
        )
        orchestrator = Orchestrator(
            adapter=_make_adapter("orchestrator", orch_provider),
            router=router,
            store=store,
        )

        state = WorkflowState(user_request="Build something")
        decision, updated_state = await orchestrator.step(state)

        assert decision.action == OrchestratorAction.PLAN
        assert updated_state.plan is not None
        assert updated_state.status == WorkflowStatus.RUNNING

        # Verify state is saved in sqlite store and can be reloaded
        loaded = await store.load_workflow(state.workflow_id)
        assert loaded.workflow_id == state.workflow_id
        assert loaded.status == WorkflowStatus.RUNNING
        assert loaded.plan is not None
        assert loaded.plan.tasks[0].id == "TASK-1"

    async def test_awaiting_approval_pauses_and_approve_resumes(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(emitted.append)

        planner = PlannerAgent(_make_adapter("planner", ScriptedProvider([])), runtime=runtime)
        executor = ExecutorAgent(_make_adapter("executor", ScriptedProvider([])), runtime=runtime)
        qa = QAAgent(_make_adapter("qa", ScriptedProvider([])), runtime=runtime)

        router = AgentRouter(
            planner=planner,
            executor=executor,
            qa=qa,
            runtime=runtime,
            events=events_bus,
        )
        orchestrator = Orchestrator(
            adapter=_make_adapter("orchestrator", ScriptedProvider([])),
            router=router,
            events=events_bus,
        )

        state = WorkflowState(
            user_request="Refactor subsystem",
            status=WorkflowStatus.AWAITING_APPROVAL,
        )

        # Step should not execute model or router, should remain paused
        decision, updated_state = await orchestrator.step(state)
        assert updated_state.status == WorkflowStatus.AWAITING_APPROVAL
        assert decision.action == OrchestratorAction.REQUEST_HUMAN_REVIEW

        # Approve sets to RUNNING and records feedback
        approved_state = await orchestrator.approve(updated_state, feedback="Proceed with caution")
        assert approved_state.status == WorkflowStatus.RUNNING
        assert "[Human Guidance]: Proceed with caution" in approved_state.user_request

        event_types = [e.type for e in emitted]
        assert EventType.APPROVAL_RESOLVED in event_types
