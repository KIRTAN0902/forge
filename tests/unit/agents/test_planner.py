from __future__ import annotations

from pathlib import Path

from forge.agents import PlannerAgent
from forge.core.contracts.plan import Plan, Task
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
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response

from ...conftest import RuntimeFactory


def _make_adapter(provider: ScriptedProvider, events: EventBus | None = None) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model="planner-test")
    return ModelAdapter(
        profile_name="planner",
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role="planner",
        events=events,
        workflow_id="wf_plan",
    )


def _sample_requirements() -> RequirementContract:
    return RequirementContract(
        summary="Add integer addition function",
        functional=[FunctionalRequirement(id="FR-001", description="Implement add(a, b)")],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                description="add(2, 3) returns 5",
                verification=VerificationMethod.AUTOMATED_TEST,
                requirement_ids=["FR-001"],
            )
        ],
    )


def _sample_plan() -> Plan:
    return Plan(
        goal="Fix addition bug in ops.py",
        approach="Correct subtraction operator to addition",
        tasks=[
            Task(
                id="TASK-1",
                description="Fix add() in ops.py",
                dependencies=[],
                acceptance_criteria=["AC-001"],
                files_hint=["src/calc/ops.py"],
            )
        ],
        acceptance_criteria=["AC-001"],
        risks=["Regression in existing ops"],
    )


class TestPlannerAgent:
    async def test_generates_requirements_and_plan(self) -> None:
        reqs = _sample_requirements()
        plan = _sample_plan()
        provider = ScriptedProvider(
            [
                reqs.model_dump_json(),
                plan.model_dump_json(),
            ]
        )
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        adapter = _make_adapter(provider, events=events_bus)
        planner = PlannerAgent(adapter=adapter, events=events_bus, workflow_id="wf_1")

        result = await planner.plan("Fix the addition operator")

        assert result.requirements.summary == "Add integer addition function"
        assert len(result.plan.tasks) == 1
        assert result.plan.tasks[0].id == "TASK-1"

        event_types = [e.type for e in emitted]
        assert EventType.REQUIREMENTS_DEFINED in event_types
        assert EventType.PLAN_CREATED in event_types

    async def test_replan_with_existing_requirements(self) -> None:
        plan = _sample_plan()
        provider = ScriptedProvider([plan.model_dump_json()])
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        adapter = _make_adapter(provider, events=events_bus)
        planner = PlannerAgent(adapter=adapter, events=events_bus, workflow_id="wf_replan")

        existing_reqs = _sample_requirements()
        result = await planner.plan(
            "Fix addition",
            requirements=existing_reqs,
            existing_plan=plan,
        )

        assert result.requirements == existing_reqs
        assert len(result.plan.tasks) == 1
        event_types = [e.type for e in emitted]
        assert EventType.PLAN_REVISED in event_types

    async def test_investigate_before_planning(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        runtime = make_runtime(sample_repo)
        reqs = _sample_requirements()
        plan = _sample_plan()

        provider = ScriptedProvider(
            [
                tool_call_response("read_file", {"path": "src/calc/ops.py"}),
                "Inspected ops.py. Ready to plan.",
                reqs.model_dump_json(),
                plan.model_dump_json(),
            ]
        )
        adapter = _make_adapter(provider)
        planner = PlannerAgent(adapter=adapter, runtime=runtime)

        result = await planner.plan(
            "Fix addition",
            investigate=True,
            max_investigation_steps=3,
        )

        assert result.requirements.summary == "Add integer addition function"
        assert result.plan.goal == "Fix addition bug in ops.py"
        # Tool call was passed in requests
        assert any(bool(r.tools) for r in provider.requests)
