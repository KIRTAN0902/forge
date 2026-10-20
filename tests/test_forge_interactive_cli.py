"""Regression tests for the adaptive 3-tier interactive intent router.

These tests lock in the Claude Code / Codex CLI style behaviour:

* a greeting stays conversational in Anvil and never spins up a cycle,
* a direct operational request executes real tools / shell without a cycle,
* a genuinely complex engineering task triggers a full Anvil -> Hammer -> Temper cycle,
* pinning a stage (Alt+2 / Alt+3) routes the turn straight to that stage's agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from forge.agents import ExecutorAgent, OrchestratorAgent, QAAgent
from forge.models import ModelAdapter, ModelCapabilities, ModelProfile
from forge.models.providers.scripted import ScriptedProvider

if TYPE_CHECKING:
    from forge.tools.runtime import ToolRuntime
from forge.ui.cycle_controller import (
    CycleVerdict,
    ExecutionMode,
    ForgeCycleController,
    classify_intent,
)
from forge.ui.state import (
    AnvilPlanEvaluation,
    ForgeStage,
    ForgeTUIState,
    PlanTurnIntent,
)


async def _approve_plan_turn(plan: Any, response: str) -> AnvilPlanEvaluation:
    return AnvilPlanEvaluation(intent=PlanTurnIntent.APPROVE_AS_IS)


def _adapter(provider: ScriptedProvider, role: str) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model=f"{role}-test", detect_capabilities=False)
    return ModelAdapter(
        profile_name=role,
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role=role,
    )


def _empty_agents() -> tuple[dict[ForgeStage, object], dict[ForgeStage, ScriptedProvider]]:
    providers: dict[ForgeStage, ScriptedProvider] = {
        ForgeStage.ANVIL: ScriptedProvider(),
        ForgeStage.HAMMER: ScriptedProvider(),
        ForgeStage.TEMPER: ScriptedProvider(),
    }
    no_runtime = cast("ToolRuntime", None)
    agents: dict[ForgeStage, Any] = {
        ForgeStage.ANVIL: OrchestratorAgent(_adapter(providers[ForgeStage.ANVIL], "orchestrator")),
        ForgeStage.HAMMER: ExecutorAgent(
            _adapter(providers[ForgeStage.HAMMER], "executor"), no_runtime
        ),
        ForgeStage.TEMPER: QAAgent(_adapter(providers[ForgeStage.TEMPER], "qa"), no_runtime),
    }
    return agents, providers


async def test_greeting_stays_conversational_in_anvil() -> None:
    state = ForgeTUIState()
    agents, providers = _empty_agents()
    providers[ForgeStage.ANVIL].add("Hello! I am Forge. How can I help you today?")
    controller = ForgeCycleController(state=state, agents=agents)

    result = await controller.run_cycle_loop("hi")

    assert controller.last_mode == ExecutionMode.CONVERSATIONAL
    assert "Forge" in result
    # A conversational turn must never advance the manufacturing cycle.
    assert state.cycle_count == 1
    assert state.stage_buffers[ForgeStage.HAMMER].final_output == ""
    assert state.stage_buffers[ForgeStage.TEMPER].final_output == ""
    assert providers[ForgeStage.HAMMER].requests == []
    assert providers[ForgeStage.TEMPER].requests == []
    # Real usage is accumulated onto the session counter.
    assert state.total_session_tokens > 0
    assert state.stage_buffers[ForgeStage.ANVIL].chat_history


async def test_direct_command_execution_runs_real_subprocess(tmp_path: Path) -> None:
    state = ForgeTUIState()
    controller = ForgeCycleController(state=state, root=tmp_path)

    result = await controller.run_cycle_loop("!echo forge-direct-tools")

    assert controller.last_mode == ExecutionMode.DIRECT_TOOL
    assert "forge-direct-tools" in result
    # Executed for real (not through the cycle) and streamed into the Anvil viewport.
    assert state.cycle_count == 1
    anvil_events = state.stage_buffers[ForgeStage.ANVIL].tool_events
    assert any("forge-direct-tools" in ev for ev in anvil_events)


def test_direct_queries_classify_as_tool_not_cycle() -> None:
    assert classify_intent("run pytest -q") == ExecutionMode.DIRECT_TOOL
    assert classify_intent("list the files in src") == ExecutionMode.DIRECT_TOOL
    assert classify_intent("git status") == ExecutionMode.DIRECT_TOOL
    assert classify_intent("!npm test") == ExecutionMode.DIRECT_TOOL


def test_unmarked_engineering_request_defaults_to_planning_cycle() -> None:
    prompt = (
        "I want you to create a beautiful, modern, and fully responsive personal "
        "portfolio website using React, Tailwind CSS, and Framer Motion. Do not use "
        "placeholder content. Please build and run the project."
    )
    assert classify_intent(prompt) == ExecutionMode.FULL_FORGE_CYCLE


def test_questions_remain_conversational() -> None:
    assert classify_intent("How should I structure this React app?") == ExecutionMode.CONVERSATIONAL
    assert classify_intent("What are the tradeoffs of Tailwind versus CSS modules?") == (
        ExecutionMode.CONVERSATIONAL
    )


async def test_creation_request_renders_pending_plan_not_code() -> None:
    state = ForgeTUIState()
    agents, providers = _empty_agents()
    providers[ForgeStage.ANVIL].add(
        "Implement the portfolio in three phases.\n\n"
        "```tsx\n"
        "function App() {\n"
        "  const title = 'Portfolio';\n"
        "  return <main>{title}</main>;\n"
        "}\n"
        "```\n\n"
        "REQ-1: The page renders a responsive hero section.\n"
        "REQ-2: The production build succeeds.\n"
        "TARGET_FILES: src/App.tsx, src/index.css"
    )
    controller = ForgeCycleController(state=state, agents=agents)

    result = await controller.run_cycle_loop(
        "I want you to create a responsive React portfolio website and run it."
    )

    assert state.pending_plan is not None
    assert "Final Plan" in result
    assert "REQ-1" in result
    assert "src/App.tsx" in result
    assert "function App" not in result
    assert providers[ForgeStage.HAMMER].requests == []


async def test_complex_task_triggers_full_forge_cycle() -> None:
    state = ForgeTUIState()
    verdicts = {"count": 0}

    def verifier(_cycle: int, _prompt: str) -> CycleVerdict:
        verdicts["count"] += 1
        return CycleVerdict(
            passed=True,
            feedback="All criteria satisfied.",
            deliverable="DELIVERABLE: authentication service implemented.",
        )

    controller = ForgeCycleController(state=state, temper_verifier=verifier)
    controller._evaluate_plan_turn = _approve_plan_turn  # type: ignore[method-assign]
    prompt = "Implement a user authentication service with login and session tokens"

    assert classify_intent(prompt) == ExecutionMode.FULL_FORGE_CYCLE

    plan = await controller.run_cycle_loop(prompt)
    assert "Final Plan" in plan
    result = await controller.run_cycle_loop("approve")

    assert controller.last_mode == ExecutionMode.FULL_FORGE_CYCLE
    assert state.cycle_count == 1
    assert verdicts["count"] == 1
    assert "DELIVERABLE" in result
    # Every manufacturing stage contributed output.
    assert state.stage_buffers[ForgeStage.ANVIL].final_output
    assert state.stage_buffers[ForgeStage.HAMMER].final_output
    assert state.stage_buffers[ForgeStage.TEMPER].final_output
    assert state.stage_buffers[ForgeStage.TEMPER].status == "passed"


async def test_direct_chat_with_pinned_stage_bypasses_cycle() -> None:
    state = ForgeTUIState()
    agents, providers = _empty_agents()
    providers[ForgeStage.HAMMER].add("Hammer here: I can implement that directly.")
    controller = ForgeCycleController(state=state, agents=agents)

    # This prompt would normally trigger a full cycle, but the developer pinned Hammer.
    result = await controller.run_cycle_loop(
        "Implement the whole module", target_stage=ForgeStage.HAMMER
    )

    assert controller.last_mode == ExecutionMode.DIRECT_TOOL
    assert "Hammer" in result
    assert len(providers[ForgeStage.HAMMER].requests) == 1
    # The cycle never started and Anvil was not consulted.
    assert state.cycle_count == 1
    assert providers[ForgeStage.ANVIL].requests == []
    assert state.stage_buffers[ForgeStage.HAMMER].chat_history
