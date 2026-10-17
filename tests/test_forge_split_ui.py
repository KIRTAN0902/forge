"""Comprehensive verification suite for Forge's Split-Pane Blacksmithing Workspace ("Anvil -> Hammer -> Temper")."""

from __future__ import annotations

from typing import Any, cast

import pytest

from forge.agents import ExecutorAgent, OrchestratorAgent, QAAgent
from forge.models import ModelAdapter, ModelCapabilities, ModelProfile
from forge.models.providers.scripted import ScriptedProvider
from forge.tools.human_question import AskUserQuestion, AskUserQuestionInput
from forge.ui.cycle_controller import CycleVerdict, ForgeCycleController
from forge.ui.split_app import (
    ForgeSplitPaneApp,
    render_sidebar_plain_text,
)
from forge.ui.state import CycleMode, ForgeStage, ForgeTUIState, UserQuestion, format_tokens


def _scripted_agent(provider: ScriptedProvider, role: str) -> Any:
    profile = ModelProfile(provider="scripted", model=f"{role}-test", detect_capabilities=False)
    adapter = ModelAdapter(
        profile_name=role,
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role=role,
    )
    if role == "orchestrator":
        return OrchestratorAgent(adapter)
    no_runtime = cast("Any", None)
    if role == "executor":
        return ExecutorAgent(adapter, no_runtime)
    return QAAgent(adapter, no_runtime)


def test_format_tokens_helper() -> None:
    """Verify clean token formatting across all orders of magnitude."""
    assert format_tokens(0) == "0"
    assert format_tokens(840) == "840"
    assert format_tokens(950) == "950"
    assert format_tokens(3150) in {"3.1k", "3.2k"}
    assert format_tokens(42100) == "42.1k"
    assert format_tokens(125400) == "125.4k"
    assert format_tokens(1400000) == "1.40M"
    assert format_tokens(-500) == "-500"


def test_default_split_pane_and_sidebar_rendering() -> None:
    """Verify sidebar renders Anvil, Hammer, Temper with the active box indicator | Temper | <-- (or [ Temper ] <--), Cycle :- 1, and both Stage: and Total: token lines."""
    state = ForgeTUIState()
    state.set_active_stage(ForgeStage.TEMPER)
    state.record_tokens(ForgeStage.TEMPER, 3100)

    plain = render_sidebar_plain_text(state)

    # 1. Three stages present
    assert "Anvil" in plain
    assert "Hammer" in plain
    assert "Temper" in plain

    # 2. Active worker stage enclosed in a clean box with a trailing arrow
    assert "│ ● Temper" in plain
    assert "<--" in plain

    # 3. Cycle counter in Purple
    assert "Cycle :- 1" in plain

    # 4. Dual token counters
    assert "Tokens :-" in plain
    assert "Stage:" in plain
    assert "Total:" in plain
    assert "3.1k" in plain or "3.10k" in plain


def test_live_stage_switch_while_hammer_working() -> None:
    """Start Hammer (active_worker_stage = ForgeStage.HAMMER), call state.select_view_stage(ForgeStage.ANVIL),

    and assert that auto_follow is False, the viewport renders Anvil's buffer exclusively,
    and background appends to Hammer continue accumulating in stage_buffers[ForgeStage.HAMMER]
    without hijacking the user's view.
    """
    state = ForgeTUIState()

    # Hammer starts working
    state.set_active_stage(ForgeStage.HAMMER)
    assert state.active_worker_stage == ForgeStage.HAMMER
    assert state.selected_view_stage == ForgeStage.HAMMER
    assert state.auto_follow is True

    # User manually switches view to Anvil to inspect plan
    state.select_view_stage(ForgeStage.ANVIL)
    assert state.auto_follow is False
    assert state.selected_view_stage == ForgeStage.ANVIL

    # Viewport buffer is exclusively Anvil
    view_buf = state.get_active_view_buffer()
    assert view_buf.stage == ForgeStage.ANVIL

    # Background Hammer worker emits thoughts and tool events
    state.append_thought(ForgeStage.HAMMER, "Writing patch to src/core/engine.py")
    state.append_tool_event(ForgeStage.HAMMER, "Tool: pytest -q --tb=short | Exit: 0")
    state.record_tokens(ForgeStage.HAMMER, 1500)

    # Hammer buffer accumulated the background updates
    hammer_buf = state.stage_buffers[ForgeStage.HAMMER]
    assert "Writing patch to src/core/engine.py" in hammer_buf.thoughts
    assert "Tool: pytest -q --tb=short | Exit: 0" in hammer_buf.tool_events
    assert hammer_buf.stage_tokens == 1500

    # User's view was NOT hijacked - still viewing Anvil exclusively!
    assert state.selected_view_stage == ForgeStage.ANVIL
    assert state.get_active_view_buffer().stage == ForgeStage.ANVIL
    assert "Writing patch to src/core/engine.py" not in state.get_active_view_buffer().thoughts

    # Sidebar displays [VIEW] badge next to Anvil while keeping a boxed Hammer <--
    sidebar_text = render_sidebar_plain_text(state)
    assert "[VIEW]" in sidebar_text
    assert "│ ● Hammer" in sidebar_text

    # Resume auto-follow
    state.resume_auto_follow()
    assert state.auto_follow is True
    assert state.selected_view_stage == ForgeStage.HAMMER


def test_dual_token_counter_tracking_across_stages_and_cycles() -> None:
    """Record 1,200 tokens in Anvil -> Assert active_stage_tokens == 1200 and total_session_tokens == 1200.

    Transition to Hammer -> Assert active_stage_tokens == 0 and total_session_tokens == 1200.
    Record 3,400 tokens in Hammer -> Assert active_stage_tokens == 3400 and total_session_tokens == 4600.
    Advance to Cycle :- 2 -> Assert cycle_count == 2, Anvil stage tokens reset to 0, and total_session_tokens == 4600.
    """
    state = ForgeTUIState()

    # 1. Record 1,200 tokens in Anvil
    state.set_active_stage(ForgeStage.ANVIL)
    state.record_tokens(ForgeStage.ANVIL, 1200)
    assert state.active_stage_tokens == 1200
    assert state.total_session_tokens == 1200

    # 2. Transition to Hammer
    state.set_active_stage(ForgeStage.HAMMER)
    assert state.active_stage_tokens == 0
    assert state.total_session_tokens == 1200

    # 3. Record 3,400 tokens in Hammer
    state.record_tokens(ForgeStage.HAMMER, 3400)
    assert state.active_stage_tokens == 3400
    assert state.total_session_tokens == 4600

    # 4. Advance to Cycle :- 2
    state.advance_to_next_cycle(temper_feedback="Regression in auth validation")
    assert state.cycle_count == 2
    assert state.stage_buffers[ForgeStage.ANVIL].stage_tokens == 0
    assert state.total_session_tokens == 4600


@pytest.mark.asyncio
async def test_cycle_loop_halts_on_temper_pass_and_loops_on_reject() -> None:
    """Mock a 2-cycle run where Temper rejects in Cycle :- 1 and approves in Cycle :- 2,

    verifying automatic transition and final termination.
    """
    state = ForgeTUIState()

    def mock_temper_verifier(cycle: int, request: str) -> CycleVerdict:
        if cycle == 1:
            return CycleVerdict(
                passed=False,
                feedback="P1 Defect: Missing error boundary for invalid JWT signature.",
            )
        return CycleVerdict(
            passed=True,
            feedback="All criteria verified.",
            deliverable="Verified JWT authentication service with test coverage.",
        )

    controller = ForgeCycleController(
        state=state,
        temper_verifier=mock_temper_verifier,
        max_cycles=5,
    )

    deliverable = await controller.run_cycle_loop("Implement JWT authentication service")

    # Verified cycle progression
    assert state.cycle_count == 2
    assert state.active_worker_stage is None
    assert "Verified JWT authentication service" in deliverable
    assert state.stage_buffers[ForgeStage.TEMPER].status == "passed"
    assert 1 in state.cycle_history


def test_stage_buffer_render_to_ansi() -> None:
    """Verify StageBuffer.render_to_ansi renders thoughts and tools without redundant header or stage output."""
    state = ForgeTUIState()
    buf = state.stage_buffers[ForgeStage.TEMPER]
    buf.thoughts.append("Performing AST mutation analysis")
    buf.tool_events.append("MutationScore: 100% (6/6 mutants killed)")
    buf.final_output = "Production grade implementation verified."
    buf.status = "passed"

    ansi_text = buf.render_to_ansi(width=80)
    assert "FORGE // TEMPER" not in ansi_text
    assert "Stage Output" not in ansi_text
    assert "Verified Final Output Deliverable" not in ansi_text
    assert "Thought Process" in ansi_text
    assert "Performing AST mutation analysis" in ansi_text
    assert "Tool & Sandbox Execution" in ansi_text


def test_split_pane_app_keybindings_and_controls() -> None:
    """Verify application initializes layout, keybindings, and state wiring cleanly."""
    state = ForgeTUIState()
    app = ForgeSplitPaneApp(state=state)

    assert app.layout is not None
    assert app.application is not None
    assert app.sidebar_control is not None
    assert app.viewport_control is not None

    # Keybinding checks: F1, F2, F3, F4, F5, etc. are bound
    keys = [binding.keys for binding in app.kb.bindings]
    assert any("f1" in k or ("escape", "1") in k for k in keys)
    assert any("f2" in k or ("escape", "2") in k for k in keys)
    assert any("f3" in k or ("escape", "3") in k for k in keys)
    assert any("f4" in k or ("escape", "0") in k for k in keys)
    assert any("f5" in k or ("escape", "m") in k for k in keys)
    assert any("pageup" in k for k in keys)
    assert any("pagedown" in k for k in keys)


@pytest.mark.asyncio
async def test_dynamic_cycle_mode_single_cycle_stops_after_cycle_1() -> None:
    """Verify that when the user specifies 'run first cycle and stop',

    the orchestrator sets Single Cycle mode and halts after Cycle 1 even if rejected.
    """
    state = ForgeTUIState()

    def mock_temper_reject_verifier(cycle: int, request: str) -> CycleVerdict:
        return CycleVerdict(passed=False, feedback="P2 Defect: Edge case missing in validator.")

    controller = ForgeCycleController(
        state=state,
        temper_verifier=mock_temper_reject_verifier,
        max_cycles=5,
    )

    result = await controller.run_cycle_loop("Refactor auth system: run first cycle and stop")

    assert controller.cycle_mode == CycleMode.SINGLE_CYCLE
    assert state.cycle_mode == CycleMode.SINGLE_CYCLE
    # Must halt after cycle 1 (does not loop to cycle 2)
    assert state.cycle_count == 1
    assert state.active_worker_stage is None
    assert "Single Cycle completed" in result
    assert "Edge case missing" in result


@pytest.mark.asyncio
async def test_dynamic_cycle_mode_continuous_until_fulfilled() -> None:
    """Verify that when the user specifies 'run until requirements get fulfilled',

    the orchestrator sets Until Fulfilled mode and loops through cycles until Temper passes.
    """
    state = ForgeTUIState()

    def mock_temper_verifier(cycle: int, request: str) -> CycleVerdict:
        if cycle == 1:
            return CycleVerdict(passed=False, feedback="Missing regression test.")
        return CycleVerdict(passed=True, deliverable="All requirements verified.")

    controller = ForgeCycleController(
        state=state,
        temper_verifier=mock_temper_verifier,
        max_cycles=5,
    )

    result = await controller.run_cycle_loop(
        "Build payment webhook: run until requirements get fulfilled"
    )

    assert controller.cycle_mode == CycleMode.UNTIL_FULFILLED
    assert state.cycle_mode == CycleMode.UNTIL_FULFILLED
    assert state.cycle_count == 2
    assert "All requirements verified" in result


@pytest.mark.asyncio
async def test_conversational_turn_behaves_like_chat_without_cycles_or_questions() -> None:
    """Verify that pure conversational inquiries (like 'hi', 'hello', 'what can you do')

    stream a real model reply on Anvil without triggering questions or running Hammer/Temper.
    """
    state = ForgeTUIState()
    asked_questions: list[UserQuestion] = []
    provider = ScriptedProvider()

    async def mock_question_handler(question: UserQuestion) -> str:
        asked_questions.append(question)
        return question.options[0]

    controller = ForgeCycleController(
        state=state,
        question_handler=mock_question_handler,
        agents={ForgeStage.ANVIL: _scripted_agent(provider, "orchestrator")},
    )

    # 1. Test "hi" — the reply is whatever the model returns, not a canned string.
    provider.add("Hello! I am Forge. How can I help you today?")
    result = await controller.run_cycle_loop("hi")

    # No questions asked
    assert len(asked_questions) == 0
    assert "Forge" in result
    assert len(provider.requests) == 1
    assert state.stage_buffers[ForgeStage.ANVIL].status == "passed"
    assert state.active_worker_stage is None
    # The full multi-turn conversation history is preserved for the next turn.
    anvil_history = controller.conversation_history[ForgeStage.ANVIL]
    assert anvil_history[0] == {"role": "user", "content": "hi"}
    assert "Forge" in anvil_history[1]["content"]

    # Hammer & Temper stages were deferred (never executed)
    assert len(state.stage_buffers[ForgeStage.HAMMER].thoughts) == 0
    assert len(state.stage_buffers[ForgeStage.HAMMER].tool_events) == 0
    assert len(state.stage_buffers[ForgeStage.TEMPER].thoughts) == 0

    # 2. Test "what can you do" — the prior turn is replayed to the model.
    provider.add("I run the Anvil -> Hammer -> Temper blacksmithing pipeline.")
    result_help = await controller.run_cycle_loop("what can you do")
    assert len(asked_questions) == 0
    assert "Anvil" in result_help or "pipeline" in result_help
    second_request = provider.requests[1]
    assert any("hi" in (m.content or "") for m in second_request.messages)


@pytest.mark.asyncio
async def test_unspecified_task_executes_without_unnecessary_question() -> None:
    """Verify that an engineering task without explicit cycle keywords does NOT trigger

    an unnecessary question; it executes smoothly using the active cycle mode.
    """
    state = ForgeTUIState()
    asked_questions: list[UserQuestion] = []

    async def mock_question_handler(question: UserQuestion) -> str:
        asked_questions.append(question)
        return question.options[0]

    def verifier(_cycle: int, _prompt: str) -> CycleVerdict:
        return CycleVerdict(
            passed=True,
            feedback="All acceptance criteria satisfied.",
            deliverable="Exported user profiles to profiles.json.",
        )

    controller = ForgeCycleController(
        state=state,
        question_handler=mock_question_handler,
        temper_verifier=verifier,
    )

    result = await controller.run_cycle_loop("Create user profile export feature")

    # Zero intrusive questions asked to user
    assert len(asked_questions) == 0
    # Active mode was preserved
    assert controller.cycle_mode == CycleMode.UNTIL_FULFILLED
    assert state.cycle_mode == CycleMode.UNTIL_FULFILLED
    # Finished successfully
    assert "Exported user profiles to profiles.json." in result


@pytest.mark.asyncio
async def test_orchestrator_generic_question_asking() -> None:
    """Verify generic question asking ('not for this but anything if it wants to know')."""
    state = ForgeTUIState()

    async def mock_question_handler(question: UserQuestion) -> str:
        # User selects option 2
        return question.options[1]

    controller = ForgeCycleController(
        state=state,
        question_handler=mock_question_handler,
    )

    answer = await controller.ask_question(
        question="Which database backend should we configure for the cache?",
        options=["Redis Cluster", "Memcached", "PostgreSQL Unlogged Table"],
        default_option="Redis Cluster",
        stage=ForgeStage.ANVIL,
    )

    assert answer == "Memcached"
    # Anvil's thought buffer recorded the question and answer
    anvil_thoughts = state.stage_buffers[ForgeStage.ANVIL].thoughts
    assert any("Which database backend" in t for t in anvil_thoughts)
    assert any("Memcached" in t for t in anvil_thoughts)


def test_question_rendering_and_input_prompt_dynamics() -> None:
    """Verify question card is rendered in ANSI viewport and input prompt switches to ? [1-N] >."""
    state = ForgeTUIState()
    app = ForgeSplitPaneApp(state=state)

    # Initial prompt is the stage-aware single-line prefix
    initial_prompt = app._render_input_prompt()
    assert initial_prompt[0][1] == "anvil ❯ "

    # Pose a question on Anvil
    q = UserQuestion(
        question="Select target API environment:",
        options=["Production", "Staging", "Local Sandbox"],
        default_option="Staging",
        stage=ForgeStage.ANVIL,
    )
    state.set_pending_question(q)

    # Prompt dynamically switches to question options indicator
    question_prompt = app._render_input_prompt()
    assert "? [1-3] > " in question_prompt[0][1]

    # Viewport renders the Rich question card
    ansi_view = app._render_viewport()
    plain_view = str(ansi_view)
    assert "Question from Orchestrator" in plain_view
    assert "Select target API environment:" in plain_view
    assert "Production" in plain_view
    assert "Staging (Default)" in plain_view

    # User answers by typing "2" (selects Staging)
    chosen = state.resolve_pending_question("2")
    assert chosen == "Staging"
    assert state.pending_question is None

    # Prompt returns to normal
    restored_prompt = app._render_input_prompt()
    assert restored_prompt[0][1] == "anvil ❯ "


@pytest.mark.asyncio
async def test_ask_user_question_tool() -> None:
    """Verify AskUserQuestion tool executes and falls back safely in test environment."""
    from forge.tools.environment import ToolEnvironment
    from forge.tools.shell import ShellRunner, resolve_shell
    from forge.tools.workspace import Workspace

    workspace = Workspace(".")
    shell = ShellRunner(spec=resolve_shell("auto"))
    env = ToolEnvironment(workspace=workspace, shell=shell)

    tool = AskUserQuestion()
    inp = AskUserQuestionInput(
        question="Choose test framework",
        options=["pytest", "unittest"],
        default_option="pytest",
    )

    # Without custom env handler, uses default option
    out = await tool.run(inp, env)
    assert out.answer == "pytest"

    # With custom env handler, invokes handler
    async def custom_ask(question: str, options: list[str], default: str | None) -> str:
        return options[1]

    env.ask_question = custom_ask
    out2 = await tool.run(inp, env)
    assert out2.answer == "unittest"
