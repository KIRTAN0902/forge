"""Comprehensive verification suite for Forge's Split-Pane Blacksmithing Workspace ("Anvil -> Hammer -> Temper")."""

from __future__ import annotations

import pytest

from forge.ui.cycle_controller import CycleVerdict, ForgeCycleController
from forge.ui.split_app import (
    ForgeSplitPaneApp,
    render_sidebar_plain_text,
)
from forge.ui.state import ForgeStage, ForgeTUIState, format_tokens


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

    # 2. Active worker stage enclosed in box with arrow
    assert "|  Temper  | <--" in plain or "[ Temper ] <--" in plain

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

    # Sidebar displays [👁 VIEWING] badge next to Anvil while keeping boxed Hammer <--
    sidebar_text = render_sidebar_plain_text(state)
    assert "[👁 VIEWING]" in sidebar_text
    assert "|  Hammer  | <--" in sidebar_text or "[ Hammer ] <--" in sidebar_text

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
    """Verify StageBuffer.render_to_ansi renders header, thoughts, tools, and deliverable."""
    state = ForgeTUIState()
    buf = state.stage_buffers[ForgeStage.TEMPER]
    buf.thoughts.append("Performing AST mutation analysis")
    buf.tool_events.append("MutationScore: 100% (6/6 mutants killed)")
    buf.final_output = "Production grade implementation verified."
    buf.status = "passed"

    ansi_text = buf.render_to_ansi(width=80)
    assert "TEMPER" in ansi_text
    assert "Thought Process" in ansi_text
    assert "Performing AST mutation analysis" in ansi_text
    assert "Tool & Sandbox Execution" in ansi_text
    assert "Verified Final Output Deliverable" in ansi_text


def test_split_pane_app_keybindings_and_controls() -> None:
    """Verify application initializes layout, keybindings, and state wiring cleanly."""
    state = ForgeTUIState()
    app = ForgeSplitPaneApp(state=state)

    assert app.layout is not None
    assert app.application is not None
    assert app.sidebar_control is not None
    assert app.viewport_control is not None

    # Keybinding checks: F1, F2, F3, F4, etc. are bound
    keys = [binding.keys for binding in app.kb.bindings]
    assert any("f1" in k or ("escape", "1") in k for k in keys)
    assert any("f2" in k or ("escape", "2") in k for k in keys)
    assert any("f3" in k or ("escape", "3") in k for k in keys)
    assert any("f4" in k or ("escape", "0") in k for k in keys)
    assert any("pageup" in k for k in keys)
    assert any("pagedown" in k for k in keys)
