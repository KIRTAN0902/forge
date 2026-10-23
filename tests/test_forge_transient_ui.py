"""Verification suite for Forge's ephemeral live-activity indicator and role-pure stage output.

Covers the Claude-Code-style transient UI:
* a single 2-line ``⠋ Thinking...`` / ``└─ {action}`` indicator that overwrites in place, and
  disappears the instant a stage finishes (leaving only the clean final turn);
* Anvil renders the plan only (long implementation code blocks collapsed to a reference);
* Hammer and Temper emit role-specific deliverables (Implementation Report / QA verdict).
"""

from __future__ import annotations

from forge.ui.state import (
    TRANSIENT_SPINNER_FRAMES,
    ForgeStage,
    ForgeTUIState,
    format_stage_deliverable,
    format_transient_action,
)


def _render(state: ForgeTUIState, stage: ForgeStage, width: int = 80) -> str:
    return state.stage_buffers[stage].render_to_ansi(width=width)


def test_transient_thinking_and_tool_line_overwrites_and_disappears() -> None:
    """The live-activity indicator shows one headline + one action line, overwrites, then vanishes."""
    state = ForgeTUIState()
    state.set_active_stage(ForgeStage.ANVIL)

    # A tool call maps to a clean, single-line Claude-Code-style action phrase.
    assert (
        format_transient_action("read_file", {"path": "forge/ui/state.py"})
        == "reading forge/ui/state.py..."
    )
    assert (
        format_transient_action("run_tests", {"command": "pytest -q"})
        == "running command `pytest -q`..."
    )
    assert (
        format_transient_action("symbol_search", {"query": "ForgeTUIState"})
        == "searching codebase for 'ForgeTUIState'..."
    )
    assert (
        format_transient_action("create_directory", {"path": "src/generated"})
        == "creating directory src/generated..."
    )

    state.set_live_activity(
        ForgeStage.ANVIL, "Thinking...", format_transient_action("read_file", {"path": "a.py"})
    )
    first = _render(state, ForgeStage.ANVIL)
    assert "Thinking..." in first
    assert "reading a.py..." in first
    # Exactly one spinner headline + one subline (2 lines total).
    assert first.count("└─") == 1
    assert any(frame in first for frame in TRANSIENT_SPINNER_FRAMES)

    # A subsequent tool call OVERWRITES the previous line (never appends a second log wall).
    state.set_live_activity(
        ForgeStage.ANVIL, "Thinking...", format_transient_action("run_command", {"command": "ls"})
    )
    second = _render(state, ForgeStage.ANVIL)
    assert "running command `ls`..." in second
    assert "reading a.py..." not in second
    assert second.count("└─") == 1

    # No transient leaks across stages.
    assert "└─" not in _render(state, ForgeStage.HAMMER)

    # Finishing the stage clears the indicator entirely; only the clean output remains.
    state.clear_live_activity(ForgeStage.ANVIL)
    state.append_chat(ForgeStage.ANVIL, "Anvil", "Final Plan: refactor the parser module.")
    final = _render(state, ForgeStage.ANVIL)
    assert "Thinking..." not in final
    assert "└─" not in final
    assert not state.stage_buffers[ForgeStage.ANVIL].is_working
    assert "Final Plan: refactor the parser module." in final


def test_anvil_strips_code_blocks_and_shows_only_plan() -> None:
    """Anvil is plan-only: long implementation code blocks collapse to a Hammer reference."""
    long_block = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(12)) + "\n```"
    short_block = "```python\nreturn 1\n```"
    raw = f"Architectural Plan:\n1. Extract the tokenizer.\n\n{long_block}\n\nInline: {short_block}"

    formatted = format_stage_deliverable(ForgeStage.ANVIL, raw)

    assert "• [Target Implementation: see Hammer stage]" in formatted
    assert "x5 = 5" not in formatted  # long block body removed
    assert "1. Extract the tokenizer." in formatted
    assert "return 1" in formatted  # short snippet (< 3 lines) preserved

    # Idempotent: formatting an already-formatted plan is a no-op.
    assert format_stage_deliverable(ForgeStage.ANVIL, formatted) == formatted

    # The rendered viewport never leaks the stripped implementation code.
    state = ForgeTUIState()
    state.append_chat(ForgeStage.ANVIL, "Anvil", formatted)
    rendered = _render(state, ForgeStage.ANVIL)
    assert "x5 = 5" not in rendered
    assert "• [Target Implementation: see Hammer stage]" in rendered


def test_hammer_and_temper_role_specific_outputs() -> None:
    """Hammer leads with an Implementation Report; Temper leads with a QA & Verification Verdict."""
    hammer = format_stage_deliverable(ForgeStage.HAMMER, "Edited src/forge/ui/state.py; tests pass.")
    assert hammer.startswith("Implementation Report")
    assert "Edited src/forge/ui/state.py; tests pass." in hammer
    assert format_stage_deliverable(ForgeStage.HAMMER, hammer) == hammer

    temper = format_stage_deliverable(ForgeStage.TEMPER, "All checks PASSED with zero defects.")
    assert temper.startswith("QA & Verification Verdict")
    assert "All checks PASSED with zero defects." in temper
    assert format_stage_deliverable(ForgeStage.TEMPER, temper) == temper

    # Empty input yields empty output (never a bare heading).
    assert format_stage_deliverable(ForgeStage.HAMMER, "") == ""
    assert format_stage_deliverable(ForgeStage.TEMPER, "   ") == ""

    # Each stage's deliverable renders in its own viewport, independent of the others.
    state = ForgeTUIState()
    state.append_chat(ForgeStage.HAMMER, "Hammer", hammer)
    state.append_chat(ForgeStage.TEMPER, "Temper", temper)
    assert "Implementation Report" in _render(state, ForgeStage.HAMMER)
    assert "QA & Verification Verdict" in _render(state, ForgeStage.TEMPER)
