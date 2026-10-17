"""Comprehensive verification suite for Forge Modals, Slash Commands, and Telemetry."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from forge.config.session_store import SessionStore
from forge.config.settings import (
    ForgeConfigManager,
    ProviderName,
)
from forge.config.usage_tracker import UsageTracker
from forge.ui.cycle_controller import ForgeCycleController
from forge.ui.split_app import (
    SLASH_COMMANDS,
    ForgeSplitPaneApp,
)
from forge.ui.state import ForgeStage, ForgeTUIState


def test_mouse_support_disabled_for_native_text_selection() -> None:
    """1. Assert that ForgeSplitPaneApp.app.mouse_support is False and hovering/moving the mouse never changes selected_view_stage."""
    state = ForgeTUIState()
    app = ForgeSplitPaneApp(state=state)

    # 1. mouse_support is explicitly False on prompt_toolkit Application
    assert not app.application.mouse_support()

    # 2. Sidebar control has show_cursor False and rendering has no mouse handler callbacks
    assert state.selected_view_stage == ForgeStage.ANVIL
    fragments = app.sidebar_control._render()
    for frag in fragments:
        # A fragment with a mouse handler is a 3-tuple (style, text, handler)
        if len(frag) > 2:
            assert frag[2] is None, f"Found active mouse handler on fragment: {frag}"

    # 3. Moving or selecting state happens strictly via keyboard/API, not mouse
    assert state.selected_view_stage == ForgeStage.ANVIL
    state.select_view_stage(ForgeStage.HAMMER)
    assert state.selected_view_stage == ForgeStage.HAMMER
    state.select_view_stage(ForgeStage.TEMPER)
    assert state.selected_view_stage == ForgeStage.TEMPER


def test_no_duplicate_stage_output_or_top_passed_banner() -> None:
    """2. Assert that StageBuffer.render_to_ansi() does NOT render FORGE // ANVIL PASSED at top and does NOT duplicate assistant reply."""
    state = ForgeTUIState()
    buf = state.stage_buffers[ForgeStage.ANVIL]
    buf.status = "passed"
    buf.stage_tokens = 450
    buf.chat_history.append(("You", "Implement distributed rate limiter"))
    buf.chat_history.append(("Anvil", "Token bucket rate limiter specification completed."))
    buf.final_output = "Token bucket rate limiter specification completed."

    rendered = buf.render_to_ansi(width=80)

    # Top status line is removed
    assert "FORGE // ANVIL" not in rendered
    assert "FORGE // ANVIL   PASSED" not in rendered

    # Duplicate ── Stage Output ── is removed
    assert "── Stage Output ──" not in rendered
    assert "Verified Final Output Deliverable" not in rendered

    # Assistant output appears exactly ONCE in the conversational transcript stream
    assert "Token bucket rate limiter specification completed." in rendered
    assert rendered.count("Token bucket rate limiter specification completed.") == 1


@pytest.mark.asyncio
async def test_multi_provider_connect_and_cross_provider_stage_models(tmp_path: Path) -> None:
    """3. Connect Provider 1 (gemini), Provider 2 (openrouter), and custom, verify /models aggregation and cross-provider stage binding without restart."""
    config_file = tmp_path / "config.json"
    manager = ForgeConfigManager(config_path=config_file, probe_ollama=False)
    state = ForgeTUIState()
    controller = ForgeCycleController(state=state, config_manager=manager)
    app = ForgeSplitPaneApp(state=state, controller=controller)

    # 1. Connect Gemini with API key only (no URL prompt needed)
    manager.set_api_key(ProviderName.GEMINI, "AIzaSyGeminiKey12345")
    assert manager.is_connected(ProviderName.GEMINI) is True

    # 2. Connect OpenRouter with API key only (no URL prompt needed)
    manager.set_api_key(ProviderName.OPENROUTER, "sk-or-v1-OpenRouterKeyABC")
    assert manager.is_connected(ProviderName.OPENROUTER) is True

    # 3. Connect Custom Provider with URL + API key
    manager.set_base_url(ProviderName.CUSTOM, "http://localhost:8000/v1")
    manager.set_api_key(ProviderName.CUSTOM, "custom-secret-key")
    assert manager.is_connected(ProviderName.CUSTOM) is True

    # Aggregated models in /models modal must gather from all connected providers
    app.modal_manager.refresh_aggregated_models()
    aggregated = app.modal_manager.cached_aggregated_models
    prov_set = {p for p, _ in aggregated}
    assert ProviderName.GEMINI in prov_set
    assert ProviderName.OPENROUTER in prov_set
    assert ProviderName.CUSTOM in prov_set

    # Cross-Provider Mixing: Anvil -> Gemini, Hammer -> OpenRouter, Temper -> Gemini
    manager.set_stage_model("anvil", "gemini/gemini-2.5-pro")
    manager.set_stage_model("hammer", "openrouter/anthropic/claude-3.5-sonnet")
    manager.set_stage_model("temper", "gemini/gemini-2.5-flash")

    # Immediate live activation in memory with zero restart
    b_anvil = manager.binding_for("anvil")
    b_hammer = manager.binding_for("hammer")
    b_temper = manager.binding_for("temper")
    assert b_anvil is not None and b_anvil.spec_string == "gemini/gemini-2.5-pro"
    assert b_hammer is not None and b_hammer.spec_string == "openrouter/anthropic/claude-3.5-sonnet"
    assert b_temper is not None and b_temper.spec_string == "gemini/gemini-2.5-flash"

    # Persisted to ~/.forge/config.json
    reloaded = ForgeConfigManager(config_path=config_file, probe_ollama=False)
    assert reloaded.binding_for("anvil").spec_string == "gemini/gemini-2.5-pro"  # type: ignore[union-attr]
    assert reloaded.binding_for("hammer").spec_string == "openrouter/anthropic/claude-3.5-sonnet"  # type: ignore[union-attr]
    assert reloaded.binding_for("temper").spec_string == "gemini/gemini-2.5-flash"  # type: ignore[union-attr]


def test_removed_key_and_singular_model_commands() -> None:
    """4. Verify /key and /model are removed from the completer, and /provider, /models, /usage, /resume, /restart are registered and functional."""
    state = ForgeTUIState()
    app = ForgeSplitPaneApp(state=state)
    completer = app.completer

    # /key and /model are removed from registry
    assert "/key" not in SLASH_COMMANDS
    assert "/model" not in SLASH_COMMANDS

    # /provider, /models, /usage, /resume, /restart are registered
    assert "/provider" in SLASH_COMMANDS
    assert "/models" in SLASH_COMMANDS
    assert "/usage" in SLASH_COMMANDS
    assert "/resume" in SLASH_COMMANDS
    assert "/restart" in SLASH_COMMANDS

    # Completer does not yield /key or /model
    all_cmds = {
        c.text for c in completer.get_completions(Document("/"), CompleteEvent())
    }
    assert "/key" not in all_cmds
    assert "/model" not in all_cmds
    assert "/provider" in all_cmds
    assert "/models" in all_cmds
    assert "/usage" in all_cmds
    assert "/resume" in all_cmds
    assert "/restart" in all_cmds

    # Placeholder hint contains new commands and no /key or /model
    hint_text = "".join(part[1] for part in app._render_input_hint())
    assert "/provider" in hint_text
    assert "/models" in hint_text
    assert "/usage" in hint_text
    assert "/resume" in hint_text
    assert "/restart" in hint_text
    assert "/key" not in hint_text
    assert "/model," not in hint_text


def test_global_session_store_and_resume(tmp_path: Path) -> None:
    """5. Simulate a conversation in directory A, verify it saves globally under ~/.forge/sessions/, and verify /resume restores state in directory B."""
    sessions_dir = tmp_path / "global_sessions"
    store = SessionStore(sessions_dir=sessions_dir)

    # 1. Active conversation in workspace A
    workspace_a = tmp_path / "workspace_a"
    workspace_a.mkdir()

    state_a = ForgeTUIState()
    state_a.cycle_count = 2
    state_a.active_stage_tokens = 920
    state_a.total_session_tokens = 4100
    state_a.append_chat(ForgeStage.ANVIL, "You", "Architect a distributed ledger consensus system")
    state_a.append_chat(ForgeStage.ANVIL, "Anvil", "Raft consensus protocol plan ready.")
    state_a.append_thought(ForgeStage.ANVIL, "Analyzing leader election timeout invariants.")
    state_a.append_chat(ForgeStage.HAMMER, "Hammer", "Wrote raft_node.py implementation.")
    state_a.append_tool_event(ForgeStage.HAMMER, "Tool: pytest tests/test_raft.py | Exit: 0")
    state_a.append_chat(ForgeStage.TEMPER, "Temper", "Consensus verified under 30% packet loss.")

    session_id = "raft_session_001"
    store.save_session(state_a, session_id=session_id, cwd=workspace_a)

    # Check global session file & index exist
    assert (sessions_dir / f"{session_id}.json").is_file()
    assert (sessions_dir / "index.json").is_file()

    # 2. Restore in workspace B
    workspace_b = tmp_path / "workspace_b"
    workspace_b.mkdir()

    state_b = ForgeTUIState()
    assert state_b.cycle_count == 1
    assert state_b.total_session_tokens == 0
    assert len(state_b.stage_buffers[ForgeStage.ANVIL].chat_history) == 0

    restored = store.restore_session(session_id, state_b)
    assert restored is True
    assert state_b.cycle_count == 2
    assert state_b.total_session_tokens == 4100
    assert state_b.active_stage_tokens == 920

    anvil_buf = state_b.stage_buffers[ForgeStage.ANVIL]
    assert len(anvil_buf.chat_history) == 2
    assert "Architect a distributed ledger" in anvil_buf.chat_history[0][1]
    assert "Raft consensus protocol plan ready" in anvil_buf.chat_history[1][1]
    assert any("Analyzing leader election timeout invariants" in t for t in anvil_buf.thoughts)

    hammer_buf = state_b.stage_buffers[ForgeStage.HAMMER]
    assert "Wrote raft_node.py implementation" in hammer_buf.chat_history[0][1]
    assert "Tool: pytest tests/test_raft.py | Exit: 0" in hammer_buf.tool_events

    temper_buf = state_b.stage_buffers[ForgeStage.TEMPER]
    assert "Consensus verified under 30% packet loss" in temper_buf.chat_history[0][1]


def test_usage_dashboard_aggregation_by_provider_and_model(tmp_path: Path) -> None:
    """6. Record multiple LLM calls across gemini and openrouter, run /usage, and assert exact breakdown per provider and model."""
    ledger_path = tmp_path / "usage_ledger.json"
    tracker = UsageTracker(ledger_path=ledger_path)

    # Record calls across Gemini, OpenRouter, and local Ollama
    tracker.record_call(
        provider="gemini",
        model="gemini-2.5-pro",
        stage="anvil",
        prompt_tokens=48210,
        completion_tokens=12400,
        latency_ms=1450.0,
        session_id="session_42",
    )
    tracker.record_call(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        stage="hammer",
        prompt_tokens=84100,
        completion_tokens=31200,
        latency_ms=2300.0,
        session_id="session_42",
    )
    tracker.record_call(
        provider="ollama",
        model="qwen2.5-coder:7b",
        stage="temper",
        prompt_tokens=62000,
        completion_tokens=19500,
        latency_ms=450.0,
        session_id="session_42",
    )

    # Provider aggregations
    by_prov = tracker.aggregate_by_provider()
    assert "gemini" in by_prov
    assert by_prov["gemini"]["requests"] == 1
    assert by_prov["gemini"]["prompt_tokens"] == 48210
    assert by_prov["gemini"]["completion_tokens"] == 12400
    assert by_prov["gemini"]["total_tokens"] == 60610
    assert by_prov["gemini"]["estimated_cost"] > 0

    assert "openrouter" in by_prov
    assert by_prov["openrouter"]["requests"] == 1
    assert by_prov["openrouter"]["total_tokens"] == 115300
    assert by_prov["openrouter"]["estimated_cost"] > 0

    assert "ollama" in by_prov
    assert by_prov["ollama"]["total_tokens"] == 81500
    assert by_prov["ollama"]["estimated_cost"] == 0.0

    # Model aggregations
    by_model = tracker.aggregate_by_model()
    model_map = {m["model"]: m for m in by_model}
    assert "gemini-2.5-pro" in model_map
    assert model_map["gemini-2.5-pro"]["total_tokens"] == 60610
    assert "anthropic/claude-3.5-sonnet" in model_map
    assert model_map["anthropic/claude-3.5-sonnet"]["total_tokens"] == 115300
    assert "qwen2.5-coder:7b" in model_map
    assert model_map["qwen2.5-coder:7b"]["estimated_cost"] == 0.0

    # Rendered dashboard text check
    dashboard = tracker.render_dashboard_text(session_id="session_42", time_window="all")
    assert "FORGE // API USAGE & COST TELEMETRY DASHBOARD" in dashboard
    assert "gemini" in dashboard
    assert "gemini-2.5-pro" in dashboard
    assert "openrouter" in dashboard
    assert "anthropic/claude-3.5-sonnet" in dashboard
    assert "ollama" in dashboard
    assert "qwen2.5-coder:7b" in dashboard
    assert "$0.0000" in dashboard
