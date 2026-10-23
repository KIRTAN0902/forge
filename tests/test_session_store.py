"""Session persistence safety, prefix resolution, and stale-state replacement."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.config.session_store import SessionStore
from forge.ui.state import ForgeStage, ForgeTUIState


def test_session_prefix_resolves_to_exact_stored_id(tmp_path: Path) -> None:
    store = SessionStore(sessions_dir=tmp_path / "sessions")
    state = ForgeTUIState()
    state.append_chat(ForgeStage.ANVIL, "You", "Build a parser")

    store.save_session(state, "abcdef123456", cwd=tmp_path)

    assert store.resolve_session_id("abcdef") == "abcdef123456"
    assert store.resolve_session_id("abcdef123456") == "abcdef123456"
    restored = ForgeTUIState()
    assert store.restore_session("abcdef", restored) is True
    assert restored.stage_buffers[ForgeStage.ANVIL].chat_history == [
        ("You", "Build a parser")
    ]


def test_restore_clears_stale_buffers_missing_from_saved_session(tmp_path: Path) -> None:
    store = SessionStore(sessions_dir=tmp_path / "sessions")
    saved = ForgeTUIState()
    saved.append_chat(ForgeStage.ANVIL, "You", "new task")
    store.save_session(saved, "fresh", cwd=tmp_path)

    target = ForgeTUIState()
    target.append_chat(ForgeStage.HAMMER, "Hammer", "stale output")
    target.append_tool_event(ForgeStage.TEMPER, "stale tool event")

    assert store.restore_session("fresh", target) is True
    assert target.stage_buffers[ForgeStage.ANVIL].chat_history == [("You", "new task")]
    assert target.stage_buffers[ForgeStage.HAMMER].chat_history == []
    assert target.stage_buffers[ForgeStage.TEMPER].tool_events == []


def test_session_ids_cannot_escape_the_store(tmp_path: Path) -> None:
    store = SessionStore(sessions_dir=tmp_path / "sessions")
    state = ForgeTUIState()

    with pytest.raises(ValueError, match="invalid session id"):
        store.save_session(state, "../escape", cwd=tmp_path)

    assert store.load_session("../escape") is None
    assert store.resolve_session_id("..\\escape") is None
    assert not (tmp_path / "escape.json").exists()
