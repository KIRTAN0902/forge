"""Unit tests for the sticky bottom input box UI and keybindings in Forge chat."""

from __future__ import annotations

from unittest.mock import MagicMock

from rich.console import Console

from forge.cli.chat import (
    ChatSessionState,
    _create_prompt_session,
    _get_bottom_toolbar,
    _get_prompt_continuation,
    _get_prompt_tokens,
    position_cursor_at_bottom,
    read_user_input,
)


def test_position_cursor_at_bottom_runs_safely() -> None:
    # Ensure it doesn't crash regardless of tty state
    position_cursor_at_bottom()


def test_get_prompt_tokens_framed_box() -> None:
    state = ChatSessionState(default_agent="orchestrator")
    tokens = _get_prompt_tokens(state)

    token_text = "".join(t[1] for t in tokens)
    assert "╭─" in token_text
    assert "ANVIL" in token_text
    assert "Tab to switch" in token_text
    assert "│" in token_text
    assert "> " in token_text

    # Cycle to executor
    state.cycle_agent()
    tokens_exec = _get_prompt_tokens(state)
    text_exec = "".join(t[1] for t in tokens_exec)
    assert "HAMMER" in text_exec

    # Cycle to QA
    state.cycle_agent()
    tokens_qa = _get_prompt_tokens(state)
    text_qa = "".join(t[1] for t in tokens_qa)
    assert "TEMPER" in text_qa


def test_get_prompt_continuation() -> None:
    prefix = _get_prompt_continuation(80, 2, False)
    text = "".join(t[1] for t in prefix)
    assert "│" in text
    assert "·" in text


def test_get_bottom_toolbar_responsive() -> None:
    state = ChatSessionState()

    # Toolbar contains essential shortcut hints
    toolbar = _get_bottom_toolbar(state)
    text = "".join(t[1] for t in toolbar)
    assert "╰─" in text
    assert "[Enter] Send" in text
    assert "[Tab] Agent" in text
    assert "─╯" in text


def test_create_prompt_session_has_multiline_and_box_bindings() -> None:
    state = ChatSessionState()
    session = _create_prompt_session(state)
    assert session is not None
    assert session.multiline is True
    assert session.placeholder is not None


def test_read_user_input_with_session() -> None:
    console = Console()
    mock_session = MagicMock()
    mock_session.prompt.return_value = "hello from bottom box"

    res = read_user_input(mock_session, console)
    assert res == "hello from bottom box"
    mock_session.prompt.assert_called_once()
