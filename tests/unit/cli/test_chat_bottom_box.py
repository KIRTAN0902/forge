"""Unit tests for the bottom-fixed proper enclosed box UI and keybindings in Forge chat."""

from __future__ import annotations

from unittest.mock import MagicMock

from rich.console import Console

from forge.cli.chat import (
    ChatSessionState,
    _create_prompt_session,
    _get_bottom_toolbar,
    _get_prompt_continuation,
    _get_prompt_tokens,
    _get_rprompt,
    position_cursor_at_bottom,
    read_user_input,
)


def test_position_cursor_at_bottom_runs_safely() -> None:
    # Ensure it executes safely across any environment (headless, tty, Windows conout)
    position_cursor_at_bottom(box_height=3)
    position_cursor_at_bottom()


def test_get_prompt_tokens_enclosed_box() -> None:
    state = ChatSessionState(default_agent="orchestrator")
    tokens = _get_prompt_tokens(state)

    token_text = "".join(t[1] for t in tokens)
    assert "╭─" in token_text
    assert "ANVIL" in token_text
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


def test_get_rprompt() -> None:
    state = ChatSessionState()
    rprompt = _get_rprompt(state)
    assert rprompt == [("class:box_border", "│")]


def test_get_prompt_continuation() -> None:
    prefix = _get_prompt_continuation(80, 2, False)
    text = "".join(t[1] for t in prefix)
    assert "│" in text
    assert "·" in text


def test_get_bottom_toolbar_enclosed() -> None:
    state = ChatSessionState()
    toolbar = _get_bottom_toolbar(state)
    assert toolbar is not None
    text = "".join(t[1] for t in toolbar)
    assert "╰─" in text
    assert "[Enter] Send" in text
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
    mock_session.prompt.return_value = "hello from proper box"

    res = read_user_input(mock_session, console)
    assert res == "hello from proper box"
    mock_session.prompt.assert_called_once()
