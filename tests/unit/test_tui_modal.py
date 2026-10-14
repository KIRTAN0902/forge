"""Unit tests for the interactive TUI modal dialog."""

from __future__ import annotations

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from forge.cli.tui_modal import CTRL_A_TOKEN, TuiItem, run_tui_modal


def test_tui_modal_select_item() -> None:
    """Pressing Enter immediately selects the first item."""
    items = [
        TuiItem(id="gpt-4o", title="GPT-4o", subtitle="OpenAI", section="Popular", is_dot=True),
        TuiItem(id="claude-3.7", title="Claude 3.7", subtitle="Anthropic", section="Popular"),
    ]
    with create_pipe_input() as pipe:
        pipe.send_text("\r")
        res = run_tui_modal(
            title="Select model",
            items=items,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert isinstance(res, TuiItem)
        assert res.id == "gpt-4o"


def test_tui_modal_arrow_down_select() -> None:
    """Down arrow moves selection to next item."""
    items = [
        TuiItem(id="gpt-4o", title="GPT-4o", subtitle="OpenAI", section="Popular"),
        TuiItem(id="claude-3.7", title="Claude 3.7", subtitle="Anthropic", section="Popular"),
    ]
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[B\r")  # Down arrow + Enter
        res = run_tui_modal(
            title="Select model",
            items=items,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert isinstance(res, TuiItem)
        assert res.id == "claude-3.7"


def test_tui_modal_search_filtering() -> None:
    """Typing text filters items dynamically."""
    items = [
        TuiItem(id="gpt-4o", title="GPT-4o", subtitle="OpenAI", section="Popular"),
        TuiItem(id="claude-3.7", title="Claude 3.7", subtitle="Anthropic", section="Popular"),
        TuiItem(id="deepseek-r1", title="DeepSeek R1", subtitle="DeepSeek", section="Popular"),
    ]
    with create_pipe_input() as pipe:
        pipe.send_text("deep\r")  # Type 'deep' + Enter
        res = run_tui_modal(
            title="Select model",
            items=items,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert isinstance(res, TuiItem)
        assert res.id == "deepseek-r1"


def test_tui_modal_custom_input() -> None:
    """Typing an unknown model and pressing Enter returns the custom string as an item."""
    items = [
        TuiItem(id="gpt-4o", title="GPT-4o", subtitle="OpenAI", section="Popular"),
    ]
    with create_pipe_input() as pipe:
        pipe.send_text("my-org/my-model:v1\r")
        res = run_tui_modal(
            title="Select model",
            items=items,
            allow_custom=True,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert isinstance(res, TuiItem)
        assert res.id == "my-org/my-model:v1"


def test_tui_modal_ctrl_a() -> None:
    """Pressing Ctrl+A returns the CTRL_A_TOKEN."""
    items = [
        TuiItem(id="gpt-4o", title="GPT-4o", subtitle="OpenAI"),
    ]
    with create_pipe_input() as pipe:
        pipe.send_text("\x01")  # Ctrl+A
        res = run_tui_modal(
            title="Select model",
            items=items,
            allow_ctrl_a=True,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert res == CTRL_A_TOKEN


def test_tui_modal_escape_cancel() -> None:
    """Pressing Escape returns None."""
    items = [
        TuiItem(id="gpt-4o", title="GPT-4o", subtitle="OpenAI"),
    ]
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b")  # Escape
        res = run_tui_modal(
            title="Select model",
            items=items,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert res is None
