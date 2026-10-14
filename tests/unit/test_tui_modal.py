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


def test_tui_modal_multi_word_search() -> None:
    """Multi-word search matches all terms regardless of order."""
    items = [
        TuiItem(id="anthropic/claude-3.7-sonnet", title="claude-3.7-sonnet", subtitle="Anthropic"),
        TuiItem(id="openai/gpt-4o", title="gpt-4o", subtitle="OpenAI"),
        TuiItem(id="nvidia/nemotron-3-super", title="nemotron-3-super", subtitle="NVIDIA"),
    ]
    with create_pipe_input() as pipe:
        pipe.send_text("sonnet 3.7\r")  # Multi-word query
        res = run_tui_modal(
            title="Select model",
            items=items,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert isinstance(res, TuiItem)
        assert res.id == "anthropic/claude-3.7-sonnet"


def test_tui_modal_backspace_search() -> None:
    """Typing characters and backspacing updates search and selection."""
    items = [
        TuiItem(id="item-a", title="Alpha"),
        TuiItem(id="item-b", title="Beta"),
    ]
    with create_pipe_input() as pipe:
        # Type 'z' (no match), backspace, then Enter selects first item Alpha
        pipe.send_text("z\x7f\r")
        res = run_tui_modal(
            title="Select model",
            items=items,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert isinstance(res, TuiItem)
        assert res.id == "item-a"


def test_tui_modal_pagedown_navigation() -> None:
    """PageDown jumps down multiple items."""
    items = [TuiItem(id=f"item-{i}", title=f"Item {i}") for i in range(12)]
    with create_pipe_input() as pipe:
        pipe.send_text("\x1b[6~\r")  # PageDown + Enter
        res = run_tui_modal(
            title="Select model",
            items=items,
            input_override=pipe,
            output_override=DummyOutput(),
        )
        assert isinstance(res, TuiItem)
        assert res.id == "item-5"


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


def test_tui_modal_headless_returns_none() -> None:
    """In headless / non-interactive mode without input override, returns None safely."""
    items = [TuiItem(id="item-1", title="Item 1")]
    res = run_tui_modal(title="Headless test", items=items)
    assert res is None
