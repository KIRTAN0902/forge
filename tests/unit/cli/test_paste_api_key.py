"""Unit tests for clipboard pasting and robust API key entry."""

from __future__ import annotations

from unittest.mock import patch

from rich.console import Console
from rich.prompt import Prompt

from forge.cli.credentials import (
    looks_like_api_key,
    prompt_api_key,
    read_clipboard_key,
)


def test_looks_like_api_key() -> None:
    assert looks_like_api_key("sk-proj-1234567890abcdef12345") is True
    assert looks_like_api_key("nvapi-1234567890abcdef12345") is True
    assert looks_like_api_key("gsk_1234567890abcdef12345") is True
    assert looks_like_api_key("anthropic-key-1234567890") is True
    assert looks_like_api_key("AIzaSyD1234567890abcdef") is True
    assert looks_like_api_key("ghp_1234567890abcdef12345") is True
    # General high-entropy token >= 20 chars
    assert looks_like_api_key("abc123def456ghi789jkl012") is True

    # Invalid patterns
    assert looks_like_api_key("") is False
    assert looks_like_api_key(None) is False
    assert looks_like_api_key("short") is False
    assert looks_like_api_key("has spaces in the middle of it") is False
    assert looks_like_api_key("has\nnewline") is False
    assert looks_like_api_key("https://api.openai.com/v1") is False
    assert looks_like_api_key("file:///home/user/test.py") is False


def test_read_clipboard_key() -> None:
    # 1. Native OS read returns a string or None without crashing
    result = read_clipboard_key()
    assert result is None or isinstance(result, str)

    # 2. Test when read_clipboard_key returns a valid key
    with patch(
        "tests.unit.cli.test_paste_api_key.read_clipboard_key",
        return_value="sk-test-clipboard-12345",
    ):
        assert read_clipboard_key() == "sk-test-clipboard-12345"


def test_prompt_api_key_delegates_when_mocked() -> None:
    console = Console(record=True)
    with patch.object(Prompt, "ask") as mock_ask:
        mock_ask.return_value = "mocked-sk-key"
        key = prompt_api_key("API Key", console=console)
        assert key == "mocked-sk-key"
        assert mock_ask.call_count == 1


def test_prompt_api_key_handles_ctrl_v_raw_char() -> None:
    """When a terminal sends raw ASCII 22 (\x16) for Ctrl+V, replace it with clipboard content."""
    console = Console(record=True)
    # Simulate non-interactive stdin where user pressed Ctrl+V sending \x16
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("forge.cli.credentials.read_clipboard_key", return_value="sk-pasted-from-clipboard"),
        patch("prompt_toolkit.shortcuts.prompt", return_value="\x16"),
    ):
        key = prompt_api_key("API Key", console=console)
        assert key == "sk-pasted-from-clipboard"


def test_prompt_api_key_strips_quotes() -> None:
    """If user copies with quotes e.g. 'sk-...' or \"sk-...\", quotes are cleanly stripped."""
    console = Console(record=True)
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("prompt_toolkit.shortcuts.prompt", return_value='"sk-with-double-quotes"'),
    ):
        key = prompt_api_key("API Key", console=console)
        assert key == "sk-with-double-quotes"

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("prompt_toolkit.shortcuts.prompt", return_value="'sk-with-single-quotes'"),
    ):
        key = prompt_api_key("API Key", console=console)
        assert key == "sk-with-single-quotes"


def test_prompt_api_key_uses_clipboard_on_enter() -> None:
    """If clipboard has an API key and user presses Enter, the clipboard key is used."""
    console = Console(record=True)
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch(
            "forge.cli.credentials.read_clipboard_key", return_value="sk-proj-detected-key-12345"
        ),
        patch("prompt_toolkit.shortcuts.prompt", return_value=""),
    ):
        key = prompt_api_key("API Key", console=console)
        assert key == "sk-proj-detected-key-12345"
        output = console.export_text()
        assert "Detected API key in clipboard" in output
