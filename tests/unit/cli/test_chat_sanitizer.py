"""Unit tests for chat response sanitization."""

from __future__ import annotations

from forge.cli.chat import sanitize_response_for_display


def test_sanitize_response_empty_and_falsy() -> None:
    assert sanitize_response_for_display("") == ""
    assert sanitize_response_for_display(None) == ""  # type: ignore[arg-type]


def test_sanitize_raw_tool_line_numbers() -> None:
    raw = (
        "Assets/README.md (lines 1-5 of 50)\n"
        " 1\t# Overview\n"
        " 2\t\n"
        " 3\tThis is a test description.\n"
        " 4\t| Col1 | Col2 |\n"
        " 5\t|---|---|\n"
        "[45 more lines; use offset=6]"
    )
    cleaned = sanitize_response_for_display(raw)

    assert "**`Assets/README.md`** *(lines 1-5 of 50)*" in cleaned
    assert " 1\t" not in cleaned
    assert "# Overview" in cleaned
    assert "This is a test description." in cleaned
    assert "| Col1 | Col2 |" in cleaned
    assert "*(45 more lines in file)*" in cleaned
    assert "[45 more lines; use offset=6]" not in cleaned


def test_sanitize_preserves_clean_markdown() -> None:
    clean_md = (
        "## Summary of Assets\n\n"
        "- **01_Portfolio_Command_Center.png**: Portfolio Command Center\n"
        "- **02_Investigation_Canvas.png**: Canvas for The Souled Store\n\n"
        "```python\n"
        "def hello():\n"
        "    return 'world'\n"
        "```"
    )
    assert sanitize_response_for_display(clean_md) == clean_md
