"""Keeping long agent conversations inside the context window.

Compaction is deterministic and preserves what matters most:

* the system prompt and the task (first user message) are never touched;
* the most recent turns are kept intact (an assistant tool call is never separated
  from its results);
* older tool results are elided first (they are usually re-readable), then older turns
  are replaced by a short note listing what was done.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from forge.context.tokens import estimate_message_tokens
from forge.models.types import Message, Role


def compact_history(
    messages: Sequence[Message],
    budget_tokens: int,
    *,
    keep_recent: int = 8,
    elided_chars: int = 300,
) -> list[Message]:
    history = list(messages)
    if estimate_message_tokens(history) <= budget_tokens:
        return history

    head_end = _head_end(history)
    tail_start = _tail_start(history, max(len(history) - keep_recent, head_end))
    head, middle, tail = history[:head_end], history[head_end:tail_start], history[tail_start:]

    # Stage 1: shorten old tool results and long assistant monologues.
    shortened = [_shorten(m, elided_chars) for m in middle]
    candidate = [*head, *shortened, *tail]
    if estimate_message_tokens(candidate) <= budget_tokens:
        return candidate

    # Stage 2: replace the middle with a summary note.
    note = Message.user(_summary_note(middle))
    candidate = [*head, note, *tail] if middle else [*head, *tail]
    if estimate_message_tokens(candidate) <= budget_tokens:
        return candidate

    # Stage 3: shorten tool results in the recent turns too, except the very latest ones.
    protected = 2
    recent = [
        _shorten(m, elided_chars) if i < len(tail) - protected else m for i, m in enumerate(tail)
    ]
    return [*head, note, *recent] if middle else [*head, *recent]


def _head_end(history: list[Message]) -> int:
    index = 0
    while index < len(history) and history[index].role == Role.SYSTEM:
        index += 1
    if index < len(history) and history[index].role == Role.USER:
        index += 1
    return index


def _tail_start(history: list[Message], start: int) -> int:
    # Never start the tail on a tool result: its assistant call must come with it.
    while start > 0 and history[start].role == Role.TOOL:
        start -= 1
    return start


def _shorten(message: Message, limit: int) -> Message:
    text = message.text
    if message.role == Role.TOOL and len(text) > limit:
        return message.model_copy(
            update={
                "content": f"{text[:limit]}\n... [{len(text) - limit} characters elided to save "
                "context; re-run the tool if you need them]"
            }
        )
    if message.role == Role.ASSISTANT and len(text) > 4 * limit:
        return message.model_copy(
            update={"content": f"{text[: 2 * limit]}\n... [rest of this message elided]"}
        )
    return message


def _summary_note(middle: Sequence[Message]) -> str:
    calls = [c for m in middle for c in m.tool_calls]
    counts = Counter(c.name for c in calls)
    paths = sorted(
        {
            str(c.arguments.get("path"))
            for c in calls
            if c.name in {"write_file", "edit_file", "delete_file"} and c.arguments.get("path")
        }
    )
    parts = [f"[{len(middle)} earlier messages were removed to fit the context window."]
    if counts:
        parts.append(
            "They included tool calls: "
            + ", ".join(f"{n} x{k}" for n, k in counts.most_common())
            + "."
        )
    if paths:
        parts.append("Files modified earlier: " + ", ".join(paths) + ".")
    parts.append("Re-read files or re-run commands if you need details.]")
    return " ".join(parts)
