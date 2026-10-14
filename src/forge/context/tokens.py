"""Model-agnostic token estimation.

Exact counts need each model's tokenizer, which Forge deliberately does not depend on.
Source code averages roughly 3.5-4 characters per token across modern BPE tokenizers;
3.5 errs on the side of over-estimating, so budgets are conservative.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

from forge.models.types import Message

CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


def estimate_message_tokens(messages: Iterable[Message]) -> int:
    total = 0
    for message in messages:
        total += MESSAGE_OVERHEAD_TOKENS + estimate_tokens(message.text)
        for call in message.tool_calls:
            total += estimate_tokens(call.name) + estimate_tokens(json.dumps(call.arguments))
    return total


def truncate_to_tokens(text: str, tokens: int, *, marker: str = "") -> str:
    """Cut ``text`` to roughly ``tokens``, on a line boundary when possible."""
    limit = int(tokens * CHARS_PER_TOKEN)
    if len(text) <= limit:
        return text
    cut = text[: max(limit - len(marker), 0)]
    newline = cut.rfind("\n")
    if newline > limit * 0.6:
        cut = cut[:newline]
    return cut + marker
