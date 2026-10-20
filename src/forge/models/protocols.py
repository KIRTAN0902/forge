"""Prompt-level protocols used when a model lacks a native capability.

* **Tool calling**: tools are described in the system prompt and the model emits
  ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks. This is the Hermes
  format many open models (Qwen, Hermes, some Llama fine-tunes) were trained on.
* **Structured output**: the JSON schema goes in the prompt; the reply is extracted
  leniently (code fences, surrounding prose) and validated, with repair round-trips.
* **Inline reasoning**: ``<think>...</think>`` blocks some models emit in the content are
  moved out of the answer so they never pollute parsing.
"""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from forge.models.types import Message, Role, ToolCall, ToolDefinition

# -- inline reasoning ------------------------------------------------------------------


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Return (answer, reasoning) with inline reasoning blocks removed from the answer."""
    from forge.providers.client import strip_thinking_tags

    answer, reasoning = strip_thinking_tags(text)
    return answer.strip(), reasoning.strip() or None


# -- tool calling ----------------------------------------------------------------------

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


def render_tool_instructions(tools: list[ToolDefinition], tool_choice: str | None) -> str:
    lines = [
        "# Tools",
        "",
        "You can call the tools below. To call a tool, output a block exactly like this:",
        "",
        TOOL_CALL_OPEN,
        '{"name": "<tool name>", "arguments": {<arguments as a JSON object>}}',
        TOOL_CALL_CLOSE,
        "",
        "Rules:",
        "- The block must contain one valid JSON object with keys `name` and `arguments`.",
        "- You may output several blocks to call several tools; they run in order.",
        "- After your tool calls, stop. Results come back in <tool_result> messages.",
    ]
    if tool_choice == "required":
        lines.append("- You MUST call at least one tool in this reply.")
    elif tool_choice and tool_choice not in {"auto", "none"}:
        lines.append(f"- You MUST call the `{tool_choice}` tool in this reply.")
    else:
        lines.append("- If no tool is needed, answer normally without any tool_call block.")
    lines += ["", "## Available tools", ""]
    for tool in tools:
        lines.append(f"### {tool.name}")
        lines.append(tool.description.strip())
        lines.append("Arguments JSON schema: " + json.dumps(tool.parameters, separators=(",", ":")))
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_text_tool_calls(
    text: str, tools: list[ToolDefinition], *, id_prefix: str = "call"
) -> tuple[str, list[ToolCall]]:
    """Extract ``<tool_call>`` blocks. Returns (remaining text, calls).

    Blocks naming unknown tools are still returned (with a parse error) so the agent
    loop can tell the model, instead of silently dropping its intent.
    """
    known = {t.name for t in tools}
    calls: list[ToolCall] = []
    for index, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        calls.append(_build_call(match.group(1), known, f"{id_prefix}_{index}"))
    remaining = _TOOL_CALL_RE.sub("", text).strip()

    if not calls:
        # Fallback for malformed output from weaker models: a bare or fenced
        # {"name": <known tool>, "arguments": {...}} object, possibly surrounded by stray
        # text or a dangling </tool_call>. Only objects naming a known tool qualify, so
        # ordinary JSON in an answer is never mistaken for a call.
        spans = _known_call_objects(text, known)
        for index, (start, end) in enumerate(spans):
            calls.append(_build_call(text[start:end], known, f"{id_prefix}_{index}"))
        if spans:
            pieces, cursor = [], 0
            for start, end in spans:
                pieces.append(text[cursor:start])
                cursor = end
            pieces.append(text[cursor:])
            remaining = re.sub(r"</?tool_call>|```(?:json)?", "", "".join(pieces)).strip()
    return remaining, calls


def _known_call_objects(text: str, known: set[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = text.find("{")
    while start != -1:
        end = _matching_bracket(text, start, "{", "}")
        if end == -1:
            break
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and obj.get("name") in known and "arguments" in obj:
            spans.append((start, end + 1))
            start = text.find("{", end + 1)
        else:
            start = text.find("{", start + 1)
    return spans


def _build_call(body: str, known: set[str], call_id: str) -> ToolCall:
    body = _strip_fence(body.strip())
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        return ToolCall(
            id=call_id, name="", raw_arguments=body, parse_error=f"invalid JSON: {exc.msg}"
        )
    if not isinstance(obj, dict):
        return ToolCall(id=call_id, name="", raw_arguments=body, parse_error="expected an object")
    name = str(obj.get("name") or obj.get("tool") or "")
    arguments = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(arguments, str):
        with contextlib.suppress(json.JSONDecodeError):
            arguments = json.loads(arguments)
    error = None
    if not isinstance(arguments, dict):
        error, arguments = "arguments must be a JSON object", {}
    elif name not in known:
        error = f"unknown tool {name!r}"
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
        parse_error=error,
    )


def render_tool_call(call: ToolCall) -> str:
    payload = {"name": call.name, "arguments": call.arguments}
    return f"{TOOL_CALL_OPEN}\n{json.dumps(payload)}\n{TOOL_CALL_CLOSE}"


def to_prompt_protocol(messages: list[Message]) -> list[Message]:
    """Rewrite native tool-call history into the text protocol.

    Assistant tool calls become ``<tool_call>`` text, and tool results become user
    messages, since models without tool support usually reject the ``tool`` role.
    Consecutive same-role messages are merged because many chat templates require
    strict user/assistant alternation.
    """
    converted: list[Message] = []
    for message in messages:
        if message.role == Role.ASSISTANT and message.tool_calls:
            parts = [message.text] if message.text else []
            parts += [render_tool_call(c) for c in message.tool_calls]
            converted.append(Message.assistant("\n".join(parts)))
        elif message.role == Role.TOOL:
            name = message.name or "tool"
            converted.append(
                Message.user(
                    f'<tool_result name="{name}" id="{message.tool_call_id}">\n'
                    f"{message.text}\n</tool_result>"
                )
            )
        else:
            converted.append(message)
    return _merge_consecutive(converted)


def _merge_consecutive(messages: list[Message]) -> list[Message]:
    merged: list[Message] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.role == message.role
            and message.role in {Role.USER, Role.ASSISTANT}
            and isinstance(previous.content, str | None)
            and isinstance(message.content, str | None)
            and not previous.tool_calls
            and not message.tool_calls
        ):
            merged[-1] = previous.model_copy(
                update={"content": f"{previous.text}\n\n{message.text}".strip()}
            )
        else:
            merged.append(message)
    return merged


def with_system_suffix(messages: list[Message], suffix: str) -> list[Message]:
    """Append ``suffix`` to the system prompt, creating one if needed."""
    if messages and messages[0].role == Role.SYSTEM:
        first = messages[0]
        return [first.model_copy(update={"content": f"{first.text}\n\n{suffix}"}), *messages[1:]]
    return [Message.system(suffix), *messages]


# -- structured output -----------------------------------------------------------------


def render_schema_instructions(schema: dict[str, Any], *, native: bool) -> str:
    if native:
        return "Respond with a single JSON object that conforms to the provided JSON schema."
    return (
        "# Output format\n\n"
        "Respond with ONLY a single JSON object (no prose, no markdown) that conforms to "
        "this JSON schema:\n\n" + json.dumps(schema, separators=(",", ":"))
    )


def render_repair_message(error: str) -> str:
    return (
        "Your previous reply could not be used: "
        f"{error}\n\nReply again with ONLY the corrected JSON object."
    )


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    return match.group(1).strip() if match else text


def extract_json(text: str) -> Any:
    """Parse the JSON value in a model reply, tolerating fences and surrounding prose."""
    text, _ = split_reasoning(text)
    candidate = _strip_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} or [...] span that parses.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        while start != -1:
            end = _matching_bracket(candidate, start, opener, closer)
            if end != -1:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    pass
            start = candidate.find(opener, start + 1)
    raise ValueError("no JSON object found in the reply")


def _matching_bracket(text: str, start: int, opener: str, closer: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return -1
