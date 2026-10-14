"""Phase 1 exit condition: two different models can be swapped without changing agent code.

``solve`` stands in for an agent. It is written once against ``ModelAdapter`` and run
unchanged against two simulated models with very different capabilities:

* a "frontier" model with native tool calling and JSON-schema output, which rejects
  prompt-protocol text;
* a "small local" model with no tool or JSON support at all, which errors if it is sent
  native tool definitions or response formats.

Only the configuration differs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from forge.models import (
    Message,
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
    ModelRegistry,
    ModelRequest,
    ModelResponse,
    Role,
    ToolDefinition,
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response

ADD = ToolDefinition(
    name="add",
    description="Add two integers.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)


class Answer(BaseModel):
    result: int
    explanation: str


async def solve(adapter: ModelAdapter) -> Answer:
    """A tiny agent: use a tool, then report a structured answer."""
    messages = [
        Message.system("You are a careful assistant. Use tools for arithmetic."),
        Message.user("What is 17 + 25?"),
    ]
    for _ in range(4):
        response = await adapter.generate(messages, tools=[ADD])
        if not response.tool_calls:
            break
        messages.append(response.message)
        for call in response.tool_calls:
            result = call.arguments["a"] + call.arguments["b"]
            messages.append(Message.tool_result(call, str(result)))
    messages.append(Message.user("Report the final answer."))
    return await adapter.structured(messages, Answer)


def _tool_results(request: ModelRequest) -> list[str]:
    return [m.text for m in request.messages if m.role == Role.TOOL]


def frontier_model(request: ModelRequest) -> ModelResponse | str:
    assert not any("<tool_call>" in m.text for m in request.messages if m.role == Role.SYSTEM)
    if request.response_format is not None:
        value = int(_tool_results(request)[-1])
        return json.dumps({"result": value, "explanation": "used the add tool"})
    if request.tools and not _tool_results(request):
        return tool_call_response("add", {"a": 17, "b": 25})
    return "The sum is 42."


def small_local_model(request: ModelRequest) -> str:
    if request.tools or request.response_format is not None or request.json_mode:
        raise AssertionError("this model cannot accept native tools or response formats")
    if any(m.role == Role.TOOL or m.tool_calls for m in request.messages):
        raise AssertionError("this model cannot accept native tool-call history")
    system = request.messages[0].text if request.messages[0].role == Role.SYSTEM else ""
    transcript = "\n".join(m.text for m in request.messages if m.role != Role.SYSTEM)
    if "# Output format" in system:
        value = transcript.split("<tool_result")[1].split(">")[1].split("<")[0].strip()
        return f'Here you go:\n```json\n{{"result": {value}, "explanation": "added"}}\n```'
    if "<tool_result" not in transcript:
        return '<tool_call>{"name": "add", "arguments": {"a": 17, "b": 25}}</tool_call>'
    return "It is 42."


MODELS: dict[str, dict[str, Any]] = {
    "frontier": {
        "responder": frontier_model,
        "capabilities": ModelCapabilities(
            tool_calling=True, structured_output=True, json_mode=True, context_window=200_000
        ),
    },
    "small-local": {"responder": small_local_model, "capabilities": ModelCapabilities()},
}


@pytest.mark.parametrize("model_name", sorted(MODELS))
async def test_same_agent_code_runs_on_different_models(model_name: str) -> None:
    spec = MODELS[model_name]
    provider = ScriptedProvider([spec["responder"]] * 10)
    registry = ModelRegistry(
        profiles={
            model_name: ModelProfile(
                provider="p",
                model=model_name,
                capabilities=spec["capabilities"],
                detect_capabilities=False,
            )
        },
        roles={},
    )
    registry.register_provider("p", provider)
    registry.bind_role("executor", model_name)

    answer = await solve(await registry.adapter("executor"))

    assert answer.result == 42
    assert registry.tracker.total.calls == len(provider.requests) == 3
