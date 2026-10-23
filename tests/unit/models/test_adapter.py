from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from forge.core.errors import ModelCapabilityError, ModelOutputError
from forge.events import Event, EventBus, EventType
from forge.models import (
    FinishReason,
    GenerationParams,
    ImagePart,
    Message,
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
    ModelRequest,
    OptimizationConfig,
    Pricing,
    Role,
    StreamDone,
    TextDelta,
    ToolDefinition,
    Usage,
)
from forge.models.providers.scripted import ScriptedProvider, text_response, tool_call_response

TOOL = ToolDefinition(
    name="read_file",
    description="Read a file",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)
WRITE_TOOL = ToolDefinition(
    name="write_file",
    description="Write a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    },
)


class Verdict(BaseModel):
    ok: bool
    reason: str


def adapter(
    provider: ScriptedProvider,
    *,
    caps: ModelCapabilities | None = None,
    optimization: OptimizationConfig | None = None,
    params: GenerationParams | None = None,
    pricing: Pricing | None = None,
    events: EventBus | None = None,
) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model="test-model", pricing=pricing)
    return ModelAdapter(
        profile_name="test",
        profile=profile,
        provider=provider,
        capabilities=caps or ModelCapabilities(),
        role="executor",
        params=params,
        optimization=optimization,
        events=events,
        workflow_id="wf_1",
    )


NATIVE = ModelCapabilities(tool_calling=True, structured_output=True, json_mode=True)


class TestToolCalling:
    async def test_native_passes_tools_through(self) -> None:
        provider = ScriptedProvider([tool_call_response("read_file", {"path": "a.py"})])
        response = await adapter(provider, caps=NATIVE).generate(
            [Message.user("read a.py")], tools=[TOOL]
        )
        request = provider.requests[0]
        assert [t.name for t in request.tools] == ["read_file"]
        assert request.parallel_tool_calls is None  # capability not declared -> not sent
        assert response.tool_calls[0].arguments == {"path": "a.py"}

    async def test_parallel_hint_follows_capability_and_batching(self) -> None:
        caps = NATIVE.model_copy(update={"parallel_tool_calls": True})
        provider = ScriptedProvider(["a", "b"])
        await adapter(provider, caps=caps).generate([Message.user("x")], tools=[TOOL])
        sequential = OptimizationConfig(tool_batching="sequential")
        await adapter(provider, caps=caps, optimization=sequential).generate(
            [Message.user("x")], tools=[TOOL]
        )
        assert [r.parallel_tool_calls for r in provider.requests] == [True, False]

    async def test_prompt_protocol_when_model_lacks_tools(self) -> None:
        provider = ScriptedProvider(
            [
                'I will read it.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}'
                "</tool_call>"
            ]
        )
        response = await adapter(provider).generate([Message.user("read a.py")], tools=[TOOL])
        request = provider.requests[0]
        assert request.tools == []  # never sends tools to a model that cannot take them
        assert request.messages[0].role == Role.SYSTEM
        assert "<tool_call>" in request.messages[0].text
        assert "read_file" in request.messages[0].text
        assert response.text == "I will read it."
        assert response.tool_calls[0].name == "read_file"
        assert response.finish_reason == FinishReason.TOOL_CALLS

    async def test_prompt_protocol_round_trip_history(self) -> None:
        provider = ScriptedProvider(
            ['<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>', "Done."]
        )
        model = adapter(provider)
        messages = [Message.system("You are an engineer."), Message.user("read a")]
        first = await model.generate(messages, tools=[TOOL])
        messages += [first.message, Message.tool_result(first.tool_calls[0], "print('hi')")]
        second = await model.generate(messages, tools=[TOOL])
        assert second.text == "Done."
        sent = provider.requests[1].messages
        assert all(m.role != Role.TOOL for m in sent)
        assert sent[0].text.startswith("You are an engineer.")
        assert "<tool_result" in sent[-1].text
        assert "print('hi')" in sent[-1].text

    async def test_forced_strategy_overrides_capability(self) -> None:
        provider = ScriptedProvider(["no tools needed"])
        await adapter(
            provider, caps=NATIVE, optimization=OptimizationConfig(tool_call_strategy="prompt")
        ).generate([Message.user("x")], tools=[TOOL])
        assert provider.requests[0].tools == []

    async def test_native_salvages_text_tool_calls(self) -> None:
        provider = ScriptedProvider(
            ['<tool_call>{"name": "read_file", "arguments": {"path": "z"}}</tool_call>']
        )
        response = await adapter(provider, caps=NATIVE).generate([Message.user("x")], tools=[TOOL])
        assert response.tool_calls[0].arguments == {"path": "z"}
        assert response.text == ""

    async def test_native_salvages_xml_tool_calls(self) -> None:
        provider = ScriptedProvider(
            [
                "<tool_call>\n"
                "<function=write_file>\n"
                "<parameter=path>src/App.tsx</parameter>\n"
                "<parameter=content>export default function App() {}</parameter>\n"
                "</function>\n"
                "</tool_call>"
            ]
        )
        response = await adapter(provider, caps=NATIVE).generate(
            [Message.user("write it")], tools=[WRITE_TOOL]
        )
        assert response.text == ""
        assert response.finish_reason == FinishReason.TOOL_CALLS
        assert response.tool_calls[0].name == "write_file"
        assert response.tool_calls[0].arguments == {
            "path": "src/App.tsx",
            "content": "export default function App() {}",
        }

    async def test_tool_choice_none_sends_no_tools(self) -> None:
        provider = ScriptedProvider(["answer"])
        await adapter(provider, caps=NATIVE).generate(
            [Message.user("x")], tools=[TOOL], tool_choice="none"
        )
        assert provider.requests[0].tools == []
        assert provider.requests[0].tool_choice is None


class TestStructured:
    async def test_native_uses_response_format(self) -> None:
        provider = ScriptedProvider(['{"ok": true, "reason": "fine"}'])
        result = await adapter(provider, caps=NATIVE).structured([Message.user("x")], Verdict)
        assert result == Verdict(ok=True, reason="fine")
        request = provider.requests[0]
        assert request.response_format is not None
        assert request.response_format.name == "Verdict"
        assert request.response_format.schema_["properties"]["ok"]["type"] == "boolean"

    async def test_json_mode_strategy(self) -> None:
        provider = ScriptedProvider(['{"ok": false, "reason": "r"}'])
        caps = ModelCapabilities(json_mode=True)
        await adapter(provider, caps=caps).structured([Message.user("x")], Verdict)
        request = provider.requests[0]
        assert request.json_mode is True
        assert request.response_format is None
        assert '"reason"' in request.messages[0].text  # schema in the prompt

    async def test_prompt_strategy_tolerates_prose_and_fences(self) -> None:
        provider = ScriptedProvider(['Sure!\n```json\n{"ok": true, "reason": "r"}\n```'])
        result = await adapter(provider).structured([Message.user("x")], Verdict)
        assert result.ok is True
        assert provider.requests[0].response_format is None
        assert provider.requests[0].json_mode is False

    async def test_repair_round_trip(self) -> None:
        provider = ScriptedProvider(['{"ok": "maybe"}', '{"ok": true, "reason": "fixed"}'])
        result = await adapter(provider).structured([Message.user("judge")], Verdict)
        assert result.reason == "fixed"
        repair = provider.requests[1].messages
        assert repair[-2].role == Role.ASSISTANT
        assert repair[-2].text == '{"ok": "maybe"}'
        assert "schema validation failed" in repair[-1].text
        assert "reason" in repair[-1].text

    async def test_repair_exhausted(self) -> None:
        provider = ScriptedProvider(["nope", "still nope", "never"])
        with pytest.raises(ModelOutputError, match="Verdict"):
            await adapter(provider).structured([Message.user("x")], Verdict)
        assert len(provider.requests) == 3  # 1 attempt + 2 repairs (default)

    async def test_downgrades_when_server_rejects_native(self) -> None:
        def reject(request: ModelRequest) -> str | None:
            if request.response_format is not None:
                return "response_format json_schema is not supported"
            return None

        provider = ScriptedProvider(
            ['{"ok": true, "reason": "a"}', '{"ok": true, "reason": "b"}'], reject=reject
        )
        model = adapter(provider, caps=NATIVE)
        assert (await model.structured([Message.user("x")], Verdict)).reason == "a"
        assert model.structured_strategy() == "json_mode"
        # Second call goes straight to the working strategy.
        assert (await model.structured([Message.user("y")], Verdict)).reason == "b"
        assert [r.response_format is not None for r in provider.requests] == [True, False, False]

    async def test_truncation_is_explained(self) -> None:
        cut = text_response('{"ok": tr')
        cut.finish_reason = FinishReason.LENGTH
        provider = ScriptedProvider([cut, '{"ok": true, "reason": "r"}'])
        await adapter(provider).structured([Message.user("x")], Verdict)
        assert "cut off" in provider.requests[1].messages[-1].text


class TestNormalizationAndAccounting:
    async def test_inline_think_moves_to_reasoning(self) -> None:
        provider = ScriptedProvider(["<think>2+2 is 4</think>The answer is 4."])
        response = await adapter(provider).generate([Message.user("2+2?")])
        assert response.text == "The answer is 4."
        assert response.message.reasoning == "2+2 is 4"

    async def test_think_does_not_break_structured_output(self) -> None:
        provider = ScriptedProvider(['<think>{"bad": 1}</think>{"ok": true, "reason": "r"}'])
        assert (await adapter(provider).structured([Message.user("x")], Verdict)).ok

    async def test_images_rejected_without_vision(self) -> None:
        provider = ScriptedProvider(["x"])
        message = Message.user([ImagePart(url="data:image/png;base64,AA")])
        with pytest.raises(ModelCapabilityError):
            await adapter(provider).generate([message])
        await adapter(provider, caps=ModelCapabilities(vision=True)).generate([message])

    async def test_param_precedence_profile_role_call(self) -> None:
        provider = ScriptedProvider(["x"])
        base = GenerationParams(temperature=0.7, max_tokens=100, extra={"a": 1})
        role = GenerationParams(temperature=0.2, extra={"b": 2})
        model = adapter(provider, params=base.merged(role))
        await model.generate([Message.user("x")], params=GenerationParams(max_tokens=5))
        params = provider.requests[0].params
        assert params.temperature == 0.2
        assert params.max_tokens == 5
        assert params.extra == {"a": 1, "b": 2}

    async def test_usage_cost_and_events(self) -> None:
        bus = EventBus()
        events: list[Event] = []
        bus.subscribe(events.append)
        provider = ScriptedProvider(
            [text_response("hi", usage=Usage(input_tokens=1_000_000, output_tokens=500_000))]
        )
        model = adapter(
            provider,
            pricing=Pricing(input_per_mtok=1.0, output_per_mtok=2.0),
            events=bus,
        )
        response = await model.generate([Message.user("x")])
        assert response.usage.cost_usd == pytest.approx(2.0)
        assert model.tracker.total.calls == 1
        assert model.tracker.by_role["executor"].usage.input_tokens == 1_000_000
        assert events[0].type == EventType.MODEL_CALLED
        assert events[0].workflow_id == "wf_1"
        assert events[0].payload["strategy"] == "text"
        assert events[0].payload["usage"]["output_tokens"] == 500_000


class TestStreaming:
    async def test_native_stream(self) -> None:
        provider = ScriptedProvider(["one two three"])
        events = [e async for e in adapter(provider).stream([Message.user("x")])]
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "one two three"
        assert isinstance(events[-1], StreamDone)
        assert events[-1].response.text == "one two three"

    async def test_prompt_tools_fall_back_to_single_chunk(self) -> None:
        provider = ScriptedProvider(
            ['<tool_call>{"name": "read_file", "arguments": {"path": "q"}}</tool_call>']
        )
        events = [e async for e in adapter(provider).stream([Message.user("x")], tools=[TOOL])]
        assert len(events) == 1  # no text, just the done event
        done = events[0]
        assert isinstance(done, StreamDone)
        assert done.response.tool_calls[0].arguments == {"path": "q"}

    async def test_no_streaming_capability(self) -> None:
        provider = ScriptedProvider(["whole"])
        events = [
            e
            async for e in adapter(provider, caps=ModelCapabilities(streaming=False)).stream(
                [Message.user("x")]
            )
        ]
        assert [type(e) for e in events] == [TextDelta, StreamDone]


def test_scripted_tool_response_is_valid_json() -> None:
    response = tool_call_response("read_file", {"path": "x"})
    assert json.loads(response.tool_calls[0].raw_arguments) == {"path": "x"}
