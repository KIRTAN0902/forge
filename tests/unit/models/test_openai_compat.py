from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from forge.models import (
    FinishReason,
    GenerationParams,
    ImagePart,
    Message,
    ModelRequest,
    ProviderConfig,
    ProviderError,
    ReasoningDelta,
    ResponseFormat,
    StreamDone,
    TextDelta,
    TextPart,
    ToolCall,
    ToolCallDelta,
    ToolDefinition,
)
from forge.models.providers.openai_compat import OpenAICompatibleProvider

Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """Mock transport that records request bodies and replays scripted responses."""

    def __init__(self, *responses: httpx.Response | Handler | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        entry = self.responses.pop(0)
        if isinstance(entry, Exception):
            raise entry
        if callable(entry):
            return entry(request)
        return entry

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def make_provider(
    recorder: Recorder, **config: Any
) -> tuple[OpenAICompatibleProvider, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    provider = OpenAICompatibleProvider(
        ProviderConfig(base_url="https://llm.test/v1", **config),
        name="test",
        api_key="sk-test",
        transport=httpx.MockTransport(recorder),
        sleep=fake_sleep,
    )
    return provider, sleeps


def completion(message: dict[str, Any], finish: str = "stop", **extra: Any) -> httpx.Response:
    body = {
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        **extra,
    }
    return httpx.Response(200, json=body)


def sse(*chunks: dict[str, Any] | str) -> httpx.Response:
    lines = []
    for chunk in chunks:
        if isinstance(chunk, str):
            lines.append(chunk)
        else:
            lines.append("data: " + json.dumps(chunk))
        lines.append("")
    return httpx.Response(
        200, content="\n".join(lines).encode(), headers={"content-type": "text/event-stream"}
    )


TOOL = ToolDefinition(
    name="read_file",
    description="Read",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
)


class TestPayload:
    def test_full_request_encoding(self) -> None:
        provider, _ = make_provider(Recorder())
        call = ToolCall(id="c1", name="read_file", arguments={"path": "a"})
        request = ModelRequest(
            messages=[
                Message.system("sys"),
                Message.user([TextPart(text="look"), ImagePart(url="data:image/png;base64,AA")]),
                Message.assistant(None, [call]),
                Message.tool_result(call, "file body"),
            ],
            tools=[TOOL],
            tool_choice="read_file",
            parallel_tool_calls=False,
            response_format=ResponseFormat(name="Out", schema_={"type": "object"}),
            params=GenerationParams(
                temperature=0.2, max_tokens=50, reasoning_effort="high", extra={"top_k": 5}
            ),
        )
        body = provider.build_payload("model-x", request, stream=False)
        assert body["model"] == "model-x"
        assert body["messages"][1]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA"},
        }
        assistant = body["messages"][2]
        assert assistant["content"] is None
        assert assistant["tool_calls"][0]["function"] == {
            "name": "read_file",
            "arguments": '{"path": "a"}',
        }
        assert body["messages"][3] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "file body",
        }
        assert body["tools"][0]["function"]["name"] == "read_file"
        assert body["tool_choice"] == {"type": "function", "function": {"name": "read_file"}}
        assert body["parallel_tool_calls"] is False
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "Out"
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 50
        assert body["reasoning_effort"] == "high"
        assert body["top_k"] == 5
        assert "stream" not in body

    def test_unset_params_are_omitted(self) -> None:
        provider, _ = make_provider(Recorder())
        body = provider.build_payload(
            "m", ModelRequest(messages=[Message.user("hi")]), stream=False
        )
        assert set(body) == {"model", "messages"}

    @pytest.mark.parametrize(
        ("style", "effort", "expected"),
        [
            ("openrouter", "low", {"reasoning": {"effort": "low"}}),
            ("openrouter", "none", {"reasoning": {"enabled": False}}),
            ("none", "high", {}),
        ],
    )
    def test_reasoning_styles(self, style: str, effort: str, expected: dict[str, Any]) -> None:
        provider, _ = make_provider(Recorder(), reasoning_style=style)
        request = ModelRequest(
            messages=[Message.user("x")],
            params=GenerationParams(reasoning_effort=effort),  # type: ignore[arg-type]
        )
        body = provider.build_payload("m", request, stream=False)
        assert {k: v for k, v in body.items() if k in {"reasoning", "reasoning_effort"}} == expected

    def test_json_mode_and_stream_options(self) -> None:
        provider, _ = make_provider(Recorder())
        request = ModelRequest(messages=[Message.user("x")], json_mode=True)
        body = provider.build_payload("m", request, stream=True)
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_extra_overrides_mapped_params(self) -> None:
        provider, _ = make_provider(Recorder())
        request = ModelRequest(
            messages=[Message.user("x")],
            params=GenerationParams(
                max_tokens=10, extra={"max_tokens": None, "max_completion_tokens": 10}
            ),
        )
        body = provider.build_payload("m", request, stream=False)
        assert body["max_tokens"] is None
        assert body["max_completion_tokens"] == 10


class TestComplete:
    async def test_text_reply_usage_and_auth(self) -> None:
        recorder = Recorder(
            completion(
                {"role": "assistant", "content": "hello", "reasoning": "thought"},
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "cost": 0.0012,
                },
            )
        )
        provider, _ = make_provider(recorder)
        response = await provider.complete("m", ModelRequest(messages=[Message.user("hi")]))
        assert response.text == "hello"
        assert response.message.reasoning == "thought"
        assert response.finish_reason == FinishReason.STOP
        assert response.usage.input_tokens == 100
        assert response.usage.reasoning_tokens == 5
        assert response.usage.cached_input_tokens == 40
        assert response.usage.cost_usd == pytest.approx(0.0012)
        assert response.latency_s is not None
        assert recorder.requests[0].headers["authorization"] == "Bearer sk-test"
        assert recorder.requests[0].url.path == "/v1/chat/completions"

    async def test_tool_calls_parsed_and_finish_normalized(self) -> None:
        recorder = Recorder(
            completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "x"}'},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{broken"},
                        },
                        {
                            # Ollama sends arguments as an object and may omit the id
                            "function": {"name": "read_file", "arguments": {"path": "y"}},
                        },
                    ],
                },
                finish="stop",  # some servers say "stop" even with tool calls
            )
        )
        provider, _ = make_provider(recorder)
        response = await provider.complete("m", ModelRequest(messages=[Message.user("hi")]))
        assert response.finish_reason == FinishReason.TOOL_CALLS
        first, broken, ollama = response.tool_calls
        assert first.arguments == {"path": "x"}
        assert broken.parse_error is not None
        assert broken.raw_arguments == "{broken"
        assert ollama.arguments == {"path": "y"}
        assert ollama.id

    async def test_http_400_is_rejected_request(self) -> None:
        recorder = Recorder(
            httpx.Response(400, json={"error": {"message": "response_format not supported"}})
        )
        provider, sleeps = make_provider(recorder)
        with pytest.raises(ProviderError) as info:
            await provider.complete("m", ModelRequest(messages=[Message.user("hi")]))
        assert info.value.status == 400
        assert info.value.is_request_rejected
        assert "response_format not supported" in str(info.value)
        assert sleeps == []  # 400 is not retried

    async def test_error_body_with_200(self) -> None:
        recorder = Recorder(
            httpx.Response(200, json={"error": {"message": "upstream", "code": 502}})
        )
        provider, _ = make_provider(recorder, max_retries=0)
        with pytest.raises(ProviderError, match="upstream") as info:
            await provider.complete("m", ModelRequest(messages=[Message.user("hi")]))
        assert info.value.status == 502

    async def test_retries_rate_limit_then_succeeds(self) -> None:
        recorder = Recorder(
            httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow down"}),
            httpx.ConnectError("boom"),
            completion({"role": "assistant", "content": "ok"}),
        )
        provider, sleeps = make_provider(recorder)
        response = await provider.complete("m", ModelRequest(messages=[Message.user("hi")]))
        assert response.text == "ok"
        assert len(sleeps) == 2
        assert sleeps[0] >= 3  # Retry-After honoured

    async def test_retries_exhausted(self) -> None:
        recorder = Recorder(*[httpx.Response(503, text="down") for _ in range(3)])
        provider, sleeps = make_provider(recorder, max_retries=2)
        with pytest.raises(ProviderError) as info:
            await provider.complete("m", ModelRequest(messages=[Message.user("hi")]))
        assert info.value.status == 503
        assert len(sleeps) == 2


class TestStream:
    async def test_assembles_text_reasoning_tools_and_usage(self) -> None:
        recorder = Recorder(
            sse(
                ": OPENROUTER PROCESSING",
                {"model": "m", "choices": [{"delta": {"reasoning": "think "}}]},
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "function": {"name": "read_file", "arguments": '{"pa'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": 'th": "a"}'}}]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
                "data: [DONE]",
            )
        )
        provider, _ = make_provider(recorder)
        events = [
            e async for e in provider.stream("m", ModelRequest(messages=[Message.user("hi")]))
        ]
        assert recorder.body()["stream"] is True
        assert [type(e) for e in events[:3]] == [ReasoningDelta, TextDelta, TextDelta]
        assert any(isinstance(e, ToolCallDelta) for e in events)
        done = events[-1]
        assert isinstance(done, StreamDone)
        response = done.response
        assert response.text == "Hello"
        assert response.message.reasoning == "think "
        assert response.tool_calls[0].arguments == {"path": "a"}
        assert response.finish_reason == FinishReason.TOOL_CALLS
        assert response.usage.output_tokens == 4

    async def test_stream_error_chunk_raises(self) -> None:
        recorder = Recorder(sse({"error": {"message": "model overloaded"}}))
        provider, _ = make_provider(recorder)
        with pytest.raises(ProviderError, match="overloaded"):
            async for _ in provider.stream("m", ModelRequest(messages=[Message.user("hi")])):
                pass

    async def test_stream_retries_before_first_byte(self) -> None:
        recorder = Recorder(
            httpx.Response(502, text="bad gateway"),
            sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
        )
        provider, sleeps = make_provider(recorder)
        events = [
            e async for e in provider.stream("m", ModelRequest(messages=[Message.user("hi")]))
        ]
        final = events[-1]
        assert isinstance(final, StreamDone)
        assert final.response.text == "ok"
        assert len(sleeps) == 1


class TestDetection:
    async def test_openrouter_metadata(self) -> None:
        recorder = Recorder(
            httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "z-ai/glm",
                            "context_length": 131072,
                            "top_provider": {"max_completion_tokens": 8192},
                            "supported_parameters": ["tools", "response_format", "reasoning"],
                            "architecture": {"input_modalities": ["text"]},
                            "pricing": {"prompt": "0.0000006", "completion": "0.0000022"},
                        }
                    ]
                },
            )
        )
        provider, _ = make_provider(recorder)
        info = await provider.describe_model("z-ai/glm")
        assert info is not None
        assert info.context_window == 131072
        assert info.max_output_tokens == 8192
        assert info.tool_calling is True
        assert info.structured_output is False
        assert info.json_mode is True
        assert info.reasoning is True
        assert info.vision is False
        assert info.input_cost_per_mtok == pytest.approx(0.6)

    async def test_ollama_show_fallback(self) -> None:
        def show(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/show"
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "tools"],
                    "model_info": {"qwen2.context_length": 32768},
                },
            )

        recorder = Recorder(
            httpx.Response(200, json={"data": [{"id": "qwen2.5:7b", "object": "model"}]}),
            show,
        )
        provider, _ = make_provider(recorder)
        info = await provider.describe_model("qwen2.5:7b")
        assert info is not None
        assert info.tool_calling is True
        assert info.vision is False
        # trained context is not reported as usable (Ollama serves with its own num_ctx)
        assert info.context_window is None

    async def test_unknown_everywhere(self) -> None:
        recorder = Recorder(
            httpx.Response(500, text="no listing"), httpx.Response(404, text="not ollama")
        )
        provider, _ = make_provider(recorder)
        assert await provider.describe_model("x") is None
