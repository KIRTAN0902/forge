"""End-to-end QA gauntlet for Forge's reasoning pipeline and TUI contracts.

Every provider wire format is exercised through the real ``UnifiedLLMClient`` code path with
``httpx.MockTransport``. The invariant under test: hidden chain-of-thought is routed only to
``on_thinking`` (and retained in ``LLMResponse.reasoning_content``) while visible answer text is
the only thing that reaches ``on_token``, stage buffers, JSON extraction, QA prompts, or patches.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from forge.config.settings import ProviderName, StageModelBinding
from forge.harness.adversarial_qa import AdversarialQAGatekeeper
from forge.harness.prosthetic import FuzzyAnchorPatcher
from forge.providers.client import (
    ThinkingStreamParser,
    UnifiedLLMClient,
    strip_thinking_tags,
)
from forge.ui.cycle_controller import reasoning_only_diagnostic
from forge.ui.split_app import ForgeSplitPaneApp
from forge.ui.state import ForgeStage, ForgeTUIState, format_stage_deliverable


def _sse(*payloads: dict[str, Any]) -> str:
    return "\n".join(f"data: {json.dumps(payload)}" for payload in payloads) + "\n\ndata: [DONE]\n"


# -- parser / helper invariants -----------------------------------------------------------


def test_parser_splits_inline_thinking_across_chunk_boundaries() -> None:
    tokens: list[str] = []
    thoughts: list[str] = []
    parser = ThinkingStreamParser(tokens.append, thoughts.append)

    for chunk in ["visible ", "<thi", "nk>hidden ", "planning", "</thi", "nk> answer"]:
        parser.feed(chunk)

    content, reasoning = parser.close()
    assert content == "visible  answer"
    assert reasoning == "hidden planning"
    assert "".join(tokens) == "visible  answer"
    assert all("hidden" not in token and "planning" not in token for token in tokens)
    assert thoughts[-1] == "hidden planning"
    assert parser.in_think_block is False
    assert parser.reasoning_incomplete is False
    assert parser.content_buffer == "visible  answer"
    assert parser.think_buffer == "hidden planning"


def test_parser_handles_orphan_close_and_truncated_think_blocks() -> None:
    orphan = ThinkingStreamParser()
    assert orphan.feed("reasoning from prompt template</thinking>Final answer") == "Final answer"
    assert orphan.close() == ("Final answer", "reasoning from prompt template")

    truncated = ThinkingStreamParser()
    assert truncated.feed("Answer start<thinking>never finished") == "Answer start"
    content, reasoning = truncated.close()
    assert content == "Answer start"
    assert reasoning == "never finished"
    assert truncated.reasoning_incomplete is True


def test_strip_thinking_tags_removes_every_hidden_variant() -> None:
    clean, hidden = strip_thinking_tags(
        "before<thinking>first thought</thinking>middle<think>second thought</think> after"
    )
    assert clean == "beforemiddle after"
    assert "first thought" in hidden
    assert "second thought" in hidden

    clean, hidden = strip_thinking_tags("reasoning from prompt template</think>Final answer")
    assert clean == "Final answer"
    assert hidden == "reasoning from prompt template"

    clean, hidden = strip_thinking_tags("plain answer only")
    assert clean == "plain answer only"
    assert hidden == ""


# -- real provider wire formats -----------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_compatible_native_and_inline_reasoning_never_leak() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "native secret",
                        "content": "<thinking>inline secret</thinking>visible answer",
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=_sse(payload))

    client = UnifiedLLMClient(
        transport=httpx.MockTransport(handler), env={"OPENAI_API_KEY": "sk-test"}
    )
    tokens: list[str] = []
    thoughts: list[str] = []
    response = await client.complete_or_stream(
        StageModelBinding(provider=ProviderName.OPENAI, model="o3-mini"),
        [{"role": "user", "content": "hi"}],
        on_token=tokens.append,
        on_thinking=thoughts.append,
    )

    assert response.content == "visible answer"
    assert "secret" in response.reasoning_content
    assert "secret" not in "".join(tokens)
    assert thoughts


@pytest.mark.asyncio
async def test_openai_compatible_stream_routes_reasoning_and_inline_tags() -> None:
    stream = _sse(
        {"choices": [{"delta": {"reasoning": "native "}}]},
        {"choices": [{"delta": {"reasoning_content": "secret"}}]},
        {"choices": [{"delta": {"content": "visible "}}]},
        {"choices": [{"delta": {"content": "<thinking>inline"}}]},
        {"choices": [{"delta": {"content": " secret</thinking>"}}]},
        {"choices": [{"delta": {"content": "answer"}}]},
        {"usage": {"prompt_tokens": 4, "completion_tokens": 6}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=stream)

    client = UnifiedLLMClient(
        transport=httpx.MockTransport(handler), env={"OPENAI_API_KEY": "sk-test"}
    )
    tokens: list[str] = []
    response = await client.stream_chat(
        StageModelBinding(provider=ProviderName.OPENROUTER, model="deepseek/r1"),
        [{"role": "user", "content": "hi"}],
        on_token=tokens.append,
    )

    assert response.content == "visible answer"
    assert "".join(tokens) == "visible answer"
    assert "secret" not in "".join(tokens)
    assert "native" in response.reasoning_content


@pytest.mark.asyncio
async def test_gemini_thought_parts_are_hidden_in_nonstream_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "streamGenerateContent" in str(request.url):
            text = _sse(
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "gemini secret", "thought": True}]}}
                    ]
                },
                {"candidates": [{"content": {"parts": [{"text": "gemini answer"}]}}]},
            )
            return httpx.Response(200, text=text)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "gemini secret", "thought": True},
                                {"text": "gemini answer"},
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 6,
                },
            },
        )

    client = UnifiedLLMClient(
        transport=httpx.MockTransport(handler), env={"GEMINI_API_KEY": "AIza-test"}
    )
    binding = StageModelBinding(provider=ProviderName.GEMINI, model="gemini-2.5-pro")

    plain = await client.complete_or_stream(binding, [{"role": "user", "content": "hi"}])
    assert plain.content == "gemini answer"
    assert "secret" in plain.reasoning_content

    tokens: list[str] = []
    streamed = await client.stream_chat(
        binding, [{"role": "user", "content": "hi"}], on_token=tokens.append
    )
    assert streamed.content == "gemini answer"
    assert "".join(tokens) == "gemini answer"


@pytest.mark.asyncio
async def test_anthropic_thinking_and_thinking_delta_are_not_visible() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            text = _sse(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "claude secret"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "claude answer"},
                },
                {"type": "message_delta", "usage": {"output_tokens": 5}},
            )
            return httpx.Response(200, text=text)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "claude secret"},
                    {"type": "text", "text": "claude answer"},
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        )

    client = UnifiedLLMClient(
        transport=httpx.MockTransport(handler), env={"ANTHROPIC_API_KEY": "sk-ant-test"}
    )
    binding = StageModelBinding(
        provider=ProviderName.ANTHROPIC, model="claude-3-7-sonnet-latest"
    )

    plain = await client.complete_or_stream(binding, [{"role": "user", "content": "hi"}])
    assert plain.content == "claude answer"
    assert "secret" in plain.reasoning_content

    tokens: list[str] = []
    streamed = await client.stream_chat(
        binding, [{"role": "user", "content": "hi"}], on_token=tokens.append
    )
    assert streamed.content == "claude answer"
    assert "".join(tokens) == "claude answer"


@pytest.mark.asyncio
async def test_ollama_thinking_field_is_hidden_in_nonstream_and_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            lines = [
                json.dumps({"message": {"thinking": "local secret"}}),
                json.dumps({"message": {"content": "local answer"}}),
                json.dumps({"done": True, "prompt_eval_count": 2, "eval_count": 3}),
            ]
            return httpx.Response(200, text="\n".join(lines))
        return httpx.Response(
            200,
            json={
                "message": {"thinking": "local secret", "content": "local answer"},
                "prompt_eval_count": 2,
                "eval_count": 3,
            },
        )

    client = UnifiedLLMClient(transport=httpx.MockTransport(handler))
    binding = StageModelBinding(provider=ProviderName.OLLAMA, model="qwen3:8b")

    plain = await client.complete_or_stream(binding, [{"role": "user", "content": "hi"}])
    assert plain.content == "local answer"
    assert "secret" in plain.reasoning_content

    tokens: list[str] = []
    streamed = await client.stream_chat(
        binding, [{"role": "user", "content": "hi"}], on_token=tokens.append
    )
    assert streamed.content == "local answer"
    assert "".join(tokens) == "local answer"


# -- downstream consumers ------------------------------------------------------------------


def test_stage_deliverable_qa_prompt_and_patch_strip_hidden_thinking() -> None:
    rendered = format_stage_deliverable(
        ForgeStage.ANVIL, "<thinking>architecture secret</thinking>Ship stage 1"
    )
    assert "Ship stage 1" in rendered
    assert "architecture secret" not in rendered

    cleaned = AdversarialQAGatekeeper.strip_chain_of_thought(
        "Thought: reviewer secret\n<thinking>nested secret</thinking>\n\nPASS"
    )
    assert "reviewer secret" not in cleaned
    assert "nested secret" not in cleaned
    assert "PASS" in cleaned

    original = "def answer():\n    return 1\n"
    patched = FuzzyAnchorPatcher.apply_patch(
        original,
        "<thinking>locate function</thinking>def answer():\n    return 1",
        "<thinking>rewrite safely</thinking>def answer():\n    return 2",
    )
    assert patched is not None
    assert "return 2" in patched
    assert "locate function" not in patched
    assert "rewrite safely" not in patched


def test_reasoning_only_diagnostic_is_actionable_and_names_the_model() -> None:
    binding = StageModelBinding(provider=ProviderName.OPENAI, model="o3-mini")
    normal = reasoning_only_diagnostic(binding)
    truncated = reasoning_only_diagnostic(binding, truncated=True)

    assert "openai/o3-mini" in normal
    assert "hidden reasoning" in normal
    assert "max_tokens" in normal
    assert "token limit" in truncated
    assert "Thought:" not in normal


# -- TUI contracts -------------------------------------------------------------------------


def test_modal_body_is_focusable_and_modal_to_trap_keystrokes() -> None:
    app = ForgeSplitPaneApp(state=ForgeTUIState())
    control = app.modal_manager.body_control

    assert control.is_focusable()
    assert control.modal is True
    app.modal_manager.open_usage_modal()
    assert app.modal_manager.is_open
    assert app.application.layout.current_control is control


@pytest.mark.asyncio
async def test_stream_redraws_are_coalesced_but_always_flushed() -> None:
    app = ForgeSplitPaneApp(state=ForgeTUIState())
    paints: list[float] = []

    def _count() -> None:
        paints.append(time.monotonic())

    app._invalidate_now = _count  # type: ignore[method-assign]
    app._last_redraw_at = time.monotonic()

    for _ in range(50):
        app._handle_state_redraw()
    assert paints == []
    assert app._redraw_handle is not None

    await asyncio.sleep(0.08)
    assert len(paints) == 1
    assert app._redraw_handle is None

    app._last_redraw_at = 0.0
    app._handle_state_redraw()
    assert len(paints) == 2
