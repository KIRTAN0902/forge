"""Provider/model config engine, unified streaming LLM client and serious TUI typography."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from forge.config.settings import (
    ForgeConfigManager,
    ModelTier,
    ProviderName,
    StageModelBinding,
    classify_model_tier,
    parse_provider_model,
)
from forge.providers.client import UnifiedLLMClient
from forge.ui.cycle_controller import ForgeCycleController
from forge.ui.split_app import ForgeSplitPaneApp, render_sidebar_plain_text
from forge.ui.state import ForgeStage, ForgeTUIState

_EMOJI = (
    "\U0001f5a6",  # 🖶
    "\U0001f525",  # 🔥
    "\U0001f533",  # 🔳
    "\U0001f534",  # 🔴
    "\U0001f7e9",  # 🟩
    "\U0001f441",  # 👁
    "\U0001f3af",  # 🎯
    "\U0001f4ac",  # 💬
    "\U0001f464",  # 👤
    "\u26a1",  # ⚡
    "\u26a0",  # ⚠
    "\U0001f527",  # 🔧
    "\U0001f4ad",  # 💭
    "\u2705",  # ✅
    "\u274c",  # ❌
    "\u2753",  # ❓
)


def _manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ForgeConfigManager:
    # Neutralise any real-world API keys so the test is hermetic.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    return ForgeConfigManager(
        config_path=tmp_path / "config.json",
        env={},
        cwd=tmp_path,
        probe_ollama=False,
    )


# -- Phase 1: persistent configuration engine --------------------------------------------


def test_config_persistence_and_key_management(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, monkeypatch)

    # Parsing infers the provider for bare names and honours explicit ``provider/model``.
    assert parse_provider_model("openrouter/anthropic/claude-3.5-sonnet") == (
        ProviderName.OPENROUTER,
        "anthropic/claude-3.5-sonnet",
    )
    assert parse_provider_model("ollama/qwen2.5-coder:7b") == (
        ProviderName.OLLAMA,
        "qwen2.5-coder:7b",
    )
    assert parse_provider_model("claude-3-5-sonnet-latest")[0] is ProviderName.ANTHROPIC

    # Tiering: local/small models are prosthetic SLMs, big remote models are amplifiers.
    assert classify_model_tier(ProviderName.OLLAMA, "qwen2.5-coder:7b") is ModelTier.PROSTHETIC_SLM
    assert (
        classify_model_tier(ProviderName.GEMINI, "gemini-2.5-pro") is ModelTier.AMPLIFIER_FRONTIER
    )

    # Saving a key persists it (masked for display) and auto-attaches a live default.
    manager.set_api_key("gemini", "AIzaSyExampleKey1234")
    assert manager.api_key_for(ProviderName.GEMINI) == "AIzaSyExampleKey1234"
    assert manager.masked_key("gemini") == "AIza...****"
    for stage in ("anvil", "hammer", "temper"):
        binding = manager.binding_for(stage)
        assert binding is not None
        assert binding.provider is ProviderName.GEMINI
        assert binding.spec_string == "gemini/gemini-2.5-flash"

    assert manager.config_path.is_file()
    payload = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert payload["api_keys"]["gemini"] == "AIzaSyExampleKey1234"
    if os.name != "nt":
        assert (manager.config_path.stat().st_mode & 0o777) == 0o600

    # A fresh manager reloads the persisted key + binding from disk.
    reloaded = ForgeConfigManager(
        config_path=manager.config_path, env={}, cwd=tmp_path, probe_ollama=False
    )
    assert reloaded.api_key_for(ProviderName.GEMINI) == "AIzaSyExampleKey1234"
    assert reloaded.binding_for("anvil") is not None
    assert reloaded.binding_for("anvil").spec_string == "gemini/gemini-2.5-flash"  # type: ignore[union-attr]

    # ``/config`` renders a masked description, never the raw secret.
    described = reloaded.describe()
    assert "Forge Configuration" in described
    assert "AIza...****" in described
    assert "AIzaSyExampleKey1234" not in described


# -- Phase 3: interactive slash commands -------------------------------------------------


@pytest.mark.asyncio
async def test_slash_commands_key_provider_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path, monkeypatch)
    controller = ForgeCycleController(state=ForgeTUIState(), config_manager=manager)

    # ``/key <provider> <key>`` saves the credential and reports it masked.
    key_out = await controller.handle_slash_command("/key deepseek sk-deepseek-secret")
    assert "API key saved" in key_out
    assert "sk-d...****" in key_out
    assert manager.api_key_for("deepseek") == "sk-deepseek-secret"

    # ``/key`` with no argument shows status only.
    status_out = await controller.handle_slash_command("/key")
    assert "Provider API keys" in status_out

    # ``/provider`` (no wizard hook) prints the provider table.
    provider_out = await controller.handle_slash_command("/provider")
    assert "forge provider" in provider_out

    # ``/config`` shows the persistent file + masked bindings.
    config_out = await controller.handle_slash_command("/config")
    assert "Forge Configuration" in config_out

    # ``/model`` persists a stage-scoped binding to the config engine.
    await controller.handle_slash_command("/model anvil gemini/gemini-2.5-pro")
    anvil = manager.binding_for("anvil")
    assert anvil is not None
    assert anvil.spec_string == "gemini/gemini-2.5-pro"

    # ``/model all`` fans the spec out across every stage.
    await controller.handle_slash_command("/model all ollama/qwen2.5-coder:7b")
    for stage in ("anvil", "hammer", "temper"):
        assert manager.binding_for(stage).spec_string == "ollama/qwen2.5-coder:7b"  # type: ignore[union-attr]


# -- Phase 2: unified streaming LLM client -----------------------------------------------


@pytest.mark.asyncio
async def test_unified_llm_client_request_formatting_and_token_accounting() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/v1/messages":
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "claude reply"}],
                    "usage": {"input_tokens": 11, "output_tokens": 5},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello there"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            },
        )

    client = UnifiedLLMClient(
        transport=httpx.MockTransport(handler),
        env={"OPENAI_API_KEY": "sk-test"},
    )

    # OpenAI-compatible non-streaming call: correct URL, model, system injection, usage.
    binding = StageModelBinding(provider=ProviderName.OPENAI, model="gpt-4o")
    response = await client.complete_or_stream(
        binding,
        [{"role": "user", "content": "hi"}],
        system_prompt="You are Forge.",
    )
    assert captured[-1].url.path == "/v1/chat/completions"
    body = json.loads(captured[-1].content)
    assert body["model"] == "gpt-4o"
    assert body["messages"][0] == {"role": "system", "content": "You are Forge."}
    assert response.content == "hello there"
    assert (response.prompt_tokens, response.completion_tokens, response.total_tokens) == (9, 4, 13)

    # Anthropic uses the Messages endpoint and its own usage field names.
    anthropic = StageModelBinding(provider=ProviderName.ANTHROPIC, model="claude-3-5-sonnet-latest")
    a_resp = await client.complete_or_stream(anthropic, [{"role": "user", "content": "hi"}])
    assert captured[-1].url.path == "/v1/messages"
    assert captured[-1].headers["anthropic-version"] == "2023-06-01"
    assert a_resp.content == "claude reply"
    assert (a_resp.prompt_tokens, a_resp.completion_tokens) == (11, 5)


@pytest.mark.asyncio
async def test_unified_llm_client_streams_ollama_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        lines = [
            json.dumps({"message": {"content": "Hel"}}),
            json.dumps({"message": {"content": "lo"}}),
            json.dumps({"done": True, "prompt_eval_count": 7, "eval_count": 2}),
        ]
        return httpx.Response(200, text="\n".join(lines))

    client = UnifiedLLMClient(transport=httpx.MockTransport(handler))
    binding = StageModelBinding(provider=ProviderName.OLLAMA, model="qwen2.5-coder:7b")

    deltas: list[str] = []
    response = await client.stream_chat(
        binding,
        [{"role": "user", "content": "hi"}],
        on_token=deltas.append,
    )
    assert deltas == ["Hel", "lo"]
    assert response.content == "Hello"
    assert (response.prompt_tokens, response.completion_tokens) == (7, 2)


# -- Phase 4: serious engineering typography ---------------------------------------------


def test_serious_tui_typography_and_no_redundant_stage_box() -> None:
    state = ForgeTUIState()
    state.set_active_stage(ForgeStage.TEMPER)
    state.append_chat(ForgeStage.TEMPER, "You", "verify the parser")
    state.append_chat(ForgeStage.TEMPER, "Temper", "everything passes")
    state.append_thought(ForgeStage.TEMPER, "checking boundary conditions")
    state.append_tool_event(ForgeStage.TEMPER, "Tool: pytest -q | Exit: 0")
    state.append_tool_event(ForgeStage.TEMPER, "[WARN] coverage below target")
    state.set_stage_output(ForgeStage.TEMPER, "Verified deliverable: parser is correct.")
    state.record_tokens(ForgeStage.TEMPER, 4200)

    rendered = state.stage_buffers[ForgeStage.TEMPER].render_to_ansi(width=80)

    assert "FORGE // TEMPER" not in rendered
    assert "── Verified Final Output Deliverable ──" not in rendered
    assert "❯ user" in rendered
    assert "● temper" in rendered
    assert "── Thought Process ──" in rendered
    assert "│ ├─ thought:" in rendered
    assert "── Tool & Sandbox Execution ──" in rendered
    assert "[WARN] coverage below target" in rendered

    # The heavy redundant panels are gone (no chat panel title) and no arcade emoji remain.
    assert "Conversation" not in rendered
    assert "✔ QA Verification Passed" not in rendered
    for glyph in _EMOJI:
        assert glyph not in rendered

    # Sidebar gains the professional workspace title and loses the old arcade header.
    sidebar = render_sidebar_plain_text(state)
    assert "FORGE // WORKSPACE" in sidebar
    assert "FORGE BLACKSMITH" not in sidebar
    assert "│ ● Temper" in sidebar


def test_single_line_input_prompt_and_vanishing_placeholder() -> None:
    app = ForgeSplitPaneApp(state=ForgeTUIState())

    prompt = app._render_input_prompt()
    assert prompt[0][1] == "anvil ❯ "
    assert list(prompt[1:]) == []

    hint = app._render_input_hint()
    assert "Type a task" in "".join(fragment[1] for fragment in hint)


def test_completer_offers_context_aware_arguments() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    app = ForgeSplitPaneApp(state=ForgeTUIState())
    completer: Any = app.input_buffer.completer

    command_texts = {
        c.text for c in completer.get_completions(Document("/p"), CompleteEvent())
    }
    assert "/provider" in command_texts
    assert "/key" not in command_texts

    model_command_texts = {
        c.text for c in completer.get_completions(Document("/m"), CompleteEvent())
    }
    assert "/models" in model_command_texts
    assert "/model" not in model_command_texts

    provider_args = {
        c.text for c in completer.get_completions(Document("/provider "), CompleteEvent())
    }
    assert "gemini" in provider_args
    assert "openrouter" in provider_args
    assert "custom" in provider_args

    stage_args = {
        c.text for c in completer.get_completions(Document("/models "), CompleteEvent())
    }
    assert "all" in stage_args
    assert "anvil" in stage_args
