"""Live-CLI verification suite.

These tests lock in Forge's anti-hardcoding law and the deterministic slash-command layer:

* every conversational turn reaches the real configured LLM and streams its actual reply,
* when no model is reachable Forge renders a credential diagnostic (never a canned sentence),
* ``/`` and ``!`` syntax is dispatched before any intent classification,
* the ``/`` autocomplete completer powers the live popup menu.
"""

from __future__ import annotations

from typing import Any

from forge.agents import OrchestratorAgent
from forge.models import (
    BUILTIN_PROVIDERS,
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
)
from forge.models.providers.scripted import ScriptedProvider
from forge.ui.cycle_controller import (
    CycleVerdict,
    ExecutionMode,
    ForgeCycleController,
    model_diagnostic,
)
from forge.ui.split_app import SLASH_COMMANDS, ForgeSplitPaneApp, SlashCommandCompleter
from forge.ui.state import ForgeStage, ForgeTUIState

# Strings a lazy implementation might fabricate instead of calling a model.
_CANNED_MARKERS = (
    "Blacksmithing Pipeline",
    "Hello! I am Forge",
    "What would you like to build today",
)


def _anvil_adapter(provider: ScriptedProvider) -> ModelAdapter:
    profile = ModelProfile(
        provider="scripted", model="orchestrator-live", detect_capabilities=False
    )
    return ModelAdapter(
        profile_name="orchestrator",
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role="orchestrator",
    )


class _FakeAdapter:
    def __init__(self, label: str) -> None:
        self._label = label
        self.model = label

    def describe(self) -> str:
        return f"fake://{self._label}"


class _FakeRegistry:
    """Minimal ModelRegistry stand-in for the live ``/model`` swap path."""

    def __init__(self) -> None:
        self.provider_configs: dict[str, Any] = {}
        self.profiles: dict[str, Any] = {}
        self.roles: dict[str, str] = {}

    def add_profile(self, name: str, profile: Any) -> None:
        self.profiles[name] = profile

    def configure_providers(self, providers: Any) -> None:
        self.provider_configs.update(providers)

    def bind_role(self, role: str, name: str) -> None:
        self.roles[role] = name

    async def adapter(self, role: str, events: Any = None) -> _FakeAdapter:
        return _FakeAdapter(self.roles.get(role, "unknown"))


# -- no hardcoded replies ---------------------------------------------------------------


async def test_conversational_turn_streams_real_model_reply() -> None:
    """A greeting is answered by the real model, not a canned string."""
    state = ForgeTUIState()
    provider = ScriptedProvider()
    provider.add("Live model online — how can I help with your codebase?")
    controller = ForgeCycleController(
        state=state,
        agents={ForgeStage.ANVIL: OrchestratorAgent(_anvil_adapter(provider))},
    )

    result = await controller.run_cycle_loop("hello")

    assert result == "Live model online — how can I help with your codebase?"
    assert len(provider.requests) == 1
    # The real reply (and only the real reply) landed in the transcript.
    speakers = [s for s, _ in state.stage_buffers[ForgeStage.ANVIL].chat_history]
    assert speakers == ["You", "Anvil"]
    assert state.stage_buffers[ForgeStage.ANVIL].chat_history[-1][1] == result
    for canned in _CANNED_MARKERS:
        assert canned not in result


async def test_no_model_attached_renders_credential_diagnostic() -> None:
    """With no live model, Forge must explain what to configure — never fabricate."""
    state = ForgeTUIState()
    controller = ForgeCycleController(state=state)

    result = await controller.run_cycle_loop("hi")

    assert result == model_diagnostic()
    assert "OPENROUTER_API_KEY" in result
    assert "OLLAMA_HOST" in result
    assert state.stage_buffers[ForgeStage.ANVIL].status == "failed"
    # Nothing fictional was streamed into the transcript.
    assert not any(canned in result for canned in _CANNED_MARKERS)


async def test_conversation_history_persists_across_turns() -> None:
    """Multi-turn context is replayed to the model on the following turn."""
    state = ForgeTUIState()
    provider = ScriptedProvider()
    provider.add("First reply.", "Second reply, now with context.")
    controller = ForgeCycleController(
        state=state,
        agents={ForgeStage.ANVIL: OrchestratorAgent(_anvil_adapter(provider))},
    )

    await controller.run_cycle_loop("hi")
    await controller.run_cycle_loop("tell me more")

    history = controller.conversation_history[ForgeStage.ANVIL]
    assert [turn["role"] for turn in history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # The second request replays the full prior dialogue.
    second = provider.requests[1]
    contents = [m.content for m in second.messages]
    assert any("hi" in (c or "") for c in contents)
    assert any("First reply." in (c or "") for c in contents)


# -- deterministic slash commands -------------------------------------------------------


async def _fresh_controller(**kwargs: Any) -> ForgeCycleController:
    state = ForgeTUIState()
    controller = ForgeCycleController(state=state, **kwargs)
    return controller


async def test_help_and_status_commands_are_immediate() -> None:
    controller = await _fresh_controller()

    help_text = await controller.handle_slash_command("/help")
    assert "/models" in help_text
    assert "/cycle" in help_text
    assert "/research" in help_text
    assert "!<shell command>" in help_text

    status_text = await controller.handle_slash_command("/status")
    assert "Forge Session Status" in status_text
    assert "Cycle" in status_text
    assert "Anvil" in status_text


async def test_mode_command_sets_and_reports_routing() -> None:
    controller = await _fresh_controller()

    set_chat = await controller.handle_slash_command("/mode chat")
    assert controller.forced_mode == ExecutionMode.CONVERSATIONAL
    assert "CHAT" in set_chat

    summary = await controller.handle_slash_command("/mode")
    assert "CHAT" in summary

    set_cycle = await controller.handle_slash_command("/mode cycle")
    assert controller.forced_mode == ExecutionMode.FULL_FORGE_CYCLE
    assert "CYCLE" in set_cycle

    auto = await controller.handle_slash_command("/mode auto")
    assert controller.forced_mode is None
    assert "AUTO" in auto


async def test_models_command_swaps_live() -> None:
    provider = ScriptedProvider()
    agent = OrchestratorAgent(_anvil_adapter(provider))
    controller = await _fresh_controller(
        agents={ForgeStage.ANVIL: agent}, registry=_FakeRegistry()
    )

    swapped = await controller.handle_slash_command(
        "/models hammer ollama/qwen2.5-coder:7b"
    )
    assert "hammer" in swapped.lower()
    assert controller.registry.roles["executor"] == "ollama/qwen2.5-coder:7b"

    bad_stage = await controller.handle_slash_command("/models nope openrouter/x")
    assert "Unknown stage" in bad_stage


async def test_shell_command_runs_for_real() -> None:
    controller = await _fresh_controller()

    result = await controller.handle_slash_command("!echo forge-live-cli")

    assert "forge-live-cli" in result
    assert controller.last_mode == ExecutionMode.DIRECT_TOOL
    assert any("forge-live-cli" in ev for ev in controller.state.stage_buffers[ForgeStage.ANVIL].tool_events)


async def test_clear_and_reset_commands_mutate_state() -> None:
    controller = await _fresh_controller()
    controller.state.append_chat(ForgeStage.ANVIL, "You", "scratch")

    cleared = await controller.handle_slash_command("/clear")
    assert "Cleared" in cleared
    # /clear re-renders its own confirmation into the (now cleared) buffer.
    anvil_chat = controller.state.stage_buffers[ForgeStage.ANVIL].chat_history
    assert all("scratch" not in text for _, text in anvil_chat)

    controller.state.append_thought(ForgeStage.HAMMER, "leftover")
    reset = await controller.handle_slash_command("/reset")
    assert "Session reset" in reset
    assert controller.state.cycle_count == 1
    assert controller.state.stage_buffers[ForgeStage.HAMMER].thoughts == []


async def test_unknown_command_is_reported_not_swallowed() -> None:
    controller = await _fresh_controller()

    result = await controller.handle_slash_command("/frobnicate")

    assert "Unknown command" in result
    assert controller.state.stage_buffers[ForgeStage.ANVIL].status == "failed"


async def test_exit_command_signals_goodbye() -> None:
    controller = await _fresh_controller()
    assert "Goodbye" in await controller.handle_slash_command("/quit")
    assert "Goodbye" in await controller.handle_slash_command("/exit")


async def test_slash_commands_bypass_intent_routing() -> None:
    """A ``/`` command never reaches the model even when agents are attached."""
    provider = ScriptedProvider()
    controller = await _fresh_controller(
        agents={ForgeStage.ANVIL: OrchestratorAgent(_anvil_adapter(provider))}
    )

    await controller.run_cycle_loop("/status")

    assert provider.requests == []
    assert controller.state.stage_buffers[ForgeStage.ANVIL].status == "passed"


# -- live ``/`` autocomplete ------------------------------------------------------------


def test_completer_offers_slash_commands() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    completer = SlashCommandCompleter()
    completions = list(completer.get_completions(Document("/mo"), CompleteEvent()))
    texts = {c.text for c in completions}
    assert {"/models", "/mode"} <= texts

    # Every advertised slash command is completable.
    all_completions = {
        c.text for c in completer.get_completions(Document("/"), CompleteEvent())
    }
    assert set(SLASH_COMMANDS) <= all_completions
    assert {"/provider", "/models"} <= all_completions


def test_completer_stays_quiet_for_plain_text() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    completer = SlashCommandCompleter()
    assert list(completer.get_completions(Document("hello world"), CompleteEvent())) == []


def test_split_app_wires_completer_and_input_bindings() -> None:
    controller = ForgeCycleController(state=ForgeTUIState())
    app = ForgeSplitPaneApp(controller=controller)

    assert app.input_buffer.completer is app.completer
    prompt_fragments = app._render_input_prompt()
    assert prompt_fragments[0][1] == "anvil ❯ "
    assert list(prompt_fragments[1:]) == []


def test_forced_cycle_mode_overrides_classification() -> None:
    state = ForgeTUIState()

    def verifier(_cycle: int, _prompt: str) -> CycleVerdict:
        return CycleVerdict(passed=True, feedback="ok", deliverable="done: forced cycle")

    controller = ForgeCycleController(state=state, temper_verifier=verifier)
    controller.set_execution_mode("cycle")
    assert controller.classify("hi") == ExecutionMode.FULL_FORGE_CYCLE

    controller.set_execution_mode("chat")
    assert controller.classify("Implement a whole new subsystem") == ExecutionMode.CONVERSATIONAL


def test_builtin_providers_expose_api_key_env() -> None:
    """The diagnostic env vars must be real provider credentials."""
    envs = {getattr(p, "api_key_env", None) for p in BUILTIN_PROVIDERS.values()}
    for required in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                     "ANTHROPIC_API_KEY"):
        assert required in envs


# -- interactive provider / model wizards ------------------------------------------------


async def test_provider_command_invokes_wizard_and_refreshes() -> None:
    calls: list[str] = []

    async def provider_configurer() -> str:
        calls.append("provider")
        return "OpenRouter connected."

    controller = ForgeCycleController(
        state=ForgeTUIState(),
        provider_configurer=provider_configurer,
    )

    result = await controller.handle_slash_command("/provider")

    assert calls == ["provider"]
    assert "OpenRouter connected." in result
    # Without a live registry/root the refresh degrades to a restart hint (never fabricated).
    assert "Restart Forge" in result
    assert controller.state.stage_buffers[ForgeStage.ANVIL].status == "passed"


async def test_models_command_passes_stage_to_picker() -> None:
    picked: list[str | None] = []

    async def model_picker(stage: str | None) -> str:
        picked.append(stage)
        return f"Selected a model for {stage}."

    controller = ForgeCycleController(state=ForgeTUIState(), model_picker=model_picker)

    await controller.handle_slash_command("/models hammer")
    await controller.handle_slash_command("/models")

    assert picked == ["Hammer", None]


async def test_models_command_rejects_unknown_stage() -> None:
    async def model_picker(_stage: str | None) -> str:  # pragma: no cover - must not run
        raise AssertionError("picker should not be called for an unknown stage")

    controller = ForgeCycleController(state=ForgeTUIState(), model_picker=model_picker)

    result = await controller.handle_slash_command("/models frobnicate")

    assert "Unknown stage" in result
    assert controller.state.stage_buffers[ForgeStage.ANVIL].status == "failed"


async def test_wizard_commands_guide_without_ui_hooks() -> None:
    controller = ForgeCycleController(state=ForgeTUIState())

    provider_result = await controller.handle_slash_command("/provider")
    models_result = await controller.handle_slash_command("/models")

    assert "forge provider" in provider_result
    assert "forge models config" in models_result


async def test_refresh_models_rebinds_every_stage(tmp_path: Any) -> None:
    import types

    (tmp_path / "forge.yaml").write_text(
        "version: 1\n"
        "models:\n"
        "  shared:\n"
        "    provider: openai\n"
        "    model: gpt-4o\n"
        "    detect_capabilities: false\n"
        "roles:\n"
        "  default: {model: shared}\n"
        "  orchestrator: {model: shared}\n"
        "  executor: {model: shared}\n"
        "  qa: {model: shared}\n",
        encoding="utf-8",
    )

    agents = {
        stage: types.SimpleNamespace(adapter=_FakeAdapter("stale"), events=None)
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER)
    }
    controller = ForgeCycleController(
        state=ForgeTUIState(), registry=_FakeRegistry(), root=tmp_path, agents=agents
    )

    summary = await controller.refresh_models()

    assert "refreshed" in summary.lower()
    for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
        assert agents[stage].adapter.describe() == "fake://shared"


def test_split_app_wires_wizard_hooks() -> None:
    controller = ForgeCycleController(state=ForgeTUIState())
    app = ForgeSplitPaneApp(controller=controller)

    assert controller.provider_configurer is not None
    assert controller.model_picker is not None
    assert app.console is not None
