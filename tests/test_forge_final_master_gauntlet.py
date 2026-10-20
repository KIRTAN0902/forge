"""Final production gauntlet for the unified Forge CLI coding harness.

Covers the complete public contract: slash surface, live provider/model bindings,
semantic plan approval, autonomous multi-cycle convergence, three-part usage
accounting, global resume, restart flushing, and the minimalist TUI language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forge.config.session_store import SessionStore
from forge.config.settings import ForgeConfigManager, ProviderName
from forge.ui.cycle_controller import (
    ForgeCycleController,
    extract_plan_requirements,
    parse_plan_evaluation,
)
from forge.ui.split_app import SLASH_COMMANDS, ForgeSplitPaneApp
from forge.ui.state import (
    AnvilPlanEvaluation,
    ForgeStage,
    ForgeTUIState,
    PendingPlan,
    PendingPlanContract,
    PlanRequirement,
    PlanTurnIntent,
    RequirementItem,
)

FORBIDDEN_EMOJI = ("🟦", "🟥", "🟩", "👁", "🎯", "💬", "👤", "⚡", "⚠", "🔍", "💾", "✔", "✖")


class _Response:
    """Minimal real-client response surface consumed by the controller."""

    def __init__(self, content: str, *, prompt: int, completion: int) -> None:
        self.content = content
        self.reasoning_content = ""
        self.reasoning_incomplete = False
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _ScriptedLiveClient:
    """Stage-aware scripted transport; every stage still performs a real client call."""

    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self.scripts = {key: list(value) for key, value in scripts.items()}
        self.calls: list[tuple[str, int]] = []

    async def stream_chat(
        self,
        binding: Any,
        messages: list[dict[str, str]],
        *,
        on_token: Any = None,
        on_thinking: Any = None,
        stage: str = "anvil",
    ) -> _Response:
        queue = self.scripts.get(stage) or ["LIVE_FALLBACK"]
        index = min(sum(1 for name, _ in self.calls if name == stage), len(queue) - 1)
        content = queue[index]
        self.calls.append((stage, index))
        if on_token is not None:
            on_token(content)
        return _Response(content, prompt=10 + index, completion=5 + index)

    async def list_live_models(self, provider: ProviderName) -> list[str]:
        return [f"{provider.value}/model-a"]

    async def test_connection(self, binding: Any) -> tuple[bool, str]:
        return True, "ok"


def _manager(tmp_path: Path) -> ForgeConfigManager:
    return ForgeConfigManager(config_path=tmp_path / "config.json", probe_ollama=False)


def _plan_text() -> str:
    return (
        "1. Add the export endpoint and storage layer.\n"
        "2. Cover it with tests.\n\n"
        "REQ-1: CSV export endpoint returns one row per profile.\n"
        "REQ-2: Regression tests cover empty and populated exports."
    )


def _evaluation(intent: PlanTurnIntent, **updates: Any) -> str:
    payload: dict[str, Any] = {
        "intent": intent.value,
        "reasoning_summary": "structured test adjudication",
    }
    payload.update(updates)
    return json.dumps(payload)


def _multicycle_scripts() -> dict[str, list[str]]:
    return {
        "anvil": [
            _plan_text(),
            _evaluation(PlanTurnIntent.APPROVE_AS_IS),
            "Cycle 1 plan: implement export endpoint.",
            "Cycle 2 remediation: add the missing empty-export test.",
        ],
        "hammer": [
            "Implemented the CSV export endpoint.",
            "Added empty-export regression coverage and fixed the row generator.",
        ],
        "temper": [
            "REQ-1: [✓] endpoint returns rows\n"
            "REQ-2: [✗] empty export test missing\n"
            "ALL_REQUIREMENTS_MET: FALSE",
            "REQ-1: [✓] endpoint returns rows\n"
            "REQ-2: [✓] empty export test present\n"
            "ALL_REQUIREMENTS_MET: TRUE",
        ],
    }


def test_public_slash_surface_is_exact() -> None:
    required = {
        "/provider",
        "/models",
        "/usage",
        "/resume",
        "/restart",
        "/help",
        "/mode",
        "/cycle",
        "/research",
        "/clear",
        "/reset",
        "/exit",
    }
    assert required == set(SLASH_COMMANDS)
    assert "/key" not in SLASH_COMMANDS
    assert "/model" not in SLASH_COMMANDS
    assert "!<shell command>" not in SLASH_COMMANDS


@pytest.mark.asyncio
async def test_provider_key_is_saved_masked_and_restricted(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.set_api_key(ProviderName.GEMINI, "AIzaSy-super-secret-key")

    payload = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert payload["api_keys"]["gemini"] == "AIzaSy-super-secret-key"
    assert "AIza" in manager.masked_key("gemini")
    assert manager.masked_key("gemini") != "AIzaSy-super-secret-key"

    controller = ForgeCycleController(state=ForgeTUIState(), config_manager=manager)
    rendered = await controller.handle_slash_command("/provider")
    assert "Forge Providers" in rendered
    assert manager.masked_key("gemini") in rendered


def test_models_aggregates_connected_providers_and_binds_stages_independently(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    manager.set_api_key(ProviderName.GEMINI, "AIzaSy-gemini-key")
    manager.set_api_key(ProviderName.OPENROUTER, "sk-or-openrouter-key")

    app = ForgeSplitPaneApp(
        state=ForgeTUIState(),
        controller=ForgeCycleController(state=ForgeTUIState(), config_manager=manager),
    )
    modal = app.modal_manager
    modal.open_models_modal()
    aggregated = {provider for provider, _model in modal.cached_aggregated_models}
    assert ProviderName.GEMINI in aggregated
    assert ProviderName.OPENROUTER in aggregated

    manager.set_stage_model("anvil", "gemini/gemini-2.5-pro")
    manager.set_stage_model("hammer", "ollama/qwen2.5-coder:7b")
    manager.set_stage_model("temper", "openrouter/anthropic/claude-3.5-sonnet")

    assert manager.binding_for("anvil").spec_string == "gemini/gemini-2.5-pro"  # type: ignore[union-attr]
    assert manager.binding_for("hammer").spec_string == "ollama/qwen2.5-coder:7b"  # type: ignore[union-attr]
    assert manager.binding_for("temper").spec_string == "openrouter/anthropic/claude-3.5-sonnet"  # type: ignore[union-attr]


def test_semantic_plan_approval_contract() -> None:
    for intent in PlanTurnIntent:
        evaluation = parse_plan_evaluation(_evaluation(intent))
        assert evaluation.intent is intent

    cancel = parse_plan_evaluation(
        _evaluation(PlanTurnIntent.CANCEL_PLAN, reasoning_summary="abandoned")
    )
    assert cancel.intent is PlanTurnIntent.CANCEL_PLAN

    contract = PendingPlanContract(
        problem_statement="Export profiles",
        plan_summary_markdown="1. Build the endpoint.",
        target_files=["src/export.py"],
        requirements_checklist=[
            RequirementItem(req_id="REQ-1", description="Export works"),
        ],
    )
    plan = PendingPlan.from_contract(contract, revision=2)
    assert plan.revision == 2
    assert plan.target_files == ["src/export.py"]
    assert plan.to_contract().requirements_checklist[0].description == "Export works"


@pytest.mark.asyncio
async def test_pending_plan_routes_anvil_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    state = ForgeTUIState()
    controller = ForgeCycleController(state=state)
    plan = PendingPlan(
        request="Ship export",
        plan_text="1. Build it.",
        requirements=[PlanRequirement("REQ-1", "Export works")],
    )
    calls: list[str] = []

    async def fake_evaluate(
        pending: PendingPlan, response: str
    ) -> AnvilPlanEvaluation:
        return AnvilPlanEvaluation(intent=PlanTurnIntent.MODIFY_AND_PROCEED)

    async def fake_revise(pending: PendingPlan, feedback: str) -> PendingPlan:
        revised = PendingPlan(
            request=pending.request,
            plan_text=f"{pending.plan_text}\n{feedback}",
            requirements=pending.requirements,
            revision=pending.revision + 1,
        )
        calls.append("revise")
        return revised

    async def fake_run(pending: PendingPlan) -> str:
        calls.append("run")
        state.set_pending_plan(None)
        return "autonomous"

    monkeypatch.setattr(controller, "_evaluate_plan_turn", fake_evaluate)
    monkeypatch.setattr(controller, "_revise_plan", fake_revise)
    monkeypatch.setattr(controller, "_run_autonomous_cycle", fake_run)
    state.set_pending_plan(plan)

    assert await controller.run_cycle_loop("proceed with the change") == "autonomous"
    assert calls == ["revise", "run"]


def test_requirement_extraction_is_stable_and_ordered() -> None:
    requirements = extract_plan_requirements(_plan_text(), "export profiles")
    assert [item.req_id for item in requirements] == ["REQ-1", "REQ-2"]
    assert "CSV export endpoint" in requirements[0].text


@pytest.mark.asyncio
async def test_autonomous_cycle_runs_to_100_percent_without_user_prompts(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.set_api_key(ProviderName.GEMINI, "AIzaSy-gemini-key")
    client = _ScriptedLiveClient(_multicycle_scripts())
    state = ForgeTUIState()
    controller = ForgeCycleController(
        state=state,
        llm_client=client,  # type: ignore[arg-type]
        config_manager=manager,
        max_cycles=5,
    )

    plan_render = await controller.run_cycle_loop("Implement CSV profile export")
    assert "Final Plan" in plan_render
    assert "REQ-1" in plan_render
    assert state.pending_plan is not None

    report = await controller.run_cycle_loop("approve")

    assert state.cycle_count == 2
    assert state.all_requirements_met is True
    assert state.pending_plan is None
    assert "Cycle :- 2" in report
    assert "[✓] REQ-1" in report
    assert "[✓] REQ-2" in report
    assert "ALL_REQUIREMENTS_MET: TRUE" in report
    assert [stage for stage, _ in client.calls].count("hammer") == 2
    assert [stage for stage, _ in client.calls].count("temper") == 2


@pytest.mark.asyncio
async def test_usage_report_has_agents_cycles_and_session_grand_total(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.set_api_key(ProviderName.GEMINI, "AIzaSy-gemini-key")
    state = ForgeTUIState()
    controller = ForgeCycleController(
        state=state,
        llm_client=_ScriptedLiveClient(_multicycle_scripts()),
        config_manager=manager,
    )

    await controller.run_cycle_loop("Implement CSV profile export")
    await controller.run_cycle_loop("approve")
    report = controller.render_usage_report()

    assert "Per Agent:" in report
    assert "Anvil" in report and "Hammer" in report and "Temper" in report
    assert "Per Cycle:" in report
    assert "Cycle :- 1" in report and "Cycle :- 2" in report
    assert "Session Grand Total:" in report
    assert state.total_session_tokens > 0


def test_resume_roundtrip_restores_plan_usage_and_totals(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    state = ForgeTUIState()
    state.cycle_count = 3
    state.total_session_tokens = 900
    state.total_session_input_tokens = 600
    state.total_session_output_tokens = 300
    state.cycle_usage = {1: {"anvil": {"input": 100, "output": 50, "total": 150}}}
    state.cycle_usage[2] = {"temper": {"input": 200, "output": 100, "total": 300}}
    state.pending_plan = PendingPlan(
        request="Ship export",
        plan_text="1. Build it.",
        requirements=[PlanRequirement("REQ-1", "Export works", met=True, evidence="tests")],
    )
    state.stage_buffers[ForgeStage.HAMMER].stage_input_tokens = 40
    state.stage_buffers[ForgeStage.HAMMER].stage_output_tokens = 20
    state.stage_buffers[ForgeStage.HAMMER].stage_tokens = 60

    store.save_session(state, "resume-session", tmp_path, title="Resume me")
    restored = ForgeTUIState()
    assert store.restore_session("resume-session", restored) is True

    assert restored.cycle_count == 3
    assert restored.total_session_tokens == 900
    assert restored.total_session_input_tokens == 600
    assert restored.total_session_output_tokens == 300
    assert restored.cycle_usage[2]["temper"]["total"] == 300
    assert restored.pending_plan is not None
    assert restored.pending_plan.requirements[0].req_id == "REQ-1"
    assert restored.pending_plan.requirements[0].met is True
    assert restored.stage_buffers[ForgeStage.HAMMER].stage_input_tokens == 40


@pytest.mark.asyncio
async def test_restart_flushes_session_and_resets_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    state = ForgeTUIState()
    state.append_chat(ForgeStage.ANVIL, "You", "before restart")
    controller = ForgeCycleController(state=state, config_manager=manager, root=tmp_path)

    import forge.config.session_store as session_store_module

    store = SessionStore(tmp_path / "sessions")
    monkeypatch.setattr(session_store_module, "get_session_store", lambda: store)

    message = await controller.handle_slash_command("/restart")

    assert "Restarting" in message or "restart" in message.lower()
    assert state.cycle_count == 1
    history_text = "\n".join(
        text for buffer in state.stage_buffers.values() for _speaker, text in buffer.chat_history
    )
    assert "before restart" not in history_text
    assert any(path.name.endswith(".json") for path in (tmp_path / "sessions").glob("*.json"))


def test_tui_language_has_no_cartoon_emoji() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "forge" / "ui"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for emoji in FORBIDDEN_EMOJI:
            if emoji in text:
                offenders.append(f"{path.name}: {emoji}")
    assert offenders == []


def test_split_app_disables_mouse_and_advertises_forge_block() -> None:
    state = ForgeTUIState()
    app = ForgeSplitPaneApp(
        state=state, controller=ForgeCycleController(state=state)
    )
    assert bool(app.application.mouse_support()) is False
    assert app.application.min_redraw_interval == pytest.approx(0.04)
    rendered = "".join(text for _style, text in app._render_header())
    assert "FORGE" in rendered
