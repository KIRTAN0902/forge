"""Unit tests for Forge AUTO Mode feature, keybindings, and autonomous workflow execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from forge.cli.chat import (
    ChatSessionState,
    _get_bottom_toolbar,
    _get_prompt_tokens,
    print_auto_mode_completion,
    run_chat_session,
)
from forge.cli.interactive import AutoApprover
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import QAReport, QAStatus
from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.policies.types import PolicyDecision, ProposedAction, RiskLevel


def test_chat_session_state_auto_mode_toggle() -> None:
    state = ChatSessionState(default_agent="orchestrator", auto_mode=False)
    assert state.auto_mode is False

    # Toggle on
    assert state.toggle_auto_mode() is True
    assert state.auto_mode is True

    # Toggle off
    assert state.toggle_auto_mode() is False
    assert state.auto_mode is False

    # Explicit set
    state.set_auto_mode(True)
    assert state.auto_mode is True
    state.set_auto_mode(False)
    assert state.auto_mode is False


def test_get_prompt_tokens_shows_auto_mode_badge() -> None:
    state = ChatSessionState(default_agent="orchestrator", auto_mode=True)
    tokens = _get_prompt_tokens(state)
    token_text = "".join(t[1] for t in tokens)

    assert "ANVIL" in token_text
    assert "[⚡ AUTO]" in token_text
    assert "Auto: ON" in token_text

    # When toggled off
    state.toggle_auto_mode()
    tokens_off = _get_prompt_tokens(state)
    text_off = "".join(t[1] for t in tokens_off)
    assert "[⚡ AUTO]" not in text_off
    assert "Auto: OFF" in text_off


def test_get_bottom_toolbar_shows_auto_mode() -> None:
    state = ChatSessionState(auto_mode=True)
    toolbar = _get_bottom_toolbar(state)
    text = "".join(t[1] for t in toolbar)

    assert "Run Auto" in text
    assert "Auto: ON" in text

    state.toggle_auto_mode()
    toolbar_off = _get_bottom_toolbar(state)
    text_off = "".join(t[1] for t in toolbar_off)
    assert "Send" in text_off
    assert "Auto: OFF" in text_off


@pytest.mark.asyncio
async def test_auto_approver_approves_without_prompt() -> None:
    console = Console(record=True)
    approver = AutoApprover(console, silent=False)

    action = ProposedAction(
        kind="tool", name="run_command", risk=RiskLevel.HIGH, arguments={"cmd": "pytest"}
    )
    decision = PolicyDecision(
        action=action,
        policy="safety_policy",
        verdict="require_approval",
        reason="Requires check",
    )

    res = await approver.request_approval(action, decision)
    assert res is True
    output = console.export_text()
    assert "Auto-approved action" in output
    assert "run_command" in output


def test_orchestrator_has_no_keyword_approval_gate() -> None:
    import forge.agents.orchestrator as orchestrator_module

    assert not hasattr(orchestrator_module, "EXECUTION_CONFIRM_TRIGGERS")


def test_print_auto_mode_completion_renders_cleanly() -> None:
    console = Console(record=True)
    state = WorkflowState(user_request="Build REST user auth endpoints")
    state.status = WorkflowStatus.COMPLETED
    state.iteration = 4
    state.plan = Plan(
        goal="Build REST user auth endpoints",
        tasks=[
            Task(
                id="TASK-1",
                title="Define user schemas",
                description="Create Pydantic models for user request and response",
                status=TaskStatus.DONE,
            ),
            Task(
                id="TASK-2",
                title="Implement JWT handler",
                description="Write JWT token signing and verification logic",
                status=TaskStatus.DONE,
            ),
        ],
    )
    state.qa_reports.append(
        QAReport(
            status=QAStatus.PASS,
            summary="All tests passed with zero blocking issues.",
        )
    )

    print_auto_mode_completion(state, console)
    output = console.export_text()

    assert "AUTO MODE COMPLETED" in output
    assert "COMPLETED" in output
    assert "Planning (ANVIL)" in output
    assert "Execution (HAMMER): 2/2 tasks completed" in output
    assert "QA Verification (TEMPER): PASS" in output


@pytest.mark.asyncio
async def test_orchestrator_auto_mode_resolves_approval_gate() -> None:
    from forge.workflow.orchestrator import Orchestrator

    mock_adapter = AsyncMock()
    mock_router = AsyncMock()

    # Step returns awaiting approval on iteration 1, then completed on iteration 2
    state = WorkflowState(user_request="Build something in AUTO mode")

    orchestrator = Orchestrator(adapter=mock_adapter, router=mock_router, max_iterations=5)

    call_count = 0

    async def fake_step(s: WorkflowState):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            s.status = WorkflowStatus.AWAITING_APPROVAL
            return Decision(
                action=OrchestratorAction.REQUEST_HUMAN_REVIEW,
                rationale="Review needed before executing",
            ), s
        s.status = WorkflowStatus.COMPLETED
        return Decision(
            action=OrchestratorAction.COMPLETE,
            rationale="Workflow completed",
        ), s

    orchestrator.step = fake_step  # type: ignore[method-assign]

    final_state = await orchestrator.run(state, auto_mode=True)
    assert final_state.status == WorkflowStatus.COMPLETED
    assert call_count == 2


def test_slash_auto_command_toggles_mode_in_chat_session(tmp_path: Path) -> None:
    inputs = iter(["/auto", "/auto off", "/auto on", "/exit"])

    with (
        patch("forge.cli.chat.ensure_credentials_configured"),
        patch("forge.cli.chat.read_user_input", side_effect=lambda *args: next(inputs)),
    ):
        run_chat_session(path=tmp_path)
