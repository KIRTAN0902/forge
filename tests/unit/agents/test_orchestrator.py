from __future__ import annotations

import pytest

from forge.agents import OrchestratorAgent
from forge.events import EventBus
from forge.models import (
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
)
from forge.models.providers.scripted import ScriptedProvider
from forge.models.types import Message


def _make_adapter(provider: ScriptedProvider, events: EventBus | None = None) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model="orch-test")
    return ModelAdapter(
        profile_name="orchestrator",
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role="orchestrator",
        events=events,
        workflow_id="wf_orch",
    )


@pytest.mark.asyncio
async def test_orchestrator_conversational_brainstorming() -> None:
    provider = ScriptedProvider()
    provider.add(
        "For database migrations, I recommend Alembic with SQLite for local and Postgres for production. "
        "Would you like me to hand this off to the Planning Agent to build it?"
    )
    adapter = _make_adapter(provider)
    agent = OrchestratorAgent(adapter=adapter)

    turn = await agent.chat(
        "What library should we use for database migrations?",
        history=[],
        project_context="Project: TestProject",
    )

    assert "Alembic" in turn.response_text
    assert turn.handoff_requirements is None
    assert len(turn.messages) >= 2


@pytest.mark.asyncio
async def test_orchestrator_handoff_with_tag() -> None:
    provider = ScriptedProvider()
    handoff_content = (
        "Implement Alembic migration environment in db/migrations with SQLite support."
    )
    provider.add(
        f"Understood. Handing off to the Planning Agent.\n\n"
        f"```forge-handoff\n{handoff_content}\n```"
    )
    adapter = _make_adapter(provider)
    agent = OrchestratorAgent(adapter=adapter)

    turn = await agent.chat(
        "Let's implement the Alembic setup now.",
        history=[],
    )

    assert "Understood" in turn.response_text
    assert "forge-handoff" not in turn.response_text
    assert turn.handoff_requirements == handoff_content


@pytest.mark.asyncio
async def test_orchestrator_handoff_with_user_confirmation() -> None:
    provider = ScriptedProvider()
    provider.add("I will proceed with the implementation plan.")
    adapter = _make_adapter(provider)
    agent = OrchestratorAgent(adapter=adapter)

    history = [
        Message.user("Can we design a user registration endpoint?"),
        Message.assistant("We can use FastAPI with Pydantic schemas and bcrypt password hashing."),
    ]

    turn = await agent.chat(
        "build it",
        history=history,
    )

    assert turn.handoff_requirements is not None
    assert "user registration endpoint" in turn.handoff_requirements
    assert "FastAPI" in turn.handoff_requirements
