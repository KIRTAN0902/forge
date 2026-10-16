"""Workflow construction and orchestration helpers for Forge CLI."""

from __future__ import annotations

from dataclasses import dataclass

from forge.agents import ExecutorAgent, OrchestratorAgent, QAAgent
from forge.config import ForgeConfig
from forge.events.bus import EventBus
from forge.models import ModelRegistry, RoleBinding
from forge.policies.approval import Approver
from forge.storage.sqlite import SQLiteStore
from forge.tools import create_tool_runtime
from forge.tools.runtime import ToolRuntime
from forge.workflow import AgentRouter, Orchestrator


@dataclass
class WorkflowHarness:
    """Encapsulates the engine runtime, agents, store, and orchestrator."""

    config: ForgeConfig
    store: SQLiteStore
    runtime: ToolRuntime
    registry: ModelRegistry
    router: AgentRouter
    orchestrator: Orchestrator
    events: EventBus

    async def aclose(self) -> None:
        """Close store and tool/model runtimes."""
        await self.runtime.aclose()
        await self.registry.aclose()
        await self.store.close()


async def build_workflow_harness(
    config: ForgeConfig,
    *,
    registry: ModelRegistry | None = None,
    approver: Approver | None = None,
    event_bus: EventBus | None = None,
    max_iterations: int | None = None,
) -> WorkflowHarness:
    """Build and wire the complete multi-agent harness from configuration."""
    events = event_bus or EventBus()
    store = await SQLiteStore.open(config.db_path)
    events.subscribe(store.append_event)

    runtime = create_tool_runtime(config, root=config.root, approver=approver, events=events)

    profiles = dict(config.models)
    roles = dict(config.roles)
    if not profiles and not registry:
        from forge.cli.credentials import resolve_default_model_profile

        pname, prof = resolve_default_model_profile()
        if prof is not None:
            profiles[pname] = prof
            roles["default"] = RoleBinding(model=pname)

    model_registry = registry or ModelRegistry(
        providers=config.providers, profiles=profiles, roles=roles
    )

    plan_role = "planner" if "planner" in model_registry.roles else "orchestrator"
    plan_adapter = await model_registry.adapter(plan_role, events=events)
    orch_adapter = await model_registry.adapter("orchestrator", events=events)
    executor_adapter = await model_registry.adapter("executor", events=events)
    qa_adapter = await model_registry.adapter("qa", events=events)

    orchestrator_agent = OrchestratorAgent(plan_adapter, runtime=runtime, events=events)
    executor = ExecutorAgent(executor_adapter, runtime=runtime, events=events)
    qa = QAAgent(qa_adapter, runtime=runtime, events=events)

    router = AgentRouter(
        orchestrator=orchestrator_agent, executor=executor, qa=qa, runtime=runtime, events=events
    )

    limit = max_iterations or config.limits.max_iterations
    orchestrator = Orchestrator(
        adapter=orch_adapter,
        router=router,
        store=store,
        events=events,
        max_iterations=limit,
    )

    return WorkflowHarness(
        config=config,
        store=store,
        runtime=runtime,
        registry=model_registry,
        router=router,
        orchestrator=orchestrator,
        events=events,
    )
