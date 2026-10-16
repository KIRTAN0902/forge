"""Planner compatibility alias pointing to OrchestratorAgent.

The Planner role has been consolidated into the OrchestratorAgent, which serves as
Forge's Technical Lead, Architecture Planner, and Conversational Orchestrator.
"""

from __future__ import annotations

from forge.agents.orchestrator import OrchestratorAgent, PlannerResult

# Backward-compatible aliases
PlannerAgent = OrchestratorAgent

__all__ = ["PlannerAgent", "PlannerResult"]
