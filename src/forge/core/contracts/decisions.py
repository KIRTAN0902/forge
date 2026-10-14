"""Orchestrator decisions.

The specification lists two slightly different action sets (§7 and §22 Phase 5). This
enum is the reconciled vocabulary:

* ``QA`` is split into ``FULL_QA`` and ``TARGETED_QA`` so the scope is always explicit.
* ``REQUEST_RESEARCH`` asks the Planner to investigate (there is no separate research
  agent); ``REQUEST_HUMAN_REVIEW`` is the spec's ``HUMAN_REVIEW``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from forge.core.contracts.base import Contract


class OrchestratorAction(StrEnum):
    PLAN = "PLAN"
    REPLAN = "REPLAN"
    EXECUTE = "EXECUTE"
    REEXECUTE = "REEXECUTE"
    TARGETED_QA = "TARGETED_QA"
    FULL_QA = "FULL_QA"
    RETRY = "RETRY"
    ROLLBACK = "ROLLBACK"
    REQUEST_RESEARCH = "REQUEST_RESEARCH"
    REQUEST_HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"
    BLOCK = "BLOCK"
    COMPLETE = "COMPLETE"


TERMINAL_ACTIONS = frozenset({OrchestratorAction.BLOCK, OrchestratorAction.COMPLETE})


class Decision(Contract):
    action: OrchestratorAction
    rationale: str
    task_ids: list[str] = Field(default_factory=list)
    finding_ids: list[str] = Field(default_factory=list)
    instructions: str = Field(default="", description="Guidance passed to the next agent.")
