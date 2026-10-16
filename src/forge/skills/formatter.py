"""Prompt and UI formatting utilities for activated skills."""

from __future__ import annotations

from forge.skills.types import SkillActivation


def format_skills_for_prompt(activations: list[SkillActivation]) -> str:
    """Format active skills into explicit markdown prompt instructions for the agent."""
    if not activations:
        return ""

    lines = [
        "## [AUTONOMOUSLY ACTIVATED SPECIALIST SKILLS]",
        "Forge's Orchestrator has dynamically activated the following specialized protocols for this step:",
    ]

    for act in activations:
        lines.append(f"\n### [{act.skill.id}] {act.skill.name}")
        lines.append(f"**Trigger Rationale**: {act.rationale}")
        lines.append(f"**Execution Protocol**:\n{act.skill.protocol}")
        lines.append(f"**Failure Guardrail**: {act.skill.failure_mode_prevented}")

    lines.append("\nYou MUST strictly adhere to the protocols above while executing this task.")
    return "\n".join(lines)


def format_skills_badge(activations: list[SkillActivation]) -> str:
    """Format a compact terminal badge displaying active skill IDs."""
    if not activations:
        return ""
    ids = ", ".join(act.skill.id for act in activations)
    return f"[bold cyan]⚡ Active Skills:[/bold cyan] [dim]{ids}[/dim]"
