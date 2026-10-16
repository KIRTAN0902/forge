"""Forge autonomous skill engine: specialist capabilities and cybernetic router."""

from __future__ import annotations

from forge.skills.anthropic_loader import (
    load_anthropic_skill_markdown,
    parse_anthropic_skills,
)
from forge.skills.design_loader import (
    load_design_skill_markdown,
    parse_design_skills,
)
from forge.skills.evaluator import SkillEvaluator
from forge.skills.formatter import format_skills_badge, format_skills_for_prompt
from forge.skills.models import ModelTier, OperationType, RepoSignals, SkillSpec
from forge.skills.registry import (
    ALL_SKILLS,
    AUTONOMOUS_SKILLS,
    AutonomousSkillRouter,
    SkillRegistry,
)
from forge.skills.types import Skill, SkillActivation, SkillCategory, SkillSignal

__all__ = [
    "ALL_SKILLS",
    "AUTONOMOUS_SKILLS",
    "AutonomousSkillRouter",
    "ModelTier",
    "OperationType",
    "RepoSignals",
    "Skill",
    "SkillActivation",
    "SkillCategory",
    "SkillEvaluator",
    "SkillRegistry",
    "SkillSignal",
    "SkillSpec",
    "format_skills_badge",
    "format_skills_for_prompt",
    "load_anthropic_skill_markdown",
    "load_design_skill_markdown",
    "parse_anthropic_skills",
    "parse_design_skills",
]
