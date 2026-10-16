"""Unit tests for Frontend & Design Engineering Skills (DSN-01 through DSN-04)."""

from __future__ import annotations

from forge.skills.design_loader import (
    load_design_skill_markdown,
    parse_design_skills,
)
from forge.skills.evaluator import SkillEvaluator
from forge.skills.registry import AutonomousSkillRouter, SkillRegistry
from forge.skills.types import SkillCategory, SkillSignal


class TestDesignEngineeringSkills:
    def test_parse_design_skills_completeness(self) -> None:
        skills = parse_design_skills()
        assert len(skills) == 4

        skill_ids = {spec.skill_id for spec, _, _ in skills}
        assert skill_ids == {"DSN-01", "DSN-02", "DSN-03", "DSN-04"}

        for spec, item, keywords in skills:
            assert spec.name.startswith("Design ")
            assert item.category == SkillCategory.DESIGN
            assert len(keywords) >= 3
            assert len(spec.execution_rules) >= 1
            assert spec.target_agent in {"orchestrator", "executor", "qa"}

    def test_load_design_skill_markdown(self) -> None:
        for dirname in ["impeccable", "ui-ux-pro-max", "shadcn-ui", "emil-design-eng"]:
            md = load_design_skill_markdown(dirname)
            assert len(md) > 200
            assert "# " in md

    def test_design_skills_registered_in_skill_registry(self) -> None:
        reg = SkillRegistry.with_anthropic()
        assert reg.get("DSN-01") is not None
        assert reg.get("DSN-02") is not None
        assert reg.get("DSN-03") is not None
        assert reg.get("DSN-04") is not None

        # Verify category query
        design_skills = reg.by_category(SkillCategory.DESIGN)
        assert len(design_skills) >= 4

    def test_design_skills_registered_in_autonomous_router(self) -> None:
        router = AutonomousSkillRouter()
        assert router.get_spec("DSN-01") is not None
        assert router.get_spec("DSN-02") is not None
        assert router.get_spec("DSN-03") is not None
        assert router.get_spec("DSN-04") is not None
        assert len(router.list_all_specs()) >= 72

    def test_evaluator_triggers_impeccable(self) -> None:
        evaluator = SkillEvaluator()
        signal = SkillSignal(user_request="Perform visual audit and eliminate UI anti-patterns in the styles")
        activations = evaluator.evaluate(signal)
        active_ids = {act.skill.id for act in activations}
        assert "DSN-01" in active_ids

    def test_evaluator_triggers_ui_ux_pro_max(self) -> None:
        evaluator = SkillEvaluator()
        signal = SkillSignal(user_request="Design an empty state with responsive mobile layouts and skeleton loader")
        activations = evaluator.evaluate(signal)
        active_ids = {act.skill.id for act in activations}
        assert "DSN-02" in active_ids

    def test_evaluator_triggers_shadcn_ui(self) -> None:
        evaluator = SkillEvaluator()
        signal = SkillSignal(user_request="Add a dialog using shadcn components and cva variants")
        activations = evaluator.evaluate(signal)
        active_ids = {act.skill.id for act in activations}
        assert "DSN-03" in active_ids

    def test_evaluator_triggers_emil_design_engineering(self) -> None:
        evaluator = SkillEvaluator()
        signal = SkillSignal(user_request="Implement micro-interaction with physics spring and layoutId morphing")
        activations = evaluator.evaluate(signal)
        active_ids = {act.skill.id for act in activations}
        assert "DSN-04" in active_ids

    def test_evaluator_triggers_anthropic_frontend_design(self) -> None:
        evaluator = SkillEvaluator()
        signal = SkillSignal(user_request="Design a bento grid with distinctive editorial typography")
        activations = evaluator.evaluate(signal)
        active_ids = {act.skill.id for act in activations}
        assert "ANT-01" in active_ids
