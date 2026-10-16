"""Unit tests for Anthropic Agent Skills (anthropics/skills) imported into Forge."""

from __future__ import annotations

from forge.skills.anthropic_loader import (
    load_anthropic_skill_markdown,
    parse_anthropic_skills,
)
from forge.skills.evaluator import SkillEvaluator
from forge.skills.registry import AutonomousSkillRouter, SkillRegistry
from forge.skills.types import SkillSignal


def test_parse_anthropic_skills_count_and_contracts() -> None:
    """Verify all 19 Anthropic skills are parsed with valid SkillSpec and Skill objects."""
    loaded = parse_anthropic_skills()
    assert len(loaded) == 19

    skill_ids = {spec.skill_id for spec, _, _ in loaded}
    expected_ids = {f"ANT-{i:02d}" for i in range(1, 20)}
    assert skill_ids == expected_ids

    # Check key skills
    spec_map = {spec.skill_id: spec for spec, _, _ in loaded}
    assert spec_map["ANT-01"].name == "Anthropic frontend-design"
    assert spec_map["ANT-01"].target_agent == "executor"
    assert "*.html" in spec_map["ANT-01"].trigger_file_globs

    assert spec_map["ANT-02"].name == "Anthropic docx"
    assert "*.docx" in spec_map["ANT-02"].trigger_file_globs

    assert spec_map["ANT-03"].name == "Anthropic xlsx"
    assert "*.xlsx" in spec_map["ANT-03"].trigger_file_globs

    assert spec_map["ANT-06"].name == "Anthropic mcp-builder"
    assert spec_map["ANT-06"].target_agent == "orchestrator"

    assert spec_map["ANT-07"].name == "Anthropic webapp-testing"
    assert spec_map["ANT-07"].target_agent == "qa"


def test_anthropic_skills_in_registries() -> None:
    """Verify SkillRegistry and AutonomousSkillRouter include all 19 Anthropic skills."""
    reg = SkillRegistry.with_anthropic()
    assert len(reg.list_all()) >= 43  # 24 core + 19 Anthropic + 4 design (47 total)
    assert reg.get("ANT-01") is not None
    assert reg.get("ANT-19") is not None

    router = AutonomousSkillRouter()
    assert len(router.list_all_specs()) >= 68  # 49 harness + 19 Anthropic + 4 design (72 total)
    assert router.get_spec("ANT-01") is not None
    assert router.get_spec("ANT-19") is not None


def test_skill_evaluator_triggers_anthropic_skills() -> None:
    """Verify SkillEvaluator automatically activates Anthropic skills for the Orchestrator."""
    evaluator = SkillEvaluator()

    # 1. Frontend design prompt
    sig_ui = SkillSignal(user_request="Create a responsive landing page layout with modern typography")
    acts_ui = evaluator.evaluate(sig_ui)
    assert any(a.skill.id == "ANT-01" for a in acts_ui)

    # 2. Spreadsheet request
    sig_excel = SkillSignal(user_request="Build an excel spreadsheet for financial forecasting")
    acts_excel = evaluator.evaluate(sig_excel)
    assert any(a.skill.id == "ANT-03" for a in acts_excel)

    # 3. Word document touched file
    sig_doc = SkillSignal(user_request="Update executive summary", touched_files=["reports/memo.docx"])
    acts_doc = evaluator.evaluate(sig_doc)
    assert any(a.skill.id == "ANT-02" for a in acts_doc)

    # 4. MCP builder request
    sig_mcp = SkillSignal(user_request="Design an MCP server for SQLite database querying")
    acts_mcp = evaluator.evaluate(sig_mcp)
    assert any(a.skill.id == "ANT-06" for a in acts_mcp)


def test_load_anthropic_skill_markdown() -> None:
    """Verify full markdown instructions are loadable from disk."""
    md = load_anthropic_skill_markdown("frontend-design")
    assert "Frontend Design" in md
    assert "Typography carries the personality of the page" in md
