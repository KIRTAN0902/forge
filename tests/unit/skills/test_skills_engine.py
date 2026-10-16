"""Tests for Forge's autonomous skill engine, registry, and evaluators."""

from __future__ import annotations

from typer.testing import CliRunner

from forge.cli.app import app
from forge.skills import (
    SkillCategory,
    SkillEvaluator,
    SkillRegistry,
    SkillSignal,
    format_skills_badge,
    format_skills_for_prompt,
)


def test_registry_contains_all_24_skills() -> None:
    """Verify that all 24 skills are properly registered with valid metadata."""
    registry = SkillRegistry()
    skills = registry.list_all()
    assert len(skills) == 24

    expected_ids = {
        "DBG-01",
        "DBG-02",
        "DBG-03",
        "REF-01",
        "REF-02",
        "REF-03",
        "DB-01",
        "DB-02",
        "DB-03",
        "API-01",
        "API-02",
        "API-03",
        "SEC-01",
        "SEC-02",
        "SEC-03",
        "PERF-01",
        "PERF-02",
        "PERF-03",
        "TST-01",
        "TST-02",
        "TST-03",
        "DOC-01",
        "DOC-02",
        "DOC-03",
    }
    registered_ids = {s.id for s in skills}
    assert registered_ids == expected_ids

    # Check categories
    for s in skills:
        assert isinstance(s.category, SkillCategory)
        assert len(s.target_agents) >= 1
        assert s.name
        assert s.protocol
        assert s.failure_mode_prevented


def test_registry_filtering_by_category_and_agent() -> None:
    """Verify filtering skills by category and agent role."""
    registry = SkillRegistry()

    debugging_skills = registry.by_category(SkillCategory.DEBUGGING)
    assert {s.id for s in debugging_skills} == {"DBG-01", "DBG-02", "DBG-03"}

    security_skills = registry.by_category("security")
    assert {s.id for s in security_skills} == {"SEC-01", "SEC-02", "SEC-03"}

    orch_skills = registry.for_agent("orchestrator")
    assert any(s.id == "DBG-03" for s in orch_skills)
    assert any(s.id == "REF-01" for s in orch_skills)
    assert any(s.id == "DOC-02" for s in orch_skills)

    qa_skills = registry.for_agent("qa")
    assert any(s.id == "TST-03" for s in qa_skills)
    assert any(s.id == "SEC-02" for s in qa_skills)


def test_evaluator_triggers_debugging_skills() -> None:
    """Verify DBG-01, DBG-02, and DBG-03 signal evaluations."""
    evaluator = SkillEvaluator()

    # DBG-01: Non-zero exit code
    sig1 = SkillSignal(test_exit_code=1, error_text="AssertionError: 2 != 3")
    act1 = evaluator.evaluate(sig1)
    assert any(a.skill.id == "DBG-01" for a in act1)

    # DBG-02: Regression report
    sig2 = SkillSignal(
        user_request="This feature used to work before the last PR, please investigate the regression"
    )
    act2 = evaluator.evaluate(sig2)
    assert any(a.skill.id == "DBG-02" for a in act2)

    # DBG-03: 3 consecutive failures
    sig3 = SkillSignal(consecutive_failures=3)
    act3 = evaluator.evaluate(sig3)
    assert any(a.skill.id == "DBG-03" for a in act3)


def test_evaluator_triggers_security_and_perf_skills() -> None:
    """Verify SEC-01, SEC-02, SEC-03, and PERF-01 signal evaluations."""
    evaluator = SkillEvaluator()

    # SEC-01: Dangerous shell=True sink
    sig_sink = SkillSignal(diff_text='+ subprocess.run("ls " + path, shell=True)')
    act_sink = evaluator.evaluate(sig_sink)
    assert any(a.skill.id == "SEC-01" for a in act_sink)

    # SEC-02: Leaked key
    sig_key = SkillSignal(diff_text='+ API_KEY = "sk-ant-api03-abcdef123456789012345678"')
    act_key = evaluator.evaluate(sig_key)
    assert any(a.skill.id == "SEC-02" for a in act_key)

    # SEC-03: Package install
    sig_pkg = SkillSignal(command_executed="uv add suspicious-pkg-name")
    act_pkg = evaluator.evaluate(sig_pkg)
    assert any(a.skill.id == "SEC-03" for a in act_pkg)

    # PERF-01: Blocking time.sleep inside async def
    sig_async = SkillSignal(diff_text="+ async def fetch_data():\n+     time.sleep(2)")
    act_async = evaluator.evaluate(sig_async)
    assert any(a.skill.id == "PERF-01" for a in act_async)


def test_evaluator_triggers_db_api_and_test_integrity_skills() -> None:
    """Verify DB-01, API-01, and TST-03 signal evaluations."""
    evaluator = SkillEvaluator()

    # DB-01: Migration files touched
    sig_db = SkillSignal(touched_files=["alembic/versions/001_initial.py"])
    act_db = evaluator.evaluate(sig_db)
    assert any(a.skill.id == "DB-01" for a in act_db)

    # API-01: Route files touched
    sig_api = SkillSignal(touched_files=["src/app/routes/users.py"])
    act_api = evaluator.evaluate(sig_api)
    assert any(a.skill.id == "API-01" for a in act_api)

    # TST-03: Mocks in tests
    sig_mock = SkillSignal(
        touched_files=["tests/test_service.py"],
        diff_text="+ with patch('my_service.execute') as mock_exec:\n+     mock_exec.assert_called()",
    )
    act_mock = evaluator.evaluate(sig_mock)
    assert any(a.skill.id == "TST-03" for a in act_mock)


def test_prompt_formatter_and_badge() -> None:
    """Verify prompt instructions and compact badges formatting."""
    evaluator = SkillEvaluator()
    sig = SkillSignal(test_exit_code=1, error_text="failed test")
    activations = evaluator.evaluate(sig)

    prompt_text = format_skills_for_prompt(activations)
    assert "AUTONOMOUSLY ACTIVATED SPECIALIST SKILLS" in prompt_text
    assert "DBG-01" in prompt_text
    assert "Stratified Fault Localization" in prompt_text

    badge = format_skills_badge(activations)
    assert "Active Skills:" in badge
    assert "DBG-01" in badge


def test_cli_skills_list_command() -> None:
    """Verify 'forge skills list' CLI command displays the skills table."""
    runner = CliRunner()
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    assert "Forge Pre-Programmed Specialist Skills" in result.stdout
    assert "DBG-01" in result.stdout
    assert "SEC-02" in result.stdout
    assert "TST-03" in result.stdout
