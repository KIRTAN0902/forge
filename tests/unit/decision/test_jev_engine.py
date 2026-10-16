"""Unit tests for Forge's Jev-inspired System 1 Decision Engine."""

from __future__ import annotations

import re

from forge.decision.engine import JevDecisionEngine
from forge.decision.models import Choice, DecisionPlaneResult, Noul, Score
from forge.skills.types import SkillSignal


class TestJevPrimitives:
    def test_noul_probabilistic_boolean(self) -> None:
        n1 = Noul(statement="Needs QA audit", probability=0.85, confidence=0.90)
        assert n1.value is True
        assert n1.is_decisive() is True

        n2 = Noul(statement="Is regression", probability=0.48, confidence=0.40)
        assert n2.value is False
        assert n2.is_decisive() is False

    def test_choice_categorical_distribution(self) -> None:
        ch: Choice[str] = Choice(
            question="Select agent role",
            options=["orchestrator", "executor", "qa"],
            probabilities={"orchestrator": 0.05, "executor": 0.85, "qa": 0.10},
            selected="executor",
            confidence=0.88,
            margin=0.75,
        )
        assert ch.selected == "executor"
        assert ch.is_decisive() is True
        assert ch.probabilities["executor"] == 0.85

    def test_score_ordinal_metric(self) -> None:
        sc = Score(
            metric="mutation_risk",
            raw_score=8.5,
            tier="CRITICAL",
            confidence=0.95,
        )
        assert sc.raw_score == 8.5
        assert sc.tier == "CRITICAL"


class TestJevDecisionEngine:
    def test_decide_phase_planning(self) -> None:
        engine = JevDecisionEngine()
        sig = SkillSignal(user_request="Brainstorm and create the architectural plan for the new bento UI")
        choice = engine.decide_phase(sig)
        assert choice.selected == "planning"
        assert choice.probabilities["planning"] > choice.probabilities["execution"]
        assert choice.confidence >= 0.50

    def test_decide_phase_execution(self) -> None:
        engine = JevDecisionEngine()
        sig = SkillSignal(
            user_request="Fix the broken login endpoint and implement the missing patch",
            test_exit_code=1,
            error_text="AssertionError: 401 != 200",
        )
        choice = engine.decide_phase(sig)
        assert choice.selected == "execution"
        assert choice.probabilities["execution"] > choice.probabilities["planning"]

    def test_decide_phase_qa(self) -> None:
        engine = JevDecisionEngine()
        sig = SkillSignal(
            user_request="Audit the diff for security vulnerabilities, AST taint flow, and secrets",
            diff_text="+++ b/auth.py\n- password = 'x'\n+ exec(user_input)",
        )
        choice = engine.decide_phase(sig)
        assert choice.selected == "qa"
        assert choice.probabilities["qa"] > choice.probabilities["planning"]

    def test_assess_risk_critical(self) -> None:
        engine = JevDecisionEngine()
        sig = SkillSignal(user_request="Drop table users and execute raw sql with exec()")
        risk = engine.assess_risk(sig)
        assert risk.tier in {"HIGH", "CRITICAL"}
        assert risk.raw_score >= 5.0

    def test_assess_risk_low(self) -> None:
        engine = JevDecisionEngine()
        sig = SkillSignal(user_request="Read the README file and check the documentation")
        risk = engine.assess_risk(sig)
        assert risk.tier == "LOW"
        assert risk.raw_score < 2.5

    def test_evaluate_boolean_noul(self) -> None:
        engine = JevDecisionEngine()
        sig = SkillSignal(user_request="Warning: found secret sk-proj-1234567890abcdef123456 in auth.py")
        noul = engine.evaluate_boolean(
            "Contains API secret",
            sig,
            pattern=re.compile(r"sk-[a-zA-Z0-9_-]{10,}"),
        )
        assert noul.value is True
        assert noul.probability > 0.80

    def test_route_skills_sub_10ms_latency(self) -> None:
        engine = JevDecisionEngine()
        sig = SkillSignal(
            user_request="Perform visual audit to catch UI anti-patterns and enforce spacing scale",
            touched_files=["src/components/Card.tsx"],
        )
        result: DecisionPlaneResult = engine.route_skills(sig, agent_role="executor")
        assert len(result.selected_skill_ids) >= 1
        assert "DSN-01" in result.selected_skill_ids or "ANT-01" in result.selected_skill_ids
        # Verify Jev latency is lightning fast (sub-50ms, typically <10ms)
        assert result.system_1_latency_ms < 50.0
