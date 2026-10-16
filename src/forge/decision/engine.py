"""Jev Decision Engine: ultra-fast System 1 decision plane for Forge orchestrators.

Implements TypeSafe AI's Jev principle:
- Zero autoregressive token generation for routine routing decisions.
- Non-generative forward pass evaluating typed questions (Noul, Choice, Score).
- Calibrated confidence gating with System 2 deliberative escalation fallback.
"""

from __future__ import annotations

import math
import re
import time

from forge.decision.models import Choice, DecisionPlaneResult, Noul, Score
from forge.skills.registry import SkillRegistry
from forge.skills.types import SkillSignal

# Phase keyword priors
PHASE_KEYWORD_WEIGHTS: dict[str, dict[str, float]] = {
    "planning": {
        "plan": 2.5,
        "architecture": 3.0,
        "design": 1.8,
        "requirements": 2.5,
        "brainstorm": 2.5,
        "research": 3.0,
        "survey": 2.5,
        "mikado": 3.0,
        "adr": 2.5,
        "proposal": 2.0,
        "bento": 2.0,
        "editorial": 2.0,
        "spec": 2.0,
    },
    "execution": {
        "code": 2.0,
        "implement": 2.5,
        "fix": 2.5,
        "write": 2.0,
        "create": 1.5,
        "build": 2.0,
        "refactor": 2.2,
        "edit": 2.0,
        "add": 1.5,
        "modify": 2.0,
        "component": 1.8,
        "spring": 2.2,
        "shadcn": 2.5,
        "style": 1.8,
        "patch": 2.5,
    },
    "qa": {
        "test": 3.0,
        "audit": 3.0,
        "verify": 2.5,
        "review": 2.2,
        "security": 3.0,
        "taint": 3.5,
        "leak": 3.0,
        "secret": 3.0,
        "race": 3.0,
        "deadlock": 3.0,
        "performance": 2.5,
        "benchmark": 2.5,
        "mutation": 3.5,
        "anti-pattern": 3.0,
        "impeccable": 3.5,
        "lint": 2.0,
        "coverage": 2.5,
    },
}


class JevDecisionEngine:
    """System 1 fast decision engine for Forge orchestrator and skill router."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.registry = registry if registry is not None else SkillRegistry.with_anthropic()
        self.confidence_threshold = confidence_threshold

    def decide_phase(self, signal: SkillSignal) -> Choice[str]:
        """Compute categorical probability distribution across agent phases.

        Returns a Jev Choice primitive in sub-1ms without token generation.
        """
        text = f"{signal.user_request} {signal.error_text} {signal.command_executed}".lower()
        scores: dict[str, float] = {"planning": 1.0, "execution": 1.0, "qa": 1.0}

        # 1. Direct environment telemetry overrides
        if signal.test_exit_code is not None and signal.test_exit_code != 0:
            scores["execution"] += 4.0
            scores["qa"] += 3.0

        if signal.diff_text and ("+++" in signal.diff_text or "---" in signal.diff_text):
            # Code diff is active; likely review or execution phase
            scores["qa"] += 2.5
            scores["execution"] += 2.0

        if signal.consecutive_failures >= 3:
            # Repeated failure triggers backtracking/re-planning
            scores["planning"] += 4.0

        # 2. Keyword feature summation
        for phase, kw_map in PHASE_KEYWORD_WEIGHTS.items():
            for kw, weight in kw_map.items():
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    scores[phase] += weight

        # 3. Softmax temperature normalization
        total_exp = sum(math.exp(s) for s in scores.values())
        probabilities = {p: round(math.exp(s) / total_exp, 4) for p, s in scores.items()}

        sorted_phases = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        winner_phase, winner_prob = sorted_phases[0]
        runner_up_prob = sorted_phases[1][1] if len(sorted_phases) > 1 else 0.0

        margin = round(winner_prob - runner_up_prob, 4)
        confidence = round(min(1.0, winner_prob + (margin * 0.5)), 4)

        return Choice(
            question="Which agent execution phase is appropriate for this request?",
            options=["planning", "execution", "qa"],
            probabilities=probabilities,
            selected=winner_phase,
            confidence=confidence,
            margin=margin,
        )

    def evaluate_boolean(
        self,
        statement: str,
        signal: SkillSignal,
        pattern: re.Pattern[str] | None = None,
    ) -> Noul:
        """Evaluate a binary condition as a Jev Noul (probabilistic boolean)."""
        text = f"{signal.user_request} {signal.error_text} {signal.diff_text}".lower()

        if pattern is not None:
            match = bool(pattern.search(text))
            prob = 0.95 if match else 0.08
            conf = 0.92
            return Noul(statement=statement, probability=prob, confidence=conf)

        # Basic presence check
        words = statement.lower().split()
        match_count = sum(1 for w in words if w in text)
        prob = min(0.95, round(match_count / max(1, len(words)), 3))
        conf = round(0.70 + (0.25 * prob), 3)

        return Noul(statement=statement, probability=prob, confidence=conf)

    def assess_risk(self, signal: SkillSignal) -> Score:
        """Assess change severity / vulnerability risk on a scalar metric."""
        raw_score = 0.0
        text = f"{signal.user_request} {signal.error_text} {signal.diff_text}".lower()

        # Check critical danger signals
        danger_ops = ("drop table", "truncate", "delete from", "rm -rf", "alembic downgrade")
        if any(w in text for w in danger_ops):
            raw_score += 4.5
        exec_hazards = ("exec(", "eval(", "shell=true", "raw sql", "injection")
        if any(w in text for w in exec_hazards):
            raw_score += 3.5
        secret_hazards = ("jwt", "secret", "private key", "bearer", "password")
        if any(w in text for w in secret_hazards):
            raw_score += 3.0
        concurrency_hazards = (
            "asyncio.sleep",
            "requests.get",
            "blocking",
            "deadlock",
            "race condition",
        )
        if any(w in text for w in concurrency_hazards):
            raw_score += 2.5

        raw_score = min(10.0, raw_score)
        if raw_score >= 8.0:
            tier = "CRITICAL"
        elif raw_score >= 5.0:
            tier = "HIGH"
        elif raw_score >= 2.5:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return Score(
            metric="system_change_risk",
            raw_score=round(raw_score, 2),
            tier=tier,
            confidence=0.88,
        )

    def route_skills(
        self,
        signal: SkillSignal,
        agent_role: str | None = None,
    ) -> DecisionPlaneResult:
        """Execute full System 1 decision pass over skills catalog in sub-10ms."""
        start_time = time.perf_counter()

        # Step 1: Decide phase
        phase_choice = self.decide_phase(signal)
        effective_role = (agent_role or phase_choice.selected).lower().strip()

        # Step 2: Rank candidate skills
        candidate_skills = self.registry.list_all()
        skill_scores: dict[str, float] = {}
        text = f"{signal.user_request} {signal.error_text} {signal.command_executed}".lower()

        for s in candidate_skills:
            score = 0.1  # baseline prior
            # Target agent match
            if effective_role in s.target_agents:
                score += 1.0

            # Match primary trigger phrases
            triggers = s.primary_trigger.lower().split()
            matching_words = sum(1 for w in triggers if len(w) > 3 and w in text)
            score += matching_words * 0.8

            # Category boost based on phase
            if (effective_role == "planning" and s.category in {"documentation", "design"}) or (
                effective_role == "execution"
                and s.category in {"debugging", "refactoring", "database", "api", "design"}
            ):
                score += 0.8
            elif effective_role == "qa" and s.category in {
                "testing",
                "security",
                "performance",
                "design",
            }:
                score += 1.2

            skill_scores[s.id] = score

        # Normalize skill confidences
        max_s = max(skill_scores.values()) if skill_scores else 1.0
        skill_confidences = {s_id: round(sc / max_s, 3) for s_id, sc in skill_scores.items()}

        # Select skills above confidence threshold (top 1-4 skills)
        sorted_skills = sorted(skill_confidences.items(), key=lambda kv: kv[1], reverse=True)
        selected_skill_ids: list[str] = [s_id for s_id, conf in sorted_skills if conf >= 0.75][:4]

        # Ensure at least top 1 is included if confidence is reasonable
        if not selected_skill_ids and sorted_skills and sorted_skills[0][1] >= 0.50:
            selected_skill_ids.append(sorted_skills[0][0])

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Step 3: Check if escalation to System 2 is warranted
        escalate = not phase_choice.is_decisive(self.confidence_threshold)

        return DecisionPlaneResult(
            phase_choice=phase_choice,
            selected_skill_ids=selected_skill_ids,
            skill_confidences={s_id: skill_confidences[s_id] for s_id in selected_skill_ids},
            escalated_to_system_2=escalate,
            system_1_latency_ms=elapsed_ms,
            metadata={
                "evaluated_candidates": len(candidate_skills),
                "target_role": effective_role,
            },
        )
