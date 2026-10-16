"""Jev-inspired System 1 decision models and typed primitives (Noul, Choice, Score).

Implements TypeSafe AI's Jev principle:
"LLMs generate (System 2). Jev decides (System 1). Software acts."
Enables sub-10ms, non-generative, probabilistic decision-making over discrete
skill routing and agent phase selection with calibrated confidence scoring.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Noul(BaseModel):
    """Probabilistic boolean primitive (TypeSafe Jev).

    Returns a continuous probability [0.0, 1.0] indicating likelihood that a
    condition or statement is true, accompanied by a calibrated confidence score.
    """

    statement: str = Field(description="The evaluated statement or hypothesis.")
    probability: float = Field(ge=0.0, le=1.0, description="Likelihood between 0.0 and 1.0.")
    confidence: float = Field(ge=0.0, le=1.0, description="Calibration certainty score.")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def value(self) -> bool:
        """Evaluate truthiness against calibrated threshold."""
        return self.probability >= self.threshold

    def is_decisive(self, min_confidence: float = 0.75) -> bool:
        """Check if decision is sufficiently confident to skip System 2 escalation."""
        return self.confidence >= min_confidence and abs(self.probability - 0.5) >= 0.25


class Choice(BaseModel, Generic[T]):
    """Categorical choice primitive with full probability distribution (TypeSafe Jev).

    Represents a decision made over an unordered set of discrete options,
    providing calibrated probabilities across all candidates.
    """

    question: str = Field(description="The typed decision question.")
    options: list[T] = Field(min_length=1, description="Candidate options.")
    probabilities: dict[str, float] = Field(description="Probability distribution over options.")
    selected: T = Field(description="The chosen winner option.")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall decision confidence.")
    margin: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence gap between top 2.")

    model_config = ConfigDict(frozen=True, extra="forbid")

    def is_decisive(self, min_confidence: float = 0.70) -> bool:
        """Check if choice is clear and does not require deliberative LLM escalation."""
        return self.confidence >= min_confidence and self.margin >= 0.15


class Score(BaseModel):
    """Ordinal ranking or scalar score primitive (TypeSafe Jev).

    Evaluates state or code changes against an ordered severity/complexity metric.
    """

    metric: str = Field(description="Metric being evaluated (e.g. risk, complexity, entropy).")
    raw_score: float = Field(ge=0.0, le=10.0, description="Score on 0.0 to 10.0 scale.")
    tier: str = Field(description="Categorical tier (e.g. LOW, MEDIUM, HIGH, CRITICAL).")
    confidence: float = Field(ge=0.0, le=1.0, description="Measurement calibration confidence.")

    model_config = ConfigDict(frozen=True, extra="forbid")


class DecisionPlaneResult(BaseModel):
    """Unified result emitted by the Jev Decision Engine for orchestrator dispatch."""

    phase_choice: Choice[str] = Field(description="Selected agent phase (planning/execution/qa).")
    selected_skill_ids: list[str] = Field(description="Skill IDs chosen by the decision plane.")
    skill_confidences: dict[str, float] = Field(description="Confidence scores per skill.")
    escalated_to_system_2: bool = Field(
        default=False, description="True if low confidence triggered LLM."
    )
    system_1_latency_ms: float = Field(ge=0.0, description="Execution time in milliseconds.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, extra="forbid")
