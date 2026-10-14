"""The requirement contract: the shared source of truth for every agent.

The fundamental QA question is "can we prove every acceptance criterion is satisfied?",
so acceptance criteria carry an explicit verification method and link back to the
requirements they prove.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from forge.core.contracts.base import Contract


class Priority(StrEnum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"


class NFRCategory(StrEnum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    USABILITY = "usability"
    COMPATIBILITY = "compatibility"
    OTHER = "other"


class VerificationMethod(StrEnum):
    AUTOMATED_TEST = "automated_test"
    COMMAND = "command"
    INSPECTION = "inspection"
    MANUAL = "manual"


class FunctionalRequirement(Contract):
    id: str = Field(pattern=r"^FR-\d{3,}$")
    description: str = Field(min_length=1)
    priority: Priority = Priority.MUST


class NonFunctionalRequirement(Contract):
    id: str = Field(pattern=r"^NFR-\d{3,}$")
    category: NFRCategory = NFRCategory.OTHER
    description: str = Field(min_length=1)


class AcceptanceCriterion(Contract):
    id: str = Field(pattern=r"^AC-\d{3,}$")
    description: str = Field(min_length=1)
    verification: VerificationMethod = VerificationMethod.AUTOMATED_TEST
    verification_detail: str | None = None
    requirement_ids: list[str] = Field(default_factory=list)


class RequirementContract(Contract):
    summary: str = Field(min_length=1)
    functional: list[FunctionalRequirement] = Field(default_factory=list)
    non_functional: list[NonFunctionalRequirement] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    approved: bool = False

    @model_validator(mode="after")
    def _check_ids(self) -> RequirementContract:
        req_ids = [r.id for r in self.functional] + [r.id for r in self.non_functional]
        ac_ids = [c.id for c in self.acceptance_criteria]
        for label, ids in (("requirement", req_ids), ("acceptance criterion", ac_ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"duplicate {label} ids: {dupes}")
        known = set(req_ids)
        for criterion in self.acceptance_criteria:
            unknown = [r for r in criterion.requirement_ids if r not in known]
            if unknown:
                raise ValueError(f"{criterion.id} references unknown requirements: {unknown}")
        return self

    def criterion(self, criterion_id: str) -> AcceptanceCriterion:
        for criterion in self.acceptance_criteria:
            if criterion.id == criterion_id:
                return criterion
        raise KeyError(criterion_id)

    @property
    def criterion_ids(self) -> list[str]:
        return [c.id for c in self.acceptance_criteria]
