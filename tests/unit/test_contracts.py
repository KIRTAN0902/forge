from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge.core.contracts import (
    AcceptanceCriterion,
    CriterionVerdict,
    FindingType,
    FunctionalRequirement,
    Plan,
    QAFinding,
    QAReport,
    QAStatus,
    RequirementContract,
    Severity,
    Task,
    TaskStatus,
    WorkflowState,
)
from forge.core.ids import new_id


def make_plan(**overrides: object) -> Plan:
    data: dict[str, object] = {
        "goal": "Add authentication",
        "tasks": [
            {"id": "AUTH-001", "description": "Create user model"},
            {"id": "AUTH-002", "description": "Hash passwords", "dependencies": ["AUTH-001"]},
            {"id": "AUTH-003", "description": "Login route", "dependencies": ["AUTH-002"]},
        ],
    }
    data.update(overrides)
    return Plan.model_validate(data)


class TestIds:
    def test_prefix_and_uniqueness(self) -> None:
        ids = {new_id("wf") for _ in range(1000)}
        assert len(ids) == 1000
        assert all(i.startswith("wf_") for i in ids)

    def test_time_ordered(self) -> None:
        first = new_id("x")
        second = new_id("x")
        assert first.split("_")[1] <= second.split("_")[1]

    def test_rejects_bad_prefix(self) -> None:
        with pytest.raises(ValueError):
            new_id("bad prefix")


class TestPlan:
    def test_spec_example_parses(self) -> None:
        # The example from FORGE_SPECIFICATION.md §4.1, verbatim shape.
        plan = Plan.model_validate(
            {
                "goal": "Add authentication",
                "tasks": [
                    {"id": "AUTH-001", "description": "Create user model", "dependencies": []},
                    {
                        "id": "AUTH-002",
                        "description": "Implement password hashing",
                        "dependencies": ["AUTH-001"],
                    },
                ],
                "acceptance_criteria": ["Users can register"],
                "risks": ["Existing API routes may be unauthenticated"],
            }
        )
        assert [t.id for t in plan.ready_tasks()] == ["AUTH-001"]

    def test_unknown_dependency_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown tasks"):
            make_plan(tasks=[{"id": "A", "description": "a", "dependencies": ["Z"]}])

    def test_cycle_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cycle"):
            make_plan(
                tasks=[
                    {"id": "A", "description": "a", "dependencies": ["B"]},
                    {"id": "B", "description": "b", "dependencies": ["A"]},
                ]
            )

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            make_plan(tasks=[{"id": "A", "description": "a"}, {"id": "A", "description": "b"}])

    def test_topological_order_puts_dependencies_first(self) -> None:
        plan = make_plan(
            tasks=[
                {"id": "C", "description": "c", "dependencies": ["B"]},
                {"id": "A", "description": "a"},
                {"id": "B", "description": "b", "dependencies": ["A"]},
            ]
        )
        assert [t.id for t in plan.topological_order()] == ["A", "B", "C"]

    def test_ready_tasks_progression(self) -> None:
        plan = make_plan()
        assert [t.id for t in plan.ready_tasks()] == ["AUTH-001"]
        plan.task("AUTH-001").status = TaskStatus.DONE
        assert [t.id for t in plan.ready_tasks()] == ["AUTH-002"]
        plan.task("AUTH-002").status = TaskStatus.DONE
        plan.task("AUTH-003").status = TaskStatus.DONE
        assert plan.is_complete

    def test_extra_fields_from_models_are_ignored(self) -> None:
        plan = make_plan(confidence="high")
        assert not hasattr(plan, "confidence")

    def test_assignment_is_validated(self) -> None:
        task = Task(id="A", description="a")
        with pytest.raises(ValidationError):
            task.status = "not-a-status"  # type: ignore[assignment]


class TestRequirementContract:
    def test_valid_contract(self) -> None:
        contract = RequirementContract(
            summary="Auth",
            functional=[FunctionalRequirement(id="FR-001", description="Users can register")],
            acceptance_criteria=[
                AcceptanceCriterion(
                    id="AC-001", description="Register returns 201", requirement_ids=["FR-001"]
                )
            ],
        )
        assert contract.criterion_ids == ["AC-001"]
        assert contract.criterion("AC-001").requirement_ids == ["FR-001"]

    def test_requires_acceptance_criteria(self) -> None:
        with pytest.raises(ValidationError):
            RequirementContract(summary="x", acceptance_criteria=[])

    def test_unknown_requirement_reference(self) -> None:
        with pytest.raises(ValidationError, match="unknown requirements"):
            RequirementContract(
                summary="x",
                acceptance_criteria=[
                    AcceptanceCriterion(id="AC-001", description="d", requirement_ids=["FR-404"])
                ],
            )

    def test_id_format_enforced(self) -> None:
        with pytest.raises(ValidationError):
            FunctionalRequirement(id="REQ1", description="d")


class TestQAReport:
    def test_blocking_and_unproven(self) -> None:
        report = QAReport(
            status=QAStatus.FAIL,
            criteria=[
                CriterionVerdict(criterion_id="AC-001", satisfied=True),
                CriterionVerdict(criterion_id="AC-002", satisfied=None),
                CriterionVerdict(criterion_id="AC-003", satisfied=False),
            ],
            findings=[
                QAFinding(
                    id="QA-017",
                    severity=Severity.P1,
                    type=FindingType.REQUIREMENT_VIOLATION,
                    description="Users can access another user's project",
                ),
                QAFinding(
                    id="QA-018",
                    severity=Severity.P3,
                    type=FindingType.QUALITY,
                    description="naming",
                ),
            ],
        )
        assert [f.id for f in report.blocking_findings] == ["QA-017"]
        assert report.unproven_criteria() == ["AC-002", "AC-003"]


class TestWorkflowState:
    def test_round_trip_json(self) -> None:
        state = WorkflowState(user_request="Add OAuth", plan=make_plan())
        state.current_task_id = "AUTH-002"
        restored = WorkflowState.model_validate_json(state.model_dump_json())
        assert restored == state
        assert restored.current_task is not None
        assert restored.current_task.id == "AUTH-002"
        assert [t.id for t in restored.tasks] == ["AUTH-001", "AUTH-002", "AUTH-003"]

    def test_empty_request_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowState(user_request="")
