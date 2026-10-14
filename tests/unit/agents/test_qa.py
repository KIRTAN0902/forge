from __future__ import annotations

from pathlib import Path

from forge.agents import QAAgent
from forge.core.contracts.qa import (
    CriterionVerdict,
    FindingType,
    QAReport,
    QAStatus,
    Severity,
)
from forge.core.contracts.requirements import (
    AcceptanceCriterion,
    FunctionalRequirement,
    RequirementContract,
    VerificationMethod,
)
from forge.events import Event, EventBus, EventType
from forge.models import (
    ModelAdapter,
    ModelCapabilities,
    ModelProfile,
)
from forge.models.providers.scripted import ScriptedProvider, tool_call_response

from ...conftest import PYTHON, RuntimeFactory

TEST_CMD = f'"{PYTHON}" -m pytest -q -p no:cacheprovider'


def _make_adapter(provider: ScriptedProvider, events: EventBus | None = None) -> ModelAdapter:
    profile = ModelProfile(provider="scripted", model="qa-test")
    return ModelAdapter(
        profile_name="qa",
        profile=profile,
        provider=provider,
        capabilities=ModelCapabilities(tool_calling=True, structured_output=True),
        role="qa",
        events=events,
        workflow_id="wf_qa",
    )


def _sample_requirements() -> RequirementContract:
    return RequirementContract(
        summary="Add integer addition function",
        functional=[FunctionalRequirement(id="FR-001", description="Implement add(a, b)")],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                description="add(2, 3) returns 5",
                verification=VerificationMethod.AUTOMATED_TEST,
                requirement_ids=["FR-001"],
            )
        ],
    )


class TestQAAgent:
    async def test_review_passes_when_all_criteria_satisfied(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        # First fix the bug in ops.py so real tests pass
        ops = sample_repo / "src/calc/ops.py"
        ops.write_text(ops.read_text().replace("a - b", "a + b"))

        runtime = make_runtime(sample_repo)
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        report_payload = QAReport(
            status=QAStatus.PASS,
            summary="All tests pass and acceptance criteria verified.",
            criteria=[
                CriterionVerdict(
                    criterion_id="AC-001",
                    satisfied=True,
                    evidence="Automated pytest passed.",
                )
            ],
            findings=[],
        )

        provider = ScriptedProvider(
            [
                tool_call_response("run_tests", {"command": TEST_CMD}),
                "Tests pass cleanly.",
                report_payload.model_dump_json(),
            ]
        )
        adapter = _make_adapter(provider, events=events_bus)
        qa = QAAgent(adapter=adapter, runtime=runtime, events=events_bus)

        reqs = _sample_requirements()
        report = await qa.review(
            requirements=reqs,
            changed_files=["src/calc/ops.py"],
            max_steps=3,
        )

        assert report.status == QAStatus.PASS
        assert len(report.findings) == 0
        assert report.criteria[0].satisfied is True

        event_types = [e.type for e in emitted]
        assert EventType.QA_STARTED in event_types
        assert EventType.QA_COMPLETED in event_types
        assert EventType.REWORK_REQUESTED not in event_types

    async def test_detects_test_tampering(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        # Simulate test tampering: modify test file to remove assertion
        test_file = sample_repo / "tests/test_ops.py"
        test_file.write_text(test_file.read_text().replace("assert add(2, 3) == 5", "pass"))

        runtime = make_runtime(sample_repo)
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        # Model might try to report PASS, but deterministic check will override/add finding
        report_payload = QAReport(
            status=QAStatus.PASS,
            summary="Pretending it passes",
            criteria=[
                CriterionVerdict(
                    criterion_id="AC-001",
                    satisfied=True,
                    evidence="Tests pass",
                )
            ],
            findings=[],
        )

        provider = ScriptedProvider(
            [
                "Investigated tests.",
                report_payload.model_dump_json(),
            ]
        )
        adapter = _make_adapter(provider, events=events_bus)
        qa = QAAgent(adapter=adapter, runtime=runtime, events=events_bus)

        reqs = _sample_requirements()
        report = await qa.review(
            requirements=reqs,
            changed_files=["tests/test_ops.py"],
            max_steps=2,
        )

        assert report.status == QAStatus.FAIL
        assert any(
            f.type == FindingType.REGRESSION and "tampering" in f.description.lower()
            for f in report.findings
        )
        event_types = [e.type for e in emitted]
        assert EventType.REWORK_REQUESTED in event_types

    async def test_detects_security_violations(
        self, sample_repo: Path, make_runtime: RuntimeFactory
    ) -> None:
        # Introduce a security violation into a file
        bad_file = sample_repo / "src/calc/insecure.py"
        bad_file.write_text("import os\nos.system('echo dangerous')\n")

        runtime = make_runtime(sample_repo)
        events_bus = EventBus()
        emitted: list[Event] = []
        events_bus.subscribe(lambda e: emitted.append(e))

        report_payload = QAReport(
            status=QAStatus.PASS,
            summary="Looks fine to me",
            criteria=[],
            findings=[],
        )
        provider = ScriptedProvider(
            [
                "Done.",
                report_payload.model_dump_json(),
            ]
        )
        adapter = _make_adapter(provider, events=events_bus)
        qa = QAAgent(adapter=adapter, runtime=runtime, events=events_bus)

        report = await qa.review(
            changed_files=["src/calc/insecure.py"],
            max_steps=2,
        )

        assert report.status == QAStatus.FAIL
        assert any(
            f.severity == Severity.P0 and f.type == FindingType.SECURITY for f in report.findings
        )
        event_types = [e.type for e in emitted]
        assert EventType.QA_FINDING_CREATED in event_types
        assert EventType.REWORK_REQUESTED in event_types
