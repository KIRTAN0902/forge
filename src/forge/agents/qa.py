"""QA specialist agent (spec §6).

Adversarial software tester and security reviewer.
Evaluates code changes against acceptance criteria, runs tests, inspects for regressions,
detects test tampering, scans for security flaws, and returns a structured QAReport.
"""

from __future__ import annotations

from collections.abc import Sequence

from forge.agents.base import AgentLoopOutcome, BaseAgent
from forge.agents.prompts import QA_SYSTEM_PROMPT
from forge.context.builder import BuiltContext
from forge.context.render import render_requirements
from forge.context.security import scan_files
from forge.core.contracts.execution import ExecutionResult
from forge.core.contracts.plan import Plan
from forge.core.contracts.qa import (
    FindingType,
    QAFinding,
    QAReport,
    QAScope,
    QAStatus,
    Severity,
)
from forge.core.contracts.requirements import RequirementContract
from forge.events.bus import EventBus
from forge.events.types import EventType
from forge.models.adapter import ModelAdapter
from forge.models.types import Message, ToolCall
from forge.tools.defaults import ROLE_TOOLSETS
from forge.tools.runtime import ToolRuntime


class QAAgent(BaseAgent):
    """Adversarial software tester and security reviewer."""

    def __init__(
        self,
        adapter: ModelAdapter,
        runtime: ToolRuntime,
        *,
        events: EventBus | None = None,
        workflow_id: str | None = None,
    ) -> None:
        super().__init__(
            role="qa",
            adapter=adapter,
            runtime=runtime,
            events=events,
            workflow_id=workflow_id,
        )

    async def review(
        self,
        *,
        context: BuiltContext | None = None,
        requirements: RequirementContract | None = None,
        plan: Plan | None = None,
        changed_files: Sequence[str] = (),
        execution_results: Sequence[ExecutionResult] = (),
        scope: QAScope = QAScope.FULL,
        max_steps: int = 10,
    ) -> QAReport:
        """Conduct adversarial review and return a structured QAReport."""
        criteria_ids = requirements.criterion_ids if requirements else []
        await self._emit(
            EventType.QA_STARTED,
            scope=scope.value,
            criteria_ids=criteria_ids,
            changed_files=list(changed_files),
        )

        deterministic_findings: list[QAFinding] = []
        deterministic_findings.extend(await self._check_security(changed_files))
        tampering = await self._check_test_tampering(changed_files, execution_results)
        deterministic_findings.extend(tampering)

        investigation_outcome = await self._run_investigation(
            context=context,
            requirements=requirements,
            plan=plan,
            changed_files=changed_files,
            execution_results=execution_results,
            scope=scope,
            max_steps=max_steps,
        )

        report = await self._synthesize_report(
            requirements=requirements,
            changed_files=changed_files,
            deterministic_findings=deterministic_findings,
            notes=investigation_outcome.final_text,
            scope=scope,
        )

        existing_ids = {f.id for f in report.findings}
        for df in deterministic_findings:
            if df.id not in existing_ids:
                report.findings.append(df)

        if report.blocking_findings:
            report.status = QAStatus.FAIL
        elif not report.findings and all(c.satisfied is True for c in report.criteria):
            report.status = QAStatus.PASS

        for finding in report.findings:
            await self._emit(
                EventType.QA_FINDING_CREATED,
                finding_id=finding.id,
                severity=finding.severity.value,
                type=finding.type.value,
                description=finding.description,
                affected_files=finding.affected_files,
            )

        if report.blocking_findings:
            rework_items = [item for f in report.blocking_findings for item in f.required_rework]
            await self._emit(
                EventType.REWORK_REQUESTED,
                rework_count=len(report.blocking_findings),
                rework_items=rework_items,
            )

        await self._emit(
            EventType.QA_COMPLETED,
            status=report.status.value,
            findings_count=len(report.findings),
            criteria_passed=sum(1 for c in report.criteria if c.satisfied is True),
            criteria_total=len(report.criteria),
        )

        return report

    async def _run_investigation(
        self,
        *,
        context: BuiltContext | None,
        requirements: RequirementContract | None,
        plan: Plan | None,
        changed_files: Sequence[str],
        execution_results: Sequence[ExecutionResult],
        scope: QAScope,
        max_steps: int,
    ) -> AgentLoopOutcome:
        messages: list[Message] = [
            Message.system(QA_SYSTEM_PROMPT),
        ]
        if context is not None:
            messages.append(context.to_message())

        parts = [
            f"Review code changes (scope: {scope.value}).",
            f"Changed files: {', '.join(changed_files) or 'None'}",
        ]
        if requirements:
            parts.append(f"Requirements to verify:\n{render_requirements(requirements)}")

        if execution_results:
            summaries = [f"- {r.task_id}: {r.summary}" for r in execution_results]
            parts.append("Execution summaries:\n" + "\n".join(summaries))

        parts.append(
            "Instructions:\n"
            "1. Run tests with `run_tests` to check for test failures or regressions.\n"
            "2. Read modified files with `read_file` to verify requirement implementation.\n"
            "3. Conclude with a detailed summary of your findings and criteria verdicts."
        )
        messages.append(Message.user("\n\n".join(parts)))

        return await self._run_tool_loop(
            messages,
            allowed_tools=ROLE_TOOLSETS["qa"],
            max_steps=max_steps,
        )

    async def _check_security(self, changed_files: Sequence[str]) -> list[QAFinding]:
        """Run static regex scanners over changed files to spot security vulnerabilities."""
        if self.runtime is None:
            return []

        findings: list[QAFinding] = []

        def read_file_sync(path: str) -> str | None:
            if self.runtime is None:
                return None
            resolved = self.runtime.env.workspace.resolve(path)
            if resolved.is_file():
                return resolved.read_text(encoding="utf-8", errors="ignore")
            return None

        hits = scan_files(changed_files, read_file_sync)
        for i, hit in enumerate(hits, 1):
            findings.append(
                QAFinding(
                    id=f"QA-SEC-{i:03d}",
                    severity=Severity.P0,
                    type=FindingType.SECURITY,
                    description=f"Security issue detected ({hit.label}): {hit.snippet}",
                    evidence=f"{hit.path}:{hit.line}",
                    affected_files=[hit.path],
                    required_rework=[f"Fix {hit.label} vulnerability at {hit.path}:{hit.line}"],
                )
            )

        return findings

    async def _check_test_tampering(
        self, changed_files: Sequence[str], execution_results: Sequence[ExecutionResult]
    ) -> list[QAFinding]:
        """Detect if test files were modified or deleted during code fixes (test gaming)."""
        findings: list[QAFinding] = []
        tampered_tests = [
            p
            for p in changed_files
            if ("test" in p.lower() or p.startswith("tests/")) and not p.endswith(".pyc")
        ]

        for file in tampered_tests:
            diff_text = ""
            if self.runtime is not None:
                try:
                    res = await self.runtime.execute(
                        ToolCall(id="chk_diff", name="git_diff", arguments={"paths": [file]})
                    )
                    diff_text = res.content if res.ok else ""
                except Exception:
                    pass

            tampered = (
                "-    assert" in diff_text
                or "-        assert" in diff_text
                or "- assert" in diff_text
            )
            if tampered:
                findings.append(
                    QAFinding(
                        id=f"QA-TAMPER-{len(findings) + 1:03d}",
                        severity=Severity.P1,
                        type=FindingType.REGRESSION,
                        description=(
                            f"Suspicious test tampering in {file}: assertions were removed."
                            " Code must be fixed to satisfy tests, not tests altered."
                        ),
                        evidence=diff_text[:300],
                        affected_files=[file],
                        required_rework=[
                            "Restore original test assertions and fix implementation."
                        ],
                    )
                )

        return findings

    async def _synthesize_report(
        self,
        *,
        requirements: RequirementContract | None,
        changed_files: Sequence[str],
        deterministic_findings: list[QAFinding],
        notes: str,
        scope: QAScope,
    ) -> QAReport:
        messages: list[Message] = [
            Message.system(QA_SYSTEM_PROMPT),
        ]
        prompt = (
            f"Synthesize the final QAReport (scope: {scope.value}).\n\n"
            f"Changed files: {', '.join(changed_files) or 'None'}\n\n"
            f"Investigation notes:\n{notes}\n\n"
        )
        if requirements:
            prompt += f"Requirement Contract:\n{render_requirements(requirements)}\n\n"

        if deterministic_findings:
            prompt += (
                "Identified security/tampering findings:\n"
                + "\n".join(f"- [{f.severity}] {f.description}" for f in deterministic_findings)
                + "\n\n"
            )

        prompt += (
            "Ensure the QAReport includes:\n"
            "1. 'status': 'PASS' (only if all criteria satisfied and zero P0/P1 findings), "
            "'FAIL', or 'INCONCLUSIVE'.\n"
            "2. 'criteria': a CriterionVerdict for each acceptance criterion with 'satisfied' "
            "(true/false) and 'evidence'.\n"
            "3. 'findings': any defects, regressions, or requirement violations found.\n"
            "4. 'summary': clear explanation of verification status."
        )
        messages.append(Message.user(prompt))

        return await self.adapter.structured(messages, QAReport)
