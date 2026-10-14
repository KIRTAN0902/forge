"""Formatting WorkflowState for LLM Orchestrator decision turns."""

from __future__ import annotations

from forge.context.render import render_findings, render_plan, render_requirements
from forge.core.contracts.state import WorkflowState


def render_workflow_state(state: WorkflowState) -> str:
    """Render a comprehensive markdown summary of current workflow state."""
    sections: list[str] = [
        f"## User Request\n{state.user_request.strip()}",
        (
            "## Workflow Status\n"
            f"- Status: {state.status.value}\n"
            f"- Iteration: {state.iteration}\n"
            f"- Current Task: {state.current_task_id or 'None'}"
        ),
    ]

    if state.requirements:
        sections.append(f"## Requirements\n{render_requirements(state.requirements)}")

    if state.plan:
        plan_text = render_plan(state.plan, current=state.current_task_id)
        sections.append(f"## Task Graph & Plan\n{plan_text}")

    if state.execution_results:
        latest_exec = state.execution_results[-1]
        exec_lines = [
            f"- Task ID: {latest_exec.task_id}",
            f"- Status: {latest_exec.status.value}",
            f"- Summary: {latest_exec.summary}",
            f"- Modified Files: {', '.join(c.path for c in latest_exec.changes) or 'None'}",
        ]
        if latest_exec.tests:
            for t in latest_exec.tests:
                res = "PASS" if t.succeeded else "FAIL"
                exec_lines.append(f"- Test: `{t.command}` -> {res} (code {t.exit_code})")
        sections.append("## Latest Execution Result\n" + "\n".join(exec_lines))

    if state.qa_reports:
        latest_qa = state.qa_reports[-1]
        qa_lines = [
            f"- Status: {latest_qa.status.value}",
            f"- Scope: {latest_qa.scope.value}",
            f"- Summary: {latest_qa.summary}",
        ]
        if latest_qa.criteria:
            for c in latest_qa.criteria:
                v = (
                    "PROVEN"
                    if c.satisfied is True
                    else ("DISPROVEN" if c.satisfied is False else "?")
                )
                qa_lines.append(f"- Criterion {c.criterion_id}: {v} ({c.evidence})")
        if latest_qa.findings:
            qa_lines.append("\nFindings:\n" + render_findings(latest_qa.findings))
        sections.append("## Latest QA Report\n" + "\n".join(qa_lines))

    if state.errors:
        err_lines = [f"- [{e.source}] {e.message}" for e in state.errors[-3:]]
        sections.append("## Recent Errors\n" + "\n".join(err_lines))

    if state.decisions:
        dec_lines = [
            f"- Iteration {i}: {d.action.value} ({d.rationale})"
            for i, d in enumerate(state.decisions[-3:], 1)
        ]
        sections.append("## Recent Decisions\n" + "\n".join(dec_lines))

    return "\n\n".join(sections)
