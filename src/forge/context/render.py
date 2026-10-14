"""Rendering contracts and files as compact, model-readable text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from forge.core.contracts import (
    ExecutionResult,
    Plan,
    QAFinding,
    RequirementContract,
    Task,
    TaskStatus,
    TestRun,
)
from forge.tools.analysis import Symbol, language_of

_STATUS_MARK = {
    TaskStatus.PENDING: "[ ]",
    TaskStatus.IN_PROGRESS: "[~]",
    TaskStatus.DONE: "[x]",
    TaskStatus.FAILED: "[!]",
    TaskStatus.BLOCKED: "[#]",
    TaskStatus.SKIPPED: "[-]",
}


def render_requirements(contract: RequirementContract) -> str:
    lines = [contract.summary]
    if contract.functional:
        lines.append("\nFunctional requirements:")
        lines += [f"- {r.id} ({r.priority}): {r.description}" for r in contract.functional]
    if contract.non_functional:
        lines.append("\nNon-functional requirements:")
        lines += [f"- {r.id} [{r.category}]: {r.description}" for r in contract.non_functional]
    lines.append("\nAcceptance criteria:")
    lines += [render_criterion_line(contract, c.id) for c in contract.acceptance_criteria]
    for title, items in (
        ("Assumptions", contract.assumptions),
        ("Out of scope", contract.out_of_scope),
        ("Open questions", contract.open_questions),
    ):
        if items:
            lines.append(f"\n{title}:")
            lines += [f"- {item}" for item in items]
    return "\n".join(lines)


def render_criterion_line(contract: RequirementContract, criterion_id: str) -> str:
    criterion = contract.criterion(criterion_id)
    verify = criterion.verification.value.replace("_", " ")
    detail = f": {criterion.verification_detail}" if criterion.verification_detail else ""
    refs = f" (proves {', '.join(criterion.requirement_ids)})" if criterion.requirement_ids else ""
    return f"- {criterion.id}: {criterion.description}{refs} [verify by {verify}{detail}]"


def render_plan(plan: Plan, *, current: str | None = None) -> str:
    lines = [f"Goal: {plan.goal}"]
    if plan.approach:
        lines.append(f"Approach: {plan.approach}")
    lines.append("Tasks:")
    for task in plan.topological_order():
        marker = " <- current" if task.id == current else ""
        deps = f" (after {', '.join(task.dependencies)})" if task.dependencies else ""
        lines.append(f"  {_STATUS_MARK[task.status]} {task.id}: {task.description}{deps}{marker}")
    if plan.risks:
        lines.append("Risks:")
        lines += [f"  - {r}" for r in plan.risks]
    return "\n".join(lines)


def render_task(task: Task, contract: RequirementContract | None, plan: Plan | None) -> str:
    lines = [f"{task.id}: {task.description}"]
    if task.attempts:
        lines.append(f"Previous attempts: {task.attempts}")
    if task.dependencies and plan is not None:
        done = [d for d in task.dependencies if plan.task(d).status == TaskStatus.DONE]
        lines.append(
            f"Depends on: {', '.join(task.dependencies)} (done: {', '.join(done) or 'none'})"
        )
    if task.files_hint:
        lines.append(f"Files expected to change: {', '.join(task.files_hint)}")
    if task.acceptance_criteria and contract is not None:
        lines.append("This task contributes to:")
        for criterion_id in task.acceptance_criteria:
            try:
                lines.append(render_criterion_line(contract, criterion_id))
            except KeyError:
                lines.append(f"- {criterion_id} (unknown criterion)")
    return "\n".join(lines)


def render_findings(findings: Iterable[QAFinding]) -> str:
    blocks = []
    for finding in findings:
        lines = [f"{finding.id} [{finding.severity} {finding.type}] {finding.description}"]
        if finding.criterion_ids:
            lines.append(f"  criteria: {', '.join(finding.criterion_ids)}")
        if finding.affected_files:
            lines.append(f"  files: {', '.join(finding.affected_files)}")
        if finding.evidence:
            lines.append("  evidence: " + finding.evidence.strip().replace("\n", "\n    "))
        if finding.required_rework:
            lines.append("  required rework:")
            lines += [f"    - {item}" for item in finding.required_rework]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_test_runs(runs: Sequence[TestRun], *, tail_chars: int = 3000) -> str:
    blocks = []
    for run in runs:
        counts = ", ".join(
            f"{value} {name}"
            for name, value in (
                ("passed", run.passed),
                ("failed", run.failed),
                ("errors", run.errors),
            )
            if value is not None
        )
        status = "PASSED" if run.succeeded else "FAILED"
        header = (
            f"$ {run.command}  -> {status} (exit {run.exit_code}{'; ' + counts if counts else ''})"
        )
        tail = run.output_tail[-tail_chars:].strip()
        blocks.append(f"{header}\n{tail}" if tail and not run.succeeded else header)
    return "\n\n".join(blocks)


def render_execution(results: Sequence[ExecutionResult]) -> str:
    lines = []
    for result in results:
        changed = ", ".join(c.path for c in result.changes) or "no file changes"
        lines.append(f"{result.task_id} [{result.status}]: {result.summary} ({changed})")
        lines += [f"  note: {n}" for n in result.notes]
    return "\n".join(lines)


# -- files -------------------------------------------------------------------------------------


def fence_language(path: str) -> str:
    return language_of(path) or ""


def render_file(path: str, text: str) -> str:
    lines = text.splitlines()
    width = len(str(len(lines)))
    body = "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, start=1))
    return f"--- {path} ({len(lines)} lines)\n```{fence_language(path)}\n{body}\n```"


def render_outline(path: str, text: str, symbols: Sequence[Symbol]) -> str:
    total = len(text.splitlines())
    if not symbols:
        return f"--- {path} ({total} lines; no outline available)"
    lines = [f"--- {path} ({total} lines; outline)"]
    for symbol in symbols:
        indent = "  " if symbol.parent else ""
        span = f"{symbol.line}-{symbol.end_line}" if symbol.end_line else str(symbol.line)
        lines.append(f"{indent}{span}: {symbol.signature or symbol.kind + ' ' + symbol.name}")
    return "\n".join(lines)


def render_snippets(
    path: str, text: str, terms: Iterable[str], *, context: int = 5, max_windows: int = 4
) -> str | None:
    """Numbered excerpts around lines that mention any of ``terms``."""
    words = [t for t in {t.lower() for t in terms} if len(t) >= 3]
    if not words:
        return None
    pattern = re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)), re.I)
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if pattern.search(line)]
    if not hits:
        return None
    windows: list[list[int]] = []
    for index in hits:
        start, end = max(0, index - context), min(len(lines), index + context + 1)
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    windows.sort(key=lambda w: -sum(1 for h in hits if w[0] <= h < w[1]))
    chosen = sorted(windows[:max_windows])
    width = len(str(len(lines)))
    parts = [f"--- {path} (excerpts)"]
    for start, end in chosen:
        parts.append(f"```{fence_language(path)}")
        parts += [f"{i + 1:>{width}}| {lines[i]}" for i in range(start, end)]
        parts.append("```")
    return "\n".join(parts)
