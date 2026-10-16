"""Forge Project Artifact Reporter (.forge/forge.md & .forge/forge.json).

Maintains human-readable living markdown documentation (forge.md) detailing:
- The overall user goal
- Full multi-agent trajectory between ANVIL, HAMMER, and TEMPER
- Requirements, architecture notes, task graph, diffs, test outputs, and QA verdicts

And structured JSON telemetry (forge.json) detailing:
- The skills used, target agent, execution phase, and the rationale (why it was used).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.core.contracts.state import WorkflowState


def build_forge_md(state: WorkflowState) -> str:
    """Generate comprehensive living markdown documentation of the multi-agent trajectory."""
    lines: list[str] = []

    # 1. Header
    lines.append(f"# Forge Multi-Agent Execution Log: {state.user_request.splitlines()[0]}")
    lines.append("")
    timestamp_str = (
        state.updated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if state.updated_at
        else datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    lines.append(
        f"> **Workflow ID:** `{state.workflow_id}`  \n"
        f"> **Status:** `{state.status.value.upper()}` | "
        f"**Iterations:** `{state.iteration}` | "
        f"**Last Updated:** `{timestamp_str}`"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. Mission Goal
    lines.append("## 🎯 Goal & Objective")
    lines.append("")
    lines.append(state.user_request.strip())
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Executive Summary
    total_tasks = len(state.tasks)
    completed_tasks = sum(
        1 for t in state.tasks if str(getattr(t, "status", "")).lower() in ("completed", "done")
    )
    latest_qa = state.latest_qa
    qa_status_val = "N/A"
    if latest_qa:
        if hasattr(latest_qa, "status"):
            qa_status_val = (
                latest_qa.status.value
                if hasattr(latest_qa.status, "value")
                else str(latest_qa.status)
            )
        elif hasattr(latest_qa, "verdict"):
            qa_status_val = (
                latest_qa.verdict.value
                if hasattr(latest_qa.verdict, "value")
                else str(latest_qa.verdict)
            )

    lines.append("## 📊 Executive Summary: What Forge Has Done")
    lines.append(f"- **Final Status:** `{state.status.value}`")
    lines.append(f"- **Total Iterations Completed:** {state.iteration}")
    lines.append(f"- **Plan Progression:** {completed_tasks}/{total_tasks} tasks marked completed")
    lines.append(f"- **Adversarial QA Status:** `{qa_status_val}`")
    lines.append(f"- **Checkpoints Preserved:** {len(state.checkpoints)}")
    lines.append(f"- **Specialist Skills Activated:** {len(state.skills_used)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. Tri-Agent Collaboration Log
    lines.append("## 🤖 Tri-Agent Protocol (ANVIL ➔ HAMMER ➔ TEMPER)")
    lines.append("")
    lines.append(
        "Below is the full chronological record of decisions, implementation steps, "
        "and QA validations across all three specialist agents."
    )
    lines.append("")

    # ANVIL section
    lines.append("### 🎯 ANVIL (Orchestrator & Lead Systems Architect)")
    lines.append("- **Role:** Autonomous Orchestrator, Planner & Strategic Controller")
    if state.architecture:
        arch_str = state.architecture.strip()
        lines.append(f"- **Architecture & System Design:**\n\n```\n{arch_str}\n```")
    if state.requirements and hasattr(state.requirements, "summary"):
        lines.append(f"- **Requirements Analysis:**\n\n> {state.requirements.summary}")

    if state.plan and state.plan.tasks:
        lines.append("\n- **Task Decomposition & Dependency Graph:**")
        for t in state.plan.tasks:
            t_status = (
                t.status.value
                if hasattr(t.status, "value")
                else str(getattr(t, "status", "pending"))
            )
            mark = "x" if t_status.lower() in ("completed", "done") else " "
            hints = f" *(hints: {', '.join(t.files_hint)})*" if t.files_hint else ""
            desc = getattr(t, "description", getattr(t, "title", t.id))
            lines.append(f"  - [{mark}] **{t.id}:** {desc}{hints} `[{t_status}]`")
    else:
        lines.append("- **Task Decomposition:** *(Awaiting planning phase)*")

    if state.decisions:
        lines.append("\n- **Decision Sequence:**")
        for i, dec in enumerate(state.decisions, 1):
            act_val = dec.action.value if hasattr(dec.action, "value") else str(dec.action)
            lines.append(f"  {i}. **Iteration {i} (`{act_val}`):** {dec.rationale}")

    lines.append("")

    # HAMMER section
    lines.append("### ⚡ HAMMER (Runtime Execution Engineer)")
    lines.append("- **Role:** Code Implementer, Refactorer & Tool Operator")
    if state.execution_results:
        lines.append("- **Execution Steps & Actions:**")
        for i, res in enumerate(state.execution_results, 1):
            tid = res.task_id or f"step_{i}"
            res_status = (
                res.status.value.upper()
                if hasattr(res.status, "value")
                else str(getattr(res, "status", "DONE")).upper()
            )
            lines.append(f"  - **Task `{tid}` [{res_status}]:**")
            if hasattr(res, "summary") and res.summary:
                lines.append(f"    - *Summary:* {res.summary}")
            files_changed = [c.path for c in getattr(res, "changes", [])]
            if not files_changed and hasattr(res, "files_modified"):
                files_changed = getattr(res, "files_modified", [])
            if files_changed:
                lines.append(f"    - *Modified Files:* `{', '.join(files_changed)}`")
            test_cmds = [t.command for t in getattr(res, "tests", [])]
            if not test_cmds and hasattr(res, "commands_run"):
                test_cmds = getattr(res, "commands_run", [])
            if test_cmds:
                lines.append(f"    - *Commands Run:* `{', '.join(test_cmds)}`")
            if hasattr(res, "diff") and res.diff:
                diff_snip = res.diff[:400] + ("\n... [truncated]" if len(res.diff) > 400 else "")
                lines.append(f"    - *Diff Preview:*\n```diff\n{diff_snip}\n```")
    else:
        lines.append("- **Execution Steps:** *(No execution actions recorded yet)*")

    lines.append("")

    # TEMPER section
    lines.append("### 🔍 TEMPER (Adversarial QA & Security Gatekeeper)")
    lines.append("- **Role:** Red-Team Auditor, AST Taint Analyzer & Test Gatekeeper")
    if state.qa_reports:
        for i, qa in enumerate(state.qa_reports, 1):
            status_val = (
                qa.status.value
                if hasattr(qa, "status") and hasattr(qa.status, "value")
                else str(getattr(qa, "status", getattr(qa, "verdict", "N/A")))
            )
            lines.append(f"- **Audit {i} Status:** `{status_val}`")
            if hasattr(qa, "blocking_findings") and qa.blocking_findings:
                lines.append(f"  - **Blocking Findings ({len(qa.blocking_findings)}):**")
                for f in qa.blocking_findings:
                    f_desc = getattr(f, "description", getattr(f, "title", ""))
                    f_sev = getattr(f, "severity", "P1")
                    lines.append(f"    - 🛑 `[{f_sev}]` `{f.id}`: {f_desc}")
            non_blocking = [
                f for f in getattr(qa, "findings", []) if not getattr(f, "blocking", False)
            ]
            if non_blocking:
                lines.append(f"  - **Advisory Findings ({len(non_blocking)}):**")
                for w in non_blocking:
                    w_desc = getattr(w, "description", getattr(w, "title", ""))
                    w_sev = getattr(w, "severity", "P2")
                    lines.append(f"    - ⚠️ `[{w_sev}]` `{w.id}`: {w_desc}")
            if hasattr(qa, "summary") and qa.summary:
                lines.append(f"  - **Audit Rationale:** {qa.summary}")
    else:
        lines.append("- **Audits & Findings:** *(No QA reports filed yet)*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. Skills Used Table
    lines.append("## ⚡ Skills Used & Rationale")
    lines.append("")
    if state.skills_used:
        lines.append("| Skill ID | Skill Name | Category | Dispatched Agent | Why / Rationale |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for sk in state.skills_used:
            lines.append(
                f"| `{sk.skill_id}` | {sk.name} | {sk.category} | `{sk.agent}` | {sk.why} |"
            )
    else:
        lines.append("*No specialized skills were activated for this session.*")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. Repository Files Touched
    touched_files: set[str] = set()
    for res in state.execution_results:
        for c in getattr(res, "changes", []):
            if hasattr(c, "path"):
                touched_files.add(c.path)
        for f in getattr(res, "files_modified", []):
            touched_files.add(f)

    lines.append("## 📁 Repository Files Touched")
    if touched_files:
        for filepath in sorted(touched_files):
            lines.append(f"- `{filepath}`")
    else:
        lines.append("*No repository files were modified.*")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by Forge in `.forge/forge.md`.*")
    lines.append("")

    return "\n".join(lines)


def build_forge_json(state: WorkflowState) -> dict[str, Any]:
    """Generate structured JSON tracking skills used, their rationale, and multi-agent summary."""
    skills_by_agent: dict[str, list[str]] = {}
    skills_by_category: dict[str, int] = {}

    skills_list: list[dict[str, Any]] = []
    for sk in state.skills_used:
        agent_key = sk.agent.upper()
        if agent_key not in skills_by_agent:
            skills_by_agent[agent_key] = []
        if sk.skill_id not in skills_by_agent[agent_key]:
            skills_by_agent[agent_key].append(sk.skill_id)

        cat_key = sk.category
        skills_by_category[cat_key] = skills_by_category.get(cat_key, 0) + 1

        at_str = sk.at.isoformat() if isinstance(sk.at, datetime) else datetime.now(UTC).isoformat()
        skills_list.append(
            {
                "skill_id": sk.skill_id,
                "name": sk.name,
                "category": sk.category,
                "agent": sk.agent,
                "why": sk.why,
                "iteration": sk.iteration,
                "timestamp": at_str,
            }
        )

    created_at_str = (
        state.created_at.isoformat() if state.created_at else datetime.now(UTC).isoformat()
    )
    updated_at_str = (
        state.updated_at.isoformat() if state.updated_at else datetime.now(UTC).isoformat()
    )

    return {
        "workflow_id": state.workflow_id,
        "goal": state.user_request,
        "status": state.status.value,
        "iterations": state.iteration,
        "created_at": created_at_str,
        "updated_at": updated_at_str,
        "summary": {
            "total_skills_used": len(state.skills_used),
            "skills_by_agent": skills_by_agent,
            "skills_by_category": skills_by_category,
        },
        "skills": skills_list,
    }


def sync_forge_artifacts(workspace_root: Path, state: WorkflowState) -> tuple[Path, Path]:
    """Persist both .forge/forge.md and .forge/forge.json in the workspace root."""
    forge_dir = workspace_root.resolve() / ".forge"
    forge_dir.mkdir(parents=True, exist_ok=True)

    forge_md_path = forge_dir / "forge.md"
    forge_json_path = forge_dir / "forge.json"

    md_content = build_forge_md(state)
    forge_md_path.write_text(md_content, encoding="utf-8")

    json_data = build_forge_json(state)
    forge_json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return forge_md_path, forge_json_path


def init_forge_artifacts(
    workspace_root: Path, project_name: str | None = None
) -> tuple[Path, Path]:
    """Initialize empty baseline .forge/forge.md and .forge/forge.json on project setup."""
    forge_dir = workspace_root.resolve() / ".forge"
    forge_dir.mkdir(parents=True, exist_ok=True)

    dummy_state = WorkflowState(
        user_request=f"Project initialized for {project_name or workspace_root.name}"
    )
    return sync_forge_artifacts(workspace_root, dummy_state)
