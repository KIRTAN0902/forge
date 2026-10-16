"""Unit tests for the Forge Artifact Reporter (.forge/forge.md & .forge/forge.json)."""

from __future__ import annotations

from pathlib import Path

from forge.config import init_project, load_config
from forge.core.contracts.decisions import Decision, OrchestratorAction
from forge.core.contracts.execution import (
    ChangeType,
    ExecutionResult,
    ExecutionStatus,
    FileChange,
    TestRun,
)
from forge.core.contracts.plan import Plan, Task, TaskStatus
from forge.core.contracts.qa import FindingType, QAFinding, QAReport, QAStatus, Severity
from forge.core.contracts.state import ActivatedSkillRecord, WorkflowState, WorkflowStatus
from forge.workflow.reporter import (
    build_forge_json,
    build_forge_md,
    sync_forge_artifacts,
)


def test_build_forge_md_and_json(tmp_path: Path) -> None:
    # 1. Construct a rich WorkflowState simulating all 3 agents (ANVIL, HAMMER, TEMPER)
    state = WorkflowState(
        user_request="Refactor the authentication flow to use JWT tokens.",
        status=WorkflowStatus.COMPLETED,
        iteration=3,
        architecture="Auth Service ➔ JWT Token Verifier ➔ Database Repository",
    )

    # ANVIL contributions
    state.plan = Plan(
        goal="JWT Refactor",
        tasks=[
            Task(
                id="TASK-1",
                description="Create JWT token generator in auth/tokens.py",
                status=TaskStatus.DONE,
                files_hint=["auth/tokens.py"],
            ),
            Task(
                id="TASK-2",
                description="Update auth middleware to verify JWT headers",
                status=TaskStatus.DONE,
                files_hint=["auth/middleware.py"],
            ),
        ],
    )
    state.decisions.append(
        Decision(
            action=OrchestratorAction.PLAN,
            rationale="Analyzed requirements and created 2-stage execution plan.",
        )
    )
    state.decisions.append(
        Decision(
            action=OrchestratorAction.EXECUTE,
            rationale="Executed TASK-1.",
            task_ids=["TASK-1"],
        )
    )

    # HAMMER contributions
    state.execution_results.append(
        ExecutionResult(
            task_id="TASK-1",
            status=ExecutionStatus.COMPLETED,
            summary="Implemented token generation and signing routines.",
            changes=[
                FileChange(
                    path="src/auth/tokens.py", change_type=ChangeType.ADDED, reason="JWT provider"
                )
            ],
            tests=[TestRun(command="pytest tests/test_tokens.py", exit_code=0, passed=3, failed=0)],
        )
    )

    # TEMPER contributions
    state.qa_reports.append(
        QAReport(
            status=QAStatus.PASS,
            summary="AST audit confirmed no token leakage in logs; 100% test coverage.",
            findings=[
                QAFinding(
                    id="SEC-001",
                    severity=Severity.P2,
                    type=FindingType.QUALITY,
                    description="Consider making token expiration configurable via env var.",
                )
            ],
        )
    )

    # Skills used by each agent with explicit "why" rationales
    state.skills_used.append(
        ActivatedSkillRecord(
            skill_id="ARCH-01",
            name="Domain-Driven Decomposition",
            category="architecture",
            agent="ANVIL",
            why="Modularize token verification cleanly away from storage layer",
            iteration=1,
        )
    )
    state.skills_used.append(
        ActivatedSkillRecord(
            skill_id="PRO-02",
            name="Secure Token Implementation",
            category="security",
            agent="HAMMER",
            why="Enforce cryptographically sound JWT signing algorithm (RS256)",
            iteration=2,
        )
    )
    state.skills_used.append(
        ActivatedSkillRecord(
            skill_id="QA-SEC-01",
            name="AST Taint Flow Security Audit",
            category="security",
            agent="TEMPER",
            why="Verify that JWT secret is not logged or exposed to client responses",
            iteration=3,
        )
    )

    # 2. Build Markdown
    md_content = build_forge_md(state)
    assert (
        "# Forge Multi-Agent Execution Log: Refactor the authentication flow to use JWT tokens."
        in md_content
    )
    assert "## 🎯 Goal & Objective" in md_content
    assert "## 📊 Executive Summary: What Forge Has Done" in md_content
    assert "- **Final Status:** `completed`" in md_content
    assert "- **Total Iterations Completed:** 3" in md_content
    assert "- **Adversarial QA Status:** `PASS`" in md_content
    assert "### 🎯 ANVIL (Orchestrator & Lead Systems Architect)" in md_content
    assert "### ⚡ HAMMER (Runtime Execution Engineer)" in md_content
    assert "### 🔍 TEMPER (Adversarial QA & Security Gatekeeper)" in md_content
    assert "Domain-Driven Decomposition" in md_content
    assert "Secure Token Implementation" in md_content
    assert "AST Taint Flow Security Audit" in md_content
    assert "src/auth/tokens.py" in md_content

    # 3. Build JSON
    json_data = build_forge_json(state)
    assert json_data["goal"] == "Refactor the authentication flow to use JWT tokens."
    assert json_data["status"] == "completed"
    assert json_data["summary"]["total_skills_used"] == 3
    assert "ANVIL" in json_data["summary"]["skills_by_agent"]
    assert "HAMMER" in json_data["summary"]["skills_by_agent"]
    assert "TEMPER" in json_data["summary"]["skills_by_agent"]

    skills_list = json_data["skills"]
    assert len(skills_list) == 3
    skill_ids = [s["skill_id"] for s in skills_list]
    assert "ARCH-01" in skill_ids
    assert "PRO-02" in skill_ids
    assert "QA-SEC-01" in skill_ids

    # Verify "why" rationale is recorded
    arch_record = next(s for s in skills_list if s["skill_id"] == "ARCH-01")
    assert arch_record["agent"] == "ANVIL"
    assert "Modularize token verification cleanly" in arch_record["why"]

    # 4. Sync artifacts into filesystem
    md_path, json_path = sync_forge_artifacts(tmp_path, state)
    assert md_path.is_file()
    assert json_path.is_file()
    assert md_path.name == "forge.md"
    assert json_path.name == "forge.json"
    assert md_path.parent.name == ".forge"
    assert not (tmp_path / "forge.yaml").exists()


def test_init_project_only_creates_forge_folder_no_yaml(tmp_path: Path) -> None:
    init_res = init_project(tmp_path, name="test-service")
    assert init_res.config_path == tmp_path / ".forge" / "config.json"
    assert not (tmp_path / "forge.yaml").exists()
    assert (tmp_path / ".forge").is_dir()
    assert (tmp_path / ".forge" / "forge.md").is_file()
    assert (tmp_path / ".forge" / "forge.json").is_file()
    assert (tmp_path / ".forge" / "config.json").is_file()

    # Load config and verify root and name
    config = load_config(tmp_path, env={})
    assert config.project.name == "test-service"
    assert config.root == tmp_path.resolve()
