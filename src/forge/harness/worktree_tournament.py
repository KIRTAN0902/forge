"""Forge Parallel Git-Worktree Architecture Tournament and Speculative Escalation Router ([HRN-06], [HRN-10])."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import shutil
import subprocess
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from forge.skills.models import ModelTier

# ==============================================================================
# AST Cyclomatic Complexity Evaluator
# ==============================================================================


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """Calculates McCabe Cyclomatic Complexity of Python AST."""

    def __init__(self) -> None:
        self.complexity: int = 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)


def calculate_ast_complexity(code_or_tree: str | ast.AST) -> int:
    """Compute deterministic cyclomatic complexity for given Python code or AST."""
    tree: ast.AST
    if isinstance(code_or_tree, str):
        try:
            tree = ast.parse(code_or_tree)
        except SyntaxError:
            return 999  # Penalize unparseable code
    else:
        tree = code_or_tree

    visitor = CyclomaticComplexityVisitor()
    visitor.visit(tree)
    return visitor.complexity


# ==============================================================================
# Tournament Data Models
# ==============================================================================


class TournamentCandidate(BaseModel):
    """Result of an individual candidate branch in the worktree tournament."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    strategy_description: str
    worktree_path: str
    passed: bool
    exit_code: int
    duration_ms: float
    lines_modified: int
    complexity_delta: int
    fitness_score: float
    diff: str = ""
    error_output: str = ""


# ==============================================================================
# Git Worktree Tournament ([HRN-06])
# ==============================================================================


class GitWorktreeTournament:
    """Manages parallel git-worktree tournaments to evaluate competing architectures."""

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.worktrees_dir = self.repo_root / ".forge" / "worktrees"

    def _ensure_clean_worktrees_dir(self) -> None:
        if self.worktrees_dir.exists():
            shutil.rmtree(self.worktrees_dir, ignore_errors=True)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)

    def _create_worktree(self, candidate_id: str) -> Path:
        """Create an isolated detached git worktree for a candidate."""
        target_dir = self.worktrees_dir / candidate_id
        cmd = ["git", "worktree", "add", "--detach", str(target_dir), "HEAD"]
        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 and not target_dir.exists():
            # Fallback for non-git directory or shadow copy
            target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def _cleanup_worktree(self, candidate_dir: Path) -> None:
        """Prune and remove git worktree."""
        try:
            cmd = ["git", "worktree", "remove", "--force", str(candidate_dir)]
            subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

        if candidate_dir.exists():
            shutil.rmtree(candidate_dir, ignore_errors=True)

    @staticmethod
    def compute_fitness(
        passed: bool, duration_ms: float, lines_modified: int, complexity_delta: int
    ) -> float:
        """Deterministic fitness scoring formula:

        Score = (pass * 1000) - (duration_ms * 0.1) - (lines_modified * 2) - (complexity_delta * 10)
        """
        pass_bonus = 1000.0 if passed else 0.0
        return pass_bonus - (duration_ms * 0.1) - (lines_modified * 2.0) - (complexity_delta * 10.0)

    async def evaluate_candidate(
        self,
        candidate_id: str,
        strategy: str,
        apply_fn: Callable[[Path], Coroutine[Any, Any, None]],
        test_command: str = "pytest",
        timeout_sec: float = 60.0,
    ) -> TournamentCandidate:
        """Run a single candidate strategy in an isolated worktree and compute fitness metrics."""
        worktree_path = self._create_worktree(candidate_id)

        start_time = time.perf_counter()
        exit_code = 1
        passed = False
        error_output = ""
        diff_text = ""
        lines_modified = 0
        complexity_delta = 0

        try:
            # 1. Apply strategy changes in the worktree
            await apply_fn(worktree_path)

            # 2. Extract git diff
            diff_proc = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "HEAD",
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            diff_out, _ = await diff_proc.communicate()
            diff_text = diff_out.decode("utf-8", errors="replace")

            # Count lines modified from diff
            for line in diff_text.splitlines():
                if (line.startswith("+") and not line.startswith("+++")) or (
                    line.startswith("-") and not line.startswith("---")
                ):
                    lines_modified += 1

            # 3. Measure AST complexity delta
            for py_file in worktree_path.rglob("*.py"):
                if py_file.is_file() and ".git" not in py_file.parts:
                    try:
                        content = py_file.read_text(encoding="utf-8")
                        complexity_delta += calculate_ast_complexity(content)
                    except Exception:
                        pass

            # 4. Execute test suite inside worktree
            cmd_parts = test_command.split()
            test_proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                _stdout_data, stderr_data = await asyncio.wait_for(
                    test_proc.communicate(), timeout=timeout_sec
                )
                exit_code = test_proc.returncode if test_proc.returncode is not None else 1
                passed = exit_code == 0
                if not passed:
                    error_output = stderr_data.decode("utf-8", errors="replace")
            except TimeoutError:
                test_proc.kill()
                exit_code = 124
                passed = False
                error_output = f"Execution timed out after {timeout_sec}s"

        except Exception as e:
            exit_code = 1
            passed = False
            error_output = str(e)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0

        fitness = self.compute_fitness(
            passed=passed,
            duration_ms=duration_ms,
            lines_modified=lines_modified,
            complexity_delta=complexity_delta,
        )

        return TournamentCandidate(
            candidate_id=candidate_id,
            strategy_description=strategy,
            worktree_path=str(worktree_path),
            passed=passed,
            exit_code=exit_code,
            duration_ms=duration_ms,
            lines_modified=lines_modified,
            complexity_delta=complexity_delta,
            fitness_score=fitness,
            diff=diff_text,
            error_output=error_output,
        )

    async def run_parallel_tournament(
        self,
        task_description: str,
        candidate_strategies: list[tuple[str, Callable[[Path], Coroutine[Any, Any, None]]]],
        test_command: str = "pytest",
        timeout_sec: float = 60.0,
    ) -> TournamentCandidate:
        """Spawn K worktrees in parallel, evaluate candidates, pick winner, and clean up."""
        self._ensure_clean_worktrees_dir()

        tasks = []
        for i, (strategy_desc, apply_fn) in enumerate(candidate_strategies):
            sha_tag = hashlib.sha256(f"{strategy_desc}_{i}".encode()).hexdigest()[:8]
            cid = f"cand_{i}_{sha_tag}"
            tasks.append(
                self.evaluate_candidate(
                    candidate_id=cid,
                    strategy=strategy_desc,
                    apply_fn=apply_fn,
                    test_command=test_command,
                    timeout_sec=timeout_sec,
                )
            )

        results = await asyncio.gather(*tasks)

        # Select candidate with highest fitness score
        winner = max(results, key=lambda c: c.fitness_score)

        # Clean up all tournament worktrees
        for c in results:
            self._cleanup_worktree(Path(c.worktree_path))

        # Prune git worktree registry
        with contextlib.suppress(Exception):
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=str(self.repo_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        return winner


# ==============================================================================
# Speculative Escalation Router ([HRN-10])
# ==============================================================================


class SpeculativeEscalationRouter:
    """Fast-paths SLM speculative patches and escalates to Frontier Tournament if unresolved."""

    @staticmethod
    async def execute_with_speculative_fallback(
        fast_slm_fn: Callable[[], Coroutine[Any, Any, bool]],
        tournament_fn: Callable[[], Coroutine[Any, Any, TournamentCandidate]],
        max_fast_time_sec: float = 3.0,
    ) -> dict[str, Any]:
        """Attempt fast speculative patch first (< 3s). Escalate to Frontier tournament if failed."""
        try:
            passed = await asyncio.wait_for(fast_slm_fn(), timeout=max_fast_time_sec)
            if passed:
                return {
                    "tier_used": ModelTier.PROSTHETIC_SLM,
                    "escalated": False,
                    "success": True,
                    "message": "Speculative SLM patch passed all verification gates directly.",
                }
        except (TimeoutError, Exception):
            pass

        # Escalate to Frontier model worktree tournament
        tournament_result = await tournament_fn()
        return {
            "tier_used": ModelTier.AMPLIFIER_FRONTIER,
            "escalated": True,
            "success": tournament_result.passed,
            "winner": tournament_result,
            "message": "Escalated to Frontier Worktree Tournament.",
        }
