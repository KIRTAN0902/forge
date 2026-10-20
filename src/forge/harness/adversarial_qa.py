"""Forge Zero-Trust Adversarial QA Gatekeeper and Diff-Scoped AST Mutation Engine ([HRN-07], [HRN-08], [SEC-01..03], [TST-01..03])."""

from __future__ import annotations

import ast
import copy
import math
import re
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# 1. Shannon-Entropy Secret Scrubber ([SEC-02])
# ==============================================================================

SECRET_PREFIX_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z-_]{35}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}"),
]


def calculate_shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy H(X) = -sum(p * log2(p)) for string data."""
    if not data:
        return 0.0
    length = len(data)
    freq: dict[str, int] = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1

    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def detect_secrets_in_text(text: str) -> list[str]:
    """Flag secrets via regex prefix matching or high Shannon entropy thresholds."""
    violations: list[str] = []

    # Check known secret regex prefixes
    for pattern in SECRET_PREFIX_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            redacted = m[:4] + "..." + m[-4:] if len(m) > 8 else "***"
            violations.append(f"High-confidence secret pattern detected: {redacted}")

    # Check quoted string literals for Shannon entropy
    literal_pattern = re.compile(r'["\']([A-Za-z0-9+/=_\-]{16,})["\']')
    for match in literal_pattern.finditer(text):
        val = match.group(1)
        # Skip common words/paths
        if "/" in val or "\\" in val or " " in val:
            continue
        entropy = calculate_shannon_entropy(val)

        # Base64 tokens (> 4.5) or Hex tokens (> 3.0 for hex only)
        is_hex = all(c in "0123456789abcdefABCDEF" for c in val)
        if is_hex and len(val) >= 20 and entropy > 3.0:
            violations.append(f"High-entropy hex literal (H={entropy:.2f}): {val[:4]}...{val[-4:]}")
        elif len(val) >= 24 and entropy > 4.5:
            violations.append(
                f"High-entropy string literal (H={entropy:.2f}): {val[:4]}...{val[-4:]}"
            )

    return violations


# ==============================================================================
# 2. AST Security & Asyncio Auditor ([SEC-01], [PERF-01], [TST-03])
# ==============================================================================


class SecurityAndAsyncVisitor(ast.NodeVisitor):
    """AST Visitor enforcing security policies, event-loop hygiene, and mock integrity."""

    def __init__(self) -> None:
        self.violations: list[str] = []
        self._in_async_def = False

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev = self._in_async_def
        self._in_async_def = True
        self.generic_visit(node)
        self._in_async_def = prev

    def visit_Call(self, node: ast.Call) -> None:
        # Check eval() / exec()
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            self.violations.append(
                f"Dangerous dynamic code execution '{node.func.id}()' at line {node.lineno}"
            )

        # Check subprocess.run(..., shell=True)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "run",
            "Popen",
            "check_output",
            "call",
        }:
            for kw in node.keywords:
                if (
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    self.violations.append(f"Subprocess call with shell=True at line {node.lineno}")

        # Check blocking calls in async def
        if self._in_async_def and isinstance(node.func, ast.Attribute):
            # time.sleep()
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"
                and node.func.attr == "sleep"
            ):
                self.violations.append(
                    f"Blocking 'time.sleep()' inside async def at line {node.lineno}"
                )
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "requests":
                self.violations.append(
                    f"Blocking 'requests.{node.func.attr}()' inside async def at line {node.lineno}"
                )

        # Check Mock Integrity: unittest.mock.patch must specify autospec=True
        is_patch_call = False
        if (isinstance(node.func, ast.Name) and node.func.id == "patch") or (
            isinstance(node.func, ast.Attribute) and node.func.attr in {"patch", "object"}
        ):
            is_patch_call = True

        if is_patch_call:
            has_autospec = any(
                kw.arg == "autospec"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            if not has_autospec:
                self.violations.append(
                    f"Mock patch at line {node.lineno} missing 'autospec=True' parameter"
                )

        self.generic_visit(node)


def audit_python_code_ast(code: str) -> list[str]:
    """Perform static AST audit checking security sinks, blocking async calls, and mock integrity."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    visitor = SecurityAndAsyncVisitor()
    visitor.visit(tree)
    return visitor.violations


# ==============================================================================
# 3. Diff-Scoped AST Mutation Engine ([TST-01], [HRN-08])
# ==============================================================================


class ASTMutator(ast.NodeTransformer):
    """Generates targeted AST mutants across binary ops, comparisons, booleans, and constants."""

    def __init__(self, mutation_id: int) -> None:
        self.mutation_id = mutation_id
        self._applied = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if not self._applied and self.mutation_id == 1 and node.ops:
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                node.ops[0] = ast.NotEq()
                self._applied = True
            elif isinstance(op, ast.NotEq):
                node.ops[0] = ast.Eq()
                self._applied = True
            elif isinstance(op, ast.Lt):
                node.ops[0] = ast.GtE()
                self._applied = True
            elif isinstance(op, ast.GtE):
                node.ops[0] = ast.Lt()
                self._applied = True
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not self._applied:
            # Mutation 2: Boolean flip
            if self.mutation_id == 2 and isinstance(node.value, bool):
                node.value = not node.value
                self._applied = True
            # Mutation 4: Constant replacement
            elif self.mutation_id == 4:
                if isinstance(node.value, int) and node.value == 0:
                    node.value = 1
                    self._applied = True
                elif isinstance(node.value, str) and node.value == "":
                    node.value = "mutant_sentinel"
                    self._applied = True
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if not self._applied and self.mutation_id == 3:
            if isinstance(node.op, ast.Add):
                node.op = ast.Sub()
                self._applied = True
            elif isinstance(node.op, ast.Sub):
                node.op = ast.Add()
                self._applied = True
            elif isinstance(node.op, ast.Mult):
                node.op = ast.FloorDiv()
                self._applied = True
        return self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> ast.AST:
        if not self._applied and self.mutation_id == 5 and node.value is not None:
            node.value = ast.Constant(value=None)
            self._applied = True
        return node


class MutationTestReport(BaseModel):
    """Result of running unit tests against diff-scoped AST mutants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_mutants: int
    killed_mutants: int
    surviving_mutants: list[str] = Field(default_factory=list)
    mutation_score: float
    passed: bool


class DiffScopedMutationEngine:
    """Synthesizes targeted AST mutants and verifies test suite sensitivity."""

    @staticmethod
    def generate_mutants(source_code: str, max_mutants: int = 5) -> list[tuple[str, str]]:
        """Generate up to max_mutants mutant code variants with descriptive labels."""
        try:
            base_tree = ast.parse(source_code)
        except SyntaxError:
            return []

        mutants: list[tuple[str, str]] = []
        labels = {
            1: "Inverted comparison operator (== <-> != or < <-> >=)",
            2: "Flipped boolean literal (True <-> False)",
            3: "Replaced binary operator (+ <-> - or * <-> //)",
            4: "Replaced constant literal (0 -> 1 or '' -> 'mutant_sentinel')",
            5: "Mutated return value to None",
        }

        for mut_id in range(1, min(6, max_mutants + 1)):
            tree_copy = copy.deepcopy(base_tree)
            mutator = ASTMutator(mutation_id=mut_id)
            mutated_tree = mutator.visit(tree_copy)
            if mutator._applied:
                ast.fix_missing_locations(mutated_tree)
                try:
                    mutated_code = ast.unparse(mutated_tree)
                    mutants.append((labels[mut_id], mutated_code))
                except Exception:
                    continue

        return mutants

    @classmethod
    async def evaluate_mutation_coverage(
        cls,
        source_code: str,
        test_runner_fn: Callable[[str], Coroutine[Any, Any, bool]],
        max_mutants: int = 5,
    ) -> MutationTestReport:
        """Run test_runner_fn against each mutant. If test passes on mutant code, mutant SURVIVES."""
        mutants = cls.generate_mutants(source_code, max_mutants=max_mutants)
        if not mutants:
            return MutationTestReport(
                total_mutants=0,
                killed_mutants=0,
                surviving_mutants=[],
                mutation_score=1.0,
                passed=True,
            )

        killed = 0
        surviving: list[str] = []

        for label, mutant_code in mutants:
            # test_runner_fn returns True if tests PASS, False if tests FAIL
            test_passed_on_mutant = await test_runner_fn(mutant_code)
            if test_passed_on_mutant:
                # Tests passed despite broken mutant logic: mutant survived!
                surviving.append(label)
            else:
                # Tests failed: mutant killed!
                killed += 1

        total = len(mutants)
        score = killed / total if total > 0 else 1.0
        return MutationTestReport(
            total_mutants=total,
            killed_mutants=killed,
            surviving_mutants=surviving,
            mutation_score=score,
            passed=len(surviving) == 0,
        )


# ==============================================================================
# 4. Adversarial QA Gatekeeper ([HRN-07])
# ==============================================================================


class AdversarialQAGatekeeper:
    """Zero-Trust QA Gatekeeper enforcing information barriers, audits, and mutation tests."""

    @staticmethod
    def strip_chain_of_thought(executor_text: str) -> str:
        """Strip Executor internal chain-of-thought to prevent confirmation bias in QA review."""
        from forge.providers.client import strip_thinking_tags

        # Strip closed, unclosed, and orphan <think>/<thinking> blocks.
        cleaned, _ = strip_thinking_tags(executor_text)
        # Strip Thought: / Reasoning: prefixes
        cleaned = re.sub(
            r"(?im)^(Thought|Reasoning|Chain of Thought):[\s\S]*?(?=\n\n|\Z)", "", cleaned
        )
        return cleaned.strip()

    @staticmethod
    def run_pre_commit_audit(diff_text: str, source_code: str = "") -> list[str]:
        """Multi-vector pre-commit audit: Secrets, AST security sinks, Asyncio event-loop, Mocks."""
        violations: list[str] = []

        # 1. Secret scrubber
        secret_violations = detect_secrets_in_text(diff_text)
        violations.extend(secret_violations)

        # 2. Static AST audit
        if source_code:
            ast_violations = audit_python_code_ast(source_code)
            violations.extend(ast_violations)

        return violations
