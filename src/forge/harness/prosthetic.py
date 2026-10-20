"""Forge Prosthetic Mode: Systems elevating Small Language Models (7B-32B) to Frontier quality."""

from __future__ import annotations

import ast
import difflib
import importlib
import inspect
import shutil
import subprocess
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from forge.skills.models import OperationType

# ==============================================================================
# 1. AST Context Slicer & Skeletonizer ([SLM-01] & [HRN-01])
# ==============================================================================


class Skeletonizer(ast.NodeTransformer):
    """AST Transformer that preserves the target symbol while skeletonizing others."""

    def __init__(self, target_symbol: str | None = None) -> None:
        self.target_symbol = target_symbol

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if self.target_symbol and node.name == self.target_symbol:
            return node
        return self._skeletonize_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if self.target_symbol and node.name == self.target_symbol:
            return node
        return self._skeletonize_func(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if self.target_symbol and node.name == self.target_symbol:
            return node

        new_body: list[ast.stmt] = []
        doc = ast.get_docstring(node)
        if doc:
            first_line = doc.strip().split("\n")[0]
            new_body.append(ast.Expr(value=ast.Constant(value=first_line)))

        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self.target_symbol and stmt.name == self.target_symbol:
                    new_body.append(stmt)
                else:
                    new_body.append(self._skeletonize_func(stmt))
            elif isinstance(stmt, (ast.AnnAssign, ast.Assign, ast.Pass)):
                new_body.append(stmt)

        if not new_body:
            new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))

        node.body = new_body
        return node

    def _skeletonize_func(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        doc = ast.get_docstring(node)
        body: list[ast.stmt] = []
        if doc:
            first_line = doc.strip().split("\n")[0]
            body.append(ast.Expr(value=ast.Constant(value=first_line)))
        body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
        node.body = body
        return node


class ASTContextSlicer:
    """Extracts target AST subtrees and skeletonizes surroundings within token budgets."""

    @staticmethod
    def slice_file(
        file_path: Path | str,
        target_symbol: str | None = None,
        max_tokens: int = 3500,
    ) -> str:
        """Read source file and skeletonize non-target functions/classes to '...'."""
        path = Path(file_path)
        if not path.exists():
            return ""

        content = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError:
            # Fallback to direct text truncation if syntax is broken
            return content[: max_tokens * 4]

        skeletonizer = Skeletonizer(target_symbol=target_symbol)
        skeletonized_tree = skeletonizer.visit(tree)
        ast.fix_missing_locations(skeletonized_tree)

        try:
            rendered = ast.unparse(skeletonized_tree)
        except Exception:
            rendered = content

        # Hard token budget enforcement (estimated tokens: len // 4)
        budget_chars = max_tokens * 4
        if len(rendered) > budget_chars:
            lines = rendered.splitlines()
            trimmed_lines: list[str] = []
            curr_len = 0
            for line in lines:
                if curr_len + len(line) + 1 > budget_chars:
                    trimmed_lines.append(f"# [AST context truncated to {max_tokens} token budget]")
                    break
                trimmed_lines.append(line)
                curr_len += len(line) + 1
            rendered = "\n".join(trimmed_lines)

        return rendered


def introspect_imported_symbols(module_names: list[str]) -> str:
    """Generate ground-truth callable signatures from modules using inspect.signature ([SLM-04])."""
    lines: list[str] = ["[GROUND TRUTH CALL SIGNATURES]"]
    for mod_name in module_names:
        clean_mod = mod_name.strip()
        if not clean_mod:
            continue
        try:
            mod = importlib.import_module(clean_mod)
            for attr_name in sorted(dir(mod)):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(mod, attr_name)
                    if callable(attr):
                        sig = inspect.signature(attr)
                        lines.append(f"{clean_mod}.{attr_name}{sig}")
                except (ValueError, TypeError, AttributeError):
                    continue
        except Exception:
            continue

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def find_golden_exemplar(repo_root: Path | str, op_type: OperationType) -> str:
    """Retrieve an idiomatic, clean function or test from the repository as few-shot anchor ([SLM-07])."""
    root = Path(repo_root)
    is_test = op_type == OperationType.WRITE_UNIT_TEST
    search_dir = (root / "tests") if is_test else (root / "src")
    if not search_dir.exists():
        search_dir = root

    pattern = "test_*.py" if is_test else "*.py"
    for py_file in search_dir.rglob(pattern):
        if not py_file.is_file() or "venv" in py_file.parts or ".git" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
                    fn_lines = content.splitlines()[node.lineno - 1 : node.end_lineno]
                    if 4 <= len(fn_lines) <= 25:
                        return "\n".join(fn_lines)
        except Exception:
            continue

    return ""


# ==============================================================================
# 2. Fuzzy Anchor Patcher ([SLM-03] & [HRN-03])
# ==============================================================================


class PatchAnchorNotFoundError(Exception):
    """Raised when search anchor cannot be located in the target source."""

    def __init__(self, message: str, best_match: str = "", similarity: float = 0.0) -> None:
        super().__init__(message)
        self.best_match = best_match
        self.similarity = similarity


class FuzzyAnchorPatcher:
    """3-Tier fuzzy search/replace anchor patcher for resilient SLM code modifications."""

    @staticmethod
    def apply_patch(source_text: str, search_block: str, replace_block: str) -> str:
        """Apply patch across 3 matching tiers: Exact, Whitespace-Normalized, difflib >= 0.92."""
        from forge.providers.client import strip_thinking_tags

        search_block = strip_thinking_tags(search_block)[0].replace("\r\n", "\n")
        replace_block = strip_thinking_tags(replace_block)[0].replace("\r\n", "\n")
        normalized_source = source_text.replace("\r\n", "\n")

        # Tier 1: Exact string match
        if search_block in normalized_source:
            return normalized_source.replace(search_block, replace_block, 1)

        source_lines = normalized_source.splitlines(keepends=True)
        search_lines = search_block.splitlines(keepends=True)

        if not search_lines or not source_lines:
            raise PatchAnchorNotFoundError("Empty search block or source text", similarity=0.0)

        # Tier 2: Whitespace & indentation normalized line matching
        clean_search = [line.strip() for line in search_lines if line.strip()]
        search_len = len(search_lines)

        for i in range(len(source_lines) - search_len + 1):
            window = source_lines[i : i + search_len]
            clean_window = [line.strip() for line in window if line.strip()]
            if clean_window == clean_search:
                # Matched normalized lines: replace window
                before = "".join(source_lines[:i])
                after = "".join(source_lines[i + search_len :])
                return (
                    before
                    + replace_block
                    + ("\n" if not replace_block.endswith("\n") else "")
                    + after
                )

        # Tier 3: Sliding window with difflib SequenceMatcher >= 0.92
        best_ratio = 0.0
        best_idx = 0
        search_str = "".join(search_lines)

        for i in range(max(1, len(source_lines) - search_len + 1)):
            window_str = "".join(source_lines[i : i + search_len])
            ratio = difflib.SequenceMatcher(None, window_str, search_str).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        if best_ratio >= 0.92:
            before = "".join(source_lines[:best_idx])
            after = "".join(source_lines[best_idx + search_len :])
            return (
                before + replace_block + ("\n" if not replace_block.endswith("\n") else "") + after
            )

        closest_candidate = "".join(source_lines[best_idx : best_idx + search_len])
        raise PatchAnchorNotFoundError(
            f"Search anchor not found. Closest candidate (similarity {best_ratio:.2f}):\n{closest_candidate}",
            best_match=closest_candidate,
            similarity=best_ratio,
        )


# ==============================================================================
# 3. Sub-100ms Compiler Reflexion & Rollback ([SLM-05], [SLM-06], [HRN-02])
# ==============================================================================


class CompilerReflexionResult(BaseModel):
    """Result of in-memory AST and fast compiler validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    error_type: str | None = None
    error_message: str | None = None
    line_number: int | None = None
    reflexion_prompt: str | None = None


class Sub100msCompilerReflexion:
    """Rapid in-memory compiler check and micro-reflexion generator."""

    @staticmethod
    def verify_syntax(code: str, filename: str = "<memory>") -> CompilerReflexionResult:
        """Perform sub-100ms in-memory syntax verification via ast.parse."""
        try:
            ast.parse(code, filename=filename)
        except SyntaxError as e:
            lineno = e.lineno or 1
            msg = e.msg or "invalid syntax"
            prompt = (
                f"SyntaxError on line {lineno}: {msg}. "
                "Fix this exact syntax error in your SEARCH/REPLACE block."
            )
            return CompilerReflexionResult(
                success=False,
                error_type="SyntaxError",
                error_message=str(e),
                line_number=lineno,
                reflexion_prompt=prompt,
            )

        # Fast linter check if ruff is available
        ruff_bin = shutil.which("ruff")
        if ruff_bin:
            try:
                proc = subprocess.run(
                    [ruff_bin, "check", "--stdin-filename", filename, "-"],
                    input=code.encode("utf-8"),
                    capture_output=True,
                    timeout=0.200,
                )
                if proc.returncode != 0:
                    err_text = proc.stdout.decode("utf-8", errors="replace").strip()
                    if err_text:
                        first_err = err_text.splitlines()[0]
                        return CompilerReflexionResult(
                            success=False,
                            error_type="LintError",
                            error_message=first_err,
                            reflexion_prompt=f"Linter error: {first_err}. Please resolve in your replacement.",
                        )
            except (subprocess.TimeoutExpired, Exception):
                pass  # Do not block on timeout or process error

        return CompilerReflexionResult(success=True)

    @classmethod
    async def micro_reflexion_loop(
        cls,
        generate_fn: Callable[[str], Coroutine[Any, Any, str]],
        initial_code: str,
        max_retries: int = 2,
    ) -> tuple[str, bool]:
        """Loop up to max_retries feeding compiler errors back to model for micro-correction."""
        current_code = initial_code
        for _ in range(max_retries):
            result = cls.verify_syntax(current_code)
            if result.success:
                return current_code, True
            if not result.reflexion_prompt:
                break
            current_code = await generate_fn(result.reflexion_prompt)

        final_check = cls.verify_syntax(current_code)
        return current_code, final_check.success


class RollbackManager:
    """Manages pre-edit backups and counterexample negative constraints ([SLM-06])."""

    def __init__(self) -> None:
        self._backups: dict[Path, str] = {}

    def backup(self, file_path: Path | str) -> None:
        """Cache the pre-edit content of a file."""
        p = Path(file_path).resolve()
        if p.exists():
            self._backups[p] = p.read_text(encoding="utf-8")

    def rollback(self, file_path: Path | str) -> bool:
        """Restore original content from backup if an edit fails repeatedly."""
        p = Path(file_path).resolve()
        if p in self._backups:
            p.write_text(self._backups[p], encoding="utf-8")
            return True
        return False

    @staticmethod
    def generate_negative_constraint(failed_diff: str, error_message: str) -> str:
        """Inject explicit counterexample constraint to prevent repetitive failure loops."""
        return (
            "[DO NOT REPEAT: Previous attempt failed compilation]\n"
            f"Error: {error_message}\n"
            "Previous invalid patch:\n"
            f"{failed_diff}\n"
            "Ensure your new patch does not repeat this syntax or logical mistake."
        )
