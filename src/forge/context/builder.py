"""Role-specific context construction (spec §13).

Each builder method gathers the material the spec assigns to that role, turns it into
prioritised sections, and packs them into the model's budget:

* Planner: requirement, repository structure, architecture, relevant source,
  dependencies, tests, configuration, git state.
* Executor: current task, relevant files, plan, acceptance criteria, previous errors,
  current git diff.
* QA: requirements, acceptance criteria, changed files, git diff, test results,
  security-sensitive areas, previous findings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from forge.context.analyzer import RepositoryAnalysis
from forge.context.budget import ContextBudget, ContextBudgetManager, Section, pack
from forge.context.render import (
    render_criterion_line,
    render_file,
    render_findings,
    render_outline,
    render_plan,
    render_requirements,
    render_snippets,
    render_task,
    render_test_runs,
)
from forge.context.security import is_sensitive_path, scan_files
from forge.context.selector import ContextSelector, FileCandidate, query_terms
from forge.core.contracts import Plan, QAFinding, RequirementContract, Task, TestRun
from forge.models.capabilities import ModelCapabilities, OptimizationConfig
from forge.models.types import Message
from forge.tools.analysis import is_manifest

CONFIG_FILE_HINTS = (
    "pyproject.toml",
    "setup.cfg",
    "package.json",
    "tsconfig.json",
    "go.mod",
    "Cargo.toml",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
    ".env.example",
    "pytest.ini",
    "tox.ini",
)
MAX_CONFIG_LINES = 60


@dataclass
class BuiltContext:
    role: str
    sections: list[tuple[str, str, str]]  # (title, content, mode)
    files: list[str]
    omitted: list[str]
    tokens: int
    budget: ContextBudget
    selection: list[FileCandidate] = field(default_factory=list)

    def render(self) -> str:
        parts = [f"## {title}\n\n{content.strip()}" for title, content, _ in self.sections]
        if self.omitted:
            parts.append(
                "## Not shown (context budget)\n\n"
                + ", ".join(self.omitted)
                + ". Use the tools to read anything you need."
            )
        return "\n\n".join(parts)

    def to_message(self, preamble: str = "") -> Message:
        body = self.render()
        return Message.user(f"{preamble.strip()}\n\n{body}" if preamble.strip() else body)

    def section(self, title: str) -> str | None:
        return next((content for t, content, _ in self.sections if t == title), None)


class ContextBuilder:
    def __init__(
        self,
        analysis: RepositoryAnalysis,
        *,
        budgets: ContextBudgetManager | None = None,
    ) -> None:
        self.analysis = analysis
        self.selector = ContextSelector(analysis)
        self.budgets = budgets or ContextBudgetManager()

    # -- planner -------------------------------------------------------------------------

    def planner(
        self,
        *,
        request: str,
        capabilities: ModelCapabilities,
        optimization: OptimizationConfig,
        requirements: RequirementContract | None = None,
        seeds: Sequence[str] = (),
    ) -> BuiltContext:
        a = self.analysis
        budget = self.budgets.budget(capabilities, optimization, "planner")
        query = request + (" " + requirements.summary if requirements else "")
        selection = self.selector.select(query, seeds=seeds, limit=_file_limit(budget))
        requirement_text = f"User request:\n{request.strip()}"
        if requirements is not None:
            requirement_text += "\n\nRequirement contract:\n" + render_requirements(requirements)

        sections = [
            Section("requirement", "Requirement", requirement_text, priority=0),
            Section(
                "structure",
                "Repository structure",
                a.tree(depth=3),
                priority=1,
                fallback=a.tree(depth=2, max_entries=80),
            ),
            Section(
                "architecture", "Architecture", _architecture(a), priority=1, fallback=a.summary()
            ),
            Section("dependencies", "Dependencies", _dependencies(a), priority=2),
            Section("tests", "Tests", _tests_overview(a), priority=2),
            Section("git", "Git state", _git_state(a), priority=2),
            *self._file_sections(selection, query, base_priority=3),
            Section(
                "config",
                "Configuration",
                self._config_files(),
                priority=5,
                fallback=", ".join(a.key_files),
            ),
        ]
        return self._finish("planner", sections, budget, selection)

    # -- executor ------------------------------------------------------------------------

    def executor(
        self,
        *,
        task: Task,
        capabilities: ModelCapabilities,
        optimization: OptimizationConfig,
        plan: Plan | None = None,
        requirements: RequirementContract | None = None,
        errors: Sequence[str] = (),
        findings: Sequence[QAFinding] = (),
        diff: str | None = None,
        diff_stat: str | None = None,
        changed_files: Sequence[str] = (),
    ) -> BuiltContext:
        budget = self.budgets.budget(capabilities, optimization, "executor")
        mentions = [*errors, *(f.evidence + " " + " ".join(f.affected_files) for f in findings)]
        seeds = [*task.files_hint, *(p for f in findings for p in f.affected_files)]
        selection = self.selector.select(
            task.description + " " + " ".join(f.description for f in findings),
            seeds=seeds,
            mentions=mentions,
            changed=changed_files,
            limit=_file_limit(budget),
        )
        problems = []
        if findings:
            problems.append("QA findings to address:\n" + render_findings(findings))
        if errors:
            problems.append("Errors from previous attempts:\n" + "\n---\n".join(errors[-5:]))

        sections = [
            Section("task", "Current task", render_task(task, requirements, plan), priority=0),
            Section("problems", "Previous errors and findings", "\n\n".join(problems), priority=0),
            Section("criteria", "Acceptance criteria", _criteria(requirements, task), priority=1),
            Section("plan", "Plan", render_plan(plan, current=task.id) if plan else "", priority=2),
            Section(
                "diff", "Current changes (git diff)", diff or "", priority=2, fallback=diff_stat
            ),
            *self._file_sections(selection, task.description, base_priority=3),
            Section(
                "map",
                "Repository map",
                self.analysis.tree(depth=2, max_entries=120),
                priority=6,
                truncatable=False,
            ),
        ]
        return self._finish("executor", sections, budget, selection)

    # -- QA ------------------------------------------------------------------------------

    def qa(
        self,
        *,
        capabilities: ModelCapabilities,
        optimization: OptimizationConfig,
        requirements: RequirementContract | None,
        changed_files: Sequence[str],
        diff: str | None = None,
        diff_stat: str | None = None,
        test_results: Sequence[TestRun] = (),
        previous_findings: Sequence[QAFinding] = (),
        plan: Plan | None = None,
        focus: str = "",
    ) -> BuiltContext:
        a = self.analysis
        budget = self.budgets.budget(capabilities, optimization, "qa")
        query = " ".join(
            [
                focus,
                requirements.summary if requirements else "",
                *(
                    c.description
                    for c in (requirements.acceptance_criteria if requirements else [])
                ),
            ]
        )
        selection = self.selector.select(
            query,
            seeds=changed_files,
            mentions=[f.evidence for f in previous_findings],
            changed=changed_files,
            limit=_file_limit(budget),
        )
        hits = scan_files(changed_files, a.read)
        sensitive = [p for p in changed_files if is_sensitive_path(p)]
        security_lines = []
        if sensitive:
            security_lines.append(
                "Changed files in security-sensitive areas: " + ", ".join(sensitive)
            )
        if hits:
            security_lines.append("Patterns that need scrutiny in changed files:")
            security_lines += [f"- {h.render()}" for h in hits]
        if not security_lines:
            security_lines.append(
                "No obviously security-sensitive changes detected; still check input "
                "validation, authorization and error handling on changed code paths."
            )
        requirement_text = (
            render_requirements(requirements)
            if requirements
            else "No requirement contract was recorded; infer intent from the plan and diff."
        )

        sections = [
            Section(
                "requirements", "Requirements and acceptance criteria", requirement_text, priority=0
            ),
            Section(
                "diff", "Git diff", diff or "(no diff available)", priority=0, fallback=diff_stat
            ),
            Section(
                "tests",
                "Test results",
                render_test_runs(test_results) or "No tests run yet.",
                priority=1,
            ),
            Section("security", "Security-sensitive areas", "\n".join(security_lines), priority=1),
            Section(
                "findings", "Previous findings", render_findings(previous_findings), priority=1
            ),
            Section("plan", "Plan", render_plan(plan) if plan else "", priority=3),
            *self._file_sections(selection, query, base_priority=2, changed=set(changed_files)),
        ]
        return self._finish("qa", sections, budget, selection)

    # -- helpers -------------------------------------------------------------------------

    def _file_sections(
        self,
        selection: Sequence[FileCandidate],
        query: str,
        *,
        base_priority: int,
        changed: set[str] | None = None,
    ) -> list[Section]:
        terms = query_terms(query)
        sections = []
        for rank, candidate in enumerate(selection):
            text = self.analysis.read(candidate.path)
            if text is None:
                continue
            outline = render_outline(
                candidate.path, text, self.analysis.symbols.symbols(candidate.path)
            )
            snippets = render_snippets(candidate.path, text, terms)
            fallback = outline + ("\n" + snippets if snippets else "")
            label = "Changed file" if changed and candidate.path in changed else "File"
            why = "; ".join(candidate.reasons[:3])
            sections.append(
                Section(
                    key=f"file:{candidate.path}",
                    title=f"{label}: {candidate.path}",
                    content=f"(selected because: {why})\n{render_file(candidate.path, text)}",
                    priority=base_priority + rank // 3,
                    fallback=f"(selected because: {why})\n{fallback}",
                    files=(candidate.path,),
                )
            )
        return sections

    def _config_files(self) -> str:
        blocks = []
        for path in self.analysis.key_files:
            name = path.rsplit("/", 1)[-1]
            if name not in CONFIG_FILE_HINTS and not is_manifest(path):
                continue
            text = self.analysis.read(path)
            if not text:
                continue
            lines = text.splitlines()
            shown = "\n".join(lines[:MAX_CONFIG_LINES])
            more = (
                f"\n... ({len(lines) - MAX_CONFIG_LINES} more lines)"
                if len(lines) > MAX_CONFIG_LINES
                else ""
            )
            blocks.append(f"--- {path}\n{shown}{more}")
        return "\n\n".join(blocks)

    def _finish(
        self,
        role: str,
        sections: list[Section],
        budget: ContextBudget,
        selection: list[FileCandidate],
    ) -> BuiltContext:
        packed = pack(sections, budget.context_tokens)
        files = [f for s in packed.sections for f in s.files]
        return BuiltContext(
            role=role,
            sections=[(s.title, s.content, s.mode) for s in packed.sections],
            files=files,
            omitted=packed.omitted,
            tokens=packed.used_tokens,
            budget=budget,
            selection=selection,
        )


def _file_limit(budget: ContextBudget) -> int:
    # Roughly one candidate file per 1.5K tokens of budget, within sensible bounds.
    return max(4, min(25, budget.context_tokens // 1500))


def _architecture(a: RepositoryAnalysis) -> str:
    text = a.summary()
    if a.readme:
        text += "\n\nREADME (excerpt):\n" + a.readme.strip()
    return text


def _dependencies(a: RepositoryAnalysis) -> str:
    if not a.manifests:
        return "No dependency manifests found."
    lines = [m.render() for m in a.manifests]
    usage = sorted(a.imports.external_usage().items(), key=lambda kv: -kv[1])[:15]
    if usage:
        lines.append("Most-imported external modules: " + ", ".join(f"{k} ({v})" for k, v in usage))
    return "\n".join(lines)


def _tests_overview(a: RepositoryAnalysis) -> str:
    tests = a.tests
    if not tests.test_files and not tests.command:
        return "No tests found."
    lines = [
        f"Framework: {tests.framework or 'unknown'}; command: {tests.command or 'not detected'}",
        f"{len(tests.test_files)} test files in: {', '.join(tests.test_dirs) or '(various)'}",
    ]
    lines += [f"- {t}" for t in tests.test_files[:25]]
    if len(tests.test_files) > 25:
        lines.append(f"- ... {len(tests.test_files) - 25} more")
    return "\n".join(lines)


def _git_state(a: RepositoryAnalysis) -> str:
    if a.git is None:
        return "Not a git repository."
    g = a.git
    lines = [
        f"Branch {g.branch or '(detached)'} at {(g.head or 'no commits')[:10]}; "
        f"{'uncommitted changes present' if g.dirty else 'clean working tree'}"
    ]
    if g.changed_files:
        lines.append("Uncommitted: " + ", ".join(g.changed_files[:30]))
    if g.recent_commits:
        lines.append("Recent commits:")
        lines += [f"  {c}" for c in g.recent_commits]
    return "\n".join(lines)


def _criteria(requirements: RequirementContract | None, task: Task) -> str:
    if requirements is None:
        return ""
    ids = task.acceptance_criteria or requirements.criterion_ids
    lines = []
    for criterion_id in ids:
        try:
            lines.append(render_criterion_line(requirements, criterion_id))
        except KeyError:
            continue
    others = [c for c in requirements.criterion_ids if c not in ids]
    if others:
        lines.append(f"(other criteria in the contract: {', '.join(others)})")
    return "\n".join(lines)
