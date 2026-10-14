from __future__ import annotations

from pathlib import Path

import pytest

from forge.context import (
    ContextBudgetManager,
    ContextBuilder,
    ContextSelector,
    RepositoryAnalysis,
    RepositoryAnalyzer,
    Section,
    compact_history,
    estimate_message_tokens,
    pack,
    query_terms,
    scan_text,
)
from forge.core.contracts import (
    AcceptanceCriterion,
    FindingType,
    FunctionalRequirement,
    Plan,
    QAFinding,
    RequirementContract,
    Severity,
    TestRun,
)
from forge.models import Message, ModelCapabilities, OptimizationConfig, ToolCall
from forge.tools import GitRepo, Workspace


async def analyze(root: Path) -> RepositoryAnalysis:
    return await RepositoryAnalyzer(Workspace(root)).analyze(await GitRepo.discover(root))


SMALL = ModelCapabilities(context_window=4096, max_output_tokens=1024)
LARGE = ModelCapabilities(context_window=200_000, max_output_tokens=8192)
TARGETED = OptimizationConfig()


class TestAnalyzer:
    async def test_overview(self, multi_repo: Path) -> None:
        a = await analyze(multi_repo)
        assert a.primary_language == "python"
        assert "pyproject.toml" in a.key_files
        assert "README.md" in a.key_files
        assert "app/main.py" in a.entry_points
        assert "shop -> app.main:run" in a.entry_points
        assert a.tests.framework == "pytest"
        assert set(a.tests.test_files) == {
            "tests/test_users.py",
            "tests/test_invoice.py",
            "tests/test_login.py",
        }
        assert a.git is not None
        assert a.git.branch == "main"
        assert not a.git.dirty
        assert a.git.recent_commits[0].endswith("initial import")
        assert a.manifests[0].dependencies == {"fastapi": ">=0.110"}
        assert "app/users/service.py" in a.imports.importers["app/users/models.py"]
        assert a.tests_for("app/billing/invoice.py") == ["tests/test_invoice.py"]
        snapshot = a.to_snapshot()
        assert snapshot.vcs == "git"
        assert "python" in snapshot.summary

    async def test_read_tracks_changes_and_hides_secrets(self, multi_repo: Path) -> None:
        (multi_repo / ".env").write_text("TOKEN=abc\n")
        a = await analyze(multi_repo)
        assert "slugify" in (a.read("app/utils/strings.py") or "")
        (multi_repo / "app/utils/strings.py").write_text("def kebab(): ...\n")
        assert "kebab" in (a.read("app/utils/strings.py") or "")
        assert a.read(".env") is None


class TestSelector:
    @pytest.mark.parametrize(
        ("query", "expected_top", "must_include", "must_not_lead"),
        [
            (
                "Cap the discount that can be applied to invoice totals",
                "app/billing/invoice.py",
                {"tests/test_invoice.py", "app/billing/tax.py"},
                {"app/users/models.py", "app/utils/strings.py"},
            ),
            (
                "Users should be able to reset their password by email",
                "app/users/service.py",
                {"app/users/models.py"},
                {"app/billing/invoice.py", "app/billing/tax.py"},
            ),
            (
                "Make URL slugs handle unicode text",
                "app/utils/strings.py",
                set(),
                {"app/billing/invoice.py"},
            ),
        ],
    )
    async def test_relevance(
        self,
        multi_repo: Path,
        query: str,
        expected_top: str,
        must_include: set[str],
        must_not_lead: set[str],
    ) -> None:
        selection = ContextSelector(await analyze(multi_repo)).select(query, limit=8)
        paths = [c.path for c in selection]
        assert paths[0] == expected_top, [(c.path, round(c.score, 1), c.reasons) for c in selection]
        assert must_include <= set(paths)
        assert not (set(paths[:2]) & must_not_lead)
        assert selection[0].reasons

    async def test_mentions_and_seeds(self, multi_repo: Path) -> None:
        selector = ContextSelector(await analyze(multi_repo))
        trace = 'Traceback:\n  File "app/billing/tax.py", line 5, in tax_rate\nKeyError'
        mentioned = selector.select("unrelated words", mentions=[trace])
        assert mentioned[0].path == "app/billing/tax.py"
        assert "mentioned in errors/findings" in mentioned[0].reasons
        seeded = selector.select("", seeds=["auth/login.py"])  # partial path resolves
        assert seeded[0].path == "app/auth/login.py"

    async def test_excludes_tests_when_asked(self, multi_repo: Path) -> None:
        selector = ContextSelector(await analyze(multi_repo))
        paths = [c.path for c in selector.select("invoice discount", include_tests=False)]
        assert not any(p.startswith("tests/") for p in paths)

    def test_query_terms(self) -> None:
        assert query_terms("Fix the createUser bug in user_service") == [
            "create",
            "user",
            "createuser",
            "service",
            "user_service",
        ]


class TestBudget:
    def test_budget_scales_with_window_and_strategy(self) -> None:
        manager = ContextBudgetManager()
        small = manager.budget(SMALL, TARGETED, "executor")
        large = manager.budget(LARGE, TARGETED, "executor")
        full = manager.budget(LARGE, OptimizationConfig(context_strategy="full"), "executor")
        minimal = manager.budget(LARGE, OptimizationConfig(context_strategy="minimal"), "executor")
        assert small.context_tokens < 2000
        assert large.context_tokens == 32_000  # targeted cap
        assert full.context_tokens > 90_000  # use the big window when asked
        assert minimal.context_tokens == 6_000
        planner = manager.budget(LARGE, OptimizationConfig(context_strategy="full"), "planner")
        assert planner.context_tokens > full.context_tokens  # executor keeps more headroom

    def test_pack_priorities_fallbacks_and_order(self) -> None:
        sections = [
            Section("a", "A", "a" * 700, priority=3, fallback="short a"),
            Section("must", "Must", "m" * 350, priority=0),
            Section("b", "B", "b" * 700, priority=1),
            Section("c", "C", "c" * 7000, priority=4, truncatable=False),
            Section("empty", "Empty", "   ", priority=0),
        ]
        result = pack(sections, budget_tokens=330)
        titles = [s.title for s in result.sections]
        assert titles == ["A", "Must", "B"]  # declaration order, not priority order
        modes = {s.title: s.mode for s in result.sections}
        assert modes == {"Must": "full", "B": "full", "A": "fallback"}
        assert result.omitted == ["C"]
        assert result.used_tokens <= 330

    def test_required_section_is_truncated_not_dropped(self) -> None:
        result = pack([Section("x", "X", "line\n" * 2000, priority=0)], budget_tokens=200)
        assert result.sections[0].mode == "truncated"
        assert "truncated" in result.sections[0].content


def contract() -> RequirementContract:
    return RequirementContract(
        summary="Cap invoice discounts at 50%",
        functional=[FunctionalRequirement(id="FR-001", description="Discounts above 50% clamp")],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-001",
                description="apply_discount(80) charges half",
                requirement_ids=["FR-001"],
            ),
            AcceptanceCriterion(id="AC-002", description="Existing tests still pass"),
        ],
    )


def plan() -> Plan:
    return Plan.model_validate(
        {
            "goal": "Cap discounts",
            "tasks": [
                {
                    "id": "T1",
                    "description": "Clamp the percent in Invoice.apply_discount",
                    "files_hint": ["app/billing/invoice.py"],
                    "acceptance_criteria": ["AC-001"],
                },
                {"id": "T2", "description": "Add tests for the clamp", "dependencies": ["T1"]},
            ],
        }
    )


class TestBuilder:
    async def test_planner_context_has_spec_sections(self, multi_repo: Path) -> None:
        builder = ContextBuilder(await analyze(multi_repo))
        ctx = builder.planner(
            request="Cap the discount on invoices at 50%",
            requirements=contract(),
            capabilities=LARGE,
            optimization=TARGETED,
        )
        titles = [t for t, _, _ in ctx.sections]
        for expected in [
            "Requirement",
            "Repository structure",
            "Architecture",
            "Dependencies",
            "Tests",
            "Git state",
            "Configuration",
        ]:
            assert expected in titles, titles
        assert "File: app/billing/invoice.py" in titles
        assert ctx.files[0] == "app/billing/invoice.py"
        assert "AC-001" in (ctx.section("Requirement") or "")
        assert "fastapi" in (ctx.section("Dependencies") or "")
        assert "Branch main" in (ctx.section("Git state") or "")
        assert ctx.tokens <= ctx.budget.context_tokens
        rendered = ctx.render()
        assert rendered.startswith("## Requirement")
        assert "def apply_discount" in rendered

    async def test_small_window_degrades_gracefully(self, multi_repo: Path) -> None:
        rules = "\n\n\n".join(
            f"def discount_rule_{i}(invoice):\n    return invoice.subtotal() * {i / 1000}"
            for i in range(300)
        )
        (multi_repo / "app/billing/rules.py").write_text(rules)
        builder = ContextBuilder(await analyze(multi_repo))
        request = "Cap the discount on invoices at 50%"
        large = builder.planner(request=request, capabilities=LARGE, optimization=TARGETED)
        small = builder.planner(request=request, capabilities=SMALL, optimization=TARGETED)

        assert small.tokens <= small.budget.context_tokens + 50
        assert small.tokens < large.tokens
        assert small.section("Requirement")  # required sections are always present
        large_modes = {t: m for t, _, m in large.sections}
        small_modes = {t: m for t, _, m in small.sections}
        # The big module is shown in full when the window allows it...
        assert large_modes["File: app/billing/rules.py"] == "full"
        # ...and reduced (outline/excerpts) or listed as omitted when it does not.
        assert small_modes.get("File: app/billing/rules.py") != "full"
        reduced = [t for t, m in small_modes.items() if m in {"fallback", "truncated"}]
        assert reduced or small.omitted
        if small.omitted:
            assert "Not shown (context budget)" in small.render()

    async def test_executor_context(self, multi_repo: Path) -> None:
        builder = ContextBuilder(await analyze(multi_repo))
        p = plan()
        finding = QAFinding(
            id="QA-001",
            severity=Severity.P1,
            type=FindingType.REQUIREMENT_VIOLATION,
            description="apply_discount(80) still charges 20%",
            evidence="tests/test_invoice.py::test_cap failed",
            affected_files=["app/billing/invoice.py"],
            required_rework=["clamp percent to 50"],
        )
        ctx = builder.executor(
            task=p.task("T1"),
            plan=p,
            requirements=contract(),
            findings=[finding],
            errors=["AssertionError: assert 20.0 == 50.0"],
            diff=(
                "--- a/app/billing/invoice.py\n+++ b/app/billing/invoice.py\n@@ -1 +1 @@\n-x\n+y\n"
            ),
            capabilities=LARGE,
            optimization=TARGETED,
        )
        titles = [t for t, _, _ in ctx.sections]
        assert titles[:2] == ["Current task", "Previous errors and findings"]
        assert "Acceptance criteria" in titles
        assert "Plan" in titles
        assert "Current changes (git diff)" in titles
        assert ctx.files[0] == "app/billing/invoice.py"
        assert "QA-001" in (ctx.section("Previous errors and findings") or "")
        assert "AC-001" in (ctx.section("Acceptance criteria") or "")
        assert "<- current" in (ctx.section("Plan") or "")

    async def test_qa_context_flags_security(self, multi_repo: Path) -> None:
        builder = ContextBuilder(await analyze(multi_repo))
        ctx = builder.qa(
            requirements=contract(),
            changed_files=["app/auth/login.py"],
            diff="diff --git a/app/auth/login.py b/app/auth/login.py\n",
            test_results=[
                TestRun(command="pytest", exit_code=1, passed=2, failed=1, output_tail="E boom")
            ],
            previous_findings=[],
            capabilities=LARGE,
            optimization=TARGETED,
        )
        security = ctx.section("Security-sensitive areas") or ""
        assert "security-sensitive areas: app/auth/login.py" in security
        assert "SQL built from strings" in security
        assert "weak hash" in security
        assert "FAILED" in (ctx.section("Test results") or "")
        assert any(t == "Changed file: app/auth/login.py" for t, _, _ in ctx.sections)


class TestSecurityScan:
    @pytest.mark.parametrize(
        ("line", "label"),
        [
            ("subprocess.run(cmd, shell=True)", "shell command with shell=True"),
            ("data = pickle.loads(blob)", "unsafe deserialization"),
            ("cfg = yaml.load(text)", "unsafe YAML load"),
            ('cursor.execute("SELECT * FROM t WHERE id = %s" % uid)', "SQL built from strings"),
            ("el.innerHTML = userInput", "HTML injection sink"),
            ("requests.get(url, verify=False)", "TLS verification disabled"),
            ('API_KEY = "sk-live-123456"', "hard-coded credential"),
            ("result = eval(expression)", "dynamic code execution"),
        ],
    )
    def test_rules(self, line: str, label: str) -> None:
        hits = scan_text("x.py", f"import os\n{line}\n")
        assert [h.label for h in hits] == [label]
        assert hits[0].line == 2

    def test_safe_code(self) -> None:
        safe = "yaml.load(text, Loader=yaml.SafeLoader)\ncursor.execute('SELECT 1', (uid,))\n"
        assert scan_text("x.py", safe) == []


class TestHistoryCompaction:
    def conversation(self, turns: int) -> list[Message]:
        messages = [Message.system("You are an engineer."), Message.user("Fix the bug in ops.py")]
        for i in range(turns):
            call = ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"f{i}.py"})
            messages.append(Message.assistant(f"Reading f{i}.py", [call]))
            messages.append(Message.tool_result(call, f"content {i} " + "x" * 3000))
        edit = ToolCall(id="e", name="edit_file", arguments={"path": "ops.py"})
        messages.append(Message.assistant("Editing", [edit]))
        messages.append(Message.tool_result(edit, "ok"))
        return messages

    def test_within_budget_untouched(self) -> None:
        messages = self.conversation(2)
        assert compact_history(messages, 100_000) == messages

    def test_elides_old_tool_output_first(self) -> None:
        messages = self.conversation(10)
        budget = estimate_message_tokens(messages) // 2
        compacted = compact_history(messages, budget, keep_recent=4)
        assert estimate_message_tokens(compacted) <= budget
        assert compacted[:2] == messages[:2]  # system + task preserved
        assert compacted[-4:] == messages[-4:]  # recent turns intact
        assert any("elided to save context" in m.text for m in compacted)

    def test_summarizes_when_still_too_large(self) -> None:
        messages = self.conversation(40)
        compacted = compact_history(messages, 1500, keep_recent=4)
        assert estimate_message_tokens(compacted) <= 1500
        note = compacted[2].text
        assert "earlier messages were removed" in note
        assert "read_file x" in note
        # The tail never starts with an orphaned tool result.
        assert compacted[3].role.value == "assistant"
