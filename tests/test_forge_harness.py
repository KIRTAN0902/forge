"""Comprehensive self-contained verification suite for Forge's Dual-Mode Cybernetic Harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.harness import (
    AdversarialQAGatekeeper,
    ASTContextSlicer,
    DiffScopedMutationEngine,
    EpisodicRepoMemory,
    FuzzyAnchorPatcher,
    GitWorktreeTournament,
    LiveRuntimeTracer,
    PatchAnchorNotFoundError,
    PreFlightFalsifier,
    RollbackManager,
    SpeculativeEscalationRouter,
    Sub100msCompilerReflexion,
    TournamentCandidate,
    calculate_ast_complexity,
    calculate_shannon_entropy,
    check_chestertons_fence,
    detect_secrets_in_text,
    find_golden_exemplar,
    introspect_imported_symbols,
)
from forge.skills import (
    AutonomousSkillRouter,
    ModelTier,
    OperationType,
    RepoSignals,
)

# ==============================================================================
# Phase 1: Schemas and AutonomousSkillRouter Tests
# ==============================================================================


def test_autonomous_skill_router_catalog_and_filtering() -> None:
    router = AutonomousSkillRouter()
    all_specs = router.list_all_specs()
    assert len(all_specs) >= 30

    # Test SLM vs Frontier tier matching
    slm_signals = RepoSignals(
        model_tier=ModelTier.PROSTHETIC_SLM,
        modified_files=["src/core/calc.py"],
    )
    slm_skills = router.select_skills(slm_signals, op_type=OperationType.WRITE_PURE_FUNCTION)
    slm_ids = {s.skill_id for s in slm_skills}
    assert "SLM-01" in slm_ids
    assert "SLM-02" in slm_ids
    assert "SLM-03" in slm_ids
    # Frontier-only skill should not be present
    assert "HRN-06" not in slm_ids

    # Test error regex trigger (DBG-01)
    err_signals = RepoSignals(
        model_tier=ModelTier.AMPLIFIER_FRONTIER,
        last_stderr="AssertionError: expected 42 but got 0\nFAILED tests/test_calc.py",
    )
    err_skills = router.select_skills(err_signals, op_type=OperationType.DEBUG_FAILURE)
    assert any(s.skill_id == "DBG-01" for s in err_skills)

    # Test consecutive failures trigger (DBG-03 & SLM-06)
    stuck_signals = RepoSignals(
        model_tier=ModelTier.PROSTHETIC_SLM,
        consecutive_failures=3,
    )
    stuck_skills = router.select_skills(stuck_signals, op_type=OperationType.DEBUG_FAILURE)
    stuck_ids = {s.skill_id for s in stuck_skills}
    assert "DBG-03" in stuck_ids or "SLM-06" in stuck_ids

    # Test migration trigger (DB-01)
    mig_signals = RepoSignals(
        model_tier=ModelTier.AMPLIFIER_FRONTIER,
        modified_files=["alembic/versions/001_users.py"],
        has_migration_files=True,
    )
    mig_skills = router.select_skills(mig_signals, op_type=OperationType.DB_MIGRATION)
    assert any(s.skill_id == "DB-01" for s in mig_skills)


def test_compile_operation_sliced_prompt_token_ceiling() -> None:
    router = AutonomousSkillRouter()
    slm_signals = RepoSignals(model_tier=ModelTier.PROSTHETIC_SLM)
    active_skills = router.select_skills(slm_signals, op_type=OperationType.WRITE_PURE_FUNCTION)

    prompt = router.compile_operation_sliced_prompt(
        base_persona="You are a senior python developer.",
        active_skills=active_skills,
        model_tier=ModelTier.PROSTHETIC_SLM,
    )

    # Hard SLM budget: < 400 tokens (token estimate: len // 4)
    estimated_tokens = len(prompt) // 4
    assert estimated_tokens < 400
    assert "[ROLE]" in prompt
    assert "[RULES]" in prompt
    assert "[ALLOWED TOOLS]" in prompt

    # Frontier prompt has rich detail
    frontier_prompt = router.compile_operation_sliced_prompt(
        base_persona="You are an autonomous Frontier Architect.",
        active_skills=active_skills,
        model_tier=ModelTier.AMPLIFIER_FRONTIER,
    )
    assert "Dynamically Activated Operational Skills" in frontier_prompt
    assert "Quality & Verification Mandate" in frontier_prompt


# ==============================================================================
# Phase 2: Prosthetic Engine Tests
# ==============================================================================


def test_ast_context_slicer_skeletonization(tmp_path: Path) -> None:
    code = (
        "import os\n"
        "from typing import List\n\n"
        "def helper_func(x: int) -> int:\n"
        "    '''Helper docstring.'''\n"
        "    temp = x * 2\n"
        "    return temp + 1\n\n"
        "def target_func(y: int) -> int:\n"
        "    '''Target function to keep intact.'''\n"
        "    return y * 10\n\n"
        "class Service:\n"
        "    '''Service class docstring.'''\n"
        "    def method_one(self) -> None:\n"
        "        print('doing work')\n"
    )
    test_file = tmp_path / "sample.py"
    test_file.write_text(code, encoding="utf-8")

    sliced = ASTContextSlicer.slice_file(test_file, target_symbol="target_func", max_tokens=1000)

    # Target function body preserved
    assert "return y * 10" in sliced
    # Non-target function skeletonized to Ellipsis
    assert "temp = x * 2" not in sliced
    assert "..." in sliced
    # Class method skeletonized
    assert "print('doing work')" not in sliced


def test_introspect_imported_symbols() -> None:
    signatures_block = introspect_imported_symbols(["math"])
    assert "[GROUND TRUTH CALL SIGNATURES]" in signatures_block
    assert "math.sin" in signatures_block
    assert "math.sqrt" in signatures_block


def test_find_golden_exemplar(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    sample_file = src_dir / "exemplar.py"
    sample_file.write_text(
        "def compute_tax(amount: float) -> float:\n"
        "    '''Calculate standard rate tax.'''\n"
        "    rate = 0.05\n"
        "    return amount * rate\n",
        encoding="utf-8",
    )

    exemplar = find_golden_exemplar(tmp_path, OperationType.WRITE_PURE_FUNCTION)
    assert "def compute_tax" in exemplar
    assert "return amount * rate" in exemplar


def test_fuzzy_anchor_patcher_all_tiers() -> None:
    original = (
        "def calculate_total(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item.price\n"
        "    return total\n"
    )

    # Tier 1: Exact Match
    search_exact = "    for item in items:\n        total += item.price"
    replace_exact = (
        "    for item in items:\n        if item.active:\n            total += item.price"
    )
    patched1 = FuzzyAnchorPatcher.apply_patch(original, search_exact, replace_exact)
    assert "if item.active:" in patched1

    # Tier 2: Whitespace / Indentation Mismatch
    search_indent = "for item in items:\ntotal += item.price"
    patched2 = FuzzyAnchorPatcher.apply_patch(original, search_indent, replace_exact)
    assert "if item.active:" in patched2

    # Tier 3: High Similarity difflib >= 0.92
    search_typo = "    for item in itemz:\n        total += item.price"
    patched3 = FuzzyAnchorPatcher.apply_patch(original, search_typo, replace_exact)
    assert "if item.active:" in patched3

    # Unmatched anchor raises PatchAnchorNotFoundError
    with pytest.raises(PatchAnchorNotFoundError):
        FuzzyAnchorPatcher.apply_patch(original, "completely nonexistent code block", "new code")


def test_sub100ms_compiler_reflexion_and_rollback(tmp_path: Path) -> None:
    # 1. Valid syntax
    valid_code = "def valid():\n    return 42\n"
    res_valid = Sub100msCompilerReflexion.verify_syntax(valid_code)
    assert res_valid.success is True

    # 2. Invalid syntax
    invalid_code = "def broken(\n    return 42\n"
    res_invalid = Sub100msCompilerReflexion.verify_syntax(invalid_code)
    assert res_invalid.success is False
    assert res_invalid.error_type == "SyntaxError"
    assert res_invalid.reflexion_prompt is not None
    assert "Fix this exact syntax error" in res_invalid.reflexion_prompt

    # 3. Rollback Manager
    file_path = tmp_path / "rollback_test.py"
    file_path.write_text(valid_code, encoding="utf-8")

    mgr = RollbackManager()
    mgr.backup(file_path)

    # Overwrite with bad content
    file_path.write_text(invalid_code, encoding="utf-8")
    assert "def broken(" in file_path.read_text(encoding="utf-8")

    # Perform rollback
    rolled_back = mgr.rollback(file_path)
    assert rolled_back is True
    assert file_path.read_text(encoding="utf-8") == valid_code

    # Negative constraint prompt
    constraint = mgr.generate_negative_constraint("- old\n+ bad", "SyntaxError: unmatched '('")
    assert "[DO NOT REPEAT: Previous attempt failed compilation]" in constraint


# ==============================================================================
# Phase 3: Amplifier Engine & Worktree Tournament Tests
# ==============================================================================


def test_live_runtime_tracer_execution() -> None:
    tracer = LiveRuntimeTracer(max_records=60)

    def target_algorithm(n: int) -> int:
        acc = 0
        for i in range(n):
            acc += i
        return acc

    with tracer:
        val = target_algorithm(4)

    assert val == 6
    summary = tracer.get_trace_summary()
    assert "[LIVE RUNTIME EXECUTION TRACE" in summary
    assert "target_algorithm" in summary
    assert "acc=" in summary


def test_preflight_falsifier() -> None:
    cond = PreFlightFalsifier.formulate_condition(
        assumption="Function raises ZeroDivisionError when divisor is 0",
        observable_falsifier="ZeroDivisionError: division by zero",
    )
    assert cond.verified is False

    # Negative verification
    res1 = PreFlightFalsifier.verify_condition_against_stderr(
        cond, "All 5 tests passed successfully"
    )
    assert res1.verified is False

    # Positive verification
    res2 = PreFlightFalsifier.verify_condition_against_stderr(
        cond, "Traceback: ZeroDivisionError: division by zero in divide()"
    )
    assert res2.verified is True


def test_chestertons_fence() -> None:
    # Test safe fallback on non-git or missing line range
    msg = check_chestertons_fence("nonexistent.py", (1, 10))
    assert "nonexistent.py" in msg


def test_ast_cyclomatic_complexity() -> None:
    simple_code = "def f():\n    return 1\n"
    assert calculate_ast_complexity(simple_code) == 1

    complex_code = (
        "def branchy(x, y):\n"
        "    if x > 0:\n"
        "        for i in range(y):\n"
        "            while i < 10:\n"
        "                pass\n"
        "    return 0\n"
    )
    # 1 base + 1 (if) + 1 (for) + 1 (while) = 4
    assert calculate_ast_complexity(complex_code) == 4


def test_git_worktree_fitness_computation() -> None:
    # Fitness = (pass * 1000) - (duration_ms * 0.1) - (lines_modified * 2) - (complexity_delta * 10)
    score_pass = GitWorktreeTournament.compute_fitness(
        passed=True,
        duration_ms=100.0,
        lines_modified=5,
        complexity_delta=2,
    )
    # 1000 - 10 - 10 - 20 = 960.0
    assert score_pass == 960.0

    score_fail = GitWorktreeTournament.compute_fitness(
        passed=False,
        duration_ms=500.0,
        lines_modified=50,
        complexity_delta=10,
    )
    # 0 - 50 - 100 - 100 = -250.0
    assert score_fail == -250.0


@pytest.mark.asyncio
async def test_speculative_escalation_router_fast_path() -> None:
    async def fast_success() -> bool:
        return True

    async def tournament_stub() -> TournamentCandidate:
        return TournamentCandidate(
            candidate_id="c1",
            strategy_description="frontier_strat",
            worktree_path="",
            passed=True,
            exit_code=0,
            duration_ms=100.0,
            lines_modified=10,
            complexity_delta=1,
            fitness_score=800.0,
        )

    res = await SpeculativeEscalationRouter.execute_with_speculative_fallback(
        fast_slm_fn=fast_success,
        tournament_fn=tournament_stub,
        max_fast_time_sec=1.0,
    )
    assert res["escalated"] is False
    assert res["tier_used"] == ModelTier.PROSTHETIC_SLM


@pytest.mark.asyncio
async def test_speculative_escalation_router_escalation() -> None:
    async def fast_failing() -> bool:
        raise ValueError("SLM compiler failed")

    async def tournament_stub() -> TournamentCandidate:
        return TournamentCandidate(
            candidate_id="c1",
            strategy_description="frontier_strat",
            worktree_path="",
            passed=True,
            exit_code=0,
            duration_ms=100.0,
            lines_modified=10,
            complexity_delta=1,
            fitness_score=800.0,
        )

    res = await SpeculativeEscalationRouter.execute_with_speculative_fallback(
        fast_slm_fn=fast_failing,
        tournament_fn=tournament_stub,
        max_fast_time_sec=1.0,
    )
    assert res["escalated"] is True
    assert res["tier_used"] == ModelTier.AMPLIFIER_FRONTIER


# ==============================================================================
# Phase 4: Adversarial QA & Episodic Memory Tests
# ==============================================================================


def test_adversarial_qa_cot_stripping() -> None:
    raw_response = (
        "<think>\n"
        "We need to rewrite this function to use a hashmap.\n"
        "Let me verify if dict satisfies the requirements.\n"
        "</think>\n"
        "def solve():\n"
        "    return {}\n"
    )
    cleaned = AdversarialQAGatekeeper.strip_chain_of_thought(raw_response)
    assert "<think>" not in cleaned
    assert "hashmap" not in cleaned
    assert "def solve():" in cleaned


def test_shannon_entropy_and_secret_scrubber() -> None:
    # Low entropy plain string
    assert calculate_shannon_entropy("aaaaaaaa") == 0.0

    # High entropy random base64 secret
    secret_token = 'api_key = "sk-proj-948f938f92j8f92j8f928j3f982j3f98j2f398j2f398jf9238fj"'
    violations = detect_secrets_in_text(secret_token)
    assert len(violations) >= 1
    assert any("secret" in v.lower() or "sk-" in v for v in violations)


def test_ast_security_and_async_auditor() -> None:
    # 1. Insecure shell=True and eval()
    bad_sec_code = (
        "import subprocess\n"
        "def run_it(cmd):\n"
        "    eval('1 + 1')\n"
        "    subprocess.run(cmd, shell=True)\n"
    )
    sec_violations = AdversarialQAGatekeeper.run_pre_commit_audit(
        diff_text="", source_code=bad_sec_code
    )
    assert any("eval" in v for v in sec_violations)
    assert any("shell=True" in v for v in sec_violations)

    # 2. Blocking time.sleep inside async def
    bad_async_code = "import time\nasync def fetch_user():\n    time.sleep(1)\n"
    async_violations = AdversarialQAGatekeeper.run_pre_commit_audit(
        diff_text="", source_code=bad_async_code
    )
    assert any("time.sleep" in v for v in async_violations)

    # 3. Mock patch missing autospec=True
    bad_mock_code = (
        "from unittest.mock import patch\n"
        "def test_it():\n"
        "    with patch('os.path.exists'):\n"
        "        pass\n"
    )
    mock_violations = AdversarialQAGatekeeper.run_pre_commit_audit(
        diff_text="", source_code=bad_mock_code
    )
    assert any("autospec" in v for v in mock_violations)


@pytest.mark.asyncio
async def test_diff_scoped_mutation_engine() -> None:
    code = (
        "def is_positive(x: int) -> bool:\n    if x > 0:\n        return True\n    return False\n"
    )

    # Generate mutants
    mutants = DiffScopedMutationEngine.generate_mutants(code)
    assert len(mutants) >= 2

    # Simulated test runner:
    # Evaluates code in memory and checks if is_positive(5) == True and is_positive(-1) == False
    async def rigorous_test_runner(mutant_code: str) -> bool:
        local_scope: dict[str, object] = {}
        try:
            exec(mutant_code, {}, local_scope)
            fn = local_scope["is_positive"]
            assert callable(fn)
            return fn(1) is True and fn(0) is False and fn(-1) is False
        except Exception:
            return False

    report = await DiffScopedMutationEngine.evaluate_mutation_coverage(code, rigorous_test_runner)
    assert report.total_mutants > 0
    # Rigorous test suite kills all mutants
    assert report.killed_mutants == report.total_mutants
    assert report.passed is True


def test_episodic_repo_memory(tmp_path: Path) -> None:
    mem = EpisodicRepoMemory(tmp_path)

    # Record heuristics
    mem.record_heuristic(
        key="ASYNC_EVENT_LOOP",
        value="Always use httpx.AsyncClient instead of requests in async routes",
        category="performance",
        associated_paths=["src/routes/api.py"],
    )

    # Retrieve matching heuristic
    matched = mem.retrieve_relevant_heuristics(["src/routes/api.py"])
    assert len(matched) == 1
    assert "ASYNC_EVENT_LOOP" in matched[0]

    # Non-matching path
    unmatched = mem.retrieve_relevant_heuristics(["other/path/calc.py"])
    assert len(unmatched) == 0

    # Record and load SFT traces
    mem.record_sft_trace(
        timestamp="2026-10-16T10:00:00",
        prompt="Fix N+1 query in user serializer",
        initial_code="for u in users: ...",
        sliced_context="models.py",
        patch_diff="+ selectinload(User.orders)",
        verification_evidence="Test query count <= 2 verified",
    )

    traces = mem.load_all_traces()
    assert len(traces) == 1
    assert traces[0].prompt == "Fix N+1 query in user serializer"
    assert "selectinload" in traces[0].patch_diff
