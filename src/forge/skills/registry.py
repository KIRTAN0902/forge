"""Central registry defining Forge's specialist skills and AutonomousSkillRouter."""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from forge.skills.models import ModelTier, OperationType, RepoSignals, SkillSpec
from forge.skills.types import Skill, SkillCategory

ALL_SKILLS: tuple[Skill, ...] = (
    # --- DEBUGGING (DBG) ---
    Skill(
        id="DBG-01",
        name="Stratified Fault Localization & Repro-First",
        category=SkillCategory.DEBUGGING,
        target_agents=("executor", "qa"),
        primary_trigger="Bug report, stack trace, or non-zero test exit",
        protocol=(
            "1. Before modifying ANY production or application code, write an isolated reproduction test that executes and FAILS with exit code != 0.\n"
            "2. Localize root cause through minimal stack frame inspection rather than speculative edits.\n"
            "3. Apply targeted fix and verify that the reproduction test now passes.\n"
            "4. Promote the reproduction test into the permanent regression suite."
        ),
        failure_mode_prevented="Speculative fixes without proving the bug existed; masking symptoms while introducing regressions.",
    ),
    Skill(
        id="DBG-02",
        name="Automated Git Bisect & Delta Forensics",
        category=SkillCategory.DEBUGGING,
        target_agents=("executor",),
        primary_trigger="Regression in previously passing test or 'used to work' report",
        protocol=(
            "1. Run automated git bisect (git bisect start, git bisect bad, git bisect good <commit>).\n"
            "2. Execute the failing test command automatically at each bisect step.\n"
            "3. Identify the exact commit SHA and surgical diff that introduced the regression.\n"
            "4. Formulate the fix specifically addressing the delta introduced in that commit."
        ),
        failure_mode_prevented="Hours of blind searching in large codebases when git history contains the exact regression point.",
    ),
    Skill(
        id="DBG-03",
        name="Loop-Break & Backtracking State Machine",
        category=SkillCategory.DEBUGGING,
        target_agents=("orchestrator",),
        primary_trigger="3 consecutive failures on the same DAG node or oscillating diffs",
        protocol=(
            "1. Detect repetition or oscillation in failure signatures across 3 steps.\n"
            "2. Freeze linear retries and execute a rollback to the last verified clean checkpoint.\n"
            "3. Replan the task decomposition: split the problematic step into smaller atomic sub-tasks or pivot architecture.\n"
            "4. If unresolved after replan, pause and request human guidance with diagnostic summary."
        ),
        failure_mode_prevented="Infinite retry loops, context exhaustion, and destructive churn when an approach is fundamentally flawed.",
    ),
    # --- REFACTORING (REF) ---
    Skill(
        id="REF-01",
        name="Mikado Graph & Strangler-Fig Engine",
        category=SkillCategory.REFACTORING,
        target_agents=("orchestrator", "executor"),
        primary_trigger="Signature change affecting greater than 5 AST call sites",
        protocol=(
            "1. Construct a Mikado dependency graph of all AST call sites before modifying interfaces.\n"
            "2. Find the leaf dependencies with zero prerequisites.\n"
            "3. Implement changes leaf-first using the Strangler-Fig pattern (add new signature alongside old).\n"
            "4. Migrate callers one-by-one with passing test checkpoints at every step.\n"
            "5. Deprecate and prune the legacy interface only after 100% call-site migration."
        ),
        failure_mode_prevented="Shotgun refactorings that break dozens of files at once, leaving the codebase unbuildable for hours.",
    ),
    Skill(
        id="REF-02",
        name="Tarjan SCC Circular Dependency Untangler",
        category=SkillCategory.REFACTORING,
        target_agents=("executor",),
        primary_trigger="ImportError cycle or AST import graph cycle (>1 Strongly Connected Component)",
        protocol=(
            "1. Run Tarjan's Strongly Connected Components algorithm on the module AST import graph.\n"
            "2. Isolate the cycle vertices and identify the shared domain types/contracts causing mutual dependence.\n"
            "3. Extract shared interfaces or value objects into a new leaf module (e.g., types.py or protocols.py).\n"
            "4. Rewire imports to flow strictly hierarchically toward the new leaf.\n"
            "5. Verify cycle elimination via import graph validation."
        ),
        failure_mode_prevented="Brittle circular import workarounds like local inline imports, runtime monkey patching, and deferred init.",
    ),
    Skill(
        id="REF-03",
        name="Reachability-Verified Dead Code Pruner",
        category=SkillCategory.REFACTORING,
        target_agents=("qa", "executor"),
        primary_trigger="Post-refactor / file deletion or zero-in-degree AST symbols",
        protocol=(
            "1. Perform whole-program AST call-graph reachability analysis from public API entrypoints and CLI commands.\n"
            "2. Flag symbols with zero in-degree that are not exported in __all__ or decorated with dynamic registration.\n"
            "3. Verify non-reachability via grep/reflection check across configs and templates.\n"
            "4. Prune dead symbols and execute the complete test suite to prove zero regression."
        ),
        failure_mode_prevented="Accumulation of phantom functions, orphaned models, and dead imports that mislead future development.",
    ),
    # --- DATABASE (DB) ---
    Skill(
        id="DB-01",
        name="Three-Phase Expand-Contract Migration Guard",
        category=SkillCategory.DATABASE,
        target_agents=("orchestrator", "qa"),
        primary_trigger="Edits to migrations/, alembic/, or ORM models",
        protocol=(
            "1. Enforce three distinct phases: Phase 1 (Expand - add nullable column / new table), Phase 2 (Migrate data with dual-writing), Phase 3 (Contract - drop old column/table in subsequent deploy).\n"
            "2. Never combine destructive column renames or drops in the same migration as column additions.\n"
            "3. Provide deterministic downgrade/rollback steps for every migration.\n"
            "4. Validate backward compatibility against the previous schema version."
        ),
        failure_mode_prevented="Database locks, failed deployments, and zero-downtime migration outages caused by destructive schema changes.",
    ),
    Skill(
        id="DB-02",
        name="ORM Query-Counter & N+1 Enforcer",
        category=SkillCategory.DATABASE,
        target_agents=("qa", "executor"),
        primary_trigger="ORM collection iterated in loop or serializer",
        protocol=(
            "1. Attach query counter to the test database session for endpoint and service tests.\n"
            "2. Assert query count <= O(1) constant bound regardless of collection size.\n"
            "3. If query count scales with collection size (N+1 detected), inject selectinload / joinedload / prefetch_related.\n"
            "4. Re-run query counter test to prove single-query or bounded-query execution."
        ),
        failure_mode_prevented="Production database CPU exhaustion and severe latency degradation under realistic dataset volumes.",
    ),
    Skill(
        id="DB-03",
        name="Synthetic Cardinality & EXPLAIN Auditor",
        category=SkillCategory.DATABASE,
        target_agents=("qa",),
        primary_trigger="New SQL joins, complex filters, or sorting predicates",
        protocol=(
            "1. Generate synthetic dataset with realistic cardinality distribution (10k+ rows).\n"
            "2. Execute EXPLAIN (ANALYZE, BUFFERS) or dialect equivalent on the generated queries.\n"
            "3. Detect Seq Scan / table scan on large relations and verify Index Scan usage.\n"
            "4. Add composite or partial indexes where query cost exceeds acceptable threshold."
        ),
        failure_mode_prevented="Queries that run in milliseconds in dev with 10 rows but trigger full table scans and timeouts in production.",
    ),
    # --- API & CONTRACT (API) ---
    Skill(
        id="API-01",
        name="Bi-Directional Schema & Contract Guard",
        category=SkillCategory.API,
        target_agents=("qa",),
        primary_trigger="Modifications to route controllers, Pydantic, or OpenAPI models",
        protocol=(
            "1. Diff the OpenAPI/JSON schema before and after changes.\n"
            "2. Detect breaking changes: removed fields, tightened validations, changed types, or new required fields.\n"
            "3. Apply Postel's Law: Be liberal in what you accept (optional defaults), strict in what you emit.\n"
            "4. Generate backward compatibility contract tests asserting previous client payloads remain accepted."
        ),
        failure_mode_prevented="Breaking downstream clients, mobile apps, or frontend integrations during routine backend API updates.",
    ),
    Skill(
        id="API-02",
        name="Idempotency & Retry Safety Enforcer",
        category=SkillCategory.API,
        target_agents=("executor", "qa"),
        primary_trigger="POST/PATCH routes, webhooks, queue tasks, external HTTP calls",
        protocol=(
            "1. Verify state-mutating endpoints accept an Idempotency-Key header or unique payload token.\n"
            "2. Implement atomic check-and-set or distributed lock with deduplication table.\n"
            "3. Verify network retry simulation: sending the exact same payload twice yields identical response without duplicate side-effects.\n"
            "4. Ensure external outgoing HTTP calls are wrapped with exponential backoff and jitter."
        ),
        failure_mode_prevented="Duplicate charges, double-booked appointments, duplicated emails, and phantom database writes on network retries.",
    ),
    Skill(
        id="API-03",
        name="Boundary Fuzzing & OpenAPI Property Testing",
        category=SkillCategory.API,
        target_agents=("qa",),
        primary_trigger="Any newly implemented or modified HTTP/RPC endpoint",
        protocol=(
            "1. Generate property-based fuzz tests across input boundary conditions (Hypothesis / Schemathesis).\n"
            "2. Test boundary values: empty strings, MAX_INT, unicode emojis, malformed JSON, SQL injection strings, oversized payloads.\n"
            "3. Assert endpoint never returns 500 Internal Server Error (must return 400/422 with structured error details).\n"
            "4. Verify rate-limiting headers and payload size limits are enforced."
        ),
        failure_mode_prevented="Uncaught exceptions, unhandled null pointers, and server 500 crashes exposed to end users.",
    ),
    # --- SECURITY (SEC) ---
    Skill(
        id="SEC-01",
        name="AST Taint-Flow Source-to-Sink Analyzer",
        category=SkillCategory.SECURITY,
        target_agents=("qa",),
        primary_trigger="Diffs with SQL queries, subprocess, file path manipulation, or URL fetching",
        protocol=(
            "1. Trace user input sources (request params, CLI args, headers) across AST flow to dangerous sinks.\n"
            "2. Check sinks: cursor.execute(f'...'), subprocess.run(..., shell=True), open(user_path), requests.get(user_url).\n"
            "3. Flag any sink reached without intermediate sanitizer / parameterized binding / safe path resolution.\n"
            "4. Block execution until parameterized queries or safe abstractions (e.g. shlex.quote, Path.resolve().is_relative_to) are applied."
        ),
        failure_mode_prevented="Critical vulnerabilities: SQL Injection, Command Injection, Path Traversal, and SSRF.",
    ),
    Skill(
        id="SEC-02",
        name="Shannon-Entropy Secret Scrubber",
        category=SkillCategory.SECURITY,
        target_agents=("qa",),
        primary_trigger="Pre-commit hook on every git commit, file write, or model generation",
        protocol=(
            "1. Calculate Shannon entropy H(X) = -sum(p * log2(p)) over string literals in code diffs.\n"
            "2. Match known secret regex patterns (AWS keys, GitHub PATs, JWT tokens, private keys, API keys sk-*, etc.).\n"
            "3. If high entropy (>4.5 for Base64, >3.0 for hex) or pattern match detected, redact value immediately.\n"
            "4. Enforce environment variable injection via .env or secrets manager."
        ),
        failure_mode_prevented="Accidental leakage of API keys, database credentials, and cloud tokens to git history and LLM context.",
    ),
    Skill(
        id="SEC-03",
        name="Anti-Slopsquatting & Supply-Chain Auditor",
        category=SkillCategory.SECURITY,
        target_agents=("orchestrator", "qa"),
        primary_trigger="pip/uv/npm/cargo install commands or dependency manifest edits",
        protocol=(
            "1. Intercept any dependency addition before executing package manager commands.\n"
            "2. Query package registry metadata (PyPI/npm) to verify package age, download count, and repository provenance.\n"
            "3. Reject packages published < 30 days ago with < 1,000 downloads or names typo-squatting popular packages.\n"
            "4. Verify package hash matches lockfile (hash-checking mode)."
        ),
        failure_mode_prevented="Installing malicious hallucinated packages or typosquatted malware created by adversaries to compromise developer machines.",
    ),
    # --- PERFORMANCE (PERF) ---
    Skill(
        id="PERF-01",
        name="Asyncio Event-Loop Blocking Call Detector",
        category=SkillCategory.PERFORMANCE,
        target_agents=("qa",),
        primary_trigger="Edits inside async def coroutines or async callees",
        protocol=(
            "1. Inspect AST of all async def functions and their synchronous callees.\n"
            "2. Detect blocking calls inside async context: time.sleep, requests.get/post, synchronous DB calls, heavy CPU loops.\n"
            "3. Enforce async equivalents (asyncio.sleep, httpx.AsyncClient) or offload via asyncio.to_thread.\n"
            "4. Run event-loop slow callback detector (loop.slow_callback_duration = 0.05) during test execution."
        ),
        failure_mode_prevented="Freezing the entire asyncio event loop, causing severe latency spikes and dropped concurrent connections.",
    ),
    Skill(
        id="PERF-02",
        name="Empirical Complexity & Allocation Ratchet",
        category=SkillCategory.PERFORMANCE,
        target_agents=("qa", "executor"),
        primary_trigger="Nested loops, 'in list' checks inside loops, or batch data pipelines",
        protocol=(
            "1. Analyze algorithmic complexity in hot loops: nested loops over collections, linear search inside iteration (O(N^2)).\n"
            "2. Benchmark execution time across input sizes N = [100, 1000, 10000].\n"
            "3. Replace linear searches with O(1) set/dict lookups and use generators/iterators to bound memory allocation.\n"
            "4. Assert asymptotic scaling does not exceed O(N log N)."
        ),
        failure_mode_prevented="Accidental O(N^2) quadratic algorithms that work in unit tests but cause severe timeouts under production load.",
    ),
    Skill(
        id="PERF-03",
        name="Concurrency Race & Barrier Stress Tester",
        category=SkillCategory.PERFORMANCE,
        target_agents=("qa",),
        primary_trigger="Shared mutable state, locks, thread/task pools, quotas, or balance updates",
        protocol=(
            "1. Identify shared mutable data structures accessed across concurrent threads or coroutines.\n"
            "2. Formulate high-concurrency race test: launch 50-100 parallel workers contending for the same resource simultaneously.\n"
            "3. Check for lost updates, check-then-act race conditions, and deadlocks.\n"
            "4. Enforce atomic CAS, locks, or transactional isolation."
        ),
        failure_mode_prevented="Heisenbugs, race conditions, negative balances, and data corruption occurring only under concurrent production traffic.",
    ),
    # --- TESTING & INTEGRITY (TST) ---
    Skill(
        id="TST-01",
        name="Diff-Scoped Mutation Testing Gate",
        category=SkillCategory.TESTING,
        target_agents=("qa",),
        primary_trigger="Post-implementation QA pass on new code and unit tests",
        protocol=(
            "1. Parse the AST of newly modified lines and generate targeted mutants: invert condition (== to !=), replace True with False, delete statement.\n"
            "2. Execute the corresponding unit tests against each AST mutant.\n"
            "3. If any mutant SURVIVES (tests still pass despite the broken logic), reject the test suite as weak.\n"
            "4. Require additional test assertions specifically targeting the surviving mutant's branch."
        ),
        failure_mode_prevented="Tautological tests and false-confidence test suites that achieve 100% line coverage without actually asserting behavior.",
    ),
    Skill(
        id="TST-02",
        name="Metamorphic & Round-Trip Property Testing",
        category=SkillCategory.TESTING,
        target_agents=("qa", "executor"),
        primary_trigger="Parsers, serializers, encoders, compilers, or mathematical transformers",
        protocol=(
            "1. Define invariant metamorphic properties: decode(encode(x)) == x (round-trip idempotency).\n"
            "2. For transforms: transform(a + b) == transform(a) + transform(b) or invariant properties under permutation.\n"
            "3. Generate 1,000 randomized inputs using Hypothesis to exercise edge-case byte sequences, empty inputs, and special characters.\n"
            "4. Assert identity holds universally across all generated inputs."
        ),
        failure_mode_prevented="Lossy serialization, floating-point drift, character encoding corruption, and silent data mutation.",
    ),
    Skill(
        id="TST-03",
        name="Mock-Integrity & Anti-Mock-Tautology Guard",
        category=SkillCategory.TESTING,
        target_agents=("qa",),
        primary_trigger="Test diffs containing mock, patch, or assertion modifications",
        protocol=(
            "1. Inspect test AST to detect mocking antipatterns: mocking the system-under-test itself, mocking standard library without reason.\n"
            "2. Enforce autospec=True or create_autospec on all unittest.mock.patch calls to catch signature drift.\n"
            "3. Detect mock tautologies: tests where mocks return values that are asserted without invoking real production code.\n"
            "4. Mandate integration or contract tests over pure mock tests for all boundary interactions."
        ),
        failure_mode_prevented="Tests that pass completely green because mocks mirror the developer's assumptions while the real code is broken.",
    ),
    # --- DOCUMENTATION (DOC) ---
    Skill(
        id="DOC-01",
        name="Executable Docstring & README Drift Verifier",
        category=SkillCategory.DOCUMENTATION,
        target_agents=("qa", "executor"),
        primary_trigger="Public API signature, CLI flag, or configuration changes",
        protocol=(
            "1. Extract code blocks from README.md, docs/, and docstring examples using doctest / markdown parser.\n"
            "2. Execute all extracted examples against the live modified codebase.\n"
            "3. Fail verification if any documented example raises an exception, uses deprecated syntax, or outputs stale results.\n"
            "4. Automatically patch outdated documentation to match current API signatures."
        ),
        failure_mode_prevented="Stale documentation, broken copy-paste examples in READMEs, and developer onboarding frustration.",
    ),
    Skill(
        id="DOC-02",
        name="Automated ADR & Constraints Ledger",
        category=SkillCategory.DOCUMENTATION,
        target_agents=("orchestrator",),
        primary_trigger="Core architecture changes, dependency additions, or DBG-03 loop pivots",
        protocol=(
            "1. Detect significant architectural inflection: new dependency, database migration, or recovery from 3-strike failure.\n"
            "2. Generate an Architectural Decision Record (ADR) under docs/adr/YYYY-MM-DD-title.md.\n"
            "3. Record Context, Decision, Rationale, and Trade-offs considered.\n"
            "4. Commit the ADR alongside the code changes for long-term project memory."
        ),
        failure_mode_prevented="Forgotten architectural context, tribal knowledge loss, and repeating past mistakes.",
    ),
    Skill(
        id="DOC-03",
        name="Semantic AST-Diff Changelog & SemVer Classifier",
        category=SkillCategory.DOCUMENTATION,
        target_agents=("orchestrator", "executor"),
        primary_trigger="Git commit generation, branch squash, or release prep",
        protocol=(
            "1. Analyze AST diffs between HEAD and base branch to classify change type:\n"
            "   - Breaking interface change -> MAJOR (feat! / fix!)\n"
            "   - New feature backward-compatible -> MINOR (feat:)\n"
            "   - Bugfix / internal change -> PATCH (fix: / refactor: / test:)\n"
            "2. Generate Conventional Commit message adhering to standard format.\n"
            "3. Append entry to CHANGELOG.md under corresponding release section."
        ),
        failure_mode_prevented="Vague commit messages ('fix stuff', 'update code'), un-tracked releases, and SemVer mistakes.",
    ),
)


class SkillRegistry:
    """In-memory indexed catalog of all built-in specialist skills."""

    def __init__(
        self,
        skills: tuple[Skill, ...] = ALL_SKILLS,
        *,
        include_anthropic: bool = False,
    ) -> None:
        raw_skills = list(skills)
        if include_anthropic:
            try:
                from forge.skills.anthropic_loader import parse_anthropic_skills

                for _, skill_item, _ in parse_anthropic_skills():
                    raw_skills.append(skill_item)
            except Exception:
                pass
            try:
                from forge.skills.design_loader import parse_design_skills

                for _, skill_item, _ in parse_design_skills():
                    raw_skills.append(skill_item)
            except Exception:
                pass
        self._skills: dict[str, Skill] = {s.id: s for s in raw_skills}

    @classmethod
    def with_anthropic(cls) -> SkillRegistry:
        """Create a SkillRegistry pre-populated with core, Anthropic, and Design skills."""
        return cls(include_anthropic=True)

    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id.upper().strip())

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def by_category(self, category: SkillCategory | str) -> list[Skill]:
        cat_str = category.value if isinstance(category, SkillCategory) else category
        return [s for s in self._skills.values() if s.category == cat_str]

    def for_agent(self, agent_role: str) -> list[Skill]:
        role = agent_role.lower().strip()
        return [s for s in self._skills.values() if role in s.target_agents]


# ==============================================================================
# AUTONOMOUS SKILL SPECIFICATIONS (Pydantic v2 Dual-Mode Engine)
# ==============================================================================

AUTONOMOUS_SKILLS: tuple[SkillSpec, ...] = (
    # --- DEBUGGING (DBG) ---
    SkillSpec(
        skill_id="DBG-01",
        name="Stratified Fault Localization & Repro-First",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEBUG_FAILURE},
        trigger_error_regexes=[
            r"(?i)AssertionError",
            r"(?i)Traceback",
            r"(?i)Error:",
            r"(?i)FAILED",
        ],
        execution_rules=[
            "Write an isolated reproduction test that triggers the failure with exit code != 0 before touching application code.",
            "Localize root cause to the minimal AST frame rather than speculating.",
            "Apply surgical fix and verify repro test passes with exit code 0.",
            "Retain repro test in the permanent regression suite.",
        ],
        allowed_tools=["search_replace_edit", "write_to_file", "run_command"],
    ),
    SkillSpec(
        skill_id="DBG-02",
        name="Automated Git Bisect & Delta Forensics",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEBUG_FAILURE},
        trigger_error_regexes=[r"(?i)regression", r"(?i)used to work", r"(?i)previously passed"],
        execution_rules=[
            "Launch automated git bisect using the failing test command.",
            "Locate the exact commit introducing the failure delta.",
            "Formulate the fix specifically addressing that commit's AST delta.",
        ],
        allowed_tools=["run_command", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="DBG-03",
        name="Loop-Break & Backtracking State Machine",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEBUG_FAILURE, OperationType.REFACTOR_SIGNATURE},
        trigger_error_regexes=[r"(?i)stuck", r"(?i)cycle", r"(?i)loop"],
        execution_rules=[
            "On 3 consecutive node failures, halt linear retries immediately.",
            "Roll back uncommitted churn to last verified git checkpoint.",
            "Replan decomposition: split atomic units or pivot implementation strategy.",
        ],
        allowed_tools=["run_command", "view_file"],
    ),
    # --- REFACTORING (REF) ---
    SkillSpec(
        skill_id="REF-01",
        name="Mikado Leaf-First Refactor",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.REFACTOR_SIGNATURE},
        execution_rules=[
            "Map AST call sites before modifying signatures.",
            "Identify zero-dependency leaf calls and migrate them first.",
            "Use Strangler-Fig dual signatures until all call sites are migrated.",
        ],
        allowed_tools=["view_file", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="REF-02",
        name="Tarjan SCC Circular Dependency Untangler",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.REFACTOR_SIGNATURE},
        trigger_error_regexes=[
            r"(?i)ImportError.*circular",
            r"(?i)cannot import name.*circular import",
        ],
        execution_rules=[
            "Run Tarjan SCC on module import ASTs to isolate cycle components.",
            "Extract shared interface/types into a dedicated leaf module.",
            "Rewire imports unidirectionally toward the leaf module.",
        ],
        allowed_tools=["view_file", "write_to_file", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="REF-03",
        name="Reachability-Verified Dead Code Pruner",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.REFACTOR_SIGNATURE},
        execution_rules=[
            "Analyze AST entrypoint call graphs for zero in-degree symbols.",
            "Verify absence of reflection or dynamic registration.",
            "Prune unreferenced symbols and execute test suite.",
        ],
        allowed_tools=["view_file", "search_replace_edit", "run_command"],
    ),
    # --- DATABASE (DB) ---
    SkillSpec(
        skill_id="DB-01",
        name="Three-Phase Expand-Contract Migration Guard",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DB_MIGRATION},
        trigger_file_globs=["*migration*", "*alembic*", "*schema*"],
        execution_rules=[
            "Split migrations into Expand, Migrate Data, Contract phases.",
            "Never drop or rename existing columns in the same step as additions.",
            "Provide bidirectional upgrade/downgrade methods.",
        ],
        allowed_tools=["view_file", "write_to_file"],
    ),
    SkillSpec(
        skill_id="DB-02",
        name="ORM Query-Counter & N+1 Enforcer",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.EDIT_ORM_QUERY},
        trigger_ast_nodes=["For", "selectinload", "joinedload"],
        execution_rules=[
            "Assert ORM query count is O(1) constant in loop/serializer queries.",
            "Inject selectinload or joinedload eager loading when N+1 is detected.",
        ],
        allowed_tools=["view_file", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="DB-03",
        name="Synthetic Cardinality & EXPLAIN Auditor",
        target_agent="qa",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.EDIT_ORM_QUERY, OperationType.DB_MIGRATION},
        execution_rules=[
            "Audit SQL execution plans for table scans on high-cardinality filters.",
            "Verify appropriate composite or partial index coverage.",
        ],
        allowed_tools=["run_command"],
    ),
    # --- API (API) ---
    SkillSpec(
        skill_id="API-01",
        name="Bi-Directional Schema & Contract Guard",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.API_ENDPOINT},
        trigger_file_globs=["*routes*", "*api*", "*controller*", "*schema*"],
        execution_rules=[
            "Detect breaking changes in payload schemas before release.",
            "Enforce Postel's law: accept optional extensions, emit strict schemas.",
        ],
        allowed_tools=["view_file", "run_command"],
    ),
    SkillSpec(
        skill_id="API-02",
        name="Idempotency & Retry Safety Enforcer",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.API_ENDPOINT},
        trigger_ast_nodes=["post", "patch", "put"],
        execution_rules=[
            "Validate Idempotency-Key support on mutating operations.",
            "Verify repeated identical payloads produce identical responses without duplication.",
        ],
        allowed_tools=["view_file", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="API-03",
        name="Boundary Fuzzing & OpenAPI Property Testing",
        target_agent="qa",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.API_ENDPOINT},
        execution_rules=[
            "Fuzz endpoints with edge cases (unicode, max int, malformed JSON).",
            "Assert endpoint never responds with unhandled 500 status.",
        ],
        allowed_tools=["run_command"],
    ),
    # --- SECURITY (SEC) ---
    SkillSpec(
        skill_id="SEC-01",
        name="AST Taint-Flow Source-to-Sink Analyzer",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.SECURITY_AUDIT, OperationType.API_ENDPOINT},
        trigger_ast_nodes=["eval", "exec", "subprocess", "shell=True", "cursor.execute"],
        execution_rules=[
            "Trace untrusted input to sensitive sinks (shell, SQL, file paths).",
            "Block execution if sanitization or parameter binding is missing.",
        ],
        allowed_tools=["view_file"],
    ),
    SkillSpec(
        skill_id="SEC-02",
        name="Shannon-Entropy Secret Scrubber",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.SECURITY_AUDIT},
        execution_rules=[
            "Compute Shannon entropy H(X) over string literals in staged diffs.",
            "Block commits containing secrets (entropy > 4.5 base64 or > 3.0 hex).",
        ],
        allowed_tools=["view_file"],
    ),
    SkillSpec(
        skill_id="SEC-03",
        name="Anti-Slopsquatting & Supply-Chain Auditor",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.SECURITY_AUDIT},
        trigger_file_globs=["pyproject.toml", "requirements.txt", "package.json"],
        execution_rules=[
            "Audit package dependencies against typosquatting and recent publication dates.",
            "Enforce hash-checking mode on package installs.",
        ],
        allowed_tools=["view_file", "run_command"],
    ),
    # --- PERFORMANCE & TESTING (PERF & TST) ---
    SkillSpec(
        skill_id="PERF-01",
        name="Asyncio Event-Loop Blocking Call Detector",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_PURE_FUNCTION, OperationType.API_ENDPOINT},
        trigger_ast_nodes=["AsyncFunctionDef", "time.sleep", "requests.get"],
        execution_rules=[
            "Detect synchronous blocking calls inside async def coroutines.",
            "Enforce asyncio equivalents or offload via asyncio.to_thread.",
        ],
        allowed_tools=["view_file", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="PERF-02",
        name="Empirical Complexity & Allocation Ratchet",
        target_agent="qa",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_PURE_FUNCTION},
        execution_rules=[
            "Detect quadratic O(N^2) loops in collection transformations.",
            "Benchmark complexity scaling across N=[100, 1000, 10000].",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="PERF-03",
        name="Concurrency Race & Barrier Stress Tester",
        target_agent="qa",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_PURE_FUNCTION},
        execution_rules=[
            "Stress concurrent access to shared state with 50+ workers.",
            "Assert zero lost updates or race conditions.",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="TST-01",
        name="Diff-Scoped Mutation Testing Gate",
        target_agent="qa",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_UNIT_TEST},
        execution_rules=[
            "Generate AST mutants (inverted operators, bool flips) on new lines.",
            "Assert test suite catches every mutant (0 surviving mutants).",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="TST-02",
        name="Metamorphic & Round-Trip Property Testing",
        target_agent="executor",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_UNIT_TEST},
        execution_rules=[
            "Verify round-trip idempotency: decode(encode(x)) == x.",
            "Use property generators to test invariant relations.",
        ],
        allowed_tools=["write_to_file", "run_command"],
    ),
    SkillSpec(
        skill_id="TST-03",
        name="Mock-Integrity & Anti-Mock-Tautology Guard",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_UNIT_TEST},
        trigger_ast_nodes=["patch", "Mock", "MagicMock"],
        execution_rules=[
            "Require autospec=True on all mock patches.",
            "Reject tests that mock the unit under test itself.",
        ],
        allowed_tools=["view_file", "search_replace_edit"],
    ),
    # --- DOCUMENTATION (DOC) ---
    SkillSpec(
        skill_id="DOC-01",
        name="Executable Docstring & README Drift Verifier",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_PURE_FUNCTION, OperationType.WRITE_UNIT_TEST},
        execution_rules=[
            "Extract code examples from README and docstrings.",
            "Execute examples to guarantee zero runtime drift.",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="DOC-02",
        name="Automated ADR & Constraints Ledger",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.REFACTOR_SIGNATURE, OperationType.DB_MIGRATION},
        execution_rules=[
            "Document significant architecture choices in docs/adr/.",
            "Record Context, Decision, and Trade-offs considered.",
        ],
        allowed_tools=["write_to_file"],
    ),
    SkillSpec(
        skill_id="DOC-03",
        name="Semantic AST-Diff Changelog & SemVer Classifier",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.REFACTOR_SIGNATURE},
        execution_rules=[
            "Classify changes as MAJOR/MINOR/PATCH from AST diffs.",
            "Generate Conventional Commit messages.",
        ],
        allowed_tools=["view_file"],
    ),
    # --- SLM PROSTHETICS (SLM-01 to SLM-09) ---
    SkillSpec(
        skill_id="SLM-01",
        name="AST Dependency Slicing & Skeletonization",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={
            OperationType.LOCATE_SYMBOL,
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.EDIT_ORM_QUERY,
            OperationType.API_ENDPOINT,
            OperationType.REFACTOR_SIGNATURE,
            OperationType.DEBUG_FAILURE,
        },
        execution_rules=[
            "Skeletonize all non-target functions to '...' preserving signatures and types.",
            "Cap context slice at 3,500 tokens to maximize SLM attention density.",
        ],
        allowed_tools=["view_file", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="SLM-02",
        name="Operation-Specific Behavioral Prompt Slicing",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={
            OperationType.LOCATE_SYMBOL,
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.EDIT_ORM_QUERY,
            OperationType.API_ENDPOINT,
            OperationType.DB_MIGRATION,
            OperationType.REFACTOR_SIGNATURE,
            OperationType.WRITE_UNIT_TEST,
            OperationType.DEBUG_FAILURE,
            OperationType.SECURITY_AUDIT,
        },
        execution_rules=[
            "Keep system prompt under 400 tokens with single-operation focus.",
            "Restrict tool exposure to exactly one action phase per turn.",
        ],
        allowed_tools=["search_replace_edit"],
    ),
    SkillSpec(
        skill_id="SLM-03",
        name="3-Tier Fuzzy Anchor Patching",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.EDIT_ORM_QUERY,
            OperationType.API_ENDPOINT,
            OperationType.REFACTOR_SIGNATURE,
            OperationType.DEBUG_FAILURE,
        },
        execution_rules=[
            "Attempt exact search match first.",
            "Fallback to whitespace-stripped anchor matching if exact match fails.",
            "Fallback to difflib ratio >= 0.92 anchor matching before rejecting.",
        ],
        allowed_tools=["search_replace_edit"],
    ),
    SkillSpec(
        skill_id="SLM-04",
        name="Ground-Truth Symbol Introspection",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={OperationType.WRITE_PURE_FUNCTION, OperationType.REFACTOR_SIGNATURE},
        execution_rules=[
            "Inject inspect.signature ground-truth signatures into the prompt before asking SLM to write calls.",
            "Eliminate signature guessing and parameter hallucinations.",
        ],
        allowed_tools=["view_file"],
    ),
    SkillSpec(
        skill_id="SLM-05",
        name="Sub-100ms In-Memory Compiler Reflexion",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.EDIT_ORM_QUERY,
            OperationType.API_ENDPOINT,
            OperationType.REFACTOR_SIGNATURE,
            OperationType.DEBUG_FAILURE,
        },
        execution_rules=[
            "Validate syntax via in-memory ast.parse before touching disk.",
            "Feed compiler error back to SLM for immediate micro-correction.",
        ],
        allowed_tools=["search_replace_edit"],
    ),
    SkillSpec(
        skill_id="SLM-06",
        name="Pre-Edit Rollback & Counterexample Constraint",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={OperationType.DEBUG_FAILURE, OperationType.REFACTOR_SIGNATURE},
        execution_rules=[
            "Roll back file state cleanly on failed edit attempt.",
            "Inject the failed diff and compiler error as an explicit negative constraint.",
        ],
        allowed_tools=["run_command", "search_replace_edit"],
    ),
    SkillSpec(
        skill_id="SLM-07",
        name="Golden Exemplar Few-Shot Injection",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={OperationType.WRITE_PURE_FUNCTION, OperationType.WRITE_UNIT_TEST},
        execution_rules=[
            "Retrieve exactly one working idiom/test exemplar from the active repository.",
            "Present exemplar as reference template to align SLM style and conventions.",
        ],
        allowed_tools=["view_file"],
    ),
    SkillSpec(
        skill_id="SLM-08",
        name="Deterministic Single-Tool Schema Masking",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={
            OperationType.LOCATE_SYMBOL,
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.EDIT_ORM_QUERY,
            OperationType.API_ENDPOINT,
            OperationType.REFACTOR_SIGNATURE,
            OperationType.WRITE_UNIT_TEST,
            OperationType.DEBUG_FAILURE,
        },
        execution_rules=[
            "Mask out all irrelevant tool schemas, exposing only the single active tool.",
            "Ensure 100% schema compliance from smaller models.",
        ],
        allowed_tools=["search_replace_edit"],
    ),
    SkillSpec(
        skill_id="SLM-09",
        name="Best-of-N Git-Worktree Voting",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM},
        operation_types={OperationType.WRITE_PURE_FUNCTION, OperationType.DEBUG_FAILURE},
        execution_rules=[
            "Generate N candidate solutions across isolated git worktrees.",
            "Select winning candidate using automated compiler and test metrics.",
        ],
        allowed_tools=["run_command"],
    ),
    # --- HARNESS ENGINES (HRN-01 to HRN-10) ---
    SkillSpec(
        skill_id="HRN-01",
        name="AST Context Slicing Engine",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={
            OperationType.LOCATE_SYMBOL,
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.DEBUG_FAILURE,
        },
        execution_rules=[
            "Extract target AST subtree and skeletonize non-target symbols within budget."
        ],
        allowed_tools=["view_file"],
    ),
    SkillSpec(
        skill_id="HRN-02",
        name="Sub-100ms Compiler Reflexion Engine",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_PURE_FUNCTION, OperationType.DEBUG_FAILURE},
        execution_rules=[
            "Verify syntax in memory before disk write and perform micro-correction loop."
        ],
        allowed_tools=["search_replace_edit"],
    ),
    SkillSpec(
        skill_id="HRN-03",
        name="Fuzzy Search/Replace Anchor Engine",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_PURE_FUNCTION, OperationType.REFACTOR_SIGNATURE},
        execution_rules=[
            "Resolve search anchors across exact, whitespace-tolerant, and difflib similarity tiers."
        ],
        allowed_tools=["search_replace_edit"],
    ),
    SkillSpec(
        skill_id="HRN-04",
        name="Live Runtime Variable Tracer",
        target_agent="executor",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEBUG_FAILURE, OperationType.WRITE_PURE_FUNCTION},
        execution_rules=[
            "Trace live variable values using sys.monitoring or sys.settrace ring buffer.",
            "Record variable deltas across executed statements to isolate logical flaws.",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="HRN-05",
        name="Pre-Flight Assumption Falsifier",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.DEBUG_FAILURE,
            OperationType.REFACTOR_SIGNATURE,
        },
        execution_rules=[
            "Formulate falsification conditions before generating implementation code.",
            "Verify assumptions against codebase reality before executing plan.",
        ],
        allowed_tools=["view_file"],
    ),
    SkillSpec(
        skill_id="HRN-06",
        name="Parallel Git-Worktree Architecture Tournament",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.REFACTOR_SIGNATURE, OperationType.WRITE_PURE_FUNCTION},
        execution_rules=[
            "Spawn K isolated git worktrees to explore distinct architectural solutions.",
            "Score implementations via deterministic matrix: exit code, complexity, test duration, diff size.",
            "Merge winning branch and cleanly remove tournament worktrees.",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="HRN-07",
        name="Information-Barrier Blind QA Red-Teaming",
        target_agent="qa",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_UNIT_TEST, OperationType.SECURITY_AUDIT},
        execution_rules=[
            "Strip Executor chain-of-thought to prevent bias.",
            "Perform independent black-box verification against interface contracts.",
        ],
        allowed_tools=["view_file", "run_command"],
    ),
    SkillSpec(
        skill_id="HRN-08",
        name="Diff-Scoped AST Mutation Testing",
        target_agent="qa",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.WRITE_UNIT_TEST},
        execution_rules=[
            "Synthesize up to 5 targeted AST mutations in modified lines.",
            "Assert that the test suite detects and fails on all mutations.",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="HRN-09",
        name="Episodic Repo Memory & SFT Store",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.AMPLIFIER_FRONTIER},
        operation_types={
            OperationType.WRITE_PURE_FUNCTION,
            OperationType.DEBUG_FAILURE,
            OperationType.REFACTOR_SIGNATURE,
        },
        execution_rules=[
            "Persist successful problem-solution traces and failure modes into repo memory.",
            "Retrieve historical heuristics during planning to prevent repeating errors.",
        ],
        allowed_tools=["view_file", "write_to_file"],
    ),
    SkillSpec(
        skill_id="HRN-10",
        name="Speculative Fast-Patch Parallel Escalation Router",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEBUG_FAILURE, OperationType.WRITE_PURE_FUNCTION},
        execution_rules=[
            "Dispatch fast speculative patch to SLM worker.",
            "Escalate to Frontier tournament if speculative patch fails compiler or tests.",
        ],
        allowed_tools=["run_command"],
    ),
    SkillSpec(
        skill_id="RES-01",
        name="STORM Multi-Perspective Inquiry",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEEP_RESEARCH},
        execution_rules=[
            "Deploy 4 orthogonal perspective personas: Theoretical Formalist, Systems Runtime Engineer, Benchmark Auditor, and Failure-Mode Critic.",
            "Execute iterative multi-hop retrieval waves until Information Gain Ratio drops below 0.10.",
        ],
        allowed_tools=["web_search", "fetch_web_page"],
    ),
    SkillSpec(
        skill_id="RES-02",
        name="Citation Graph Snowballing",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEEP_RESEARCH},
        execution_rules=[
            "Execute backward snowballing by parsing references of seed nodes for co-cited foundational papers.",
            "Execute forward snowballing to harvest cutting-edge recent citations from academic graphs.",
            "Generate deterministic BibTeX keys and blocks directly from verified metadata.",
        ],
        allowed_tools=["web_search", "fetch_web_page"],
    ),
    SkillSpec(
        skill_id="RES-03",
        name="Zero-Trust Claim-to-Quote & BibTeX Verifier",
        target_agent="qa",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEEP_RESEARCH},
        execution_rules=[
            "Verify DOI, arXiv ID, or Crossref metadata against public registries.",
            "Reject ungrounded citations missing from the research ledger.",
            "Check numeric consistency between drafted claims and verbatim extracted evidence quotes.",
        ],
        allowed_tools=["web_search", "fetch_web_page"],
    ),
    SkillSpec(
        skill_id="RES-04",
        name="Contradiction & Confounder Matrix",
        target_agent="orchestrator",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEEP_RESEARCH},
        execution_rules=[
            "Detect competing or conflicting empirical results across harvested papers.",
            "Build contradiction matrices normalizing across datasets, compute scales, and evaluation metrics.",
        ],
        allowed_tools=["web_search", "fetch_web_page"],
    ),
    SkillSpec(
        skill_id="RES-05",
        name="Sandbox Empirical Hypothesis Verification",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEEP_RESEARCH},
        execution_rules=[
            "Formulate testable algorithmic or performance hypotheses from research synthesis.",
            "Execute seed-locked benchmark scripts inside the sandbox and capture stdout metrics tables.",
        ],
        allowed_tools=["run_command", "view_file", "write_to_file"],
    ),
    SkillSpec(
        skill_id="RES-06",
        name="Section-by-Section LaTeX/IEEE Manuscript Compiler",
        target_agent="executor",
        applicable_tiers={ModelTier.PROSTHETIC_SLM, ModelTier.AMPLIFIER_FRONTIER},
        operation_types={OperationType.DEEP_RESEARCH},
        execution_rules=[
            "Synthesize manuscript sequentially section-by-section to prevent context truncation.",
            "Audit every draft section against the ZeroTrustCitationAuditor before proceeding.",
            "Compile paper.md, paper.tex, and verified references.bib artifacts.",
        ],
        allowed_tools=["write_to_file", "view_file"],
    ),
)


class AutonomousSkillRouter:
    """Autonomous router that dynamically adapts execution topology and rules."""

    def __init__(self, specs: tuple[SkillSpec, ...] | list[SkillSpec] | None = None) -> None:
        if specs is not None:
            raw_specs = list(specs)
        else:
            raw_specs = list(AUTONOMOUS_SKILLS)
            try:
                from forge.skills.anthropic_loader import parse_anthropic_skills

                for spec, _, _ in parse_anthropic_skills():
                    raw_specs.append(spec)
            except Exception:
                pass
            try:
                from forge.skills.design_loader import parse_design_skills

                for spec, _, _ in parse_design_skills():
                    raw_specs.append(spec)
            except Exception:
                pass
        self._specs: dict[str, SkillSpec] = {s.skill_id.upper().strip(): s for s in raw_specs}

    def get_spec(self, skill_id: str) -> SkillSpec | None:
        return self._specs.get(skill_id.upper().strip())

    def list_all_specs(self) -> list[SkillSpec]:
        return list(self._specs.values())

    def decide_with_jev(self, signal: Any) -> Any:
        """Run Jev System 1 fast decision pass over registered skill specifications."""
        from forge.decision.engine import JevDecisionEngine

        engine = JevDecisionEngine(SkillRegistry.with_anthropic())
        return engine.route_skills(signal)

    def select_skills(self, signals: RepoSignals, op_type: OperationType) -> list[SkillSpec]:
        """Deterministically match repository signals and operation phase to active skills."""
        selected: list[SkillSpec] = []

        for spec in self._specs.values():
            # 1. Tier compatibility
            if signals.model_tier not in spec.applicable_tiers:
                continue

            # 2. Operation phase match
            if op_type not in spec.operation_types:
                continue

            # 3. Consecutive failures override
            if signals.consecutive_failures >= 3 and spec.skill_id in {"DBG-03", "SLM-06"}:
                selected.append(spec)
                continue

            # 4. Migration files signal override
            if signals.has_migration_files and spec.skill_id == "DB-01":
                selected.append(spec)
                continue

            # 5. File globs match
            file_match = False
            if spec.trigger_file_globs and signals.modified_files:
                for file_path in signals.modified_files:
                    for pattern in spec.trigger_file_globs:
                        if fnmatch.fnmatch(file_path, pattern):
                            file_match = True
                            break
                    if file_match:
                        break

            # 6. Error regexes match
            error_match = False
            if spec.trigger_error_regexes and signals.last_stderr:
                for regex in spec.trigger_error_regexes:
                    if re.search(regex, signals.last_stderr):
                        error_match = True
                        break

            # 7. AST nodes/imports match
            ast_match = False
            if spec.trigger_ast_nodes and signals.ast_imports:
                for node_name in spec.trigger_ast_nodes:
                    if any(node_name.lower() in imp.lower() for imp in signals.ast_imports):
                        ast_match = True
                        break

            has_triggers = bool(
                spec.trigger_file_globs or spec.trigger_error_regexes or spec.trigger_ast_nodes
            )

            # If any explicit trigger matched, or if skill has no explicit triggers (always applicable for tier/op)
            if (file_match or error_match or ast_match) or not has_triggers:
                selected.append(spec)

        return selected

    def compile_operation_sliced_prompt(
        self,
        base_persona: str,
        active_skills: list[SkillSpec],
        model_tier: ModelTier,
    ) -> str:
        """Compile an operational prompt sliced for the active phase and model tier.

        When model_tier is PROSTHETIC_SLM, enforces a strict < 400 token ceiling.
        """
        if model_tier == ModelTier.PROSTHETIC_SLM:
            # Compact base persona
            persona_brief = "You are Forge SLM Executor. Surgical precision, zero fluff."
            rules: list[str] = []
            allowed_tools: set[str] = set()

            for skill in active_skills:
                for rule in skill.execution_rules:
                    if rule not in rules:
                        rules.append(rule)
                allowed_tools.update(skill.allowed_tools)

            tools_str = ", ".join(sorted(allowed_tools)) if allowed_tools else "search_replace_edit"

            # Build strictly compact prompt
            prompt_lines = [
                f"[ROLE] {persona_brief}",
                "[RULES]",
            ]
            for i, rule in enumerate(rules[:6], start=1):
                prompt_lines.append(f"{i}. {rule}")
            prompt_lines.append(f"[ALLOWED TOOLS] {tools_str}")

            sliced_prompt = "\n".join(prompt_lines)

            # Strict < 400 tokens ceiling (len // 4 budget: 400 tokens ~= 1600 chars)
            max_chars = 1400  # Leaves comfortable margin under 400 tokens (~350 tokens)
            if len(sliced_prompt) > max_chars:
                sliced_prompt = sliced_prompt[:max_chars].rsplit("\n", 1)[0]

            return sliced_prompt

        # AMPLIFIER_FRONTIER Mode: rich context and deep reasoning directives
        lines = [
            f"# Active Persona: {base_persona}",
            "",
            "## Dynamically Activated Operational Skills",
        ]
        for skill in active_skills:
            lines.append(f"### [{skill.skill_id}] {skill.name} (Target: {skill.target_agent})")
            for rule in skill.execution_rules:
                lines.append(f"- {rule}")
            if skill.allowed_tools:
                lines.append(f"- Focused Tools: {', '.join(skill.allowed_tools)}")
            lines.append("")

        lines.extend(
            [
                "## Quality & Verification Mandate",
                "- Pre-flight check: Verify assumptions before issuing destructive commands.",
                "- Verification gate: Assert test and compiler pass before marking task complete.",
            ]
        )

        return "\n".join(lines)
