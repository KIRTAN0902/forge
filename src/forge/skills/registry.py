"""Central registry defining Forge's 24 built-in specialist skills."""

from __future__ import annotations

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
            "2. Create parallel non-breaking adapter/facade layer alongside existing code (Strangler-Fig).\n"
            "3. Migrate callers one leaf at a time while keeping the test suite green on every commit.\n"
            "4. Deprecate and remove legacy implementation only after all callers have verified migration."
        ),
        failure_mode_prevented="Catastrophic multi-file breakage where refactoring leaves the codebase in an uncompilable state.",
    ),
    Skill(
        id="REF-02",
        name="Tarjan SCC Circular Dependency Untangler",
        category=SkillCategory.REFACTORING,
        target_agents=("executor",),
        primary_trigger="ImportError cycle or AST import graph cycle (>1 strongly connected component)",
        protocol=(
            "1. Run Tarjan's Strongly Connected Components algorithm across the module import graph.\n"
            "2. Extract shared abstractions, types, or protocols into dedicated leaf modules.\n"
            "3. Replace direct circular imports with dependency inversion or lazy runtime injection.\n"
            "4. Verify cyclic graph is converted into a clean Directed Acyclic Graph (DAG)."
        ),
        failure_mode_prevented="Runtime import crashes, monkey-patching hacks, and un-testable tightly coupled modules.",
    ),
    Skill(
        id="REF-03",
        name="Reachability-Verified Dead Code Pruner",
        category=SkillCategory.REFACTORING,
        target_agents=("qa", "executor"),
        primary_trigger="Post-refactor / file deletion or zero-in-degree AST symbols",
        protocol=(
            "1. Perform global symbol reachability analysis starting from entry points and public interfaces.\n"
            "2. Flag symbols, functions, and classes with zero in-degree call paths.\n"
            "3. Verify symbols are not dynamically dispatched via getattr or reflection.\n"
            "4. Safely remove unreferenced code and verify test suite passes without regressions."
        ),
        failure_mode_prevented="Accumulation of phantom code, dead functions, and misleading zombie implementations.",
    ),

    # --- DATABASE (DB) ---
    Skill(
        id="DB-01",
        name="Three-Phase Expand-Contract Migration Guard",
        category=SkillCategory.DATABASE,
        target_agents=("orchestrator", "qa"),
        primary_trigger="Edits to migrations/, alembic/, or ORM models",
        protocol=(
            "1. Enforce phase 1 (Expand): Add new columns/tables as nullable or with safe defaults without breaking existing code.\n"
            "2. Enforce phase 2 (Dual-write/Backfill): Read from legacy or new, write to both.\n"
            "3. Enforce phase 3 (Contract): Only drop old columns or add NOT NULL constraints in a distinct subsequent migration.\n"
            "4. Forbid destructive operations (DROP TABLE, DROP COLUMN) in a single-step feature PR."
        ),
        failure_mode_prevented="Production downtime, broken database locks, and rollback impossibilities during deployments.",
    ),
    Skill(
        id="DB-02",
        name="ORM Query-Counter & N+1 Enforcer",
        category=SkillCategory.DATABASE,
        target_agents=("qa", "executor"),
        primary_trigger="ORM collection iterated in loop or serializer",
        protocol=(
            "1. Inspect loop constructs traversing relational attributes on model instances.\n"
            "2. Verify explicit eager loading (e.g. selectinload, joinedload, prefetch_related, select_related).\n"
            "3. Assert that queries executed within loop bodies remain constant O(1) regardless of collection size.\n"
            "4. Add test assertions enforcing max query count for the endpoint."
        ),
        failure_mode_prevented="Silent N+1 query storms that pass unit tests with 2 items but collapse in production with 1,000 items.",
    ),
    Skill(
        id="DB-03",
        name="Synthetic Cardinality & EXPLAIN Auditor",
        category=SkillCategory.DATABASE,
        target_agents=("qa",),
        primary_trigger="New SQL joins, filters, or sorting predicates",
        protocol=(
            "1. Review query predicates against existing table indexes.\n"
            "2. Verify absence of full table scans (Seq Scan) on tables with high cardinality.\n"
            "3. Ensure composite index columns match filter and order-by prefix order.\n"
            "4. Flag missing foreign key indexes that cause lock contention."
        ),
        failure_mode_prevented="Missing indexes causing table locks, slow queries, and database CPU spikes.",
    ),

    # --- API & CONTRACTS (API) ---
    Skill(
        id="API-01",
        name="Bi-Directional Schema & Contract Guard",
        category=SkillCategory.API,
        target_agents=("qa",),
        primary_trigger="Modifications to route controllers, Pydantic, or OpenAPI schemas",
        protocol=(
            "1. Extract OpenAPI / JSON-Schema delta between base branch and current changes.\n"
            "2. Detect breaking changes: deleted fields, renamed fields, tighter validation, or new required fields.\n"
            "3. Require version increment or backward-compatible fallback if breaking contract detected.\n"
            "4. Verify both request serializers and response models match documented schemas."
        ),
        failure_mode_prevented="Silent breaking changes in public APIs causing frontend and client consumer failures.",
    ),
    Skill(
        id="API-02",
        name="Idempotency & Retry Safety Enforcer",
        category=SkillCategory.API,
        target_agents=("executor", "qa"),
        primary_trigger="POST/PATCH routes, webhooks, queue tasks, or outbound HTTP calls",
        protocol=(
            "1. Inspect state-mutating endpoints and webhook handlers for idempotency key handling.\n"
            "2. Ensure database operations within message handlers are atomic (database transaction).\n"
            "3. Verify third-party outbound HTTP requests include exponential backoff and jitter.\n"
            "4. Verify duplicate message delivery produces identical result without double-charging or double-mutating."
        ),
        failure_mode_prevented="Duplicate transaction processing, double-charges, and webhook retry corruption.",
    ),
    Skill(
        id="API-03",
        name="Boundary Fuzzing & OpenAPI Property Testing",
        category=SkillCategory.API,
        target_agents=("qa",),
        primary_trigger="Any newly implemented or modified HTTP/RPC endpoint",
        protocol=(
            "1. Generate property-based fuzz payloads targeting boundary values (empty strings, huge payloads, unicode, null bytes, max integers).\n"
            "2. Verify endpoint handles edge cases with clean 4xx validation errors rather than unhandled 500 exceptions.\n"
            "3. Verify headers, Content-Type, and query parameters are strictly validated.\n"
            "4. Confirm memory or CPU does not spike during malformed input parsing."
        ),
        failure_mode_prevented="Unhandled 500 server crashes and denial-of-service vulnerabilities on unexpected user inputs.",
    ),

    # --- SECURITY (SEC) ---
    Skill(
        id="SEC-01",
        name="AST Taint-Flow Source-to-Sink Analyzer",
        category=SkillCategory.SECURITY,
        target_agents=("qa",),
        primary_trigger="Diffs with SQL, subprocess, file paths, or URL fetching",
        protocol=(
            "1. Trace untrusted user input sources (request params, headers, files, stdin).\n"
            "2. Check if untrusted data flows into critical sinks: exec(), eval(), raw SQL queries, shell=True, or open().\n"
            "3. Enforce parameterized queries, safe subprocess lists (shell=False), and path traversal sanitization.\n"
            "4. Fail validation immediately if unsanitized user data reaches a sensitive sink."
        ),
        failure_mode_prevented="Remote code execution (RCE), SQL injection, and path traversal vulnerabilities.",
    ),
    Skill(
        id="SEC-02",
        name="Shannon-Entropy Secret Scrubber",
        category=SkillCategory.SECURITY,
        target_agents=("qa",),
        primary_trigger="Pre-commit hook on every git commit or file write",
        protocol=(
            "1. Compute Shannon entropy on all new string literals and diff additions.\n"
            "2. Scan for signature patterns: AWS keys, JWTs, private keys, OpenRouter/OpenAI API keys.\n"
            "3. Abort the commit/write if a secret token is detected in source code or configuration.\n"
            "4. Replace hardcoded secrets with environment variable references (e.g. os.environ)."
        ),
        failure_mode_prevented="Accidental leaking of private credentials and production keys into Git history.",
    ),
    Skill(
        id="SEC-03",
        name="Anti-Slopsquatting & Supply-Chain Auditor",
        category=SkillCategory.SECURITY,
        target_agents=("orchestrator", "qa"),
        primary_trigger="pip/uv/npm/cargo install commands or manifest edits",
        protocol=(
            "1. Extract any new dependency package names proposed by agent.\n"
            "2. Query package index registry to verify package existence, release age, and download activity.\n"
            "3. Detect hallucinated package names or typosquatting of popular libraries.\n"
            "4. Block installation of unverified or suspect single-day-old packages."
        ),
        failure_mode_prevented="Installing hallucinated or malicious packages from untrusted public package registries.",
    ),

    # --- PERFORMANCE & CONCURRENCY (PERF) ---
    Skill(
        id="PERF-01",
        name="Asyncio Event-Loop Blocking Call Detector",
        category=SkillCategory.PERFORMANCE,
        target_agents=("qa",),
        primary_trigger="Edits inside async def coroutines or async callees",
        protocol=(
            "1. Inspect AST inside async def coroutines for synchronous blocking calls.\n"
            "2. Flag time.sleep(), blocking requests.get(), urllib, or synchronous file open/read.\n"
            "3. Mandate asyncio.sleep(), httpx/aiohttp, or asyncio.to_thread() for blocking calls.\n"
            "4. Verify no CPU-heavy operations starve the event loop."
        ),
        failure_mode_prevented="Freezing the entire asyncio event loop and degrading server throughput to zero.",
    ),
    Skill(
        id="PERF-02",
        name="Empirical Complexity & Allocation Ratchet",
        category=SkillCategory.PERFORMANCE,
        target_agents=("qa", "executor"),
        primary_trigger="Nested loops, 'in list' checks in loops, or data processing pipelines",
        protocol=(
            "1. Scan AST for linear scans inside iterations (e.g. 'if item in my_list' inside a loop -> O(n^2)).\n"
            "2. Replace with set/dict lookups for O(1) membership testing.\n"
            "3. Enforce generators or streaming readers when processing files or collections > 10,000 items.\n"
            "4. Verify memory allocation stays O(1) or O(log n) rather than accumulating in RAM."
        ),
        failure_mode_prevented="Silent quadratic complexity (O(n^2)) bottlenecks and out-of-memory crashes on large inputs.",
    ),
    Skill(
        id="PERF-03",
        name="Concurrency Race & Barrier Stress Tester",
        category=SkillCategory.PERFORMANCE,
        target_agents=("qa",),
        primary_trigger="Shared mutable state, locks, quotas, or counter/balance updates",
        protocol=(
            "1. Identify shared mutable data structures accessed across threads or async tasks.\n"
            "2. Verify explicit synchronization primitives (Locks, Semaphores, Atomic operations).\n"
            "3. Generate stress tests spawning 50+ concurrent coroutines updating the same state.\n"
            "4. Assert final invariant equals exact expected count without race conditions."
        ),
        failure_mode_prevented="Data corruption, negative account balances, and intermittent race condition bugs.",
    ),

    # --- TESTING INTEGRITY (TST) ---
    Skill(
        id="TST-01",
        name="Diff-Scoped Mutation Testing Gate",
        category=SkillCategory.TESTING,
        target_agents=("qa",),
        primary_trigger="Post-implementation QA pass on new code + unit tests",
        protocol=(
            "1. Introduce targeted mutations to modified AST nodes (e.g. invert conditionals, replace return values with None/0, alter binary operators).\n"
            "2. Execute the corresponding unit test suite against the mutant.\n"
            "3. Verify that the test suite FAILS for each mutation (mutant is killed).\n"
            "4. If tests still pass with mutant code, fail QA and require stronger test assertions."
        ),
        failure_mode_prevented="Tests that pass unconditionally without actually testing the underlying implementation.",
    ),
    Skill(
        id="TST-02",
        name="Metamorphic & Round-Trip Property Testing",
        category=SkillCategory.TESTING,
        target_agents=("qa", "executor"),
        primary_trigger="Parsers, serializers, encoders, compilers, or transformers",
        protocol=(
            "1. Define round-trip algebraic invariant: decode(encode(x)) == x for all valid x.\n"
            "2. Generate diverse arbitrary inputs using property testing strategies.\n"
            "3. Verify metamorphic relations (e.g. sorting an already sorted list produces identical output).\n"
            "4. Assert idempotency and state stability across successive transformations."
        ),
        failure_mode_prevented="Data loss during serialization/deserialization and subtle parser edge-case corruption.",
    ),
    Skill(
        id="TST-03",
        name="Mock-Integrity & Anti-Mock-Tautology Guard",
        category=SkillCategory.TESTING,
        target_agents=("qa",),
        primary_trigger="Test diffs containing mock, patch, or assertion edits",
        protocol=(
            "1. Scan test diffs for unittest.mock, patch, or Mock objects.\n"
            "2. Verify that the System Under Test (SUT) itself is NEVER mocked.\n"
            "3. Forbid tautological assertions (e.g. asserting mock.method() == mock_return_value).\n"
            "4. Allow mocks strictly for external third-party I/O boundaries (Stripe, Twilio, AWS)."
        ),
        failure_mode_prevented="Tests that pass 100% green but test nothing because the system under test was mocked out.",
    ),

    # --- DOCUMENTATION & GOVERNANCE (DOC) ---
    Skill(
        id="DOC-01",
        name="Executable Docstring & README Drift Verifier",
        category=SkillCategory.DOCUMENTATION,
        target_agents=("qa", "executor"),
        primary_trigger="Public API signature, CLI flag, or user-facing interface changes",
        protocol=(
            "1. Inspect public function signatures, parameters, return types, and CLI options.\n"
            "2. Verify all docstrings match current parameter names and types.\n"
            "3. Extract code snippets from README.md and execute them as doctests.\n"
            "4. Synchronize documentation before completing the task."
        ),
        failure_mode_prevented="Stale documentation, broken copy-paste README examples, and confusing developer onboarding.",
    ),
    Skill(
        id="DOC-02",
        name="Automated ADR & Constraints Ledger",
        category=SkillCategory.DOCUMENTATION,
        target_agents=("orchestrator",),
        primary_trigger="Core architecture/dependency changes or [DBG-03] pivots",
        protocol=(
            "1. Detect architectural pivots, new major dependencies, or design pattern shifts.\n"
            "2. Automatically scaffold a markdown Architecture Decision Record (ADR) under docs/adr/.\n"
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

    def __init__(self, skills: tuple[Skill, ...] = ALL_SKILLS) -> None:
        self._skills: dict[str, Skill] = {s.id: s for s in skills}

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
