# Forge Architecture

Implementation notes for [`FORGE_SPECIFICATION.md`](../FORGE_SPECIFICATION.md). The spec
says *what* to build. This file records *how* it was built, the decisions taken where the
spec was ambiguous, and the status of each phase.

> North star: given the same model, Forge should produce better software than using the
> model directly. Every design choice below either helps a model do more (native
> capabilities, fallbacks, context) or keeps it safe (deterministic authority).

## Phase status

| Phase | Scope | Status | Exit condition evidence |
|---|---|---|---|
| 0 | Foundation: packages, contracts, config, events, storage, policies, CLI skeleton | Done | 54 tests; `test_architecture.py` enforces layering |
| 1 | Model runtime | Done | `test_model_swap.py` (simulated) + `test_live_models.py` (real: Ollama qwen2.5:7b native and prompt protocols, OpenRouter Nemotron-3-Super) |
| 2 | Tool runtime | Done | `test_tool_runtime_exit.py` (scripted model fixes a bug; destructive command, secret read and workspace escape all blocked) + `test_live_tools.py` (Nemotron fixes the bug through the tools) |
| 3 | Context engine | Done | `test_context_exit.py` (Planner/Executor/QA contexts for an 8K and a 200K model: within budget, right files in, unrelated files out) + live: qwen2.5:7b names the right file from the planner context alone |
| 4 | Specialist agents | Done | `test_agents_exit.py` (Planner, Executor, QA independently perform roles) + 9 unit tests |
| 5 | LLM orchestrator | Done | `test_orchestrator_exit.py` (full PLAN->EXECUTE->QA->COMPLETE cycle driven dynamically) + 8 unit tests |
| 6 | Autonomous workflow and recovery | Done | `test_recovery_exit.py` (autonomous recovery from QA failure, rework loops, checkpoints and human gates) |
| 7 | CLI | Done | `test_cli_exit.py` (full CLI lifecycle: `init`, `run`, `status`, `history`, `diff`, `rollback`) + 8 unit tests |
| 8 | Dogfooding | Next | |
| 9 | Benchmarking | | |
| 10 | Web control plane | | |

## Package layering

```
cli ──────────────► (anything below; nothing imports cli)
workflow ─► agents ─► context ─► tools ─► config ─► models, policies
                 └──► models ─► events ─► core
storage ─► events, core
core: errors, ids, time, logging, globs, contracts (imports nothing else in forge)
```

`tests/unit/test_architecture.py` parses imports and fails the build if:

- `forge.core` imports any other Forge package
- anything except `forge.cli` imports `forge.cli` (spec rules 2 and 18: headless engine)
- `forge.agents` imports `forge.models.providers` (rule 1: agents are never coupled to providers)
- `forge.policies` imports models, tools, context, agents or workflow (authority stays
  deterministic)
- `forge.models` imports tools, context, agents, workflow or storage
- `forge.tools` imports context, agents or workflow; `forge.context` imports agents or
  workflow

## Phase 0: foundation

- **Contracts** (`forge.core.contracts`) are Pydantic models shared by all components
  (rules 4 and 5): `RequirementContract`, `Plan`/`Task`, `ExecutionResult`, `QAReport`,
  `Decision`, `WorkflowState`.
  - Contracts *ignore* unknown fields, so a model adding a stray key doesn't fail a
    valid plan. Human-written config *rejects* unknown keys, so typos surface.
  - `Plan` validates its own task graph (unknown dependencies, self-dependencies,
    cycles) and provides `topological_order()` and `ready_tasks()`.
  - Acceptance criteria carry a `verification` method and link to requirement ids,
    because the QA question is "can we *prove* every criterion?".
- **Events** (`forge.events`): a typed `EventType` vocabulary (spec §18 plus decision,
  approval, checkpoint and model-call events) and an async `EventBus`. A failing
  subscriber is logged and never breaks the workflow.
- **Storage** (`forge.storage`): SQLite through the stdlib driver on a worker thread.
  The workflow state is stored as JSON alongside an append-only event log. Schema
  migrations use `PRAGMA user_version`, and a newer schema is refused.
- **Policies** (`forge.policies`): `RiskLevel` (semantically ordered),
  `Permission`, `ProposedAction`, `PolicyEngine`. The most restrictive verdict wins, and
  an action no policy claims *fails closed* (requires approval). `NETWORK` permission is
  off by default.
- **Config** (`forge.config`): defaults < `forge.yaml` (searched upward from cwd) <
  `FORGE__SECTION__KEY` environment variables.

## Phase 1: model runtime

The five abstractions from the spec, each with one job:

| Abstraction | Job |
|---|---|
| `ModelCapabilities` | What the model can do (context, tools, JSON schema, JSON mode, reasoning, streaming, vision) |
| `ModelProfile` | One configured model: provider, model id, capabilities, params, optimization, pricing |
| `ModelProvider` | Transport only: neutral `ModelRequest` to wire format and back. `OpenAICompatibleProvider` covers OpenAI, OpenRouter, Ollama, vLLM, LM Studio, GLM, Qwen, DeepSeek |
| `ModelRegistry` | Resolves role → profile → provider, detects capabilities, builds adapters |
| `ModelAdapter` | The only thing agents hold. Chooses strategies, applies fallbacks, records usage and events |

### Use native capabilities, fall back when they're missing

The adapter chooses a strategy per capability. `auto` means "native when capable":

| Need | Native | Fallback |
|---|---|---|
| Tool calling | `tools` parameter | Hermes-style `<tool_call>{json}</tool_call>` protocol in the system prompt; tool history rewritten as text |
| Structured output | `response_format: json_schema` | `json_object` mode, then schema-in-prompt plus lenient extraction |
| Reasoning | `reasoning_effort` (OpenAI style) or `reasoning.effort` (OpenRouter style) | Inline `<think>` blocks moved into `message.reasoning` |
| Streaming | SSE | A single chunk plus the final response |

Robustness behaviours:

- **Repair loop**: invalid structured output is sent back with the exact validation
  errors (and a note if the output was truncated), up to `max_repair_attempts`.
- **Auto-downgrade**: if a server rejects a strategy (HTTP 400/404/422), the adapter
  steps down (native → JSON mode → prompt) and *stays* downgraded for that adapter.
- **Tool-call salvage**: a model given native tools that still answers in `<tool_call>`
  text has those calls recovered instead of lost.
- **History hygiene**: a model without native tool support never receives `tool`-role
  messages or `tool_calls`, even on turns that offer no tools. A test caught this case.
- **Malformed tool arguments** are kept on the `ToolCall` (`parse_error`, `raw_arguments`)
  so the agent loop can report them back to the model instead of crashing.
- **Retries**: 408/409/425/429/5xx and transport errors, with exponential backoff that
  honours `Retry-After`. Streams retry only before the first byte.

### Capability detection

`ModelRegistry.capabilities()` asks the provider (`describe_model`) and overlays the
answer onto the profile. **Explicitly configured values always win.**

- OpenRouter: `context_length`, `supported_parameters` (tools, structured outputs,
  response_format, reasoning), input modalities, pricing.
- Ollama: `/api/show` capabilities (tools, vision, thinking). The trained context length
  is deliberately **not** reported as usable: Ollama's OpenAI endpoint serves with the
  server's `num_ctx` and silently truncates beyond it. Set `context_window` to match
  `OLLAMA_CONTEXT_LENGTH`.
- `parallel_tool_calls` is never inferred, because some endpoints reject the parameter.

### Roles

`roles:` binds `default | orchestrator | planner | executor | qa` to profiles, with
per-role `params` and `optimization` overrides. Parameters layer as
profile < role < call.

## Phase 2: tool runtime

Every tool call goes through one gate, `ToolRuntime.execute`:

```
ToolCall -> look up tool (and check the role is allowed to use it)
         -> validate arguments against the tool's input schema
         -> tool.assess(): the *concrete* risk, permissions, paths and command
         -> PolicyEngine (deterministic) -> allow | require approval (Approver) | deny
         -> run with a timeout -> redact secrets -> truncate -> audit events
```

Failures never raise into the agent loop. Unknown tools, bad arguments, denials, timeouts
and tool crashes all come back as `ToolResult(ok=False)` with an error the model can read
and correct.

| Category | Tools |
|---|---|
| Filesystem | `read_file` (paged, numbered), `write_file`, `edit_file` (exact unique match, keeps CRLF files CRLF), `delete_file`, `list_directory`, `search` (regex, .gitignore-aware), `find_files` |
| Terminal | `run_command`, `run_tests` (auto-detects pytest, npm, pnpm, yarn, go, cargo, maven, gradle; parses counts), `start_process`, `process_output`, `stop_process`, `list_processes` |
| Git | `git_status`, `git_diff`, `git_log`, `git_branch`, `git_checkpoint`, `git_list_checkpoints`, `git_rollback` |
| Analysis | `repository_structure`, `symbol_search`, `file_outline`, `dependency_analysis` |

Role toolsets (`ROLE_TOOLSETS`): the Planner is read-only, the Executor gets everything,
and QA can read and run but not edit the code it is judging.

### Safety layers

| Layer | Mechanism |
|---|---|
| Sandbox | `Workspace.resolve` makes `..`, absolute paths and outward symlinks impossible, not merely forbidden |
| Command risk | `assess_command` splits on `&&`, `\|\|`, `;`, `\|` and newlines (outside quotes), looks inside `bash -c`, and takes the riskiest segment. Read-only commands are LOW; tests and edits MEDIUM; recursive deletes, `git reset --hard`, migrations and `sudo` HIGH; deploys, force-push and publishing CRITICAL |
| Forbidden commands | `rm -rf /`, `curl \| sh`, fork bombs, `mkfs`, `dd of=/dev/…`, `shutdown`… are denied outright (configurable regexes) |
| Paths | Writes to `protected_paths` (`.git/**`, `.env`, keys) and reads of `secret_paths` (`.env*` except `.env.example`, keys, `.npmrc`…) need approval. `search` skips secret files. `Workspace` defaults to the secret patterns, so the safe behaviour needs no configuration |
| Permissions | NETWORK is not granted by default, so `pip install` and `curl` are denied until a project opts in |
| Approval | HIGH and above go to the `Approver`; with no approver (non-interactive) they are rejected |
| Environment | Commands run without secret-looking environment variables (`*KEY*`, `*TOKEN*`, …); any secret value that still appears in output is redacted |
| Processes | Timeouts kill the whole process tree (`taskkill /T` on Windows, process groups on POSIX); output is captured in bounded head-and-tail buffers; background processes are stopped when the session ends |

### Checkpoints without touching the user's git state

`git_checkpoint` snapshots the entire worktree, including untracked non-ignored files, as
a dangling commit built in a temporary index and pinned at
`refs/forge/checkpoints/<id>`. The user's branch, HEAD, index and stash are never
modified. `git_rollback` first checkpoints the current state (so a rollback can be
undone), removes files created since the checkpoint, and writes the checkpoint's files
back through another temporary index. Ignored files (caches, `.venv`) are left alone.

### Shell choice

The shell is resolved once and named in `run_command`'s description, so the model writes
commands for the shell that will actually run them. On Windows, Git Bash is preferred and
the WSL launcher (`System32\bash.exe`) is avoided, falling back to PowerShell.

### What the live runs showed

- **Nemotron-3-Super (OpenRouter)** explored, read the code, fixed `add()` and re-ran the
  tests: 7 steps.
- **qwen2.5:7b (Ollama)** repeatedly edited the *test* instead of the code. The tool
  runtime kept it safe, but it gamed the test rather than fixing the bug. Catching this is
  the job of adversarial QA (Phase 4+), and the live test is marked `xfail` with that
  reason.
- Qwen also emitted a malformed call (stray text, JSON, a dangling `</tool_call>`). The
  text protocol now recovers bare `{"name": <known tool>, "arguments": …}` objects. Only
  known tool names qualify, so ordinary JSON in an answer is never mistaken for a call.

## Phase 3: context engine

| Component | Responsibility |
|---|---|
| `RepositoryAnalyzer` | One picture of the repo: files (language, test, generated), key files, entry points (including `[project.scripts]`), manifests, symbol index, import graph, test setup, git state, README. `read()` re-reads files that changed and never returns secrets |
| `ContextSelector` | Ranks files by explicit seeds > mentions in errors or findings > changed files > relevance (filename and directory terms, symbol names, TF-IDF content) > structure (import neighbours, matching tests). Every pick carries its reasons |
| `ContextBudgetManager` | Budget from the model's real window, minus output and conversation headroom (the executor keeps the most), capped by `context_strategy` (`minimal` 6K, `targeted` 32K, `full` uncapped) |
| `pack` | Includes sections by priority and renders them in declaration order. Each file offers a fallback (outline plus relevant excerpts) that is used before anything is dropped. Required sections are truncated, never dropped. Anything omitted is listed so the model knows to fetch it with tools |
| `ContextBuilder` | The spec §13 contexts, one method per role |
| `scan_files` | Points QA at injection, unsafe deserialization, `shell=True`, weak hashes, hard-coded credentials, disabled TLS or JWT checks, and permissive CORS in changed files |
| `compact_history` | Keeps long agent conversations in the window: system prompt and task untouched, recent turns intact (never splitting a tool call from its result), old tool output elided first, then older turns summarized |

| Role | Sections (spec §13) |
|---|---|
| Planner | Requirement, repository structure, architecture (summary + README), relevant source, dependencies, tests, configuration, git state |
| Executor | Current task (with its acceptance criteria), previous errors and QA findings, acceptance criteria, plan (current task marked), current git diff, relevant files, repository map |
| QA | Requirements and acceptance criteria, git diff, test results, security-sensitive areas, previous findings, plan, changed and related files |

### Lessons from making selection work

The first version of selection failed in three instructive ways, each now covered by a
test:

1. Tests outranked the code they test. Two files that import each other boosted each
   other in a loop. Structural boosts now come from a snapshot of direct scores, and test
   files carry half weight.
2. A naive stemmer turned `invoices` into `invoic` and left `invoice` alone, so one word
   counted twice.
3. With a big budget, weak matches crept in (`client.ts` "mentions return"). Programming
   keywords are now stopwords, candidates below 20% of the best score are dropped, a
   file keeps only its strongest structural link (no hub inflation), and filename matches
   outweigh directory matches.

`forge context "<request>"` shows what the Planner would get: selected files with reasons,
each section's mode and size, and the budget.

## Phase 4: specialist agents

| Agent | Responsibility | Tools | Contract Output |
|---|---|---|---|
| `PlannerAgent` | Elicits requirements and breaks down the goal into a DAG of tasks. Can explore using read-only tools before finalizing. | Read-only tools (`ROLE_TOOLSETS["planner"]`) | `PlannerResult` (`RequirementContract` + `Plan`) |
| `ExecutorAgent` | Implements an approved task. Iteratively investigates, edits files, runs tests, fixes failures, and verifies changes. | Full tools (`ROLE_TOOLSETS["executor"]`) | `ExecutionResult` (`FileChange`s + `TestRun`s) |
| `QAAgent` | Adversarial tester and security reviewer. Proves acceptance criteria, runs test suites, scans for security patterns, and catches test tampering. Never edits code. | Read-only + terminal/tests (`ROLE_TOOLSETS["qa"]`) | `QAReport` (`CriterionVerdict`s + `QAFinding`s) |

### Key Agent Behaviors

- **Stateless workers**: Agents communicate through structured Pydantic contracts rather than free-form conversation. Shared state and orchestration belong to the control plane.
- **Ground-truth audit recording**: `ExecutorAgent` harvests actual files changed and test outputs directly from `ToolResult`s, ensuring file changes and test exit codes in `ExecutionResult` reflect ground truth on disk.
- **Deterministic QA guards**: `QAAgent` pairs model reasoning with deterministic checks:
  - `_check_security`: invokes `scan_files` to catch injection, hardcoded credentials, and unsafe deserialization.
  - `_check_test_tampering`: inspects git diffs of test files to detect when tests are edited or assertions removed to game requirements, immediately flagging a blocking P1 regression.
- **Event-driven**: All agents emit structured lifecycle events (`REQUIREMENTS_DEFINED`, `PLAN_CREATED`, `TASK_STARTED`, `TASK_COMPLETED`, `QA_STARTED`, `QA_FINDING_CREATED`, `QA_COMPLETED`, `REWORK_REQUESTED`).

## Phase 5: LLM orchestrator

| Component | Responsibility |
|---|---|
| `Orchestrator` | Control plane: prompts model with rendered state markdown, evaluates next action, applies deterministic policy validation, steps through execution, and persists state after every turn. |
| `WorkflowPolicyValidator` | Deterministic guardrails: clamps iteration limit breaches to `BLOCK`, enforces plan prerequisites, resolves next ready tasks, and prevents completing with unaddressed blocking findings or without QA. |
| `AgentRouter` | Dispatches validated decisions (`PLAN`, `EXECUTE`, `QA`, etc.) to the appropriate specialist agent, dynamically builds role-specific context via `ContextBuilder`, and updates `WorkflowState`. |
| `render_workflow_state` | Converts `WorkflowState` (request, status, requirements, tasks, execution results, QA findings, errors, recent decisions) into structured Markdown for model turns. |

### Hybrid Architecture Principle

> "The LLM has intelligence. The system has authority."

The Orchestrator pairs flexible model reasoning with rigid deterministic safety invariants:
- The model evaluates complex engineering tradeoffs and decides whether to plan, execute, re-execute, or verify.
- The policy gate ensures the workflow never completes prematurely, never executes tasks with unresolved dependencies, never ignores failing QA findings, and never exceeds execution bounds.
- State is persisted after every step to `SQLiteStore`, guaranteeing durability and resumption capability.

## Phase 6: autonomous workflow & recovery

| Component | Responsibility |
|---|---|
| `CheckpointManager` | Working tree snapshotting and restoration. Captures lightweight Git commits without altering active branches or index, enabling automated and manual rollbacks. |
| `ConvergenceTracker` | Tracks failure trajectories across iterations. Fingerprints failures and detects repeated identical failures (>= 3), oscillating fixes (A -> B -> A -> B), and stagnation. |
| `compute_failure_signature` | Computes deterministic, normalized SHA-256 fingerprints of failure states (QA findings, test output tails, and error records). |
| Human Approval Gates | Orchestrator pauses on `AWAITING_APPROVAL`, emits `ApprovalRequested`, and provides `approve()` and `resume()` APIs that inject human guidance before continuing. |
| Autonomous Rework Loops | Coordinates QA rejection -> `ReworkRequested` event -> Executor context injection -> targeted fix -> verification QA -> completion. |

### Recovery & Thrashing Guardrails

1. **Safety Checkpoints**: Before any mutating task execution, `AgentRouter` automatically snapshots the repository via `CheckpointManager.create_checkpoint()`.
2. **Deterministic Rework Redirection**: When QA fails or blocking findings exist, `WorkflowPolicyValidator` clamps any completion attempts to `REEXECUTE` and passes specific `finding_ids` directly to the `ExecutorAgent`.
3. **Loop & Thrashing Intervention**: If `ConvergenceTracker` detects thrashing (consecutive identical failures, oscillation, or stagnation), `WorkflowPolicyValidator` converts the decision into `REQUEST_HUMAN_REVIEW` and pauses the autonomous execution loop to prevent token exhaustion.

## Phase 7: CLI

| Component | Responsibility |
|---|---|
| `forge.cli.app` | Typer command suite (`init`, `run`, `status`, `history`, `diff`, `resume`, `approve`, `rollback`, `models`, `config`, `context`, `version`). Serves as a headless presentation layer without embedding engine logic. |
| `forge.cli.render` | Rich console rendering: `CliEventPrinter` streams live execution events (`WorkflowStarted`, `PlanCreated`, `TaskCompleted`, `QACompleted`, etc.); formatters render workflow overviews, task graphs, QA findings, and timelines. |
| `forge.cli.interactive` | `CliApprover` prompts terminal users with risk levels, commands, and policy reasons whenever high-risk actions or human review is required. |
| `forge.cli.workflow` | `build_workflow_harness()` constructs and wires `SQLiteStore`, `ToolRuntime`, `ModelRegistry`, agents, `AgentRouter`, and `Orchestrator` cleanly from configuration. |

### CLI Execution & Headless Design

1. **Strict Layering**: The CLI is strictly a thin client. All workflows, state mutations, and safety policies are driven by the headless engine. Package layering tests enforce that no engine package imports `forge.cli`.
2. **Event Streaming**: Rather than polling or scraping logs, the CLI attaches `CliEventPrinter` to the engine's `EventBus`, printing formatted, color-coded step transitions, tool invocations, and QA audits as they occur.
3. **Execution Safety**: The `_run_async()` bridge allows CLI commands to execute seamlessly in standalone terminal processes, pytest test runners, and nested asyncio loops.
4. **Offline Isolation**: Operations like `diff` and `rollback` operate directly against Git and the SQLite storage layer without requiring model provider credentials or active network access.

## Decisions on spec ambiguities

1. **Orchestrator actions**: §7 and §22 list different action sets. The reconciled
   enum is `PLAN, REPLAN, EXECUTE, REEXECUTE, TARGETED_QA, FULL_QA, RETRY, ROLLBACK,
   REQUEST_RESEARCH, REQUEST_HUMAN_REVIEW, BLOCK, COMPLETE`. `QA` is split by scope.
   `REQUEST_RESEARCH` is a Planner investigation (there is no research agent).
2. **Requirement contract ownership**: the Planner will produce it as its first output
   (Phase 4), with an optional human approval gate (`RequirementContract.approved`).
3. **Package location**: the spec's `forge/` tree lives at the repository root
   (`src/forge`). `forge.config` was added as its own package for configuration.
4. **Static analysis lives in `forge.tools.analysis`** so the analysis tools (Phase 2)
   and the context engine (Phase 3) share one implementation.
5. **Test tampering** (seen live with a 7B model) is a QA concern. Phase 4's QA must
   treat edits to tests made while fixing code as suspect.

## Development

```bash
uv sync
uv run pytest                                   # offline suite (live tests deselected)
FORGE_LIVE_TESTS=1 uv run pytest -m live -s     # real models: Ollama and/or OpenRouter
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run forge models --detect                    # inspect configured models
uv run forge models ping planner                # smoke-test a role's model
uv run forge context "add rate limiting" --show # preview the planner context
```
