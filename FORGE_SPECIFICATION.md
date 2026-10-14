# Forge — Model-Agnostic Multi-Agent Coding Harness

## 1. Vision

**Forge** is a CLI-first, model-agnostic AI coding harness designed to make any capable AI model perform substantially better at real software engineering.

The goal is **not** to build another standalone coding model.

The goal is to build the environment around an AI model that maximizes its software-engineering capability through:

- intelligent orchestration
- specialized agents
- high-quality context engineering
- powerful development tools
- iterative execution
- intensive QA
- structured feedback loops
- memory and workflow state
- Git checkpoints and recovery
- model-specific capability utilization
- safety and permission boundaries

### North-star principle

> Given the same underlying AI model, Forge should enable that model to produce better software than using the model directly.

If a user plugs in Llama, Forge should allow Llama to perform at its best.

If the user switches to GLM, Qwen, GPT, Claude, or a future frontier model, Forge should adapt and allow that model to perform at its own capability ceiling without requiring the entire system to be redesigned.

---

# 2. What Forge Is

Forge is a **software-engineering harness**, not simply a multi-agent chatbot.

The underlying model provides intelligence.

Forge provides:

```text
Context
+ Tools
+ Memory
+ Planning
+ Orchestration
+ Verification
+ Feedback
+ Recovery
+ Safety
+ Observability
```

Conceptually:

```text
                         USER
                          │
                          ▼
                 ┌───────────────────┐
                 │    FORGE CLI      │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   ORCHESTRATOR    │
                 │                   │
                 │ LLM-driven        │
                 │ Decision making   │
                 │ Workflow state    │
                 │ Routing           │
                 │ Context control   │
                 └─────────┬─────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           PLANNER      EXECUTOR        QA
              │            │            │
              └────────────┼────────────┘
                           ▼
                    AGENT RUNTIME
                           │
                    MODEL ADAPTER
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
        Llama              GLM          Frontier Model
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                         TOOLS
              ┌────────────┼─────────────┐
              ▼            ▼             ▼
         Filesystem     Terminal        Git
              │            │             │
              └────────────┼─────────────┘
                           ▼
                     User Repository
```

---

# 3. Core Philosophy

## 3.1 Model is replaceable

Forge must never be architecturally dependent on a particular model.

The same agent implementation should be able to use:

- Llama
- GLM
- Qwen
- GPT
- Claude
- Gemini
- OpenRouter models
- OmniRoute models
- local models
- future frontier models

without rewriting the agent architecture.

---

## 3.2 Never unnecessarily constrain the model

We cannot literally guarantee "100% of a model's intelligence", but Forge should aim to:

> Never unnecessarily constrain the reasoning, context, tool usage, or autonomy available to the underlying model.

Forge should expose model capabilities instead of reducing every model to:

```text
prompt → text response
```

Where supported, Forge should preserve:

- tool calling
- structured output
- streaming
- reasoning modes
- parallel tool calls
- large context windows
- vision/multimodal capabilities
- provider-specific capabilities
- model-specific inference controls

---

## 3.3 Intelligence vs authority

The LLM can make intelligent decisions.

The system controls what the LLM is actually allowed to do.

```text
LLM
 │
 │ proposed action
 ▼
Policy / Safety Gate
 │
 ├── Allowed → Execute
 ├── Approval Required → Human
 └── Forbidden → Reject
```

This is especially important for:

- destructive commands
- production changes
- secrets
- database migrations
- file deletion
- deployment
- permission changes
- security-sensitive operations

---

# 4. Agent Architecture

Forge initially contains three specialist agents.

## 4.1 Planning / Reasoning Agent

Role:

> Senior software architect and problem solver.

Responsibilities:

- understand the user requirement
- analyze the repository
- understand existing architecture
- inspect dependencies
- inspect tests
- identify constraints
- identify risks
- determine implementation strategy
- break work into tasks
- define dependencies
- define acceptance criteria
- reason about architectural choices

The Planner should not immediately write implementation code.

Its primary output is a structured plan.

Example:

```json
{
  "goal": "Add authentication",
  "tasks": [
    {
      "id": "AUTH-001",
      "description": "Create user model",
      "dependencies": []
    },
    {
      "id": "AUTH-002",
      "description": "Implement password hashing",
      "dependencies": ["AUTH-001"]
    }
  ],
  "acceptance_criteria": [
    "Users can register",
    "Passwords are never stored plaintext",
    "Unauthorized requests are rejected"
  ],
  "risks": [
    "Existing API routes may be unauthenticated"
  ]
}
```

---

# 5. Execution Agent

Role:

> Senior implementation engineer.

Responsibilities:

- receive an approved task
- investigate relevant code
- follow the approved plan
- use repository tools
- modify code
- run tests
- inspect failures
- debug implementation issues
- verify changes
- report exactly what changed and why

The Executor should operate iteratively:

```text
Reason
  ↓
Investigate
  ↓
Tool
  ↓
Observe
  ↓
Reason
  ↓
Tool
  ↓
Test
  ↓
Fix
  ↓
Verify
```

The Executor must not consider a task complete simply because code was written.

---

# 6. QA Agent

Role:

> Adversarial software tester and security reviewer.

The QA agent should assume the implementation may be wrong until it can prove otherwise.

It must validate against the requirements, not merely inspect whether the code "looks good."

QA responsibilities:

### Requirement validation

- verify every acceptance criterion
- verify intended user behavior
- detect missing functionality

### Functional testing

- happy paths
- invalid inputs
- boundary cases
- failure cases
- unexpected states

### Integration testing

- API interactions
- database interactions
- frontend/backend interactions
- external service interactions

### Regression testing

- ensure existing functionality still works
- run relevant existing tests
- identify unintended behavior changes

### Security testing

- secrets exposure
- authentication bypass
- authorization bypass
- injection
- unsafe input handling
- command execution risks
- insecure defaults
- permission problems
- dependency vulnerabilities where applicable

### Quality

- error handling
- maintainability
- architectural consistency
- performance issues
- race/concurrency issues where relevant

QA produces structured findings.

Example:

```json
{
  "status": "FAIL",
  "findings": [
    {
      "id": "QA-017",
      "severity": "P1",
      "type": "REQUIREMENT_VIOLATION",
      "description": "Users can access another user's project",
      "evidence": "...",
      "affected_files": [
        "backend/routes/projects.py"
      ],
      "required_rework": [
        "Add ownership validation",
        "Add authorization tests"
      ]
    }
  ]
}
```

---

# 7. Orchestrator

The Orchestrator is the control plane of Forge.

It is intentionally **LLM-driven**, because real software-engineering workflows cannot realistically be encoded entirely as deterministic `if/else` rules.

The Orchestrator answers:

> Given the current state, requirements, agent outputs, failures, history, and constraints, what should happen next?

Possible actions include:

```text
PLAN
REPLAN
EXECUTE
REEXECUTE
QA
TARGETED_QA
FULL_QA
RETRY
ROLLBACK
REQUEST_RESEARCH
REQUEST_HUMAN_REVIEW
BLOCK
COMPLETE
```

The Orchestrator can dynamically choose workflows.

For example:

```text
Planner → Executor → QA → Complete
```

or:

```text
Planner → Executor → QA → Planner → Executor → QA
```

or:

```text
Planner → Planner
```

if the plan itself is incoherent.

Or:

```text
QA → Executor → Targeted QA → Full QA
```

for a localized issue.

The Orchestrator should not itself become a fourth implementation agent.

It controls the workflow.

---

# 8. LLM Orchestrator + Deterministic Guardrails

The architecture should be hybrid.

```text
                    LLM ORCHESTRATOR
                           │
                           ▼
                    Proposed Action
                           │
                           ▼
                  POLICY / SAFETY GATE
                    /             \
                 allow            reject
                   │                 │
                   ▼                 ▼
                execute           stop/review
```

The LLM provides:

- contextual reasoning
- prioritization
- routing
- strategy selection
- failure interpretation
- workflow decisions

Deterministic systems enforce:

- permissions
- destructive-operation restrictions
- security requirements
- iteration limits
- secret protection
- production restrictions
- approval requirements
- sandbox boundaries
- tool permissions
- termination rules

### Principle

> The LLM has intelligence. The system has authority.

---

# 9. Requirement Contract

Every workflow should begin by turning the user request into a requirement contract.

```text
Requirement Contract

Functional Requirements
- FR-001
- FR-002
- FR-003

Non-functional Requirements
- NFR-001 Security
- NFR-002 Performance
- NFR-003 Reliability

Acceptance Criteria
- AC-001
- AC-002
- AC-003
```

The requirement contract becomes the shared source of truth for:

```text
Planner
Executor
QA
Orchestrator
```

The fundamental QA question becomes:

> Can we prove every acceptance criterion is satisfied?

---

# 10. Shared Workflow State

Agents should not communicate primarily through free-form conversation.

Forge maintains structured shared state.

```text
WorkflowState
│
├── workflow_id
├── user_request
├── status
├── iteration
├── requirements
├── repository_snapshot
├── architecture
├── plan
├── tasks
├── current_task
├── execution_results
├── test_results
├── qa_reports
├── decisions
├── errors
└── checkpoints
```

Agents are mostly stateless workers.

The Orchestrator owns the workflow truth.

---

# 11. Model Runtime

The Model Runtime is a core part of Forge.

It must be provider-agnostic and capability-aware.

Core abstractions:

```text
ModelAdapter
ModelProvider
ModelRegistry
ModelProfile
ModelCapabilities
```

A model should expose capabilities such as:

```text
context_window
tool_calling
structured_output
parallel_tool_calls
reasoning
streaming
vision
max_output_tokens
provider_specific_features
```

Forge should avoid reducing all models to a minimal common denominator.

Instead:

```text
                 MODEL ABSTRACTION
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   Common API      Capability API    Native API
        │               │                │
   generate()      tool_call()      provider features
   stream()        reasoning()
   structured()    vision()
```

This provides portability without throwing away model-specific capabilities.

---

# 12. Model Profiles

Forge should support model profiles.

Example:

```yaml
model:
  provider: openrouter
  name: example-model

capabilities:
  tool_calling: true
  structured_output: true
  reasoning: true
  parallel_tools: true
  vision: false

optimization:
  context_strategy: targeted
  tool_batching: adaptive
```

The same underlying model may have different runtime configurations depending on its role:

```text
Llama
 ├── Planner profile
 ├── Executor profile
 └── QA profile
```

This allows Forge to exploit model strengths rather than treating every model identically.

---

# 13. Context Engineering

Context engineering is one of Forge's biggest potential advantages.

Never blindly send the entire repository to every agent.

Instead create role-specific contexts.

## Planner Context

```text
User requirement
+
Repository structure
+
Architecture
+
Relevant source
+
Dependencies
+
Tests
+
Configuration
+
Git state
```

## Executor Context

```text
Current task
+
Relevant files
+
Plan
+
Acceptance criteria
+
Previous errors
+
Current git diff
```

## QA Context

```text
Requirements
+
Acceptance criteria
+
Changed files
+
Git diff
+
Test results
+
Security-sensitive areas
+
Previous findings
```

Context should be dynamically constructed.

Large context windows should be used when useful rather than aggressively summarizing everything.

---

# 14. Tool Runtime

Agents interact with repositories through structured tools.

Core categories:

## Filesystem

```text
read_file
write_file
edit_file
delete_file
list_directory
search
```

## Terminal

```text
run_command
run_tests
process_management
```

## Git

```text
status
diff
log
checkpoint
rollback
branch
```

## Analysis

```text
repository_structure
symbol_search
dependency_analysis
```

Future tools:

```text
browser
documentation_search
security_scanner
dependency_scanner
database_tools
container_tools
```

Every tool should define:

```text
input schema
output schema
risk level
permission level
timeout
audit metadata
```

---

# 15. Git and Recovery

Git should be an important safety mechanism.

Workflow:

```text
Checkpoint
    ↓
Executor
    ↓
Changes
    ↓
QA
    ↓
Pass / Rework / Rollback
```

Forge should maintain checkpoints before important operations.

Recovery capabilities:

```text
checkpoint
rollback
diff
restore
iteration history
```

---

# 16. Iteration and Convergence

Autonomous loops must have hard boundaries.

Example:

```text
MAX_ITERATIONS = 10
```

But iteration count alone is insufficient.

Track:

```text
iteration count
same failure count
time elapsed
token usage
cost
files changed
progress
```

Detect agent thrashing.

Example:

```text
Iteration 1 → QA failure
Iteration 2 → QA failure
Iteration 3 → same failure
Iteration 4 → same failure
```

Then:

```text
BLOCKED
```

and request human intervention.

Also detect:

- repeated fixes
- oscillating fixes
- same failure signatures
- no measurable progress

---

# 17. Human Approval

Forge should support human gates for risky actions.

### Low risk

Automatically execute.

### Medium risk

Execute with logging.

### High risk

Require approval.

Examples:

```text
database migration
production configuration
destructive command
large file deletion
authentication architecture change
deployment
secret-related operation
```

The user should be able to approve from the CLI initially.

A future web UI can provide remote approval.

---

# 18. Event System

Forge should record important workflow events.

Examples:

```text
WorkflowStarted
RepositoryAnalyzed
PlanCreated
PlanApproved
TaskStarted
ToolCalled
FileModified
TestStarted
TestCompleted
TaskCompleted
QAStarted
QAFindingCreated
ReworkRequested
PlanRevised
RollbackPerformed
WorkflowCompleted
WorkflowBlocked
```

This provides:

- debugging
- observability
- workflow history
- reproducibility
- future web UI support
- analytics
- benchmarking

---

# 19. CLI-First Product Strategy

Forge will initially be CLI-first.

Example:

```bash
forge "Add OAuth authentication"
```

Possible commands:

```bash
forge init
forge run "<request>"
forge status
forge history
forge diff
forge resume
forge approve
forge rollback
forge models
forge config
forge benchmark
```

The CLI is only the first interface.

The core engine must remain headless.

Future interfaces can include:

```text
CLI
Web dashboard
VS Code extension
Desktop application
Remote control interface
```

All should use the same core engine.

---

# 20. Future Web Architecture

The future web UI should not replace the engine.

It should control and observe it.

```text
                 Local Forge Engine
                        │
                  API / WebSocket
                        │
                        ▼
                 Web Dashboard
```

The web UI could provide:

- workflow monitoring
- agent traces
- diffs
- approvals
- logs
- model selection
- token/cost analytics
- QA reports
- benchmark results
- workflow history

This is intentionally a later phase.

---

# 21. Benchmarking

Forge must eventually prove its core thesis.

Benchmark:

```text
Same model + direct usage
```

against:

```text
Same model + Forge
```

Measure:

```text
task success
requirement satisfaction
tests passed
security issues
regression rate
iterations
tokens
latency
cost
human interventions
```

Example:

```text
Model        Direct     Forge    Improvement
Llama        61%        79%      +18%
GLM          67%        84%      +17%
Qwen         64%        81%      +17%
Frontier     82%        91%       +9%
```

The numbers above are only illustrative.

The important objective is:

> Forge should improve the engineering outcome of the same underlying model.

---

# 22. Development Roadmap

## Phase 0 — Architecture & Repository Foundation

Create:

```text
forge/
├── pyproject.toml
├── README.md
├── src/
│   └── forge/
│       ├── cli/
│       ├── core/
│       ├── agents/
│       ├── models/
│       ├── tools/
│       ├── context/
│       ├── workflow/
│       ├── policies/
│       ├── events/
│       └── storage/
└── tests/
```

Establish:

- Python project
- async-first architecture
- Pydantic contracts
- testing
- logging
- configuration
- clean package boundaries

**Do not implement agents yet.**

---

## Phase 1 — Model Runtime

Implement:

```text
ModelProvider
ModelAdapter
ModelRegistry
ModelProfile
ModelCapabilities
```

Support initially one OpenAI-compatible provider and keep the abstraction provider-independent.

Support:

- normal generation
- streaming
- structured output
- tool calling
- model metadata
- usage tracking
- capability detection/configuration

**Exit condition:** two different models can be swapped without changing agent code.

---

## Phase 2 — Tool Runtime

Implement:

- filesystem tools
- terminal tools
- Git tools
- repository analysis tools

Add:

- schemas
- permission levels
- risk levels
- timeouts
- audit logging

**Exit condition:** a model can safely inspect and modify a test repository.

---

## Phase 3 — Context Engine

Implement:

```text
Repository Analyzer
Context Builder
Context Selector
Context Budget Manager
```

Create role-specific context:

```text
PlannerContext
ExecutorContext
QAContext
```

**Exit condition:** each agent can receive targeted repository context.

---

## Phase 4 — Specialist Agents

Implement:

```text
PlannerAgent
ExecutorAgent
QAAgent
```

Each must use structured input/output contracts.

**Exit condition:** each agent can independently perform its role on a test project.

---

## Phase 5 — LLM Orchestrator

Implement:

```text
WorkflowState
Orchestrator
Decision schema
Agent router
Policy validation
```

The Orchestrator decides:

```text
PLAN
REPLAN
EXECUTE
QA
TARGETED_QA
FULL_QA
RETRY
ROLLBACK
HUMAN_REVIEW
BLOCK
COMPLETE
```

**Exit condition:** a full workflow can run without hardcoding every scenario.

---

## Phase 6 — Autonomous Workflow & Recovery

Implement:

- Planner → Executor → QA loop
- rework
- checkpoints
- rollback
- iteration limits
- convergence detection
- failure signatures
- agent thrashing detection
- human approval
- workflow persistence

**Exit condition:** Forge can autonomously complete a non-trivial coding task and recover from QA failures.

---

## Phase 7 — CLI

Implement:

```bash
forge init
forge run
forge status
forge history
forge diff
forge resume
forge approve
forge rollback
forge models
forge config
```

Build a high-quality terminal UX.

**Exit condition:** Forge is usable as a real local coding agent.

---

## Phase 8 — Dogfooding

Use Forge to modify Forge itself.

Start with small tasks:

```text
bug fixes
new tools
new tests
CLI improvements
model adapters
```

Measure:

- success
- failures
- QA effectiveness
- orchestration quality
- recovery quality

**Exit condition:** Forge can reliably contribute to its own development.

---

## Phase 9 — Benchmarking

Create benchmark tasks covering:

```text
bug fixing
feature implementation
refactoring
debugging
security
requirements
integration
```

Compare:

```text
Direct model
vs
Model + Forge
```

Run across multiple models.

**Exit condition:** measurable evidence that the harness improves model performance.

---

## Phase 10 — Web Control Plane

Only after the CLI engine is mature.

Add:

```text
API
WebSocket
Workflow dashboard
Approval interface
Diff viewer
Agent traces
Benchmark dashboard
```

The web layer must remain a client of the existing engine.

---

# 23. Recommended Initial Technology Stack

### Core

```text
Python
asyncio
Pydantic
```

### CLI

Choose a mature Python CLI framework such as:

```text
Typer
```

or an equivalent lightweight CLI framework.

### Storage

Start simple:

```text
SQLite
```

Use structured workflow/event storage.

Do not introduce PostgreSQL until the architecture requires it.

### Git

Use the system Git executable or a robust Git library.

### Testing

```text
pytest
pytest-asyncio
```

### Configuration

Environment variables + project configuration file.

Example:

```text
forge.yaml
```

### Model providers

Start with an OpenAI-compatible interface, then add additional adapters as needed.

---

# 24. Important Engineering Rules

1. **Do not couple agents to providers.**
2. **Do not couple the CLI to the core engine.**
3. **Do not make the Orchestrator directly modify files.**
4. **Do not allow agents to communicate only through natural language.**
5. **Use structured contracts between components.**
6. **Do not dump the entire repository into every model call.**
7. **Do not blindly normalize away model-specific capabilities.**
8. **Do not let the LLM bypass safety policies.**
9. **Do not allow infinite agent loops.**
10. **Every autonomous change must be observable.**
11. **Every important operation should be recoverable.**
12. **Requirement satisfaction matters more than superficial code quality.**
13. **QA should be adversarial.**
14. **Use Git checkpoints for recovery.**
15. **Benchmark the harness against direct model usage.**
16. **Implement one phase at a time and test before moving forward.**
17. **Prefer simple architecture over premature abstractions.**
18. **The engine must remain headless and interface-independent.**

---

# 25. Definition of Success

Forge succeeds if this becomes true:

```text
             SAME MODEL
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 Direct Usage            Forge
       │                   │
       ▼                   ▼
   Software             Software
    output                output
       │                   │
       ▼                   ▼
    baseline          consistently
                      better outcome
```

The objective is not to claim that Forge is smarter than the model.

The objective is to build an environment in which:

> **The model can reason, investigate, implement, test, receive feedback, recover from mistakes, and continue working with minimal unnecessary friction.**

That is the core product.

---

# 26. Immediate Next Step

Do not ask a coding agent to build the entire Forge system yet.

The first coding-agent task should be:

> Analyze this specification and design Phase 0 in detail. Define the repository structure, core interfaces, data contracts, dependency choices, testing strategy, and implementation sequence. Do not implement Phases 1–10.

After reviewing and approving that architecture:

```text
Phase 0
  ↓
test
  ↓
Phase 1
  ↓
test
  ↓
Phase 2
  ↓
test
  ↓
...
```

This keeps the development agent from prematurely inventing the architecture and gives Forge a controlled path from a clean foundation to a genuinely autonomous coding harness.
