"""Adaptive 3-Tier Intent Router & ReAct execution engine for the Forge workspace.

Every developer input is classified *before* any manufacturing stage runs:

* ``ExecutionMode.CONVERSATIONAL``  - greetings, questions, brainstorming. Streams a real
  multi-turn LLM reply into the current stage (usually Anvil) and never touches Hammer or
  Temper, never increments the Cycle counter.
* ``ExecutionMode.DIRECT_TOOL``     - direct operational requests ("run the program",
  "run pytest", ``!git status``). The active agent runs a real ReAct tool loop (read_file,
  list_directory, run_command, ...) and streams live stdout/stderr into the viewport.
* ``ExecutionMode.FULL_FORGE_CYCLE`` - complex engineering tasks. ``Anvil`` creates the plan,
  ``Hammer`` executes, ``Temper`` adversarially verifies, looping until it passes.
* ``ExecutionMode.DEEP_RESEARCH``   - ``/research <topic>`` deep research mode.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from forge.core.logging import get_logger
from forge.harness.adversarial_qa import AdversarialQAGatekeeper
from forge.harness.memory import EpisodicRepoMemory
from forge.models import Message
from forge.models.profile import ModelProfile
from forge.skills import AutonomousSkillRouter
from forge.ui.state import (
    AnvilPlanEvaluation,
    CycleMode,
    ForgeStage,
    ForgeTUIState,
    PendingPlan,
    PendingPlanContract,
    PlanRequirement,
    PlanTurnIntent,
    RequirementItem,
    UserQuestion,
    format_stage_deliverable,
    format_transient_action,
)

PlanIntent = PlanTurnIntent

# A complete project implementation needs more than a handful of ReAct steps: scaffolding,
# writing each file, installing dependencies, and fixing build failures all consume calls.
_CYCLE_HAMMER_TOOL_STEPS = 50
_CYCLE_TEMPER_TOOL_STEPS = 24

log = get_logger("ui.cycle_controller")


def _diff_stats(diff: str) -> tuple[int, int]:
    """Count added and removed lines in a unified diff, ignoring file headers."""
    added = removed = 0
    for line in (diff or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _tail_text(text: str, *, max_lines: int, max_chars: int) -> str:
    """Keep the useful end of command output without flooding the viewport."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if len(lines) > max_lines:
        removed = len(lines) - max_lines
        lines = [f"... [{removed} earlier lines omitted] ...", *lines[-max_lines:]]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = f"... [{len(clipped) - max_chars} earlier characters omitted] ...\n" + (
            clipped[-max_chars:]
        )
    return clipped

__all__ = [
    "AnvilPlanEvaluation",
    "CycleVerdict",
    "ExecutionMode",
    "ForgeCycleController",
    "PendingPlanContract",
    "PlanIntent",
    "PlanTurnIntent",
    "RequirementItem",
    "apply_requirement_audit",
    "classify_intent",
    "classify_plan_response",
    "extract_plan_requirements",
    "extract_target_files",
    "is_conversational_turn",
    "parse_plan_evaluation",
]

if TYPE_CHECKING:  # pragma: no cover
    from forge.config.settings import ForgeConfigManager, StageModelBinding
    from forge.providers.client import UnifiedLLMClient


class ExecutionMode(StrEnum):
    """How a single developer input should be executed."""

    CONVERSATIONAL = "conversational"
    DIRECT_TOOL = "direct_tool"
    FULL_FORGE_CYCLE = "full_forge_cycle"
    DEEP_RESEARCH = "deep_research"


# Stage <-> role aliases used by `/models` to route a binding to the right agent.
_STAGE_ALIASES: dict[str, ForgeStage] = {
    "anvil": ForgeStage.ANVIL,
    "orchestrator": ForgeStage.ANVIL,
    "orch": ForgeStage.ANVIL,
    "hammer": ForgeStage.HAMMER,
    "executor": ForgeStage.HAMMER,
    "exec": ForgeStage.HAMMER,
    "temper": ForgeStage.TEMPER,
    "qa": ForgeStage.TEMPER,
    "verifier": ForgeStage.TEMPER,
}

_STAGE_ROLES: dict[ForgeStage, str] = {
    ForgeStage.ANVIL: "orchestrator",
    ForgeStage.HAMMER: "executor",
    ForgeStage.TEMPER: "qa",
}

_STAGE_BY_NAME: dict[str, ForgeStage] = {stage.value.lower(): stage for stage in ForgeStage}


def _mode_label(mode: ExecutionMode | None) -> str:
    """Human-readable label for the forced execution mode (or ``AUTO``)."""
    if mode is ExecutionMode.CONVERSATIONAL:
        return "CHAT"
    if mode is ExecutionMode.FULL_FORGE_CYCLE:
        return "CYCLE"
    return "AUTO"


# -- Deterministic zero-latency intent pre-filter -----------------------------------------

_GREETING_VERBS = {"hi", "hello", "hey", "greetings", "howdy", "yo"}

_FULL_CYCLE_LEADING = {
    "implement",
    "build",
    "create",
    "refactor",
    "rewrite",
    "rearchitect",
    "overhaul",
    "scaffold",
    "migrate",
    "integrate",
    "develop",
    "generate",
    "design",
    "architect",
    "port",
    "modernize",
}

_EDIT_LEADING = {
    "add",
    "update",
    "remove",
    "delete",
    "rename",
    "move",
    "change",
    "tweak",
    "edit",
    "set",
    "patch",
    "fix",
    "debug",
    "resolve",
    "diagnose",
    "upgrade",
    "bump",
}

_DIRECT_LEADING = {
    "run",
    "execute",
    "list",
    "ls",
    "cat",
    "read",
    "show",
    "open",
    "display",
    "check",
    "status",
    "git",
    "pytest",
    "grep",
    "find",
    "search",
    "diff",
    "log",
    "install",
    "test",
    "python",
    "python3",
    "py",
    "node",
    "npm",
    "pip",
    "uv",
    "cargo",
    "go",
    "make",
    "curl",
    "echo",
    "print",
    "wc",
    "head",
    "tail",
    "type",
    "where",
    "which",
    "pwd",
    "cd",
    "dir",
    "tree",
}

_COMPLEX_CONTEXT_RE = re.compile(
    r"\b(module|class|feature|service|api|endpoint|database|db|schema|migration|"
    r"test|tests|unit test|integration|refactor|regression|multi-file|package|library|"
    r"component|system|pipeline|architecture|authentication|auth|engine|handler|router|"
    r"controller|worker|queue|cache|rate ?limit|algorithm|protocol|framework|interface)\b",
    re.IGNORECASE,
)

_COMPLEX_DEBUG_RE = re.compile(
    r"\b(debug|fix|resolve|diagnose)\b.*\b(bug|error|failure|failing|test|issue|"
    r"regression|crash|exception|traceback|leak|race)\b",
    re.IGNORECASE,
)

_DIRECT_QUERY_RE = re.compile(
    r"\b(what files|which files|list files|list the files|show files|show me the|"
    r"show the|read the|read file|check the|check git|git status|git log|git diff|"
    r"how many files|who owns|where is)\b",
    re.IGNORECASE,
)

_QUESTION_LEADING_RE = re.compile(
    r"^(what|why|how|who|where|when|which|should|explain|describe|compare|"
    r"tell me|walk me through|what's|whats|how's|hows)\b",
    re.IGNORECASE,
)


def is_conversational_turn(prompt: str) -> bool:
    """Return True for obvious greetings / one-word conversational inquiries."""
    p_clean = prompt.strip().lower().rstrip(".!?,;: ").strip()
    if not p_clean:
        return True

    conversational_exact = {
        "hi",
        "hello",
        "hey",
        "howdy",
        "yo",
        "sup",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "how are you",
        "how are you doing",
        "how's it going",
        "hows it going",
        "whats up",
        "what's up",
        "who are you",
        "what are you",
        "what is forge",
        "what can you do",
        "help",
        "info",
        "thanks",
        "thank you",
        "thx",
        "cool",
        "awesome",
        "great",
        "nice",
        "bye",
        "goodbye",
        "see you",
    }
    if p_clean in conversational_exact:
        return True

    words = p_clean.split()
    if len(words) <= 3 and words[0] in _GREETING_VERBS:
        action_indicators = {
            "build",
            "create",
            "fix",
            "implement",
            "refactor",
            "test",
            "run",
            "add",
            "update",
            "delete",
            "remove",
            "write",
            "debug",
            "make",
            "patch",
            "deploy",
            "install",
            "git",
            "pr",
            "commit",
        }
        if not any(w in action_indicators for w in words):
            return True

    return False


def _planner_chat_prompt() -> str:
    """Return Anvil's plan-only prompt without introducing an import cycle."""
    from forge.agents.prompts import PLANNER_CHAT_SYSTEM_PROMPT

    return PLANNER_CHAT_SYSTEM_PROMPT


def classify_intent(prompt: str, *, forced: ExecutionMode | None = None) -> ExecutionMode:
    """Deterministically classify a prompt with zero LLM round-trips.

    Explicit slash / shell syntaxes short-circuit; otherwise the leading verb plus the
    surrounding engineering context decides whether this is chat, a quick tool action, or
    a task that genuinely needs the full Anvil -> Hammer -> Temper cycle.
    """
    if forced is not None:
        return forced

    raw = prompt.strip()
    if not raw:
        return ExecutionMode.CONVERSATIONAL
    low = raw.lower()

    if low.startswith("/research") or low.startswith("--research"):
        return ExecutionMode.DEEP_RESEARCH
    if low.startswith("/cycle"):
        return ExecutionMode.FULL_FORGE_CYCLE
    if raw.startswith("!"):
        return ExecutionMode.DIRECT_TOOL
    if raw.startswith("/"):
        # Other slash commands are handled by the command dispatcher, not the router.
        return ExecutionMode.CONVERSATIONAL

    if is_conversational_turn(raw):
        return ExecutionMode.CONVERSATIONAL

    words = re.split(r"\W+", low)
    leading = words[0] if words else ""

    if leading in _FULL_CYCLE_LEADING:
        return ExecutionMode.FULL_FORGE_CYCLE
    if leading in _EDIT_LEADING:
        if _COMPLEX_CONTEXT_RE.search(raw):
            return ExecutionMode.FULL_FORGE_CYCLE
        return ExecutionMode.DIRECT_TOOL
    if leading in _DIRECT_LEADING or _DIRECT_QUERY_RE.search(raw):
        return ExecutionMode.DIRECT_TOOL
    if _COMPLEX_DEBUG_RE.search(raw):
        return ExecutionMode.FULL_FORGE_CYCLE
    if _QUESTION_LEADING_RE.search(raw):
        return ExecutionMode.CONVERSATIONAL

    # Default to the planning cycle. A request that is not a greeting, a direct shell
    # query, or an explicit question must produce an Anvil plan before implementation.
    return ExecutionMode.FULL_FORGE_CYCLE


# Provider API-key environment variables surfaced in the offline diagnostic panel. Forge
# NEVER fabricates an assistant reply: if no live model is reachable we explain exactly what
# to configure instead of returning a canned sentence.
_API_KEY_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("OPENROUTER_API_KEY", "OpenRouter (openrouter.ai)"),
    ("OPENAI_API_KEY", "OpenAI (api.openai.com)"),
    ("GEMINI_API_KEY", "Google Gemini"),
    ("ANTHROPIC_API_KEY", "Anthropic Claude"),
    ("NVIDIA_API_KEY", "NVIDIA NIM"),
)


def model_diagnostic(reason: str = "No live LLM model is attached to this session.") -> str:
    """Actionable diagnostic shown instead of a fabricated reply when no model is reachable."""
    lines = [
        f"[WARN] {reason}",
        "",
        "Forge does not fake model output — every conversational turn must reach a real LLM.",
        "Configure a provider API key, then restart Forge (or run `forge login`):",
    ]
    lines.extend(f"  - {env}  ->  {label}" for env, label in _API_KEY_ENV_VARS)
    lines.append(
        "  - OLLAMA_HOST (optional) for a local Ollama server "
        "(default http://localhost:11434)"
    )
    lines.append("")
    lines.append("Or connect a provider and choose models at any time:")
    lines.append("  /provider                   open the Provider Connection Manager")
    lines.append("  /models                     assign models to Anvil, Hammer, Temper")
    return "\n".join(lines)


def reasoning_only_diagnostic(binding: StageModelBinding, *, truncated: bool = False) -> str:
    """Clear diagnostic for a model that spent its output budget on hidden reasoning."""
    cause = (
        "The output token limit was reached inside a hidden reasoning block."
        if truncated
        else "The model returned hidden reasoning but no final answer."
    )
    return "\n".join(
        [
            f"[WARN] {cause}",
            "",
            "Forge discarded the internal chain-of-thought instead of displaying it as a reply.",
            f"  Model: {binding.spec_string}",
            "  Remedy: increase max_tokens, lower reasoning effort, or choose a non-thinking model.",
            "  Then retry the request.",
        ]
    )


_REQ_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:REQ[-_ ]?(\d+)|(\d+)[.)])\s*[:\-]?\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TARGET_FILES_RE = re.compile(
    r"^\s*TARGET[_ ]FILES\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)


def extract_target_files(plan_text: str) -> list[str]:
    """Extract Anvil's declared target-file list from the plan summary."""
    match = _TARGET_FILES_RE.search(plan_text)
    if match is None:
        return []
    raw = match.group(1).strip().strip("`")
    if not raw or raw.upper() in {"NONE", "N/A", "UNKNOWN"}:
        return []
    files: list[str] = []
    for candidate in re.split(r"[,;]", raw):
        path = candidate.strip().strip("`'\"")
        if path and path not in files:
            files.append(path)
    return files


def parse_plan_evaluation(raw_text: str) -> AnvilPlanEvaluation:
    """Parse Anvil's structured JSON adjudication without keyword heuristics."""
    from forge.providers.client import strip_thinking_tags

    text = strip_thinking_tags(raw_text or "")[0].strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Anvil returned no JSON plan evaluation")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Anvil plan evaluation must be a JSON object")
    return AnvilPlanEvaluation.model_validate(payload)


def classify_plan_response(response: str) -> PlanTurnIntent:
    """Deprecated synchronous shim; plan replies require live Anvil adjudication."""
    raise RuntimeError(
        "classify_plan_response() cannot decide approval keywords; use the controller's "
        "AnvilPlanEvaluation flow instead."
    )


def extract_plan_requirements(plan_text: str, request: str) -> list[PlanRequirement]:
    """Extract a stable ``REQ-1..REQ-N`` contract from a model-authored plan."""
    explicit: list[str] = []
    numbered: list[str] = []
    for match in _REQ_LINE_RE.finditer(plan_text):
        explicit_id, number, text = match.groups()
        cleaned = text.strip().strip("-*_").strip()
        if not cleaned:
            continue
        if explicit_id is not None:
            explicit.append(cleaned)
        elif number is not None:
            numbered.append(cleaned)

    candidates = explicit or numbered
    requirements: list[PlanRequirement] = []
    for index, text in enumerate(candidates, start=1):
        requirements.append(PlanRequirement(req_id=f"REQ-{index}", text=text))
    if requirements:
        return requirements

    fallback = request.strip().splitlines()[0][:240] if request.strip() else "Requested change"
    return [PlanRequirement(req_id="REQ-1", text=fallback)]


def apply_requirement_audit(plan: PendingPlan, audit_text: str) -> bool:
    """Update the plan checklist from Temper's structured audit and return completion."""
    lines = audit_text.splitlines()
    for requirement in plan.requirements:
        token = requirement.req_id
        matched = next((line for line in lines if token in line), "")
        if not matched:
            requirement.met = False
            requirement.evidence = ""
            continue
        lowered = matched.lower()
        failed = (
            "✗" in matched
            or re.search(r"\[(?:fail|x| )\]", lowered) is not None
            or re.search(
                r"\b(fail|failed|missing|unmet|incomplete|pending|partial)\b", lowered
            )
            is not None
            or re.search(r"\bnot\s+(?:met|verified|complete|completed|passing|passed)\b", lowered)
            is not None
        )
        passed = (
            "✓" in matched
            or re.search(r"\[(?:pass|done|✓|x|yes)\]", lowered) is not None
            or re.search(r"\b(pass|passed|met|verified|complete|completed)\b", lowered) is not None
        )
        requirement.met = passed and not failed
        evidence = re.sub(
            r"^\s*[-*]?\s*" + re.escape(token) + r"\s*[:：-]?\s*",
            "",
            matched,
            flags=re.IGNORECASE,
        )
        evidence = re.sub(r"^\s*(?:\[[ xX✓✗]\]|[✓✗]|PASS|FAIL)\b\s*", "", evidence)
        requirement.evidence = evidence.strip()[:180]

    explicit = re.search(
        r"ALL_REQUIREMENTS_MET\s*[:=]\s*(TRUE|FALSE|YES|NO)", audit_text, re.IGNORECASE
    )
    if explicit:
        return explicit.group(1).upper() in {"TRUE", "YES"}
    if any(req.met for req in plan.requirements):
        return all(req.met for req in plan.requirements)
    return bool(
        re.search(r"\b(PASS|APPROVED|VERIFIED|ALL REQUIREMENTS MET)\b", audit_text, re.I)
    ) and not re.search(r"\b(FAIL|FAILED|REJECTED)\b", audit_text, re.I)


@dataclass
class CycleVerdict:
    """Verdict result from Temper (QA Gatekeeper stage)."""

    passed: bool
    feedback: str = ""
    defects: list[str] = field(default_factory=list)
    deliverable: str = ""


class ForgeCycleController:
    """Adaptive intent router driving conversational, direct-tool & full-cycle execution."""

    def __init__(
        self,
        state: ForgeTUIState,
        router: Any = None,
        harness: Any = None,
        max_cycles: int = 25,
        temper_verifier: Callable[[int, str], CycleVerdict] | None = None,
        cycle_mode: CycleMode = CycleMode.UNTIL_FULFILLED,
        question_handler: Callable[[UserQuestion], Awaitable[str]] | None = None,
        *,
        agents: Mapping[ForgeStage, Any] | None = None,
        runtime: Any = None,
        registry: Any = None,
        root: Path | None = None,
        project_context: str = "",
        workflow_runner: Callable[[str], Awaitable[Any]] | None = None,
        research_runner: Callable[[str], Awaitable[str]] | None = None,
        intent_classifier: Callable[[str], ExecutionMode] | None = None,
        max_tool_steps: int = 12,
        provider_configurer: Callable[[], Awaitable[str]] | None = None,
        model_picker: Callable[[str | None], Awaitable[str]] | None = None,
        config_manager: ForgeConfigManager | None = None,
        llm_client: UnifiedLLMClient | None = None,
    ) -> None:
        self.state = state
        self.router = router
        self.harness = harness
        self.max_cycles = max_cycles
        self.temper_verifier = temper_verifier
        self.cycle_mode = cycle_mode
        self.state.cycle_mode = cycle_mode
        self.question_handler = question_handler

        self.agents: dict[ForgeStage, Any] = dict(agents or {})
        self.runtime = runtime
        self.registry = registry
        self.root = root
        self.project_context = project_context
        self.workflow_runner = workflow_runner
        self.research_runner = research_runner
        self.intent_classifier = intent_classifier
        self.max_tool_steps = max_tool_steps
        # UI-injected hooks for the interactive provider/model wizards (set by the split-pane app).
        self.provider_configurer = provider_configurer
        self.model_picker = model_picker
        self.session_resumer: Callable[[], Awaitable[Any]] | None = None
        self.usage_viewer: Callable[[], Awaitable[Any]] | None = None
        self.session_id: str = uuid.uuid4().hex[:12]
        # Persistent provider/model configuration engine + native streaming LLM client.
        self.config_manager = config_manager
        self.llm_client = llm_client

        self.skill_router = AutonomousSkillRouter()
        self.qa_gatekeeper = AdversarialQAGatekeeper()
        self.memory = EpisodicRepoMemory()

        # Canonical provider-neutral Message history per stage (fed to the real model) plus a
        # lightweight role/content transcript used for introspection and the live viewport.
        self._history: dict[ForgeStage, list[Message]] = {
            ForgeStage.ANVIL: [],
            ForgeStage.HAMMER: [],
            ForgeStage.TEMPER: [],
        }
        self.conversation_history: dict[ForgeStage, list[dict[str, str]]] = {
            ForgeStage.ANVIL: [],
            ForgeStage.HAMMER: [],
            ForgeStage.TEMPER: [],
        }
        self.forced_mode: ExecutionMode | None = None
        self.last_mode: ExecutionMode = ExecutionMode.CONVERSATIONAL
        self._current_task: asyncio.Task[Any] | None = None
        self._cancelled: bool = False

    # -- public API --------------------------------------------------------------------

    def set_cycle_mode(self, mode: CycleMode) -> None:
        """Dynamically set the active cycle execution policy."""
        self.cycle_mode = mode
        self.state.set_cycle_mode(mode)

    @property
    def is_running(self) -> bool:
        """Return True if a cycle execution loop is currently running."""
        return self._current_task is not None and not self._current_task.done()

    def cancel(self) -> None:
        """Cancel the active cycle execution loop."""
        self._cancelled = True
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    def set_execution_mode(self, name: str) -> str:
        """Set the router execution mode: ``auto`` (adaptive), ``chat``, or ``cycle``."""
        normalized = name.strip().lower()
        if normalized in {"auto", "adaptive", "default"}:
            self.forced_mode = None
            return "Execution mode: AUTO (adaptive routing)"
        if normalized in {"chat", "conversation", "talk"}:
            self.forced_mode = ExecutionMode.CONVERSATIONAL
            return "Execution mode: CHAT (stay conversational — no automatic cycles)"
        if normalized in {"cycle", "forge", "build"}:
            self.forced_mode = ExecutionMode.FULL_FORGE_CYCLE
            return "Execution mode: CYCLE (force Anvil -> Hammer -> Temper)"
        return f"Unknown mode: {name!r}. Use one of: auto | chat | cycle"

    def classify(self, prompt: str, target_stage: ForgeStage | None = None) -> ExecutionMode:
        """Classify a prompt, honoring an explicitly pinned stage and any forced mode."""
        if self.intent_classifier is not None:
            return self.intent_classifier(prompt)
        if self.forced_mode is not None:
            return self.forced_mode
        if target_stage is not None and target_stage != ForgeStage.ANVIL:
            # The developer selected a specific stage: talk directly to that agent.
            base = classify_intent(prompt)
            if base == ExecutionMode.FULL_FORGE_CYCLE:
                return ExecutionMode.DIRECT_TOOL
            return base
        return classify_intent(prompt)

    async def run_cycle_loop(
        self,
        user_prompt: str,
        target_stage: ForgeStage | None = None,
    ) -> str:
        """Single adaptive entry point for every developer input."""
        self._cancelled = False

        stripped = user_prompt.strip()
        if not stripped:
            return ""

        # 1. Explicit slash / shell syntax handlers run BEFORE any intent classification.
        if stripped.startswith("!") or stripped.startswith("/"):
            return await self.handle_slash_command(stripped)

        # 2. A live plan contract intercepts the next natural-language reply.
        if self.state.pending_plan is not None:
            return await self._handle_pending_plan_response(stripped)

        # 3. Deterministic intent routing
        mode = self.classify(stripped, target_stage)
        self.last_mode = mode

        if mode == ExecutionMode.DEEP_RESEARCH:
            return await self._run_research(stripped)
        if mode == ExecutionMode.CONVERSATIONAL:
            stage = target_stage or ForgeStage.ANVIL
            return await self._run_agent_turn(stripped, stage, allow_edits=False)
        if mode == ExecutionMode.DIRECT_TOOL:
            stage = target_stage or self._default_direct_stage()
            return await self._run_agent_turn(stripped, stage, allow_edits=True)
        return await self._run_full_cycle(stripped)

    async def handle_user_turn(
        self, user_prompt: str, target_stage: ForgeStage | None = None
    ) -> str:
        """Alias for :meth:`run_cycle_loop` (Claude Code / Codex style input handling)."""
        return await self.run_cycle_loop(user_prompt, target_stage=target_stage)

    async def escalate_to_forge_cycle(self, requirement_contract: str) -> str:
        """Hand off a requirement contract into the plan-approval state machine.

        The handoff never starts manufacturing before the user has approved the plan (or
        supplied modifications that explicitly say to proceed).
        """
        return await self._run_full_cycle(requirement_contract)

    async def ask_question(
        self,
        question: str,
        options: list[str],
        default_option: str | None = None,
        stage: ForgeStage = ForgeStage.ANVIL,
    ) -> str:
        """Pose an interactive question to the user with selectable options."""
        user_question = UserQuestion(
            question=question,
            options=options,
            default_option=default_option,
            stage=stage,
        )
        self.state.append_thought(
            stage,
            f"Question for User: {question} [Options: {', '.join(options)}]",
        )

        if self.question_handler is not None:
            ans = await self.question_handler(user_question)
            self.state.append_thought(stage, f"User Selected: {ans}")
            return ans

        try:
            is_tty = sys.stdin.isatty()
        except Exception:
            is_tty = False

        if not is_tty and self.state._on_redraw is None:
            fallback = default_option or (options[0] if options else "")
            self.state.append_thought(stage, f"Default Applied (Non-Interactive): {fallback}")
            return fallback

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        user_question.future = future
        self.state.set_pending_question(user_question)

        try:
            chosen = await future
            self.state.append_thought(stage, f"User Selected: {chosen}")
            return chosen
        except asyncio.CancelledError:
            self.state.append_thought(stage, "Question was cancelled by user.")
            raise

    # -- conversational / direct tool (real agent ReAct loop) ---------------------------

    def _default_direct_stage(self) -> ForgeStage:
        if ForgeStage.HAMMER in self.agents:
            return ForgeStage.HAMMER
        return ForgeStage.ANVIL

    def _agent_for(self, stage: ForgeStage) -> Any | None:
        return self.agents.get(stage)

    def _toolset_for(self, stage: ForgeStage, allow_edits: bool) -> tuple[str, ...] | None:
        from forge.tools.defaults import INTERACTIVE_EXECUTOR_TOOLS, INTERACTIVE_SAFE_TOOLS

        if stage == ForgeStage.HAMMER and allow_edits:
            return INTERACTIVE_EXECUTOR_TOOLS
        return INTERACTIVE_SAFE_TOOLS

    def _speaker(self, stage: ForgeStage) -> str:
        return {
            ForgeStage.ANVIL: "Anvil",
            ForgeStage.HAMMER: "Hammer",
            ForgeStage.TEMPER: "Temper",
        }.get(stage, stage.value)

    async def _invoke_agent_chat(
        self,
        agent: Any,
        prompt: str,
        stage: ForgeStage,
        *,
        allow_edits: bool,
        on_token: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        system_prompt: str | None = None,
        max_tool_steps: int | None = None,
    ) -> Any:
        """Call an agent's ``chat`` with only the kwargs it actually supports."""
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(agent.chat).parameters
            names = set(params)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            names, has_var_kw = set(), True

        def accepts(name: str) -> bool:
            return has_var_kw or name in names

        if accepts("system_prompt") and system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        if accepts("history"):
            kwargs["history"] = list(self._history[stage])
        if accepts("project_context"):
            kwargs["project_context"] = self.project_context
        if accepts("max_tool_steps"):
            kwargs["max_tool_steps"] = (
                max_tool_steps if max_tool_steps is not None else self.max_tool_steps
            )
        if accepts("allowed_tools"):
            kwargs["allowed_tools"] = self._toolset_for(stage, allow_edits)
        if accepts("on_tool_call"):
            kwargs["on_tool_call"] = lambda call: self._stream_tool_call(stage, call)
        if accepts("on_tool_result"):
            kwargs["on_tool_result"] = lambda result: self._stream_tool_result(stage, result)
        if accepts("on_token") and on_token is not None:
            kwargs["on_token"] = on_token
        if accepts("on_thinking") and on_thinking is not None:
            kwargs["on_thinking"] = on_thinking

        return await agent.chat(prompt, **kwargs)

    def _stream_tool_call(self, stage: ForgeStage, call: Any) -> None:
        """Show the requested tool immediately, before it runs."""
        tool = str(getattr(call, "name", "") or "tool")
        arguments = getattr(call, "arguments", None)
        self.state.set_live_activity(
            stage,
            self._stage_work_headline(stage),
            format_transient_action(tool, arguments if isinstance(arguments, Mapping) else ""),
        )

    def _stream_tool_result(self, stage: ForgeStage, result: Any) -> None:
        """Update the transient status and commit persistent file/command evidence.

        Only the last status line is transient. File mutations and terminal executions are
        appended to the stage buffer so the viewport keeps the real diff and command output.
        """
        tool = str(getattr(result, "tool", "tool") or "tool")
        ok = bool(getattr(result, "ok", False))
        data = getattr(result, "data", None)
        payload: Mapping[str, Any] = data if isinstance(data, Mapping) else {}
        content = (getattr(result, "content", "") or "").strip()
        error = str(getattr(result, "error", "") or "").strip()
        argument = (
            payload.get("path")
            or payload.get("command")
            or payload.get("query")
            or (content.splitlines()[0] if content else "")
        )
        self.state.set_live_activity(
            stage,
            self._stage_work_headline(stage),
            format_transient_action(tool, payload or argument),
        )
        block = self._tool_event_block(tool, payload, ok=ok, error=error)
        if block:
            self.state.append_tool_event(stage, block)

    @staticmethod
    def _tool_event_block(
        tool: str, data: Mapping[str, Any], *, ok: bool, error: str
    ) -> str:
        """Build a persistent, richly renderable event for a tool result."""
        if not ok:
            detail = error or "tool execution failed"
            return f"[WARN] {tool}: {detail}"

        if tool == "write_file":
            path = str(data.get("path") or "file")
            diff = str(data.get("diff") or "")
            added, removed = _diff_stats(diff)
            verb = "Created" if data.get("created") else "Modified"
            header = f"[FILE] {verb} {path} (+{added} -{removed})"
            return f"{header}\n{diff}".rstrip()

        if tool == "edit_file":
            path = str(data.get("path") or "file")
            diff = str(data.get("diff") or "")
            added, removed = _diff_stats(diff)
            header = f"[FILE] Modified {path} (+{added} -{removed})"
            return f"{header}\n{diff}".rstrip()

        if tool == "delete_file":
            path = str(data.get("path") or "file")
            count = data.get("deleted_files")
            suffix = f" ({count} files)" if count is not None else ""
            return f"[FILE] Deleted {path}{suffix}"

        if tool == "create_directory":
            path = str(data.get("path") or "directory")
            verb = "Created" if data.get("created") else "Already exists"
            return f"[DIR] {verb} {path}"

        if tool in {"run_tests", "run_command"}:
            command = str(data.get("command") or tool)
            exit_code = data.get("exit_code")
            duration = data.get("duration_s")
            header = f"[CMD] $ {command}"
            if exit_code is not None:
                header += f"\n[exit {exit_code}"
                if isinstance(duration, int | float):
                    header += f", {duration:.1f}s"
                header += "]"
            if tool == "run_tests":
                counts = ", ".join(
                    f"{name}={data[key]}"
                    for name, key in (
                        ("passed", "passed"),
                        ("failed", "failed"),
                        ("errors", "errors"),
                        ("skipped", "skipped"),
                    )
                    if data.get(key) is not None
                )
                if counts:
                    header += f"\n[{counts}]"
                output = str(data.get("output") or "")
            else:
                output = "\n".join(
                    part
                    for part in (
                        str(data.get("stdout") or "").rstrip(),
                        str(data.get("stderr") or "").rstrip(),
                    )
                    if part
                )
            tail = _tail_text(output, max_lines=30, max_chars=6000)
            return f"{header}\n{tail}".rstrip() if tail else header

        if tool in {"start_process", "stop_process", "process_output"}:
            state = "running" if data.get("running") else f"exited ({data.get('exit_code')})"
            output = _tail_text(str(data.get("output") or ""), max_lines=16, max_chars=3000)
            header = f"[CMD] {data.get('command') or data.get('id') or tool}: {state}"
            return f"{header}\n{output}".rstrip() if output else header

        return ""

    def _thinking_handler(self, stage: ForgeStage) -> Callable[[str], None]:
        """Return a reasoning callback that only updates the ephemeral status line."""

        def _on_thinking(latest_thought_line: str) -> None:
            self.state.set_live_activity(
                stage,
                self._stage_work_headline(stage),
                format_transient_action("", latest_thought_line),
            )

        return _on_thinking

    def _tracking_thinking_handler(
        self, stage: ForgeStage
    ) -> tuple[Callable[[str], None], Callable[[], bool]]:
        """Thinking callback plus a predicate telling whether reasoning was observed."""
        seen = False
        base = self._thinking_handler(stage)

        def _on_thinking(latest_thought_line: str) -> None:
            nonlocal seen
            if latest_thought_line.strip():
                seen = True
            base(latest_thought_line)

        return _on_thinking, lambda: seen

    @staticmethod
    def _stage_work_headline(stage: ForgeStage) -> str:
        """Headline for the ephemeral live-activity indicator, per stage role."""
        if stage is ForgeStage.TEMPER:
            return "Verifying..."
        if stage is ForgeStage.HAMMER:
            return "Implementing..."
        return "Thinking..."

    def _begin_stage_activity(self, stage: ForgeStage, subline: str = "") -> None:
        """Show the transient 2-line indicator the moment a stage starts working."""
        self.state.set_live_activity(stage, self._stage_work_headline(stage), subline)

    def _commit_stage_output(
        self,
        stage: ForgeStage,
        text: str,
        *,
        status: Literal["idle", "running", "passed", "failed"] = "passed",
    ) -> str:
        """Clear the transient indicator and commit the role-pure final output for a stage."""
        formatted = format_stage_deliverable(stage, text)
        self.state.clear_live_activity(stage)
        self.state.set_stage_output(stage, formatted, status=status)
        return formatted

    @staticmethod
    def _turn_usage(turn: Any) -> tuple[int, int, int]:
        """Return ``(prompt, completion, total)`` from any agent/client result shape."""
        usage = getattr(turn, "usage", None)
        prompt = getattr(turn, "prompt_tokens", None)
        completion = getattr(turn, "completion_tokens", None)
        total = getattr(turn, "total_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "prompt_tokens", None)
        if completion is None:
            completion = getattr(usage, "completion_tokens", None)
        if total is None:
            total = getattr(usage, "total_tokens", None)
        prompt_i = int(prompt) if isinstance(prompt, int) and prompt > 0 else 0
        completion_i = int(completion) if isinstance(completion, int) and completion > 0 else 0
        total_i = int(total) if isinstance(total, int) and total > 0 else 0
        if total_i == 0:
            total_i = prompt_i + completion_i
        return prompt_i, completion_i, total_i

    @classmethod
    def _turn_tokens(cls, turn: Any) -> int:
        return cls._turn_usage(turn)[2]

    def _cycle_banner(self, cycle: int | None = None) -> str:
        """Stable cycle heading used across plans, implementation and QA reports."""
        return f"Cycle :- {self.state.cycle_count if cycle is None else cycle}"

    def _render_pending_plan(self, plan: PendingPlan) -> str:
        """Render the approval contract shown before autonomous execution starts."""
        lines = [
            self._cycle_banner(),
            f"Final Plan (revision {plan.revision})",
            "",
            plan.plan_text.strip(),
            "",
        ]
        if plan.target_files:
            lines.extend(
                ["Target Files:", ", ".join(plan.target_files), ""]
            )
        lines.extend(
            [
                "Acceptance Requirements:",
                plan.checklist_text(),
                "",
                "Reply APPROVE to launch, describe modifications to revise, or ask a question.",
            ]
        )
        return "\n".join(lines).strip()

    async def _evaluate_plan_turn(
        self, plan: PendingPlan, response: str
    ) -> AnvilPlanEvaluation:
        """Ask Anvil to semantically adjudicate the developer's reply to the plan."""
        contract = plan.to_contract().model_dump_json(indent=2)
        instruction = (
            "You are Anvil, adjudicating a developer's reply to a pending implementation "
            "plan. Decide what the developer authorizes NOW, not which words they used. "
            "Return ONLY a JSON object, with no markdown fence or commentary, matching this "
            "schema:\n"
            "{\n"
            '  "intent": "APPROVE_AS_IS" | "MODIFY_AND_PROCEED" | '
            '"MODIFY_AND_WAIT" | "QUESTION_OR_CHAT" | "CANCEL_PLAN",\n'
            '  "reasoning_summary": "short explanation",\n'
            '  "updated_plan_markdown": null or "complete revised plan",\n'
            '  "updated_target_files": null or ["path"],\n'
            '  "updated_requirements": null or [\n'
            '    {"req_id": "REQ-1", "description": "...", "completed": false, '
            '"verification_note": ""}\n'
            "  ],\n"
            '  "conversational_reply": null or "answer for QUESTION_OR_CHAT"\n'
            "}\n\n"
            "Decision rules:\n"
            "- APPROVE_AS_IS: the developer authorizes executing the current plan unchanged. "
            "Dismissive hypotheses, jokes, or statements about what they would have preferred "
            "do not modify the plan when the developer then authorizes the current plan.\n"
            "- MODIFY_AND_PROCEED: the developer requests a concrete change AND authorizes "
            "immediate execution of the revised plan. Return the revised plan and full "
            "checklist.\n"
            "- MODIFY_AND_WAIT: the developer requests a concrete change but does not authorize "
            "immediate execution. Return the revised plan and full checklist, then wait.\n"
            "- QUESTION_OR_CHAT: the developer asks a question, discusses tradeoffs, or is "
            "ambiguous. Answer in conversational_reply and keep the plan pending.\n"
            "- CANCEL_PLAN: the developer abandons the plan without authorizing execution.\n"
            "- Merely mentioning an alternative is not authorization. Authorization must be "
            "semantically present in this reply.\n\n"
            "REFERENCE CASES:\n"
            '1. "proceed, but instead of A use B" -> MODIFY_AND_PROCEED with revised plan.\n'
            '2. "instead of A use B" -> MODIFY_AND_WAIT with revised plan.\n'
            '3. "you should use B instead of A, but anyways proceed" -> APPROVE_AS_IS with no '
            "plan changes, because the developer explicitly authorizes the current plan.\n"
            '4. "I would have used B, but whatever, go ahead" -> APPROVE_AS_IS with no changes.\n'
            '5. "What does REQ-2 verify?" -> QUESTION_OR_CHAT with an answer.\n\n'
            f"CURRENT PLAN CONTRACT:\n{contract}\n\n"
            f"DEVELOPER REPLY:\n{response}"
        )
        raw, _usage = await self._stage_instruction(
            ForgeStage.ANVIL, instruction, allow_edits=False
        )
        try:
            evaluation = parse_plan_evaluation(raw)
        except (ValueError, json.JSONDecodeError) as first_error:
            repair = (
                f"{instruction}\n\n"
                "Your previous reply could not be parsed: "
                f"{first_error}. Return ONLY the JSON object now."
            )
            raw, _usage = await self._stage_instruction(
                ForgeStage.ANVIL, repair, allow_edits=False
            )
            try:
                evaluation = parse_plan_evaluation(raw)
            except (ValueError, json.JSONDecodeError):
                evaluation = AnvilPlanEvaluation(
                    intent=PlanTurnIntent.QUESTION_OR_CHAT,
                    reasoning_summary=(
                        "Anvil could not produce a valid structured plan adjudication; the "
                        "plan remains pending for safety."
                    ),
                    conversational_reply=(
                        "I could not safely resolve that reply against the pending plan. "
                        "The plan is still awaiting your approval; please restate whether to "
                        "approve, revise, or cancel it."
                    ),
                )
        self.state.append_thought(
            ForgeStage.ANVIL,
            f"Anvil semantic plan adjudication: {evaluation.intent.value}.",
        )
        return evaluation

    def _apply_plan_evaluation(
        self, plan: PendingPlan, evaluation: AnvilPlanEvaluation
    ) -> PendingPlan | None:
        """Apply Anvil's revised plan contract, or return None when it supplied no plan."""
        revised_text = (evaluation.updated_plan_markdown or "").strip()
        if not revised_text:
            return None

        if evaluation.updated_requirements is not None:
            requirements = [
                PlanRequirement.from_item(item)
                for item in evaluation.updated_requirements
            ]
        else:
            requirements = extract_plan_requirements(revised_text, plan.request)

        target_files = (
            list(evaluation.updated_target_files)
            if evaluation.updated_target_files is not None
            else list(plan.target_files)
        )
        revised = PendingPlan(
            request=plan.request,
            plan_text=revised_text,
            requirements=requirements,
            revision=plan.revision + 1,
            target_files=target_files,
        )
        self.state.set_pending_plan(revised)
        rendered = self._render_pending_plan(revised)
        self.state.set_stage_output(ForgeStage.ANVIL, rendered, status="passed")
        self.state.finalize_stream(ForgeStage.ANVIL, rendered)
        self.state.append_thought(
            ForgeStage.ANVIL,
            f"Plan revised to revision {revised.revision} from Anvil's semantic contract.",
        )
        return revised

    async def _handle_pending_plan_response(self, response: str) -> str:
        """Resolve a natural-language reply against the pending plan contract."""
        plan = self.state.pending_plan
        if plan is None:
            return ""
        self.last_mode = ExecutionMode.FULL_FORGE_CYCLE
        evaluation = await self._evaluate_plan_turn(plan, response)
        intent = evaluation.intent

        if intent is PlanTurnIntent.CANCEL_PLAN:
            self.state.set_pending_plan(None)
            self.state.all_requirements_met = False
            notice = "Plan cancelled. No implementation was started."
            self.state.append_thought(ForgeStage.ANVIL, notice)
            self.state.set_stage_output(ForgeStage.ANVIL, notice, status="idle")
            self.auto_save_session()
            return notice

        if intent is PlanTurnIntent.QUESTION_OR_CHAT:
            answer = (evaluation.conversational_reply or "").strip()
            if not answer:
                answer = await self._run_agent_turn(
                    response, ForgeStage.ANVIL, allow_edits=False
                )
            self.state.append_thought(
                ForgeStage.ANVIL,
                "Plan remains pending approval after the user's question.",
            )
            self.state.set_stage_output(
                ForgeStage.ANVIL, self._render_pending_plan(plan), status="passed"
            )
            return answer

        if intent in {
            PlanTurnIntent.MODIFY_AND_PROCEED,
            PlanTurnIntent.MODIFY_AND_WAIT,
        }:
            revised = self._apply_plan_evaluation(plan, evaluation)
            if revised is None:
                revised = await self._revise_plan(plan, response)
            if intent is PlanTurnIntent.MODIFY_AND_WAIT:
                return self._render_pending_plan(revised)
            return await self._run_autonomous_cycle(revised)

        return await self._run_autonomous_cycle(plan)

    async def _generate_plan(self, prompt: str) -> PendingPlan:
        """Ask the live Anvil model for a plan plus an explicit REQ contract."""
        instruction = (
            "You are the Anvil orchestrator. Produce an implementation plan for the task "
            "below. Do not write implementation code. End with a requirements contract in "
            "this exact shape, one requirement per line:\n"
            "REQ-1: <independently verifiable requirement>\n"
            "REQ-2: <independently verifiable requirement>\n"
            "TARGET_FILES: <comma-separated repository paths, or NONE>\n\n"
            f"TASK:\n{prompt}"
        )
        text, _usage = await self._stage_instruction(
            ForgeStage.ANVIL,
            instruction,
            allow_edits=False,
            system_prompt=_planner_chat_prompt(),
        )
        plan_text = format_stage_deliverable(ForgeStage.ANVIL, text)
        plan = PendingPlan(
            request=prompt,
            plan_text=plan_text,
            requirements=extract_plan_requirements(plan_text, prompt),
            target_files=extract_target_files(plan_text),
        )
        self.state.all_requirements_met = False
        self.state.set_pending_plan(plan)
        rendered = self._render_pending_plan(plan)
        self.state.set_stage_output(ForgeStage.ANVIL, rendered, status="passed")
        self.state.finalize_stream(ForgeStage.ANVIL, rendered)
        self.state.append_thought(
            ForgeStage.ANVIL,
            f"{self._cycle_banner()} plan revision {plan.revision} is awaiting approval.",
        )
        return plan

    async def _revise_plan(self, plan: PendingPlan, feedback: str) -> PendingPlan:
        """Ask Anvil to revise the plan from semantic user feedback."""
        instruction = (
            "Revise the implementation plan using the user's feedback. Preserve all "
            "requirements that still apply and return a complete revised plan followed by "
            "the full requirements contract (REQ-1, REQ-2, ...) and a "
            "TARGET_FILES: <comma-separated paths, or NONE> line.\n\n"
            f"ORIGINAL REQUEST:\n{plan.request}\n\n"
            f"CURRENT PLAN:\n{plan.plan_text}\n\n"
            f"USER FEEDBACK:\n{feedback}"
        )
        text, _usage = await self._stage_instruction(
            ForgeStage.ANVIL,
            instruction,
            allow_edits=False,
            system_prompt=_planner_chat_prompt(),
        )
        plan_text = format_stage_deliverable(ForgeStage.ANVIL, text)
        revised = PendingPlan(
            request=plan.request,
            plan_text=plan_text,
            requirements=extract_plan_requirements(plan_text, plan.request),
            revision=plan.revision + 1,
            target_files=extract_target_files(plan_text),
        )
        self.state.set_pending_plan(revised)
        rendered = self._render_pending_plan(revised)
        self.state.set_stage_output(ForgeStage.ANVIL, rendered, status="passed")
        self.state.finalize_stream(ForgeStage.ANVIL, rendered)
        self.state.append_thought(
            ForgeStage.ANVIL,
            f"Plan revised to revision {revised.revision}; awaiting approval "
            "when the user requested a pause.",
        )
        return revised

    async def _stage_instruction(
        self,
        stage: ForgeStage,
        instruction: str,
        *,
        allow_edits: bool,
        system_prompt: str | None = None,
        max_tool_steps: int | None = None,
    ) -> tuple[str, tuple[int, int, int]]:
        """Execute one real stage call through its agent or the live streaming client.

        No conversational or manufacturing stage ever substitutes a canned reply. When no
        live model is reachable the caller receives an actionable diagnostic instead.
        """
        self.state.set_active_stage(stage, reset_tokens=False)
        self._begin_stage_activity(stage)
        agent = self._agent_for(stage)

        def _on_token(delta: str) -> None:
            self.state.append_stream(stage, delta)

        on_thinking, saw_reasoning = self._tracking_thinking_handler(stage)
        self.state.start_stream(stage, self._speaker(stage))
        usage = (0, 0, 0)
        reply = ""
        failed = False
        agent_failure: Exception | None = None

        if agent is not None:
            try:
                turn = await self._invoke_agent_chat(
                    agent,
                    instruction,
                    stage,
                    allow_edits=allow_edits,
                    on_token=_on_token,
                    on_thinking=on_thinking,
                    system_prompt=system_prompt,
                    max_tool_steps=max_tool_steps,
                )
                reply = (getattr(turn, "response_text", "") or "").strip()
                usage = self._turn_usage(turn)
                messages = getattr(turn, "messages", None)
                if messages:
                    self._history[stage] = list(messages)
                if not reply:
                    binding = self._config_binding(stage)
                    reply = (
                        reasoning_only_diagnostic(binding)
                        if saw_reasoning() and binding is not None
                        else model_diagnostic("The live model returned no visible content.")
                    )
            except Exception as exc:  # never silently fall back to a no-tools call
                agent_failure = exc
                log.exception("%s agent tool loop failed", stage.value)

        if agent is not None and agent_failure is not None:
            failed = True
            reply = model_diagnostic(
                f"{stage.value} agent tool loop failed: {type(agent_failure).__name__}: "
                f"{agent_failure}"
            )
        elif agent is None:
            binding = self._config_binding(stage)
            if binding is None:
                reply = model_diagnostic()
                failed = True
            else:
                client = self._llm_client()
                messages = [{"role": "user", "content": instruction}]
                session_kwargs = (
                    {"session_id": self.session_id}
                    if "session_id" in inspect.signature(client.stream_chat).parameters
                    else {}
                )
                try:
                    response = await client.stream_chat(
                        binding,
                        messages,
                        on_token=_on_token,
                        on_thinking=on_thinking,
                        stage=stage.value.lower(),
                        **session_kwargs,
                    )
                    reply = (response.content or "").strip()
                    usage = (
                        response.prompt_tokens,
                        response.completion_tokens,
                        response.total_tokens,
                    )
                    if not reply:
                        reply = (
                            reasoning_only_diagnostic(
                                binding, truncated=response.reasoning_incomplete
                            )
                            if response.reasoning_content
                            else model_diagnostic(
                                "The live model returned no visible content."
                            )
                        )
                except Exception as exc:
                    failed = True
                    reply = model_diagnostic(
                        f"Live model call failed ({binding.spec_string}): {exc}"
                    )

        prompt_tokens, completion_tokens, total_tokens = usage
        self.state.record_tokens(
            stage,
            total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        formatted = format_stage_deliverable(stage, reply)
        self.state.finalize_stream(stage, formatted)
        self.conversation_history[stage].append(
            {"role": "assistant", "content": formatted}
        )
        self._commit_stage_output(stage, reply, status="failed" if failed else "passed")
        self.state.set_active_stage(None)
        return formatted, usage

    async def _cycle_anvil(
        self, plan: PendingPlan, feedback: str = ""
    ) -> str:
        """Produce the initial plan or a delta-remediation plan for the current cycle."""
        if not feedback:
            rendered = self._render_pending_plan(plan)
            self.state.set_stage_output(ForgeStage.ANVIL, rendered, status="passed")
            return rendered
        unmet_lines = [
            requirement.checklist_line()
            for requirement in plan.requirements
            if not requirement.met
        ]
        unmet_text = "\n".join(unmet_lines) or plan.checklist_text()
        instruction = (
            f"{self._cycle_banner()} delta remediation. Do not repeat the whole plan; "
            "resolve the unmet requirements below and state the exact corrective steps "
            "Hammer must take.\n\n"
            f"ORIGINAL REQUEST:\n{plan.request}\n\n"
            f"BASE PLAN:\n{plan.plan_text}\n\n"
            f"UNMET REQUIREMENTS:\n{unmet_text}\n\n"
            f"TEMPER FEEDBACK:\n{feedback}"
        )
        text, _usage = await self._stage_instruction(
            ForgeStage.ANVIL, instruction, allow_edits=False
        )
        rendered = f"{self._cycle_banner()} Remediation Plan\n\n{text}"
        self.state.set_stage_output(ForgeStage.ANVIL, rendered, status="passed")
        self.state.finalize_stream(ForgeStage.ANVIL, rendered)
        return rendered

    async def _cycle_hammer(self, plan: PendingPlan, feedback: str = "") -> str:
        """Ask the live Hammer model to implement the complete plan in the repository."""
        target_files = ", ".join(plan.target_files) or "Infer every file required by the plan."
        instruction = (
            f"{self._cycle_banner()} implementation. Implement the ENTIRE approved plan "
            "in the repository now. You are the only agent allowed to write code. Use the "
            "write_file and edit_file tools to create or update every file required for a "
            "complete, runnable project, not a skeleton and not a code snippet in chat. "
            "Do not stop after one file. Continue working through the plan until every "
            "task is implemented. Run the project's install/build/test commands with the "
            "available tools and fix failures caused by your changes. Do not ask the user "
            "questions and do not merely describe what you would do. Your first action must "
            "be a tool call, and you must keep calling tools until the real files exist on "
            "disk. Finish only after a real test/build command has run successfully.\n\n"
            f"APPROVED PLAN:\n{plan.plan_text}\n\n"
            f"ACCEPTANCE REQUIREMENTS:\n{plan.checklist_text()}\n\n"
            f"TARGET FILES:\n{target_files}\n\n"
            "When finished, report the files created or modified, the exact commands run, "
            "the test/build result, and any remaining blockers."
        )
        if feedback:
            instruction += f"\n\nUNRESOLVED DEFECTS FROM LAST CYCLE:\n{feedback}"
        text, _usage = await self._stage_instruction(
            ForgeStage.HAMMER,
            instruction,
            allow_edits=True,
            max_tool_steps=_CYCLE_HAMMER_TOOL_STEPS,
        )
        rendered = f"{self._cycle_banner()} Implementation Report\n\n{text}"
        self.state.set_stage_output(ForgeStage.HAMMER, rendered, status="passed")
        self.state.finalize_stream(ForgeStage.HAMMER, rendered)
        return rendered

    async def _cycle_temper(
        self, plan: PendingPlan, cycle: int
    ) -> tuple[str, bool, str]:
        """Audit every requirement and return ``(report, all_met, audit_text)``."""
        instruction = (
            f"{self._cycle_banner(cycle)} adversarial audit. Independently verify every "
            "requirement against the actual repository state. Do not trust Hammer's report "
            "as evidence. You must run the project's real test suite and build/typecheck "
            "commands with run_tests or run_command, inspect the relevant files, and base "
            "each verdict on command output or direct file evidence. If tests/build cannot "
            "run, report the exact error and treat the requirement as unmet.\n\n"
            "Reply with exactly one line per requirement in this form:\n"
            "REQ-1: PASS - evidence   (or FAIL - missing evidence)\n"
            "REQ-2: PASS - evidence\n"
            "Then finish with: ALL_REQUIREMENTS_MET: TRUE or FALSE.\n"
            "Set ALL_REQUIREMENTS_MET to FALSE if any requirement is unmet, any test/build "
            "command fails, or you could not verify the result.\n\n"
            f"APPROVED PLAN:\n{plan.plan_text}\n\n"
            f"REQUIREMENTS:\n{plan.checklist_text()}"
        )
        text, _usage = await self._stage_instruction(
            ForgeStage.TEMPER,
            instruction,
            allow_edits=False,
            max_tool_steps=_CYCLE_TEMPER_TOOL_STEPS,
        )

        audit_text = text
        all_met = apply_requirement_audit(plan, audit_text)
        if all_met:
            for requirement in plan.requirements:
                requirement.met = True
                requirement.evidence = requirement.evidence or "verified in cycle audit"
        if self.temper_verifier is not None:
            verdict = self.temper_verifier(cycle, plan.request)
            all_met = verdict.passed
            audit_text = verdict.feedback or text
            for requirement in plan.requirements:
                requirement.met = verdict.passed
                if not verdict.passed:
                    requirement.evidence = verdict.feedback[:180]
            if verdict.deliverable:
                text = verdict.deliverable

        report_lines = [
            self._cycle_banner(cycle),
            "QA & Verification Verdict",
            "",
            plan.checklist_text(),
            "",
            f"ALL_REQUIREMENTS_MET: {'TRUE' if all_met else 'FALSE'}",
        ]
        if audit_text and audit_text not in text:
            report_lines.extend(["", audit_text])
        if text and text not in report_lines:
            report_lines.extend(["", text])
        report = "\n".join(report_lines).strip()
        if all_met:
            self.state.set_stage_output(ForgeStage.TEMPER, report, status="passed")
        else:
            self.state.set_stage_output(ForgeStage.TEMPER, report, status="failed")
        self.state.finalize_stream(ForgeStage.TEMPER, report)
        return report, all_met, audit_text

    async def _run_autonomous_cycle(self, plan: PendingPlan) -> str:
        """Execute plan -> implement -> audit loops without further user prompts."""
        self.state.set_pending_plan(plan)
        for requirement in plan.requirements:
            requirement.met = False
            requirement.evidence = ""
        self.state.all_requirements_met = False
        self.auto_save_session()
        feedback = ""
        final_report = ""

        while not self._cancelled:
            cycle = self.state.cycle_count
            self.state.append_thought(
                ForgeStage.ANVIL,
                f"{self._cycle_banner(cycle)} autonomous execution started.",
            )
            await self._cycle_anvil(plan, feedback)
            if self._cancelled:
                break
            await self._cycle_hammer(plan, feedback)
            if self._cancelled:
                break
            final_report, all_met, audit_text = await self._cycle_temper(plan, cycle)
            self.state.all_requirements_met = all_met

            if all_met:
                self.state.append_thought(
                    ForgeStage.TEMPER,
                    f"{self._cycle_banner(cycle)} ALL_REQUIREMENTS_MET == True.",
                )
                self._commit_stage_output(
                    ForgeStage.TEMPER, final_report, status="passed"
                )
                self.state.set_pending_plan(None)
                self.auto_save_session()
                return final_report

            unmet = [
                request
                for request in plan.requirements
                if not request.met
            ]
            feedback = (
                f"{self._cycle_banner(cycle)} unmet requirements:\n"
                + "\n".join(req.checklist_line() for req in unmet)
                + f"\n\nTemper audit:\n{audit_text}"
            )
            self._commit_stage_output(
                ForgeStage.TEMPER,
                f"{final_report}\n\n{feedback}",
                status="failed",
            )

            live_failure = (
                "No live LLM model is attached" in final_report
                or "Live model call failed" in final_report
            )
            if self.temper_verifier is None and live_failure:
                self.state.append_thought(
                    ForgeStage.TEMPER,
                    f"{self._cycle_banner(cycle)} no live model is reachable; "
                    "autonomous remediation cannot continue.",
                )
                self.state.set_active_stage(None)
                return final_report
            if self.cycle_mode == CycleMode.SINGLE_CYCLE:
                self.state.append_thought(
                    ForgeStage.ANVIL,
                    f"{self._cycle_banner(cycle)} Single Cycle policy: halting.",
                )
                self.state.set_active_stage(None)
                return final_report
            if cycle >= self.max_cycles:
                self.state.append_thought(
                    ForgeStage.TEMPER,
                    f"Cycle ceiling ({self.max_cycles}) reached; requirements remain unmet.",
                )
                self.state.set_active_stage(None)
                return final_report

            self.state.advance_to_next_cycle(feedback)

        self.state.set_active_stage(None)
        return final_report

    async def _run_agent_turn(
        self,
        prompt: str,
        stage: ForgeStage,
        *,
        allow_edits: bool,
    ) -> str:
        """Run one real agent chat / ReAct tool turn strictly inside ``stage``.

        The prompt is sent to the stage's real configured LLM (never a canned string), streamed
        token-by-token into the viewport, and both the user and assistant turns are recorded in
        the persistent per-stage ``conversation_history``. When no model is attached we render an
        actionable credential diagnostic instead of fabricating a reply.
        """
        self.state.set_active_stage(stage, reset_tokens=False)
        self.state.append_chat(stage, "You", prompt)
        self._begin_stage_activity(stage, "analyzing request...")
        self.conversation_history[stage].append({"role": "user", "content": prompt})

        agent = self._agent_for(stage)
        if agent is None:
            binding = self._config_binding(stage)
            if binding is not None:
                return await self._run_llm_turn(prompt, stage, binding)
            reply = model_diagnostic()
            formatted = self._commit_stage_output(stage, reply, status="failed")
            self.state.append_chat(stage, self._speaker(stage), formatted)
            self.conversation_history[stage].append({"role": "assistant", "content": formatted})
            self.state.set_active_stage(None)
            return formatted

        def _on_token(delta: str) -> None:
            self.state.append_stream(stage, delta)

        on_thinking, saw_reasoning = self._tracking_thinking_handler(stage)
        self.state.start_stream(stage, self._speaker(stage))
        try:
            turn = await self._invoke_agent_chat(
                agent,
                prompt,
                stage,
                allow_edits=allow_edits,
                on_token=_on_token,
                on_thinking=on_thinking,
            )
            reply = (getattr(turn, "response_text", "") or "").strip()
            prompt_tokens, completion_tokens, total_tokens = self._turn_usage(turn)
            messages = getattr(turn, "messages", None)
            if messages:
                self._history[stage] = list(messages)
            handoff = getattr(turn, "handoff_requirements", None)
            if not reply:
                binding = self._config_binding(stage)
                reply = (
                    reasoning_only_diagnostic(binding)
                    if saw_reasoning() and binding is not None
                    else model_diagnostic("The live model returned no visible content.")
                )

            self.state.record_tokens(
                stage,
                total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            formatted = format_stage_deliverable(stage, reply)
            self.state.finalize_stream(stage, formatted)
            self.conversation_history[stage].append({"role": "assistant", "content": formatted})
            self._commit_stage_output(stage, reply, status="passed")
            self.state.set_active_stage(None)

            if handoff:
                return await self.escalate_to_forge_cycle(handoff)
            return formatted
        except Exception as exc:
            log.exception("%s agent turn failed", stage.value)
            binding = self._config_binding(stage)
            if not allow_edits and binding is not None:
                return await self._run_llm_turn(prompt, stage, binding)
            reply = model_diagnostic(
                f"{stage.value} agent tool loop failed: {type(exc).__name__}: {exc}"
            )
            formatted = format_stage_deliverable(stage, reply)
            self.state.finalize_stream(stage, formatted)
            self.conversation_history[stage].append(
                {"role": "assistant", "content": formatted}
            )
            self._commit_stage_output(stage, reply, status="failed")
            self.state.set_active_stage(None)
            return formatted

    async def _run_llm_turn(
        self, prompt: str, stage: ForgeStage, binding: StageModelBinding
    ) -> str:
        """Stream a real reply from the native unified client when no agent adapter is attached.

        The user turn is already recorded by :meth:`_run_agent_turn`; this issues a real
        streaming HTTP request through :class:`UnifiedLLMClient`, streams deltas into the
        viewport, records the real token usage, and appends the assistant turn to history.
        """
        client = self._llm_client()
        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in self.conversation_history[stage]
        ]

        def _on_token(delta: str) -> None:
            self.state.append_stream(stage, delta)

        self._begin_stage_activity(stage, "streaming live model...")
        self.state.start_stream(stage, self._speaker(stage))
        session_kwargs = (
            {"session_id": self.session_id}
            if "session_id" in inspect.signature(client.stream_chat).parameters
            else {}
        )
        try:
            response = await client.stream_chat(
                binding,
                messages,
                on_token=_on_token,
                on_thinking=self._thinking_handler(stage),
                stage=stage.value.lower(),
                **session_kwargs,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            reply = model_diagnostic(f"Live model call failed ({binding.spec_string}): {exc}")
            formatted = format_stage_deliverable(stage, reply)
            self.state.finalize_stream(stage, formatted)
            self.conversation_history[stage].append({"role": "assistant", "content": formatted})
            self._commit_stage_output(stage, reply, status="failed")
            self.state.set_active_stage(None)
            return formatted

        reply = (response.content or "").strip()
        if not reply:
            if response.reasoning_content:
                reply = reasoning_only_diagnostic(
                    binding, truncated=response.reasoning_incomplete
                )
            else:
                reply = model_diagnostic("The live model returned no visible content.")
        self.state.record_tokens(
            stage,
            response.total_tokens,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        formatted = format_stage_deliverable(stage, reply)
        self.state.finalize_stream(stage, formatted)
        self.conversation_history[stage].append({"role": "assistant", "content": formatted})
        self._commit_stage_output(stage, reply, status="passed")
        self.state.set_active_stage(None)
        return formatted

    # -- real subprocess execution (!shell) ----------------------------------------------

    async def execute_shell_command(
        self, command: str, stage: ForgeStage = ForgeStage.ANVIL
    ) -> str:
        """Run a real shell command, streaming stdout/stderr live into the viewport."""
        if not command:
            return "Usage: !<shell command>"

        self.state.set_active_stage(stage, reset_tokens=False)
        self.state.append_chat(stage, "You", f"!{command}")
        self.state.append_tool_event(stage, f"$ {command}")

        output, exit_code = await self._stream_subprocess(command, stage)

        self.state.append_tool_event(stage, f"[exit {exit_code}]")
        self.state.set_stage_output(stage, output or "(no output)", status="passed")
        self.state.append_chat(stage, self._speaker(stage), output or "(no output)")
        self.state.set_active_stage(None)
        return output

    async def _stream_subprocess(self, command: str, stage: ForgeStage) -> tuple[str, int]:
        cwd = str(self.root) if self.root is not None else None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )
        except Exception as exc:  # pragma: no cover - platform dependent
            message = f"Failed to launch command: {exc}"
            self.state.append_tool_event(stage, message)
            return message, 1

        lines: list[str] = []
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            lines.append(line)
            self.state.append_tool_event(stage, line)
        exit_code = await proc.wait()
        return "\n".join(lines), exit_code

    # -- full Anvil -> Hammer -> Temper cycle --------------------------------------------

    def _apply_cycle_mode_from_prompt(self, prompt: str) -> None:
        """Honour explicit one-cycle or until-fulfilled wording from the user."""
        lowered = prompt.lower()
        single_keywords = (
            "run first cycle and stop",
            "run 1 cycle and stop",
            "first cycle and stop",
            "run first cycle",
            "run 1 cycle",
            "single cycle",
            "stop after first cycle",
            "stop after 1 cycle",
            "step by step",
            "run one cycle",
            "pause after cycle 1",
            "1 cycle only",
        )
        continuous_keywords = (
            "run until requirements get fulfilled",
            "until requirements get fulfilled",
            "until requirements are fulfilled",
            "until requirements fulfilled",
            "run until fulfilled",
            "run until done",
            "continuous",
            "auto mode",
            "loop until pass",
            "full cycle loop",
            "fix until passing",
            "run all cycles",
        )
        if any(keyword in lowered for keyword in single_keywords):
            self.set_cycle_mode(CycleMode.SINGLE_CYCLE)
        elif any(keyword in lowered for keyword in continuous_keywords):
            self.set_cycle_mode(CycleMode.UNTIL_FULFILLED)

    async def _run_full_cycle(self, prompt: str) -> str:
        """Generate a plan and enter the semantic approval state machine.

        Manufacturing never starts until the user approves the plan or explicitly asks to
        apply modifications and proceed.
        """
        self._apply_cycle_mode_from_prompt(prompt)
        self.state.append_chat(ForgeStage.ANVIL, "You", prompt)
        plan = await self._generate_plan(prompt)
        return self._render_pending_plan(plan)

    async def _run_real_cycle(self, prompt: str) -> str:
        """Delegate to the real multi-agent workflow orchestrator and mirror its outcome."""
        self.state.set_active_stage(ForgeStage.ANVIL, reset_tokens=False)
        self._begin_stage_activity(ForgeStage.ANVIL, "planning multi-agent workflow...")
        self.state.append_thought(
            ForgeStage.ANVIL, f"Escalating to full Forge cycle: {prompt[:120]}"
        )
        assert self.workflow_runner is not None
        final_state = await self.workflow_runner(prompt)

        status = getattr(getattr(final_state, "status", None), "value", "completed")
        tasks = getattr(getattr(final_state, "plan", None), "tasks", []) or []
        qa_reports = getattr(final_state, "qa_reports", []) or []
        if qa_reports:
            qa_status = getattr(getattr(qa_reports[-1], "status", None), "value", "n/a")
        else:
            qa_status = "n/a"

        self.state.append_thought(
            ForgeStage.ANVIL, f"Workflow finished with status '{status}'."
        )
        plan_line = f"Planned {len(tasks)} task(s) for:\n{prompt}"
        self.state.append_chat(
            ForgeStage.ANVIL,
            self._speaker(ForgeStage.ANVIL),
            format_stage_deliverable(ForgeStage.ANVIL, plan_line),
        )
        self._commit_stage_output(ForgeStage.ANVIL, plan_line, status="passed")

        exec_line = f"Executed {len(tasks)} task(s)."
        self.state.append_chat(
            ForgeStage.HAMMER,
            self._speaker(ForgeStage.HAMMER),
            format_stage_deliverable(ForgeStage.HAMMER, exec_line),
        )
        self._commit_stage_output(ForgeStage.HAMMER, exec_line, status="passed")

        summary = (
            f"Full Forge cycle finished for: '{prompt}'\n"
            f"• Workflow status: {status}\n"
            f"• Tasks: {len(tasks)}\n"
            f"• QA verdict: {qa_status.upper()}"
        )
        temper_ok = qa_status.lower() in {"pass", "passed"}
        self.state.append_chat(
            ForgeStage.TEMPER,
            self._speaker(ForgeStage.TEMPER),
            format_stage_deliverable(ForgeStage.TEMPER, summary),
        )
        self._commit_stage_output(
            ForgeStage.TEMPER, summary, status="passed" if temper_ok else "failed"
        )
        self.state.set_active_stage(None)
        return summary

    async def _run_simulated_cycle(self, prompt: str) -> str:
        """Compatibility entry point for an already-approved in-process cycle."""
        plan = self.state.pending_plan
        if plan is None:
            plan = await self._generate_plan(prompt)
        return await self._run_autonomous_cycle(plan)

    async def _simulated_anvil(self, current_request: str) -> None:
        """Run Anvil through the live model and commit its plan/remediation output."""
        plan = self.state.pending_plan or PendingPlan(
            request=current_request,
            plan_text=current_request,
            requirements=extract_plan_requirements(current_request, current_request),
        )
        await self._cycle_anvil(plan)

    async def _simulated_hammer(self, current_request: str) -> None:
        """Run Hammer through the live model and commit its implementation report."""
        plan = self.state.pending_plan or PendingPlan(
            request=current_request,
            plan_text=current_request,
            requirements=extract_plan_requirements(current_request, current_request),
        )
        await self._cycle_hammer(plan)

    async def _simulated_temper(self, original_prompt: str) -> tuple[CycleVerdict, str]:
        """Run Temper's live requirement audit and return a cycle verdict."""
        plan = self.state.pending_plan or PendingPlan(
            request=original_prompt,
            plan_text=original_prompt,
            requirements=extract_plan_requirements(original_prompt, original_prompt),
        )
        report, all_met, audit_text = await self._cycle_temper(
            plan, self.state.cycle_count
        )
        return (
            CycleVerdict(passed=all_met, feedback=audit_text, deliverable=report),
            report,
        )

    def _finish_temper_turn(self, stage: ForgeStage, text: str, streamed: bool) -> None:
        """Commit Temper's role-pure QA verdict as the permanent (transient-free) viewport turn."""
        formatted = format_stage_deliverable(stage, text)
        if streamed:
            self.state.finalize_stream(stage, formatted)
        else:
            self.state.append_chat(stage, self._speaker(stage), formatted)

    # -- deep research -------------------------------------------------------------------

    async def _run_research(self, prompt: str) -> str:
        topic = prompt
        low = prompt.strip().lower()
        if low.startswith("/research"):
            topic = prompt.strip()[len("/research") :].strip()
        elif low.startswith("--research"):
            topic = prompt.strip()[len("--research") :].strip()

        stage = ForgeStage.ANVIL
        self.state.set_active_stage(stage, reset_tokens=False)
        self.state.append_chat(stage, "You", f"/research {topic}")

        if self.research_runner is not None and topic:
            self.state.append_thought(stage, f"Deep Research Mode engaged for: {topic}")
            result = await self.research_runner(topic)
        elif topic:
            result = (
                f"Deep Research mode is not wired to a live pipeline in this session.\n"
                f"Topic requested: {topic}\n"
                f"Run `forge research \"{topic}\"` for full STORM-style multi-source synthesis."
            )
        else:
            result = "Usage: /research <topic>"

        self.state.append_chat(stage, self._speaker(stage), result)
        self.state.set_stage_output(stage, result, status="passed")
        self.state.set_active_stage(None)
        return result

    # -- slash commands ------------------------------------------------------------------

    def _view_stage(self) -> ForgeStage:
        """The stage whose viewport a command's output should be rendered into."""
        if self.state.active_worker_stage is not None:
            return self.state.active_worker_stage
        return self.state.selected_view_stage

    def _emit_command_output(
        self,
        command: str,
        text: str,
        *,
        status: Literal["idle", "running", "passed", "failed"] = "passed",
    ) -> str:
        """Render a command's output into the active stage buffer and return the text."""
        stage = self._view_stage()
        self.state.append_chat(stage, "You", command)
        self.state.append_chat(stage, "Forge", text)
        self.state.set_stage_output(stage, text, status=status)
        self.auto_save_session()
        return text

    def auto_save_session(self) -> None:
        """Persist current conversation session globally to ~/.forge/sessions/."""
        try:
            from forge.config.session_store import get_session_store

            get_session_store().save_session(
                self.state, self.session_id, self.root or Path.cwd()
            )
        except Exception:
            pass

    async def handle_slash_command(self, raw: str) -> str:
        """Deterministic dispatcher for explicit ``/command`` and ``!shell`` syntax.

        Runs BEFORE the intent router so command output is always immediate and visible in the
        active stage viewport — never swallowed by classification.
        """
        stripped = raw.strip()
        if stripped.startswith("!"):
            self.last_mode = ExecutionMode.DIRECT_TOOL
            return await self.execute_shell_command(
                stripped[1:].strip(), stage=self._view_stage()
            )

        command, _, argument = stripped.partition(" ")
        command = command.strip().lower()
        argument = argument.strip()

        if command == "/help":
            return self._emit_command_output("/help", self._help_text())
        if command == "/shortcuts":
            return self._emit_command_output("/shortcuts", self._shortcuts_text())
        if command in {"/status"} and not argument:
            return self._emit_command_output(command, self._status_text())
        if command == "/provider":
            return await self._handle_provider_command(argument)
        if command == "/models":
            return await self._handle_models_command(argument)
        if command == "/usage":
            return await self._handle_usage_command(argument)
        if command == "/resume":
            return await self._handle_resume_command(argument)
        if command == "/restart":
            return await self._handle_restart_command()
        if command == "/config":
            return await self._handle_config_command()
        if command == "/mode":
            text = (
                self.set_execution_mode(argument)
                if argument
                else self._mode_summary()
            )
            return self._emit_command_output("/mode", text)
        if command in {"/cycle", "/forge"}:
            if not argument:
                text = "Usage: /cycle <engineering task> — force Anvil -> Hammer -> Temper."
                return self._emit_command_output(command, text, status="failed")
            self.last_mode = ExecutionMode.FULL_FORGE_CYCLE
            return await self._run_full_cycle(argument)
        if command == "/research":
            if not argument:
                text = "Usage: /research <topic> — run the STORM deep research pipeline."
                return self._emit_command_output(command, text, status="failed")
            self.last_mode = ExecutionMode.DEEP_RESEARCH
            return await self._run_research(f"/research {argument}")
        if command == "/clear":
            stage = self._view_stage()
            self.state.clear_stage(stage)
            return self._emit_command_output("/clear", f"Cleared the {stage.value} viewport.")
        if command == "/reset":
            self.state.reset_view()
            return self._emit_command_output(
                "/reset", "Session reset: Cycle #1, all stage buffers and history cleared."
            )
        if command in {"/exit", "/quit"}:
            return "Goodbye! The Forge workspace is standing by whenever you return."

        text = f"Unknown command: {command}. Type /help to see available commands."
        return self._emit_command_output(command, text, status="failed")

    # Backwards-compatible private alias (older call sites).
    async def _handle_slash_command(self, raw: str) -> str:
        return await self.handle_slash_command(raw)

    def _mode_summary(self) -> str:
        label = _mode_label(self.forced_mode)
        return (
            f"Execution mode: {label}\n"
            "  /mode auto   adaptive routing (chat or cycle as appropriate)\n"
            "  /mode chat   always stay conversational in the active stage\n"
            "  /mode cycle  always force Anvil -> Hammer -> Temper"
        )

    async def _handle_provider_command(self, argument: str = "") -> str:
        """List / switch providers, or launch the interactive wizard when available.

        * ``/provider``            - inside the full-screen app, opens the add-provider wizard;
          otherwise prints the provider table (auth status, base URL, default model).
        * ``/provider <name>``     - points all three stages at that provider's default model.
        * ``/provider <name> <url>`` - same, plus a custom base URL (vLLM / LM Studio / proxy).
        """
        tokens = argument.split()
        if not tokens:
            if self.provider_configurer is not None:
                summary = await self.provider_configurer()
                refresh = await self.refresh_models()
                return self._emit_command_output("/provider", f"{summary}\n\n{refresh}")
            return self._emit_command_output("/provider", self._provider_table_text())

        from forge.config.settings import ProviderName

        try:
            name = ProviderName(tokens[0].lower())
        except ValueError:
            return self._emit_command_output(
                "/provider",
                f"Unknown provider {tokens[0]!r}. Supported: {', '.join(p.value for p in ProviderName)}.",
                status="failed",
            )

        manager = self._config_manager()
        if len(tokens) > 1 and tokens[1].strip():
            manager.set_base_url(name, tokens[1].strip())

        binding = manager.set_stage_model("all", f"{name.value}/{manager.provider(name).default_model}")
        await self.refresh_models()
        text = (
            f"[OK] All stages now target {binding.provider.value} "
            f"({manager.provider(name).base_url}).\n"
            f"  anvil | hammer | temper -> {binding.spec_string} [{binding.model_tier.value}]"
        )
        return self._emit_command_output("/provider", text)

    async def _handle_config_command(self) -> str:
        """Render the full (masked) persistent configuration and its file path (``/config``)."""
        manager = self._config_manager()
        return self._emit_command_output("/config", manager.describe())

    async def _handle_models_command(self, argument: str = "") -> str:
        """Interactively choose a model from connected providers, or assign via arguments."""
        tokens = argument.split()
        if not tokens:
            if self.model_picker is None:
                return self._emit_command_output("/models", self._model_matrix_text())
            summary = await self.model_picker(None)
            refresh = await self.refresh_models()
            return self._emit_command_output("/models", f"{summary}\n\n{refresh}")

        head = tokens[0].lower()
        if head == "list":
            provider_token = tokens[1] if len(tokens) > 1 else None
            text = await self._list_live_models(provider_token)
            return self._emit_command_output("/models", text)
        if head == "test":
            text = await self._test_stage_models()
            return self._emit_command_output("/models", text)

        if len(tokens) == 1:
            stage = _STAGE_ALIASES.get(head)
            if stage is not None:
                if self.model_picker is not None:
                    summary = await self.model_picker(stage.value)
                    refresh = await self.refresh_models()
                    return self._emit_command_output("/models", f"{summary}\n\n{refresh}")
                matrix = self._model_matrix_text()
                return self._emit_command_output(
                    "/models",
                    f"Stage: {stage.value.lower()}\n\n{matrix}",
                )
            if "/" in tokens[0]:
                text = await self._apply_model_spec("all", tokens[0])
                return self._emit_command_output("/models", text)
            return self._emit_command_output(
                "/models",
                f"Unknown stage {head!r}. Use one of: anvil | hammer | temper "
                "(aliases: orchestrator | executor | qa).",
                status="failed",
            )

        stage = _STAGE_ALIASES.get(head)
        target = stage.value.lower() if stage else head
        text = await self._apply_model_spec(target, tokens[1])
        return self._emit_command_output("/models", text)

    def render_usage_report(self) -> str:
        """Render the required three-part accounting: agents, cycles, session total."""
        agents = {stage.value.lower(): {"input": 0, "output": 0, "total": 0} for stage in ForgeStage}
        for cycle_stages in self.state.cycle_usage.values():
            for stage_name, usage in cycle_stages.items():
                target = agents.setdefault(
                    stage_name, {"input": 0, "output": 0, "total": 0}
                )
                target["input"] += int(usage.get("input", 0))
                target["output"] += int(usage.get("output", 0))
                target["total"] += int(usage.get("total", 0))

        lines = [
            "Forge Usage Accounting",
            f"Session: {self.session_id}   Cycle :- {self.state.cycle_count}",
            "",
            "Per Agent:",
        ]
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            usage = agents[stage.value.lower()]
            lines.append(
                f"  {stage.value:<7} input {usage['input']:>10,}  "
                f"output {usage['output']:>10,}  total {usage['total']:>10,}"
            )

        lines.extend(["", "Per Cycle:"])
        cycle_items = sorted(self.state.cycle_usage.items())
        if cycle_items:
            for cycle, cycle_stages in cycle_items:
                cycle_input = sum(int(item.get("input", 0)) for item in cycle_stages.values())
                cycle_output = sum(int(item.get("output", 0)) for item in cycle_stages.values())
                cycle_total = sum(int(item.get("total", 0)) for item in cycle_stages.values())
                lines.append(
                    f"  Cycle :- {cycle:<3} input {cycle_input:>10,}  "
                    f"output {cycle_output:>10,}  total {cycle_total:>10,}"
                )
        else:
            lines.append("  (no cycles executed in this session)")

        lines.extend(
            [
                "",
                "Session Grand Total:",
                f"  input {self.state.total_session_input_tokens:>12,}  "
                f"output {self.state.total_session_output_tokens:>12,}  "
                f"total {self.state.total_session_tokens:>12,}",
            ]
        )
        return "\n".join(lines)

    async def _handle_usage_command(self, argument: str = "") -> str:
        """Display per-agent, per-cycle and session accounting (``/usage``)."""
        if self.usage_viewer is not None and not argument.strip():
            await self.usage_viewer()
            return "Opened API Usage & Telemetry Dashboard."
        window = argument.strip().lower() or "session"
        if window not in {"all", "today", "session"}:
            return self._emit_command_output(
                "/usage",
                "Usage window must be one of: all, today, session.",
                status="failed",
            )
        from forge.config.usage_tracker import get_usage_tracker

        ledger = get_usage_tracker().render_dashboard_text(
            session_id=self.session_id, time_window=window
        )
        return self._emit_command_output(
            "/usage", f"{self.render_usage_report()}\n\n{ledger}"
        )

    async def _handle_resume_command(self, argument: str = "") -> str:
        """Resume conversation from global ~/.forge/sessions/ store (``/resume``)."""
        arg = argument.strip()
        from forge.config.session_store import get_session_store

        store = get_session_store()
        if not arg:
            if self.session_resumer is not None:
                await self.session_resumer()
                return "Opened Global Session Picker modal."
            sessions = store.list_sessions()
            if not sessions:
                return self._emit_command_output(
                    "/resume", "No past sessions found in global session store (~/.forge/sessions/)."
                )
            lines = ["Global Conversation Sessions (~/.forge/sessions/):", ""]
            lines.append(
                f"{'SESSION ID':<14} {'LAST ACTIVE':<18} {'TITLE':<32} {'TOKENS':>8} {'CYCLES':>6}"
            )
            lines.append("─" * 84)
            for s in sessions[:15]:
                ts = s.updated_at[:16].replace("T", " ") if s.updated_at else ""
                lines.append(
                    f"{s.session_id:<14} {ts:<18} {s.title[:30]:<32} {s.total_tokens:>8,} {s.cycle_count:>6}"
                )
            lines.append("")
            lines.append("To resume a session, run: /resume <session_id>")
            return self._emit_command_output("/resume", "\n".join(lines))

        ok = store.restore_session(arg, self.state)
        if ok:
            resolved_id = store.resolve_session_id(arg) or arg
            self.session_id = resolved_id
            for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
                transcript = self.state.stage_buffers[stage].chat_history
                self.conversation_history[stage] = [
                    {
                        "role": "user" if speaker.strip().lower() in {"you", "user"} else "assistant",
                        "content": message,
                    }
                    for speaker, message in transcript
                ]
                self._history[stage] = [
                    Message.user(message)
                    if speaker.strip().lower() in {"you", "user"}
                    else Message.assistant(message)
                    for speaker, message in transcript
                ]
            text = f"[OK] Restored session [{resolved_id}] into live Forge workspace."
            return self._emit_command_output("/resume", text)
        return self._emit_command_output(
            "/resume", f"Session [{arg}] not found in ~/.forge/sessions/.", status="failed"
        )

    async def _handle_restart_command(self) -> str:
        """Cleanly flush state, save configuration, and restart Forge workspace (``/restart``)."""
        from forge.config.session_store import get_session_store
        from forge.config.usage_tracker import get_usage_tracker

        get_session_store().save_session(self.state, self.session_id, self.root or Path.cwd())
        get_usage_tracker()._save()
        if self.config_manager is not None:
            self.config_manager.save()

        self.state.reset_view()
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            self.conversation_history[stage].clear()
            self._history[stage].clear()
        if "PYTEST_CURRENT_TEST" in os.environ:
            return self._emit_command_output("/restart", "Restarting Forge workspace fresh from scratch.")

        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except Exception as exc:
            return self._emit_command_output(
                "/restart", f"Restart execution failed: {exc}", status="failed"
            )
        return "Restarting Forge workspace..."

    async def refresh_models(self) -> str:
        """Reload configuration and rebuild every stage's adapter in place (no restart)."""
        if self.registry is None and self.root is None and self.config_manager is None:
            return "Model configuration saved. Restart Forge to activate the changes."

        from forge.cli.credentials import init_credentials_env
        from forge.config.loader import load_config
        from forge.models.profile import ModelProfile, ProviderConfig
        from forge.models.registry import ModelRegistry

        init_credentials_env()

        if self.registry is None:
            self.registry = ModelRegistry()

        has_project_file = False
        if self.root is not None and (self.root / "forge.yaml").is_file():
            try:
                config = load_config(self.root)
                self.registry.configure_providers(config.providers)
                for name, profile in config.models.items():
                    self.registry.add_profile(name, profile)
                for r_name, r_binding in config.roles.items():
                    self.registry.bind_role(r_name, r_binding.model)
                has_project_file = True
            except Exception as exc:
                return f"Saved, but could not reload the configuration: {exc}"

        if not has_project_file and self.config_manager is not None:
            manager = self._config_manager()
            for prov_enum, prov_cfg in manager.providers.items():
                if prov_cfg.api_key or prov_cfg.base_url:
                    env_var = prov_cfg.api_key_env
                    if env_var and prov_cfg.api_key:
                        os.environ[env_var] = prov_cfg.api_key
                    self.registry.provider_configs[prov_enum.value] = ProviderConfig(
                        base_url=prov_cfg.base_url,
                        api_key=prov_cfg.api_key,
                        api_key_env=prov_cfg.api_key_env,
                    )
            for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
                stage_binding = self._config_binding(stage)
                if stage_binding is not None:
                    role = _STAGE_ROLES[stage]
                    profile_name = stage_binding.spec_string
                    self.registry.add_profile(
                        profile_name,
                        ModelProfile(
                            provider=stage_binding.provider.value,
                            model=stage_binding.model,
                            detect_capabilities=False,
                        ),
                    )
                    self.registry.bind_role(role, profile_name)

        lines = ["Live model bindings refreshed:"]
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            agent = self._agent_for(stage)
            events = getattr(agent, "events", None) if agent else None
            role = _STAGE_ROLES[stage]
            try:
                adapter = await self.registry.adapter(role, events=events)
                if agent is not None:
                    agent.adapter = adapter
                lines.append(f"  • {stage.value}: {adapter.describe()}")
            except Exception as exc:
                stage_binding = self._config_binding(stage)
                label = stage_binding.spec_string if stage_binding else "no adapter"
                lines.append(f"  • {stage.value}: {label} ({exc})")
        return "\n".join(lines)

    async def _apply_model_spec(self, target: str, spec: str) -> str:
        """Persist a ``provider/model`` binding to a stage (or all) and rebind live adapters."""
        from forge.config.settings import parse_provider_model

        if target != "all" and target not in _STAGE_BY_NAME:
            return (
                f"Unknown stage '{target}'. Valid stages: anvil, hammer, temper, all."
            )

        try:
            provider_enum, model = parse_provider_model(spec)
        except ValueError:
            return (
                "Usage: /models <stage|all> <provider>/<model>   "
                "e.g. /models anvil gemini/gemini-2.5-pro or "
                "/models ollama/qwen2.5-coder:7b"
            )

        provider_value = provider_enum.value
        lines: list[str] = []
        if self.config_manager is not None:
            binding = self.config_manager.set_stage_model(target, spec)
            lines.append(
                f"[OK] Persisted {binding.spec_string} [{binding.model_tier.value}] to "
                f"{self.config_manager.config_path}"
            )

        if self.registry is not None:
            stages = list(ForgeStage) if target == "all" else [_STAGE_BY_NAME[target]]
            for stage in stages:
                lines.append(await self.swap_stage_model(stage, provider_value, model))

        if not lines:
            lines.append(f"Selected {provider_value}/{model} for {target}.")
        return "\n".join(lines)

    async def _list_live_models(self, provider_token: str | None) -> str:
        """Query and display a provider's live model catalog (``/models list [provider]``)."""
        from forge.config.settings import ProviderName

        if provider_token:
            try:
                provider = ProviderName(provider_token.lower())
            except ValueError:
                return f"Unknown provider {provider_token!r}. Supported: {', '.join(p.value for p in ProviderName)}."
            targets = [provider]
        else:
            targets = [
                name
                for name in ProviderName
                if name is ProviderName.OLLAMA
                or (self.config_manager is not None and self.config_manager.api_key_for(name))
            ]

        client = self._llm_client()
        lines = ["Live models:"]
        for name in targets:
            models = await client.list_live_models(name)
            rendered = ", ".join(models) if models else "(none reachable)"
            lines.append(f"  {name.value:<10}: {rendered}")
        return "\n".join(lines)

    async def _test_stage_models(self) -> str:
        """Ping every configured stage model and report live latency (``/models test``)."""
        client = self._llm_client()
        lines = ["Connection tests:"]
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            binding = self._config_binding(stage)
            if binding is None:
                lines.append(f"  ○ {stage.value:<7}: (no model attached)")
                continue
            ok, detail = await client.test_connection(binding)
            mark = "[OK]" if ok else "[X]"
            lines.append(f"  {mark} {stage.value:<7}: {binding.spec_string} — {detail}")
        return "\n".join(lines)

    async def swap_stage_model(
        self, stage: ForgeStage, provider: str | None, model: str
    ) -> str:
        """Rebind a stage to a new provider/model at runtime and rebuild its adapter."""
        if self.registry is None:
            from forge.models.registry import ModelRegistry

            self.registry = ModelRegistry()

        role = _STAGE_ROLES[stage]
        agent = self._agent_for(stage)
        current = getattr(getattr(agent, "adapter", None), "profile", None)
        resolved_provider = provider or getattr(current, "provider", None)
        if not resolved_provider:
            return (
                "Cannot resolve a provider for that model. Use "
                "<provider>/<model> (e.g. openrouter/google/gemini-2.5-flash)."
            )
        profile_name = f"{resolved_provider}/{model}"
        try:
            self.registry.add_profile(
                profile_name,
                ModelProfile(
                    provider=resolved_provider, model=model, detect_capabilities=False
                ),
            )
            self.registry.bind_role(role, profile_name)
            events = getattr(agent, "events", None) if agent else None
            adapter = await self.registry.adapter(role, events=events)
            if agent is not None:
                agent.adapter = adapter
            return f"[OK] {stage.value} now uses {adapter.describe()}"
        except Exception:
            return f"[OK] {stage.value} bound to {resolved_provider}/{model} (unified client active)"

    def current_model_label(self) -> str:
        """Short model id for the currently viewed stage (falls back to any attached stage)."""
        binding = self._config_binding(self.state.selected_view_stage)
        if binding is not None:
            return binding.spec_string
        for stage in (
            self.state.selected_view_stage,
            ForgeStage.ANVIL,
            ForgeStage.HAMMER,
            ForgeStage.TEMPER,
        ):
            adapter = getattr(self._agent_for(stage), "adapter", None)
            if adapter is not None:
                return str(getattr(adapter, "model", "unknown"))
        return "(no live model attached)"

    def _config_binding(self, stage: ForgeStage) -> StageModelBinding | None:
        """The persistent ``provider/model`` binding for a stage, if a config engine is attached."""
        if self.config_manager is None:
            return None
        return self.config_manager.binding_for(stage.value.lower())

    def _config_manager(self) -> ForgeConfigManager:
        """Return the attached config engine, lazily creating the user-level one if needed."""
        if self.config_manager is None:
            from forge.config.settings import ForgeConfigManager

            self.config_manager = ForgeConfigManager()
        else:
            self.config_manager.reload()
        return self.config_manager

    def _llm_client(self) -> UnifiedLLMClient:
        """Return the native streaming client, wiring it to the active config engine."""
        if self.llm_client is None:
            from forge.providers.client import UnifiedLLMClient

            manager = self._config_manager()
            self.llm_client = UnifiedLLMClient(
                api_key_resolver=manager.api_key_for,
                base_url_resolver=lambda p: manager.provider(p).base_url,
            )
        return self.llm_client

    def _model_matrix_text(self) -> str:
        """Aggregate every connected provider's models and show independent stage bindings."""
        manager = self._config_manager()
        lines = ["Forge Model Matrix", ""]
        connected = manager.get_connected_providers()
        lines.append(
            "Connected providers: "
            + (", ".join(provider.value for provider in connected) or "(none)")
        )
        aggregated = manager.get_aggregated_models()
        lines.append("")
        lines.append("Available models:")
        if aggregated:
            for provider, model in aggregated:
                lines.append(f"  {provider.value}/{model}")
        else:
            lines.append("  (connect a provider with /provider to aggregate its models)")
        lines.append("")
        lines.append("Stage bindings:")
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            binding = manager.binding_for(stage.value.lower())
            label = binding.spec_string if binding is not None else "(no model attached)"
            lines.append(f"  {stage.value:<7}: {label}")
        lines.extend(
            [
                "",
                "Use `/models <stage> <provider>/<model>` to rebind one stage, or "
                "`/models all <provider>/<model>` to fan out.",
                "Run `forge models config` for the terminal picker.",
            ]
        )
        return "\n".join(lines)

    def _model_list_text(self) -> str:
        lines = ["Forge Model Bindings:"]
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            binding = self._config_binding(stage)
            agent = self._agent_for(stage)
            adapter = getattr(agent, "adapter", None)
            if binding is not None:
                label = binding.spec_string
                tier = f"  [{binding.model_tier.value}]"
            elif adapter is not None:
                try:
                    label = adapter.describe()
                except Exception:
                    label = str(getattr(adapter, "model", "unknown"))
                tier = ""
            else:
                label = "(no live model attached)"
                tier = ""
            lines.append(f"  {stage.value:<7}: {label}{tier}")

        providers: list[str] = []
        if self.registry is not None:
            providers = sorted(getattr(self.registry, "provider_configs", {}) or {})
        if not providers:
            from forge.models.profile import BUILTIN_PROVIDERS

            providers = sorted(BUILTIN_PROVIDERS)
        lines.append("")
        lines.append(f"Providers: {', '.join(providers)}")
        lines.append("Stages: anvil | hammer | temper | all")
        lines.append("Examples:")
        lines.append("  /models gemini/gemini-2.5-flash        (all stages)")
        lines.append("  /models anvil gemini/gemini-2.5-pro    (one stage)")
        lines.append("  /models hammer ollama/qwen2.5-coder:7b")
        lines.append("  /models list openrouter   ·   /models test")
        return "\n".join(lines)

    def _provider_table_text(self) -> str:
        """Provider table with auth status, base URL and default model (``/provider``)."""
        from forge.config.settings import ProviderName

        manager = self._config_manager()
        width = max(len(name.value) for name in ProviderName)
        lines = [
            "Forge Providers  (connect with `forge provider` or the `/provider` wizard):",
            "",
        ]
        for name in ProviderName:
            cfg = manager.provider(name)
            status = cfg.masked_key() if cfg.key_configured else "NOT SET"
            lines.append(
                f"  {name.value:<{width}}  {status:<12}  {cfg.base_url}  (default: {cfg.default_model})"
            )
        lines.append("")
        lines.append("Connect: /provider   ·   Bind models: /models")
        return "\n".join(lines)

    def _help_text(self) -> str:
        rows = [
            ("/help", "Show this command reference"),
            ("/provider", "Connect, configure or disconnect model providers"),
            ("/models [stage]", "Assign connected models to Anvil, Hammer, Temper"),
            ("/usage [all|today|session]", "Per-agent, per-cycle and session usage accounting"),
            ("/resume [session_id]", "Resume a conversation from ~/.forge/sessions/"),
            ("/restart", "Flush state and restart Forge in place"),
            ("/mode [auto|chat|cycle]", "Switch adaptive execution mode"),
            ("/cycle <task>", "Plan a task, then run Anvil -> Hammer -> Temper autonomously"),
            ("/research <topic>", "Run the deep research pipeline"),
            ("/clear", "Clear the active stage viewport"),
            ("/reset", "Reset the whole session (Cycle #1, all buffers)"),
            ("/exit", "Exit the Forge workspace"),
            ("!<shell command>", "Run a real shell command (e.g. !pytest -q)"),
        ]
        width = max(len(name) for name, _ in rows)
        lines = ["Forge Interactive Commands:", ""]
        lines.extend(f"  {name:<{width}}  {desc}" for name, desc in rows)
        lines.append("")
        lines.append(self._shortcuts_text())
        return "\n".join(lines)

    def _shortcuts_text(self) -> str:
        return (
            "Shortcuts: Alt+1/F1 Anvil | Alt+2/F2 Hammer | Alt+3/F3 Temper | Alt+0/F4 Auto-follow\n"
            "           F5 Toggle mode | Alt+Left/Right History | PgUp/PgDn Scroll | Tab Complete\n"
            "           Ctrl+C Cancel Task | Ctrl+Q Exit"
        )

    def _provider_key_status(self, provider_name: str) -> tuple[str, str | None]:
        """Return (status, env_var) describing whether a provider looks reachable."""
        config = None
        if self.registry is not None:
            config = (getattr(self.registry, "provider_configs", {}) or {}).get(provider_name)
        env_var = getattr(config, "api_key_env", None)
        if env_var is None:
            from forge.models.profile import BUILTIN_PROVIDERS

            builtin = BUILTIN_PROVIDERS.get(provider_name)
            env_var = getattr(builtin, "api_key_env", None)
        if env_var is None:
            return "READY (no key required)", None
        return ("CONNECTED" if os.environ.get(env_var) else "MISSING KEY", env_var)

    def _status_text(self) -> str:
        lines = ["Forge Session Status", ""]
        lines.append(f"  Cycle      : #{self.state.cycle_count}  [{self.state.cycle_mode.value}]")
        mode_label = _mode_label(self.forced_mode)
        lines.append(f"  Routing    : {mode_label}")
        worker = self.state.active_worker_stage.value if self.state.active_worker_stage else "IDLE"
        lines.append(f"  Worker     : {worker}")
        lines.append(f"  View       : {self.state.selected_view_stage.value}")
        lines.append("")
        lines.append("  Stage models & connections:")
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            agent = self._agent_for(stage)
            adapter = getattr(agent, "adapter", None)
            if adapter is not None:
                try:
                    label = adapter.describe()
                except Exception:
                    label = str(getattr(adapter, "model", "unknown"))
                provider_name = getattr(getattr(adapter, "profile", None), "provider", "")
                status, env_var = self._provider_key_status(provider_name)
                if env_var and status != "CONNECTED":
                    suffix = f" [{status} · ${env_var}]"
                else:
                    suffix = f" [{status}]"
            else:
                label = "(no live model attached)"
                suffix = " [WARN · NO MODEL]"
            tokens = self.state.stage_buffers[stage].stage_tokens
            lines.append(f"    {stage.value:<7}: {label}{suffix}  [tokens: {tokens:,}]")
        lines.append("")
        lines.append(
            f"  Session tokens: {self.state.total_session_tokens:,} "
            f"(input {self.state.total_session_input_tokens:,} / "
            f"output {self.state.total_session_output_tokens:,})"
        )
        if self.state.pending_plan is not None:
            lines.append(
                f"  Plan: revision {self.state.pending_plan.revision} awaiting approval "
                f"({len(self.state.pending_plan.requirements)} requirements)"
            )
        return "\n".join(lines)
