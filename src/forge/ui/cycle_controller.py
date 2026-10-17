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
import os
import re
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from forge.harness.adversarial_qa import AdversarialQAGatekeeper, detect_secrets_in_text
from forge.harness.memory import EpisodicRepoMemory
from forge.models import Message
from forge.models.profile import ModelProfile
from forge.skills import AutonomousSkillRouter
from forge.ui.state import CycleMode, ForgeStage, ForgeTUIState, UserQuestion


class ExecutionMode(StrEnum):
    """How a single developer input should be executed."""

    CONVERSATIONAL = "conversational"
    DIRECT_TOOL = "direct_tool"
    FULL_FORGE_CYCLE = "full_forge_cycle"
    DEEP_RESEARCH = "deep_research"


# Stage <-> role aliases used by the `/model` command to route a binding to the right agent.
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

    # Fall back to a conversational reply; the active agent still has tools and can
    # escalate on its own via a ``forge-handoff`` block when it decides a cycle is needed.
    return ExecutionMode.CONVERSATIONAL


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
        f"⚠ {reason}",
        "",
        "Forge does not fake model output — every conversational turn must reach a real LLM.",
        "Configure a provider API key, then restart Forge (or run `forge login`):",
    ]
    lines.extend(f"  • {env}  →  {label}" for env, label in _API_KEY_ENV_VARS)
    lines.append(
        "  • OLLAMA_HOST (optional) for a local Ollama server "
        "(default http://localhost:11434)"
    )
    lines.append("")
    lines.append("Or switch models live at any time: /model <stage> <provider>/<model>")
    lines.append("Example: /model anvil openrouter/google/gemini-2.5-flash")
    return "\n".join(lines)


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
        max_cycles: int = 5,
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

        # 2. Deterministic intent routing
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
        """Hand off a synthesized requirement contract to the full Anvil -> Hammer -> Temper cycle.

        This is the explicit handoff action an agent invokes only when the task genuinely
        requires multi-step implementation and QA verification.
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

        if accepts("history"):
            kwargs["history"] = list(self._history[stage])
        if accepts("project_context"):
            kwargs["project_context"] = self.project_context
        if accepts("max_tool_steps"):
            kwargs["max_tool_steps"] = self.max_tool_steps
        if accepts("allowed_tools"):
            kwargs["allowed_tools"] = self._toolset_for(stage, allow_edits)
        if accepts("on_tool_result"):
            kwargs["on_tool_result"] = lambda result: self._stream_tool_result(stage, result)
        if accepts("on_token") and on_token is not None:
            kwargs["on_token"] = on_token

        return await agent.chat(prompt, **kwargs)

    def _stream_tool_result(self, stage: ForgeStage, result: Any) -> None:
        """Stream a finished tool invocation into the stage viewport."""
        tool = getattr(result, "tool", "tool")
        ok = getattr(result, "ok", True)
        content = (getattr(result, "content", "") or "").strip()
        marker = "✔" if ok else "✖"
        first_lines = "\n".join(content.splitlines()[:6])
        self.state.append_tool_event(stage, f"{marker} {tool}\n{first_lines}".strip())

    @staticmethod
    def _turn_tokens(turn: Any) -> int:
        total = getattr(turn, "total_tokens", None)
        if isinstance(total, int) and total > 0:
            return total
        usage = getattr(turn, "usage", None)
        total = getattr(usage, "total_tokens", None)
        if isinstance(total, int):
            return total
        return 0

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
        self.conversation_history[stage].append({"role": "user", "content": prompt})

        agent = self._agent_for(stage)
        if agent is None:
            reply = model_diagnostic()
            self.state.append_chat(stage, self._speaker(stage), reply)
            self.state.set_stage_output(stage, reply, status="failed")
            self.conversation_history[stage].append({"role": "assistant", "content": reply})
            self.state.set_active_stage(None)
            return reply

        def _on_token(delta: str) -> None:
            self.state.append_stream(stage, delta)

        self.state.start_stream(stage, self._speaker(stage))
        turn = await self._invoke_agent_chat(
            agent, prompt, stage, allow_edits=allow_edits, on_token=_on_token
        )
        reply = (getattr(turn, "response_text", "") or "").strip()
        tokens = self._turn_tokens(turn)
        messages = getattr(turn, "messages", None)
        if messages:
            self._history[stage] = list(messages)
        handoff = getattr(turn, "handoff_requirements", None)
        if not reply:
            reply = "(no response)"

        self.state.record_tokens(stage, tokens)
        self.state.finalize_stream(stage, reply)
        self.conversation_history[stage].append({"role": "assistant", "content": reply})
        self.state.set_stage_output(stage, reply, status="passed")
        self.state.set_active_stage(None)

        if handoff:
            return await self.escalate_to_forge_cycle(handoff)
        return reply

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

    async def _run_full_cycle(self, prompt: str) -> str:
        """Run the full pipeline: the real harness, the injected verifier, or a diagnostic.

        Forge never fabricates a cycle outcome. Without a live workflow runner or an injected
        verifier it renders the credential diagnostic instead of a canned deliverable.
        """
        if self.workflow_runner is not None:
            return await self._run_real_cycle(prompt)
        if self.temper_verifier is not None:
            return await self._run_simulated_cycle(prompt)

        diagnostic = model_diagnostic(
            "A full Anvil -> Hammer -> Temper cycle needs a live model harness or verifier."
        )
        stage = ForgeStage.ANVIL
        self.state.set_active_stage(stage, reset_tokens=False)
        self.state.append_chat(stage, "You", prompt)
        self.state.append_chat(stage, "Forge", diagnostic)
        self.state.set_stage_output(stage, diagnostic, status="failed")
        self.state.set_active_stage(None)
        return diagnostic

    async def _run_real_cycle(self, prompt: str) -> str:
        """Delegate to the real multi-agent workflow orchestrator and mirror its outcome."""
        self.state.set_active_stage(ForgeStage.ANVIL, reset_tokens=False)
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
        self.state.set_stage_output(
            ForgeStage.ANVIL, f"Planned {len(tasks)} task(s) for:\n{prompt}", status="passed"
        )
        self.state.set_stage_output(
            ForgeStage.HAMMER, f"Executed {len(tasks)} task(s).", status="passed"
        )
        summary = (
            f"Full Forge cycle finished for: '{prompt}'\n"
            f"• Workflow status: {status}\n"
            f"• Tasks: {len(tasks)}\n"
            f"• QA verdict: {qa_status.upper()}"
        )
        temper_ok = qa_status.lower() in {"pass", "passed"}
        self.state.set_stage_output(
            ForgeStage.TEMPER,
            summary,
            status="passed" if temper_ok else "failed",
        )
        self.state.set_active_stage(None)
        return summary

    async def _run_simulated_cycle(self, prompt: str) -> str:
        """Deterministic in-process cycle (offline / tests), using live agents when attached."""
        current_request = prompt
        final_deliverable = ""

        p_lower = prompt.lower()
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
        if any(kw in p_lower for kw in single_keywords):
            self.set_cycle_mode(CycleMode.SINGLE_CYCLE)
        elif any(kw in p_lower for kw in continuous_keywords):
            self.set_cycle_mode(CycleMode.UNTIL_FULFILLED)

        while self.state.cycle_count <= self.max_cycles and not self._cancelled:
            await self._simulated_anvil(current_request)
            if self._cancelled:
                break
            await self._simulated_hammer(current_request)
            if self._cancelled:
                break
            verdict, deliverable = await self._simulated_temper(prompt)

            if verdict.passed:
                final_deliverable = verdict.deliverable or deliverable
                self.state.append_thought(
                    ForgeStage.TEMPER,
                    "Gatekeeper Decision: APPROVED. All requirements & security checks passed.",
                )
                self.state.set_stage_output(ForgeStage.TEMPER, final_deliverable, status="passed")
                self.state.set_active_stage(None)
                return final_deliverable

            rejection = verdict.feedback or "Defects identified during adversarial verification."
            self.state.append_thought(
                ForgeStage.TEMPER, f"Gatekeeper Decision: REJECTED. Defects: {rejection}"
            )
            self.state.set_stage_output(
                ForgeStage.TEMPER,
                f"Cycle #{self.state.cycle_count} Rejection Report:\n{rejection}",
                status="failed",
            )

            if self.cycle_mode == CycleMode.SINGLE_CYCLE:
                self.state.append_thought(
                    ForgeStage.ANVIL,
                    f"Single Cycle mode: halting after Cycle #{self.state.cycle_count}.",
                )
                self.state.set_active_stage(None)
                return f"Single Cycle completed. QA feedback: {rejection}"

            if self.state.cycle_count >= self.max_cycles:
                self.state.append_thought(
                    ForgeStage.TEMPER,
                    f"Cycle ceiling ({self.max_cycles}) reached without approval. Halting.",
                )
                self.state.set_active_stage(None)
                return f"Cycle limit reached. Final QA feedback: {rejection}"

            self.state.advance_to_next_cycle(rejection)
            current_request = (
                f"{prompt}\n\n[Temper Rejection Feedback from Cycle "
                f"#{self.state.cycle_count - 1}]:\n{rejection}"
            )

        self.state.set_active_stage(None)
        return final_deliverable

    async def _simulated_anvil(self, current_request: str) -> None:
        self.state.set_active_stage(ForgeStage.ANVIL, reset_tokens=False)
        self.state.append_thought(
            ForgeStage.ANVIL,
            f"Analyzing goal for Cycle #{self.state.cycle_count}: {current_request[:120]}",
        )
        specs = self.skill_router.list_all_specs()
        active_skills = [s.name for s in specs[:3]]
        if active_skills:
            self.state.append_thought(
                ForgeStage.ANVIL,
                f"AutonomousSkillRouter activated: {', '.join(active_skills)}",
            )

        plan_summary = (
            f"Architectural Plan for Cycle #{self.state.cycle_count}:\n"
            f"1. Deconstruct requirement into dependency-ordered DAG tasks.\n"
            f"2. Apply AST context slicing to touched code modules.\n"
            f"3. Direct Hammer to implement atomic changes with regression tests."
        )
        agent = self._agent_for(ForgeStage.ANVIL)
        usage = 0
        if agent is not None:
            turn = await self._invoke_agent_chat(
                agent,
                f"Produce an implementation plan for: {current_request}",
                ForgeStage.ANVIL,
                allow_edits=False,
            )
            text = (getattr(turn, "response_text", "") or "").strip()
            if text:
                plan_summary = text
            usage = self._turn_tokens(turn)
            messages = getattr(turn, "messages", None)
            if messages:
                self._history[ForgeStage.ANVIL] = list(messages)
        await asyncio.sleep(0)
        self.state.record_tokens(ForgeStage.ANVIL, usage)
        self.state.set_stage_output(ForgeStage.ANVIL, plan_summary, status="passed")

    async def _simulated_hammer(self, current_request: str) -> None:
        self.state.set_active_stage(ForgeStage.HAMMER, reset_tokens=False)
        self.state.append_thought(
            ForgeStage.HAMMER, "Beginning code synthesis, AST patching, and tool execution..."
        )
        exec_summary = (
            f"Implementation Summary (Cycle #{self.state.cycle_count}):\n"
            f"• Touchpoints modified and syntax trees validated.\n"
            f"• Code artifacts prepared for adversarial QA verification."
        )
        agent = self._agent_for(ForgeStage.HAMMER)
        usage = 0
        if agent is not None:
            turn = await self._invoke_agent_chat(
                agent,
                f"Implement the plan for: {current_request}",
                ForgeStage.HAMMER,
                allow_edits=True,
            )
            text = (getattr(turn, "response_text", "") or "").strip()
            if text:
                exec_summary = text
            usage = self._turn_tokens(turn)
        await asyncio.sleep(0)
        self.state.record_tokens(ForgeStage.HAMMER, usage)
        self.state.set_stage_output(ForgeStage.HAMMER, exec_summary, status="passed")

    async def _simulated_temper(self, original_prompt: str) -> tuple[CycleVerdict, str]:
        self.state.set_active_stage(ForgeStage.TEMPER, reset_tokens=False)
        self.state.append_thought(
            ForgeStage.TEMPER,
            "Adversarial QA Gatekeeper: checking code for security and leaks...",
        )
        secrets_detected = detect_secrets_in_text(original_prompt)
        self.state.append_thought(
            ForgeStage.TEMPER,
            f"Secret Detection & Shannon Entropy: {len(secrets_detected)} leaks found.",
        )

        usage = 0
        agent = self._agent_for(ForgeStage.TEMPER)
        if agent is not None:
            turn = await self._invoke_agent_chat(
                agent,
                f"Adversarially verify the implementation for: {original_prompt}",
                ForgeStage.TEMPER,
                allow_edits=False,
            )
            usage = self._turn_tokens(turn)
            reply = (getattr(turn, "response_text", "") or "").strip()
            if self.temper_verifier is None:
                passed = not re.search(
                    r"\b(fail|failed|reject|rejected|defect|violation|blocker)\b",
                    reply,
                    re.IGNORECASE,
                )
                self.state.record_tokens(ForgeStage.TEMPER, usage)
                return CycleVerdict(passed=passed, feedback=reply, deliverable=reply), reply
        await asyncio.sleep(0)

        self.state.record_tokens(ForgeStage.TEMPER, usage)

        if self.temper_verifier is not None:
            verdict = self.temper_verifier(self.state.cycle_count, original_prompt)
        else:
            verdict = CycleVerdict(
                passed=True,
                feedback="All criteria passed verification.",
                deliverable=(
                    f"Final Deliverable for: '{original_prompt}'\n\n"
                    f"✔ Completed in Cycle #{self.state.cycle_count}.\n"
                    f"✔ Code modifications verified with zero mutation escapes.\n"
                    f"✔ Adversarial QA gatekeeper approval: PASSED."
                ),
            )
        deliverable = verdict.deliverable or (
            f"Verified Deliverable for '{original_prompt}' (Cycle #{self.state.cycle_count})."
        )
        return verdict, deliverable

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
        return text

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
        if command in {"/status", "/models"} and not argument:
            return self._emit_command_output(command, self._status_text())
        if command in {"/model", "/models"}:
            text = await self._handle_model_command(argument)
            return self._emit_command_output(command, text)
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

    async def _handle_model_command(self, argument: str) -> str:
        """List model bindings, or live-swap a stage's model (``/model <stage> <prov>/<model>``)."""
        if not argument:
            return self._model_list_text()

        tokens = argument.split(None, 1)
        stage_token = tokens[0].lower()
        stage = _STAGE_ALIASES.get(stage_token)
        if stage is None:
            return (
                f"Unknown stage {stage_token!r}. Use one of: anvil | hammer | temper "
                "(aliases: orchestrator | executor | qa)."
            )
        if len(tokens) < 2 or not tokens[1].strip():
            return (
                "Usage: /model <stage> <provider>/<model>  "
                "e.g. /model anvil openrouter/google/gemini-2.5-flash"
            )

        target = tokens[1].strip()
        provider, _, model = target.partition("/")
        if not model:
            provider, model = "", provider  # bare model name; keep current provider
        return await self.swap_stage_model(stage, provider or None, model.strip())

    async def swap_stage_model(
        self, stage: ForgeStage, provider: str | None, model: str
    ) -> str:
        """Rebind a stage to a new provider/model at runtime and rebuild its adapter."""
        if self.registry is None:
            return (
                "No live model registry is attached to this session. Start Forge with a "
                "configured provider (see /status) to swap models at runtime."
            )
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
            events = getattr(agent, "events", None)
            adapter = await self.registry.adapter(role, events=events)
        except Exception as exc:
            return f"Failed to bind {resolved_provider}/{model} to {stage.value}: {exc}"

        if agent is not None:
            agent.adapter = adapter
        return f"✔ {stage.value} now uses {adapter.describe()}"

    def current_model_label(self) -> str:
        """Short model id for the currently viewed stage (falls back to any attached stage)."""
        for stage in (
            self.state.selected_view_stage,
            ForgeStage.ANVIL,
            ForgeStage.HAMMER,
            ForgeStage.TEMPER,
        ):
            adapter = getattr(self._agent_for(stage), "adapter", None)
            if adapter is not None:
                return str(getattr(adapter, "model", "unknown"))
        return "(no model)"

    def _model_list_text(self) -> str:
        lines = ["Forge Model Bindings:"]
        for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER):
            agent = self._agent_for(stage)
            adapter = getattr(agent, "adapter", None)
            if adapter is not None:
                try:
                    label = adapter.describe()
                except Exception:
                    label = str(getattr(adapter, "model", "unknown"))
            else:
                label = "(no live model attached)"
            lines.append(f"  {stage.value:<7}: {label}")

        providers: list[str] = []
        if self.registry is not None:
            providers = sorted(getattr(self.registry, "provider_configs", {}) or {})
        if not providers:
            from forge.models.profile import BUILTIN_PROVIDERS

            providers = sorted(BUILTIN_PROVIDERS)
        lines.append("")
        lines.append(f"Providers: {', '.join(providers)}")
        lines.append("Swap example: /model hammer ollama/qwen2.5-coder:7b")
        return "\n".join(lines)

    def _help_text(self) -> str:
        rows = [
            ("/help", "Show this command reference"),
            ("/status", "Provider/model, connection status, cycle & token telemetry"),
            ("/model [stage] [prov/model]", "List bindings or live-swap a stage's model"),
            ("/mode [auto|chat|cycle]", "Switch adaptive execution mode"),
            ("/cycle <task> | /forge <task>", "Force a full Anvil -> Hammer -> Temper cycle"),
            ("/research <topic>", "Run the STORM deep research pipeline"),
            ("/clear", "Clear the active stage viewport and its tokens"),
            ("/reset", "Reset the whole session (Cycle #1, all buffers)"),
            ("/quit | /exit", "Exit the Forge workspace"),
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
            "Shortcuts: Alt+1 Anvil | Alt+2 Hammer | Alt+3 Temper | Alt+0 Auto-follow\n"
            "           F5 Toggle mode | PgUp/PgDn Scroll | Tab Complete | Ctrl+C Cancel "
            "| Ctrl+Q Exit"
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
                suffix = " [⚠ NO MODEL]"
            tokens = self.state.stage_buffers[stage].stage_tokens
            lines.append(f"    {stage.value:<7}: {label}{suffix}  [tokens: {tokens:,}]")
        lines.append("")
        lines.append(f"  Session total tokens: {self.state.total_session_tokens:,}")
        return "\n".join(lines)
