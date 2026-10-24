"""Reactive TUI State, Stage Buffers & Dual Token Counters for Forge Split-Pane Workspace."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import io
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.text import Text


class ForgeStage(StrEnum):
    """The 3 blacksmith stages in Forge's cybernetic pipeline."""

    ANVIL = "Anvil"
    HAMMER = "Hammer"
    TEMPER = "Temper"


class CycleMode(StrEnum):
    """Dynamic cycle loop execution policy."""

    SINGLE_CYCLE = "Single Cycle"
    UNTIL_FULFILLED = "Until Fulfilled"


@dataclass
class UserQuestion:
    """Interactive question posed to the user with selectable options or custom text."""

    question: str
    options: list[str] = field(default_factory=list)
    default_option: str | None = None
    allow_custom: bool = True
    stage: ForgeStage = ForgeStage.ANVIL
    future: asyncio.Future[str] | None = None


class IntentCategory(StrEnum):
    """Broad categories for intent-aware decision making."""

    APPROVAL = "approval"
    REVISION = "revision"
    CLARIFICATION = "clarification"
    REJECTION = "rejection"
    CANCELLATION = "cancellation"
    INFORMATIONAL = "informational"


class PlanTurnIntent(StrEnum):
    """Semantic meaning of a developer's reply to a pending plan."""

    APPROVE_AS_IS = "APPROVE_AS_IS"
    MODIFY_AND_PROCEED = "MODIFY_AND_PROCEED"
    MODIFY_AND_WAIT = "MODIFY_AND_WAIT"
    QUESTION_OR_CHAT = "QUESTION_OR_CHAT"
    CANCEL_PLAN = "CANCEL_PLAN"
    REJECTION = "REJECTION"
    CLARIFICATION = "CLARIFICATION"
    INFORMATIONAL = "INFORMATIONAL"


class RequirementItem(BaseModel):
    """One independently verifiable item in Anvil's acceptance contract."""

    req_id: str
    description: str
    completed: bool = False
    verification_note: str = ""


class PendingPlanContract(BaseModel):
    """Structured plan contract presented to the developer for approval."""

    problem_statement: str
    plan_summary_markdown: str
    target_files: list[str] = Field(default_factory=list)
    requirements_checklist: list[RequirementItem] = Field(default_factory=list)


class AnvilPlanEvaluation(BaseModel):
    """Anvil's semantic adjudication of one reply to a pending plan."""

    intent: PlanTurnIntent
    category: IntentCategory = IntentCategory.APPROVAL
    approval_granted: bool = False
    plan_must_change: bool = False
    requested_modifications: list[str] = Field(default_factory=list)
    scope_restrictions: list[str] = Field(default_factory=list)
    outstanding_questions: list[str] = Field(default_factory=list)
    execution_authorized: bool = False
    clarification_necessary: bool = False
    confidence: float = 1.0
    explanation: str = ""
    reasoning_summary: str = ""
    updated_plan_markdown: str | None = None
    updated_target_files: list[str] | None = None
    updated_requirements: list[RequirementItem] | None = None
    conversational_reply: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.explanation and self.reasoning_summary:
            self.explanation = self.reasoning_summary
        elif not self.reasoning_summary and self.explanation:
            self.reasoning_summary = self.explanation

        # Infer defaults based on intent if not explicitly set
        if self.intent in {PlanTurnIntent.APPROVE_AS_IS}:
            self.category = IntentCategory.APPROVAL
            self.approval_granted = True
            self.execution_authorized = True
        elif self.intent in {PlanTurnIntent.MODIFY_AND_PROCEED}:
            self.category = IntentCategory.REVISION
            self.approval_granted = True
            self.plan_must_change = True
            self.execution_authorized = True
        elif self.intent in {PlanTurnIntent.MODIFY_AND_WAIT}:
            self.category = IntentCategory.REVISION
            self.approval_granted = False
            self.plan_must_change = True
            self.execution_authorized = False
        elif self.intent in {PlanTurnIntent.CANCEL_PLAN}:
            self.category = IntentCategory.CANCELLATION
            self.approval_granted = False
            self.execution_authorized = False
        elif self.intent in {PlanTurnIntent.REJECTION}:
            self.category = IntentCategory.REJECTION
            self.approval_granted = False
            self.execution_authorized = False
        elif self.intent in {PlanTurnIntent.CLARIFICATION, PlanTurnIntent.QUESTION_OR_CHAT}:
            self.category = IntentCategory.CLARIFICATION
            self.approval_granted = False
            self.execution_authorized = False
            self.clarification_necessary = True
        elif self.intent in {PlanTurnIntent.INFORMATIONAL}:
            self.category = IntentCategory.INFORMATIONAL
            self.approval_granted = False
            self.execution_authorized = False



@dataclass
class PlanRequirement:
    """A single acceptance requirement extracted from an approved plan."""

    req_id: str
    text: str
    met: bool = False
    evidence: str = ""

    def checklist_line(self) -> str:
        """Render a deterministic, audit-friendly checklist line."""
        mark = "[✓]" if self.met else "[✗]"
        suffix = f"  {self.evidence.strip()}" if self.evidence.strip() else ""
        return f"{mark} {self.req_id}: {self.text}{suffix}"

    @classmethod
    def from_item(cls, item: RequirementItem) -> PlanRequirement:
        """Adapt a Pydantic checklist item to the internal plan requirement."""
        return cls(
            req_id=item.req_id,
            text=item.description,
            met=item.completed,
            evidence=item.verification_note,
        )

    def to_item(self) -> RequirementItem:
        """Adapt this requirement to Anvil's structured contract."""
        return RequirementItem(
            req_id=self.req_id,
            description=self.text,
            completed=self.met,
            verification_note=self.evidence,
        )


@dataclass
class PendingPlan:
    """Structured plan awaiting semantic approval before autonomous execution."""

    request: str
    plan_text: str
    requirements: list[PlanRequirement] = field(default_factory=list)
    revision: int = 1
    target_files: list[str] = field(default_factory=list)

    @property
    def all_requirements_met(self) -> bool:
        """True only when every acceptance requirement has been verified."""
        return bool(self.requirements) and all(req.met for req in self.requirements)

    def checklist_text(self) -> str:
        """Render the full requirement checklist in stable REQ order."""
        return "\n".join(req.checklist_line() for req in self.requirements)

    def render(self) -> str:
        """Render the plan plus its requirement contract for the viewport."""
        checklist = self.checklist_text()
        if checklist:
            return f"{self.plan_text.strip()}\n\nRequirements:\n{checklist}"
        return self.plan_text.strip()

    def to_contract(self) -> PendingPlanContract:
        """Expose the plan as the Pydantic contract Anvil adjudicates."""
        return PendingPlanContract(
            problem_statement=self.request,
            plan_summary_markdown=self.plan_text,
            target_files=list(self.target_files),
            requirements_checklist=[req.to_item() for req in self.requirements],
        )

    @classmethod
    def from_contract(
        cls,
        contract: PendingPlanContract,
        *,
        revision: int = 1,
    ) -> PendingPlan:
        """Build the internal pending plan from Anvil's structured contract."""
        return cls(
            request=contract.problem_statement,
            plan_text=contract.plan_summary_markdown,
            requirements=[
                PlanRequirement.from_item(item)
                for item in contract.requirements_checklist
            ],
            revision=revision,
            target_files=list(contract.target_files),
        )


def format_tokens(count: int) -> str:
    """Format token count into compact human-readable string (e.g. 840, 3.2k, 125.4k, 1.40M)."""
    if count < 0:
        return f"-{format_tokens(-count)}"
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        val = count / 1000.0
        return f"{val:.1f}k"
    val = count / 1_000_000.0
    return f"{val:.2f}M"


def _turn_marker(speaker: str) -> tuple[str, str]:
    """Map a transcript speaker to a minimalist marker + style (no emoji)."""
    low = speaker.strip().lower()
    if low in {"you", "user", "developer"}:
        return "❯ user", "bold #e2e8f0"
    if low == "anvil":
        return "● anvil", "bold #3b82f6"
    if low == "hammer":
        return "● hammer", "bold #ef4444"
    if low == "temper":
        return "● temper", "bold #10b981"
    return "◆ forge", "bold #94a3b8"


# Braille spinner frames for the ephemeral 2-line live-activity indicator.
TRANSIENT_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Stage accent colours (Anvil blue, Hammer red, Temper green).
_STAGE_ACCENT: dict[ForgeStage, str] = {
    ForgeStage.ANVIL: "#3b82f6",
    ForgeStage.HAMMER: "#ef4444",
    ForgeStage.TEMPER: "#10b981",
}

# Human-readable verb templates for the transient live-activity subline.
_LIST_TOOLS = frozenset({"list_files", "list_dir", "list_directory", "ls", "list"})
_READ_TOOLS = frozenset({"read_file", "read", "view_file", "open_file"})
_SEARCH_TOOLS = frozenset(
    {
        "search_code",
        "search_symbols",
        "symbol_search",
        "search",
        "grep",
        "find",
        "find_files",
    }
)
_WRITE_TOOLS = frozenset(
    {"write_file", "edit_file", "quick_edit", "apply_patch", "patch", "write", "edit"}
)
_MKDIR_TOOLS = frozenset({"create_directory", "mkdir", "make_directory"})
_SHELL_TOOLS = frozenset(
    {"run_shell", "run_command", "run_tests", "shell", "bash", "execute", "cmd"}
)
_AST_TOOLS = frozenset(
    {
        "ast_slice",
        "skeletonize",
        "ast_grep",
        "file_outline",
        "outline",
        "dependency_analysis",
        "repository_structure",
        "repo_structure",
    }
)
_MUTATION_TOOLS = frozenset({"mutation_test", "mutate", "mutation"})

_ARG_KEYS = (
    "path",
    "file",
    "file_path",
    "filepath",
    "query",
    "pattern",
    "command",
    "cmd",
    "dir",
    "directory",
    "target",
    "name",
)

_MARKDOWN_NOISE = re.compile(r"[`*_>#]")
_WHITESPACE = re.compile(r"\s+")


def _extract_arg(value: object) -> str:
    """Best-effort extraction of the salient argument (path / query / command) from a tool call."""
    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in _ARG_KEYS:
            candidate = value.get(key)
            if candidate:
                return str(candidate).strip()
        for candidate in value.values():
            if isinstance(candidate, (str, int, float)) and str(candidate).strip():
                return str(candidate).strip()
        return ""
    return str(value).strip()


def _clean_thought(text: str) -> str:
    """Flatten a raw streaming thought chunk into a single clean < 70 char clause."""
    flat = re.sub(r"[\r\n]+", " ", text or "")
    flat = _MARKDOWN_NOISE.sub("", flat)
    flat = _WHITESPACE.sub(" ", flat).strip(" -•·\t")
    if not flat:
        return "thinking..."
    for sep in (". ", "! ", "? ", "; ", " - "):
        head = flat.split(sep, 1)[0].strip()
        if head:
            flat = head
            break
    flat = flat[:67].strip()
    if not flat:
        return "thinking..."
    flat = flat[0].lower() + flat[1:]
    return flat if flat.endswith("...") else f"{flat}..."


def format_transient_action(tool_name: str, args_or_thought: object = "") -> str:
    """Convert a tool call or raw thought chunk into a clean 1-line Claude-Code status phrase.

    Examples::

        format_transient_action("read_file", {"path": "forge/ui/state.py"})
            -> "reading forge/ui/state.py..."
        format_transient_action("run_shell", {"command": "pytest -q"})
            -> "running command `pytest -q`..."
        format_transient_action("", "Evaluating module dependencies. Next...")
            -> "evaluating module dependencies..."
    """
    name = (tool_name or "").strip()
    low = name.lower()
    arg = _extract_arg(args_or_thought)

    if low in _LIST_TOOLS:
        return f"accessing folder {arg or '.'}..."
    if low in _READ_TOOLS:
        return f"reading {arg or 'file'}..."
    if low in _SEARCH_TOOLS:
        return f"searching codebase for '{arg or 'symbols'}'..."
    if low in _WRITE_TOOLS:
        return f"applying changes to {arg or 'file'}..."
    if low in _MKDIR_TOOLS:
        return f"creating directory {arg or '.'}..."
    if low in _SHELL_TOOLS:
        return f"running command `{arg or name or 'command'}`..."
    if low in _AST_TOOLS:
        return f"analyzing AST dependencies{f' for {arg}' if arg else ''}..."
    if low in _MUTATION_TOOLS:
        return "running mutation tests on diff..."

    combined = " ".join(part for part in (name, arg) if part)
    return _clean_thought(combined)


_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _strip_long_code_blocks(text: str, max_lines: int = 3) -> str:
    """Replace any fenced code block longer than ``max_lines`` with a clean 1-line reference."""

    def _replace(match: re.Match[str]) -> str:
        body = match.group(1).rstrip("\n")
        if len(body.splitlines()) > max_lines:
            return "• [Target Implementation: see Hammer stage]"
        return match.group(0)

    return _FENCE_RE.sub(_replace, text)


def _ensure_heading(text: str, heading: str) -> str:
    if not text:
        return heading
    if text.lstrip().lower().startswith(heading.lower()):
        return text
    return f"{heading}\n{text}"


def format_stage_deliverable(stage: ForgeStage, raw_text: str) -> str:
    """Enforce role-pure final output for each stage (Anvil plan, Hammer report, Temper verdict).

    * **Anvil** never shows implementation source dumps: any fenced code block longer than a
      few lines is replaced with a clean reference to the Hammer stage.
    * **Hammer** leads with an ``Implementation Report`` heading (files/diff/commands).
    * **Temper** leads with a ``QA & Verification Verdict`` heading.

    Inline ``<think>`` blocks are stripped here as a last line of defence so hidden
    chain-of-thought can never reach the viewport, the transcript, or a stage deliverable.
    """
    from forge.providers.client import strip_thinking_tags

    text = strip_thinking_tags(raw_text or "")[0].strip()
    if not text:
        return ""
    if stage is ForgeStage.ANVIL:
        return _strip_long_code_blocks(text).strip()
    if stage is ForgeStage.HAMMER:
        return _ensure_heading(text, "Implementation Report")
    if stage is ForgeStage.TEMPER:
        return _ensure_heading(text, "QA & Verification Verdict")
    return text


def _render_structured_tool_event(console: Console, event: str) -> bool:
    """Render persistent file/command evidence with real diff colouring.

    Returns True when the event was handled here; legacy plain-text tool events still go
    through the compact ``exec:`` renderer in :meth:`StageBuffer.render_to_ansi`.
    """
    lines = event.splitlines()
    if not lines:
        return False
    first = lines[0]
    if first.startswith("[FILE]"):
        console.print(Text(first, style="bold #f87171"))
        for line in lines[1:]:
            if line.startswith(("+++", "---")):
                style = "#94a3b8"
            elif line.startswith("+"):
                style = "#22c55e"
            elif line.startswith("-"):
                style = "#ef4444"
            elif line.startswith("@@"):
                style = "bold #38bdf8"
            else:
                style = "#94a3b8"
            console.print(Text(line, style=style))
        return True
    if first.startswith("[CMD]"):
        console.print(Text(first, style="bold #f59e0b"))
        for line in lines[1:]:
            if line.startswith("[exit 0"):
                style = "bold #22c55e"
            elif line.startswith("[exit "):
                style = "bold #ef4444"
            elif line.startswith("[passed=") or line.startswith("[failed="):
                style = "#94a3b8"
            else:
                style = "#cbd5e1"
            console.print(Text(line, style=style))
        return True
    return False


@dataclass
class StageBuffer:
    """Isolated, async-safe buffer storing execution state and thoughts for a single stage."""

    stage: ForgeStage
    thoughts: list[str] = field(default_factory=list)
    tool_events: list[str] = field(default_factory=list)
    chat_history: list[tuple[str, str]] = field(default_factory=list)
    final_output: str = ""
    stage_tokens: int = 0
    stage_input_tokens: int = 0
    stage_output_tokens: int = 0
    scroll_line: int = 0
    scroll_offset: int = 0
    total_rendered_lines: int = 0
    status: Literal["idle", "running", "passed", "failed"] = "idle"
    is_working: bool = False
    spinner_index: int = 0
    live_headline: str = "Thinking..."
    live_subline: str = ""

    def render_to_ansi(
        self,
        width: int = 80,
        pending_question: UserQuestion | None = None,
        height: int | None = None,
    ) -> str:
        """Render the stage buffer into clean, flat ANSI-formatted text (no boxes, no emoji)."""
        effective_width = max(30, width)
        sio = io.StringIO()
        console = Console(
            file=sio,
            force_terminal=True,
            color_system="truecolor",
            width=effective_width,
            highlight=False,
        )

        def rule() -> str:
            return "─" * (effective_width - 1)

        def print_rule() -> None:
            console.print(Text(rule(), style="#334155"))

        # 1. Conversational transcript, edge-to-edge with a subtle rule between turns.
        for speaker, message in self.chat_history:
            label, style = _turn_marker(speaker)
            console.print(Text(label, style=style))
            body = message.strip("\n")
            if body:
                for para in body.splitlines():
                    console.print(Text(f"  {para}", style="#e2e8f0"))
            print_rule()

        # 2. Thought process.
        if self.thoughts:
            console.print(Text("── Thought Process ──", style="bold #64748b"))
            for thought in self.thoughts:
                row = Text()
                row.append("│ ├─ thought: ", style="italic #64748b")
                row.append(thought.strip(), style="#cbd5e1")
                console.print(row)

        # 3. Tool & sandbox execution.
        if self.tool_events:
            console.print(Text("── Tool & Sandbox Execution ──", style="bold #64748b"))
            for event in self.tool_events:
                if _render_structured_tool_event(console, event):
                    continue
                stripped = event.strip()
                row = Text()
                if stripped.startswith("[WARN]") or "error" in stripped.lower():
                    row.append("│ ├─ ", style="#64748b")
                    row.append(
                        f"[WARN] {stripped.removeprefix('[WARN]').strip()}",
                        style="bold #f59e0b",
                    )
                else:
                    row.append("│ ├─ exec: ", style="#64748b")
                    row.append(stripped, style="#cbd5e1")
                console.print(row)

        # 6. Interactive question card (flat).
        if pending_question is not None and pending_question.stage == self.stage:
            console.print(Text("Question from Orchestrator", style="bold #00e5ff"))
            console.print(Text(pending_question.question, style="bold #e2e8f0"))
            for idx, opt in enumerate(pending_question.options, start=1):
                is_default = opt == pending_question.default_option
                suffix = " (Default)" if is_default else ""
                row = Text()
                row.append(f"  [{idx}] ", style="bold #00e5ff")
                row.append(f"{opt}{suffix}", style="bold #67e8f9" if is_default else "#e2e8f0")
                console.print(row)
            console.print(
                Text(
                    f"Select [1-{len(pending_question.options)}] or type your answer below:",
                    style="italic #94a3b8",
                )
            )

        # 7. Ephemeral 2-line live-activity indicator (vanishes the instant work finishes).
        if self.is_working:
            accent = _STAGE_ACCENT.get(self.stage, "#3b82f6")
            frame = TRANSIENT_SPINNER_FRAMES[
                self.spinner_index % len(TRANSIENT_SPINNER_FRAMES)
            ]
            headline = (self.live_headline or "Thinking...").strip()
            console.print(Text(f"{frame} {headline}", style=f"bold {accent}"))
            subline = (self.live_subline or "").strip()
            if subline:
                # Truncate so the subline never wraps onto a second line.
                limit = max(4, effective_width - 8)
                if len(subline) > limit:
                    subline = subline[: limit - 1].rstrip() + "…"
                console.print(Text(f"  └─ {subline}", style="dim #94a3b8"))

        full_output = sio.getvalue()
        lines = full_output.splitlines()
        self.total_rendered_lines = len(lines)

        # If no explicit height constraint is given (e.g. in test assertions),
        # return full output unless scroll_offset / scroll_line was explicitly requested.
        if height is None:
            if self.scroll_offset > 0 and len(lines) > self.scroll_offset:
                start = max(0, len(lines) - self.scroll_offset)
                self.scroll_line = start
                return "\n".join(lines[start:])
            if self.scroll_line > 0 and self.scroll_line < len(lines):
                return "\n".join(lines[self.scroll_line :])
            return full_output

        # Handle vertical scrolling with dynamic viewport height (auto-scroll to bottom by default)
        total = len(lines)
        if total <= height:
            self.scroll_offset = 0
            self.scroll_line = 0
            return full_output

        max_offset = total - height
        self.scroll_offset = min(max_offset, max(0, self.scroll_offset))
        start = total - height - self.scroll_offset
        end = start + height
        self.scroll_line = start
        return "\n".join(lines[start:end])


class ForgeTUIState:
    """Reactive state container for the Forge split-pane blacksmithing workspace."""

    def __init__(self) -> None:
        self.active_worker_stage: ForgeStage | None = None
        self.selected_view_stage: ForgeStage = ForgeStage.ANVIL
        self.auto_follow: bool = True
        self.cycle_count: int = 1
        self.cycle_mode: CycleMode = CycleMode.UNTIL_FULFILLED
        self.pending_question: UserQuestion | None = None
        self.pending_plan: PendingPlan | None = None
        self.all_requirements_met: bool = False
        self.viewing_history_cycle: int | None = None
        self.active_stage_tokens: int = 0
        self.total_session_tokens: int = 0
        self.total_session_input_tokens: int = 0
        self.total_session_output_tokens: int = 0
        # Real per-cycle usage: ``{cycle_number: {stage: {input, output, total}}}``.
        self.cycle_usage: dict[int, dict[str, dict[str, int]]] = {}

        self.stage_buffers: dict[ForgeStage, StageBuffer] = {
            ForgeStage.ANVIL: StageBuffer(stage=ForgeStage.ANVIL),
            ForgeStage.HAMMER: StageBuffer(stage=ForgeStage.HAMMER),
            ForgeStage.TEMPER: StageBuffer(stage=ForgeStage.TEMPER),
        }
        self.cycle_history: dict[int, dict[ForgeStage, StageBuffer]] = {}
        self._on_redraw: Callable[[], None] | None = None

    def set_active_stage(self, stage: ForgeStage | None, *, reset_tokens: bool = True) -> None:
        """Set the stage actively executing work in the background.

        ``reset_tokens`` zeroes the stage's turn counter when a new manufacturing stage
        begins. Conversational / direct-tool turns pass ``reset_tokens=False`` so the
        sidebar keeps accumulating the real tokens spent across the dialogue.
        """
        self.active_worker_stage = stage
        if stage is not None:
            buffer = self.stage_buffers[stage]
            if reset_tokens:
                # Reset current-cycle stage token counters for this stage
                buffer.stage_tokens = 0
                buffer.stage_input_tokens = 0
                buffer.stage_output_tokens = 0
            self.active_stage_tokens = buffer.stage_tokens
            buffer.status = "running"
            if self.auto_follow:
                self.selected_view_stage = stage
        else:
            # Idle: active_stage_tokens reflects the selected view stage's tokens
            self.active_stage_tokens = self.stage_buffers[self.selected_view_stage].stage_tokens
        self._notify_redraw()

    def append_chat(self, stage: ForgeStage, speaker: str, text: str) -> None:
        """Append a turn to a stage's persistent conversational transcript."""
        self.stage_buffers[stage].chat_history.append((speaker, text))
        self._notify_redraw()

    def start_stream(self, stage: ForgeStage, speaker: str) -> None:
        """Open a new streaming assistant turn that deltas will fill in live."""
        self.stage_buffers[stage].chat_history.append((speaker, ""))
        self._notify_redraw()

    def append_stream(self, stage: ForgeStage, delta: str) -> None:
        """Append a streamed token delta onto the currently open assistant turn."""
        if not delta:
            return
        history = self.stage_buffers[stage].chat_history
        if not history:
            history.append(("assistant", ""))
        speaker, text = history[-1]
        history[-1] = (speaker, text + delta)
        self._notify_redraw()

    def finalize_stream(self, stage: ForgeStage, text: str) -> None:
        """Close the open streamed turn (created by :meth:`start_stream`) with final text."""
        history = self.stage_buffers[stage].chat_history
        if history:
            speaker, _ = history[-1]
            history[-1] = (speaker, text)
        else:
            history.append(("assistant", text))
        self._notify_redraw()

    def clear_stage(self, stage: ForgeStage) -> None:
        """Clear a single stage's transcript, thoughts, tool events and tokens (``/clear``)."""
        buffer = self.stage_buffers[stage]
        buffer.chat_history.clear()
        buffer.thoughts.clear()
        buffer.tool_events.clear()
        buffer.final_output = ""
        buffer.stage_tokens = 0
        buffer.stage_input_tokens = 0
        buffer.stage_output_tokens = 0
        buffer.scroll_line = 0
        buffer.scroll_offset = 0
        buffer.total_rendered_lines = 0
        buffer.status = "idle"
        if self.selected_view_stage == stage and self.active_worker_stage is None:
            self.active_stage_tokens = 0
        self._notify_redraw()

    def reset_view(self) -> None:
        """Clear every stage buffer's transcript and reset the cycle view (``/clear``)."""
        self.cycle_history.clear()
        self.cycle_count = 1
        self.viewing_history_cycle = None
        self.active_worker_stage = None
        self.active_stage_tokens = 0
        self.selected_view_stage = ForgeStage.ANVIL
        self.auto_follow = True
        self.pending_question = None
        self.pending_plan = None
        self.all_requirements_met = False
        self.total_session_tokens = 0
        self.total_session_input_tokens = 0
        self.total_session_output_tokens = 0
        self.cycle_usage.clear()
        self.stage_buffers = {
            stage: StageBuffer(stage=stage)
            for stage in (ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER)
        }
        self._notify_redraw()

    def select_view_stage(self, stage: ForgeStage) -> None:
        """Manually pin the viewport to inspect a specific stage without interrupting workers."""
        self.selected_view_stage = stage
        self.auto_follow = False
        self.viewing_history_cycle = None
        self._notify_redraw()

    def resume_auto_follow(self) -> None:
        """Resume auto-follow mode, snapping the viewport back to the active worker stage."""
        self.auto_follow = True
        self.viewing_history_cycle = None
        if self.active_worker_stage is not None:
            self.selected_view_stage = self.active_worker_stage
        self._notify_redraw()

    def append_thought(self, stage: ForgeStage, text: str) -> None:
        """Append a thought to a stage's buffer."""
        self.stage_buffers[stage].thoughts.append(text)
        self._notify_redraw()

    def append_tool_event(self, stage: ForgeStage, text: str) -> None:
        """Append a tool or sandbox execution event to a stage's buffer."""
        self.stage_buffers[stage].tool_events.append(text)
        self._notify_redraw()

    def set_live_activity(
        self, stage: ForgeStage, headline: str = "Thinking...", subline: str = ""
    ) -> None:
        """Overwrite a stage's transient live-activity status (never appends).

        Sets ``is_working`` so the 2-line ``⠋ Thinking...`` / ``└─ action`` indicator renders at
        the bottom of the stream, replaces the previous tool/thought subline in place, advances
        the spinner, and triggers a redraw.
        """
        buffer = self.stage_buffers[stage]
        buffer.is_working = True
        buffer.live_headline = headline or "Thinking..."
        buffer.live_subline = subline
        buffer.spinner_index += 1
        self._notify_redraw()

    def clear_live_activity(self, stage: ForgeStage) -> None:
        """Clear a stage's transient indicator so only the clean final output remains."""
        buffer = self.stage_buffers[stage]
        buffer.is_working = False
        buffer.live_headline = ""
        buffer.live_subline = ""
        self._notify_redraw()

    def set_stage_output(
        self,
        stage: ForgeStage,
        text: str,
        status: Literal["idle", "running", "passed", "failed"] = "passed",
    ) -> None:
        """Set the deliverable or verdict output for a stage."""
        self.stage_buffers[stage].final_output = text
        self.stage_buffers[stage].status = status
        self._notify_redraw()

    def record_tokens(
        self,
        stage: ForgeStage,
        delta_tokens: int,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record real usage across the stage, active cycle, and session totals."""
        total = max(0, delta_tokens)
        input_tokens = max(0, prompt_tokens)
        output_tokens = max(0, completion_tokens)
        if total == 0:
            total = input_tokens + output_tokens
        if input_tokens == 0 and output_tokens == 0:
            output_tokens = total
        elif input_tokens + output_tokens != total:
            output_tokens = max(0, total - input_tokens)

        buffer = self.stage_buffers[stage]
        buffer.stage_tokens += total
        buffer.stage_input_tokens += input_tokens
        buffer.stage_output_tokens += output_tokens
        if self.active_worker_stage == stage or (
            self.active_worker_stage is None and self.selected_view_stage == stage
        ):
            self.active_stage_tokens = buffer.stage_tokens
        self.total_session_tokens += total
        self.total_session_input_tokens += input_tokens
        self.total_session_output_tokens += output_tokens

        cycle = self.cycle_usage.setdefault(self.cycle_count, {})
        stage_key = stage.value.lower()
        usage = cycle.setdefault(stage_key, {"input": 0, "output": 0, "total": 0})
        usage["input"] += input_tokens
        usage["output"] += output_tokens
        usage["total"] += total
        self._notify_redraw()

    def set_pending_plan(self, plan: PendingPlan | None) -> None:
        """Attach or clear the structured plan awaiting semantic approval."""
        self.pending_plan = plan
        self._notify_redraw()

    def advance_to_next_cycle(self, temper_feedback: str) -> None:
        """Archive current stage buffers into cycle_history and start the next cycle."""
        self.cycle_history[self.cycle_count] = {
            s: copy.copy(buf) for s, buf in self.stage_buffers.items()
        }
        self.cycle_count += 1
        self.all_requirements_met = False
        self.stage_buffers = {
            ForgeStage.ANVIL: StageBuffer(stage=ForgeStage.ANVIL),
            ForgeStage.HAMMER: StageBuffer(stage=ForgeStage.HAMMER),
            ForgeStage.TEMPER: StageBuffer(stage=ForgeStage.TEMPER),
        }
        self.append_thought(
            ForgeStage.ANVIL,
            f"[Cycle {self.cycle_count} Triggered by Temper Feedback]: {temper_feedback}",
        )
        if self.auto_follow:
            self.selected_view_stage = ForgeStage.ANVIL
        self._notify_redraw()

    def view_cycle_history(self, delta: int) -> None:
        """Navigate cycle history backward or forward."""
        if not self.cycle_history:
            return
        curr = (
            self.cycle_count if self.viewing_history_cycle is None else self.viewing_history_cycle
        )
        target = max(1, min(self.cycle_count, curr + delta))
        if target == self.cycle_count:
            self.viewing_history_cycle = None
        else:
            self.viewing_history_cycle = target
        self.auto_follow = False
        self._notify_redraw()

    def scroll_up(self, lines: int = 3) -> None:
        """Scroll the viewed stage buffer up toward earlier output."""
        buf = self.get_active_view_buffer()
        buf.scroll_offset += lines
        self._notify_redraw()

    def scroll_down(self, lines: int = 3) -> None:
        """Scroll the viewed stage buffer down toward newer output."""
        buf = self.get_active_view_buffer()
        buf.scroll_offset = max(0, buf.scroll_offset - lines)
        self._notify_redraw()

    def scroll_to_top(self) -> None:
        """Scroll to the very top of the viewed stage buffer."""
        buf = self.get_active_view_buffer()
        buf.scroll_offset = 999999
        self._notify_redraw()

    def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the viewed stage buffer (auto-scroll live view)."""
        buf = self.get_active_view_buffer()
        buf.scroll_offset = 0
        self._notify_redraw()

    def get_active_view_buffer(self) -> StageBuffer:
        """Get the StageBuffer currently viewed in the main viewport."""
        if (
            self.viewing_history_cycle is not None
            and self.viewing_history_cycle in self.cycle_history
        ):
            return self.cycle_history[self.viewing_history_cycle][self.selected_view_stage]
        return self.stage_buffers[self.selected_view_stage]

    def get_viewed_stage_tokens(self) -> int:
        """Get the token count of the currently viewed stage."""
        return self.get_active_view_buffer().stage_tokens

    def set_cycle_mode(self, mode: CycleMode) -> None:
        """Set the active cycle execution policy (Single Cycle vs Until Fulfilled)."""
        self.cycle_mode = mode
        self._notify_redraw()

    def toggle_cycle_mode(self) -> CycleMode:
        """Toggle between Single Cycle and Until Fulfilled modes."""
        if self.cycle_mode == CycleMode.SINGLE_CYCLE:
            self.cycle_mode = CycleMode.UNTIL_FULFILLED
        else:
            self.cycle_mode = CycleMode.SINGLE_CYCLE
        self._notify_redraw()
        return self.cycle_mode

    def set_pending_question(self, question: UserQuestion | None) -> None:
        """Register an active question awaiting user selection."""
        self.pending_question = question
        if question is not None and self.auto_follow:
            self.selected_view_stage = question.stage
        self._notify_redraw()

    def resolve_pending_question(self, raw_input: str) -> str:
        """Resolve the pending user question using the submitted input."""
        if self.pending_question is None:
            return ""
        q = self.pending_question
        clean = raw_input.strip()

        # Match numeric option index
        chosen: str
        if clean.isdigit():
            idx = int(clean) - 1
            chosen = q.options[idx] if 0 <= idx < len(q.options) else clean
        elif not clean and q.default_option:
            chosen = q.default_option
        else:
            # Match prefix or exact text in options
            lower_clean = clean.lower()
            matched = None
            for opt in q.options:
                if opt.lower().startswith(lower_clean) or lower_clean in opt.lower():
                    matched = opt
                    break
            chosen = matched if matched is not None else clean

        if q.future is not None and not q.future.done():
            q.future.set_result(chosen)

        self.pending_question = None
        self._notify_redraw()
        return chosen

    def cancel_pending_question(self) -> None:
        """Cancel the pending question if the user interrupts with Ctrl+C."""
        if self.pending_question is not None:
            if self.pending_question.future and not self.pending_question.future.done():
                self.pending_question.future.cancel()
            self.pending_question = None
            self._notify_redraw()

    def _notify_redraw(self) -> None:
        """Trigger TUI redraw if callback is registered."""
        if self._on_redraw is not None:
            with contextlib.suppress(Exception):
                self._on_redraw()
