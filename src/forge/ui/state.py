"""Reactive TUI State, Stage Buffers & Dual Token Counters for Forge Split-Pane Workspace."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import io
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from rich.console import Console
from rich.panel import Panel
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


@dataclass
class StageBuffer:
    """Isolated, async-safe buffer storing execution state and thoughts for a single stage."""

    stage: ForgeStage
    thoughts: list[str] = field(default_factory=list)
    tool_events: list[str] = field(default_factory=list)
    chat_history: list[tuple[str, str]] = field(default_factory=list)
    final_output: str = ""
    stage_tokens: int = 0
    scroll_line: int = 0
    status: Literal["idle", "running", "passed", "failed"] = "idle"

    def render_to_ansi(self, width: int = 80, pending_question: UserQuestion | None = None) -> str:
        """Render the stage buffer into an ANSI-formatted string using Rich."""
        effective_width = max(30, width)
        sio = io.StringIO()
        console = Console(
            file=sio,
            force_terminal=True,
            color_system="truecolor",
            width=effective_width,
            highlight=False,
        )

        stage_color = {
            ForgeStage.ANVIL: "#3b82f6",
            ForgeStage.HAMMER: "#ef4444",
            ForgeStage.TEMPER: "#22c55e",
        }.get(self.stage, "#3b82f6")

        stage_icon = {
            ForgeStage.ANVIL: "🎯",
            ForgeStage.HAMMER: "⚡",
            ForgeStage.TEMPER: "🔍",
        }.get(self.stage, "🔨")

        stage_subtitle = {
            ForgeStage.ANVIL: "Orchestrator & Architecture Planning",
            ForgeStage.HAMMER: "Code Synthesis, AST Mutation & Tool Execution",
            ForgeStage.TEMPER: "Adversarial QA Gatekeeper & Verification",
        }.get(self.stage, "")

        status_style = {
            "idle": "dim white",
            "running": "bold yellow",
            "passed": "bold green",
            "failed": "bold red",
        }.get(self.status, "white")

        # 1. Header Banner
        header_text = Text()
        header_text.append(f"{stage_icon} {self.stage.value.upper()} ", style=f"bold {stage_color}")
        header_text.append(f"— {stage_subtitle}\n", style="dim white")
        header_text.append("Status: ", style="bold white")
        header_text.append(f"[{self.status.upper()}]  ", style=status_style)
        header_text.append("Tokens: ", style="bold white")
        header_text.append(
            f"{format_tokens(self.stage_tokens)} ({self.stage_tokens:,})\n", style="bold #f97316"
        )

        console.print(
            Panel(
                header_text,
                title=f"[bold {stage_color}]{self.stage.value} Stage[/bold {stage_color}]",
                border_style=stage_color,
                expand=True,
            )
        )

        # 2. Conversational Transcript (persistent multi-turn chat scrollback)
        if self.chat_history:
            console.print()
            chat_text = Text()
            for speaker, message in self.chat_history:
                if speaker.lower() in {"you", "user", "developer"}:
                    chat_text.append(f"👤 {speaker}: ", style="bold #38bdf8")
                else:
                    chat_text.append(f"{stage_icon} {speaker}: ", style=f"bold {stage_color}")
                chat_text.append(f"{message.strip()}\n\n", style="white")

            console.print(
                Panel(
                    chat_text,
                    title=f"[bold {stage_color}]💬 Conversation[/bold {stage_color}]",
                    border_style=stage_color,
                    expand=True,
                )
            )

        # 3. Thought Process Section
        if self.thoughts:
            console.print()
            thought_text = Text()
            for idx, t in enumerate(self.thoughts, start=1):
                thought_text.append(f"  [{idx}] ", style=f"bold {stage_color}")
                thought_text.append(f"{t.strip()}\n", style="white")

            console.print(
                Panel(
                    thought_text,
                    title=(
                        f"[bold {stage_color}]💭 Thought Process "
                        f"({len(self.thoughts)} steps)[/bold {stage_color}]"
                    ),
                    border_style="dim cyan",
                    expand=True,
                )
            )

        # 4. Tool & Sandbox Execution Section
        if self.tool_events:
            console.print()
            events_text = Text()
            for ev in self.tool_events:
                events_text.append(f"  • {ev.strip()}\n", style="yellow")

            console.print(
                Panel(
                    events_text,
                    title="[bold yellow]🛠 Tool & Sandbox Execution[/bold yellow]",
                    border_style="yellow",
                    expand=True,
                )
            )

        # 5. Stage Output / Final Deliverable / Defect Report
        if self.final_output:
            console.print()
            if self.stage == ForgeStage.TEMPER and self.status == "passed":
                deliverable_text = Text()
                deliverable_text.append(
                    "✔ QA Verification Passed — All Requirements Satisfied!\n\n", style="bold green"
                )
                deliverable_text.append(self.final_output.strip(), style="white")
                console.print(
                    Panel(
                        deliverable_text,
                        title="[bold green]✅ Verified Final Output Deliverable[/bold green]",
                        border_style="bold green",
                        expand=True,
                    )
                )
            elif self.status == "failed":
                defect_text = Text()
                defect_text.append(
                    "✖ Defects Identified — Triggering Automatic Next Cycle:\n\n", style="bold red"
                )
                defect_text.append(self.final_output.strip(), style="white")
                console.print(
                    Panel(
                        defect_text,
                        title="[bold red]❌ Rejection & Defect Report[/bold red]",
                        border_style="bold red",
                        expand=True,
                    )
                )
            else:
                console.print(
                    Panel(
                        Text(self.final_output.strip(), style="white"),
                        title=f"[bold {stage_color}]Stage Output[/bold {stage_color}]",
                        border_style=stage_color,
                        expand=True,
                    )
                )

        # 6. Interactive Question Card (when Orchestrator or stage asks user for input)
        if pending_question is not None and pending_question.stage == self.stage:
            console.print()
            q_text = Text()
            q_text.append(f"{pending_question.question}\n\n", style="bold white")
            for idx, opt in enumerate(pending_question.options, start=1):
                is_default = opt == pending_question.default_option
                def_marker = " (Default)" if is_default else ""
                q_text.append(f"  [{idx}] ", style="bold #00e5ff")
                q_text.append(
                    f"{opt}{def_marker}\n",
                    style="bright_cyan bold" if is_default else "white",
                )
            q_text.append(
                f"\nSelect [1-{len(pending_question.options)}] or type your answer below:",
                style="italic #94a3b8",
            )
            console.print(
                Panel(
                    q_text,
                    title="[bold #00e5ff]❓ Question from Orchestrator[/bold #00e5ff]",
                    border_style="#00e5ff",
                    expand=True,
                )
            )

        full_output = sio.getvalue()

        # Handle vertical scrolling
        lines = full_output.splitlines()
        if self.scroll_line > 0 and self.scroll_line < len(lines):
            return "\n".join(lines[self.scroll_line :])
        return full_output


class ForgeTUIState:
    """Reactive state container for the Forge split-pane blacksmithing workspace."""

    def __init__(self) -> None:
        self.active_worker_stage: ForgeStage | None = None
        self.selected_view_stage: ForgeStage = ForgeStage.ANVIL
        self.auto_follow: bool = True
        self.cycle_count: int = 1
        self.cycle_mode: CycleMode = CycleMode.UNTIL_FULFILLED
        self.pending_question: UserQuestion | None = None
        self.viewing_history_cycle: int | None = None
        self.active_stage_tokens: int = 0
        self.total_session_tokens: int = 0

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
                # Reset current-cycle stage token counter to 0 for this stage
                buffer.stage_tokens = 0
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

    def record_tokens(self, stage: ForgeStage, delta_tokens: int) -> None:
        """Record token usage: updates stage buffer, active stage tokens, and session total."""
        self.stage_buffers[stage].stage_tokens += delta_tokens
        if self.active_worker_stage == stage or (
            self.active_worker_stage is None and self.selected_view_stage == stage
        ):
            self.active_stage_tokens = self.stage_buffers[stage].stage_tokens
        self.total_session_tokens += delta_tokens
        self._notify_redraw()

    def advance_to_next_cycle(self, temper_feedback: str) -> None:
        """Archive current stage buffers into cycle_history and start the next cycle."""
        self.cycle_history[self.cycle_count] = {
            s: copy.copy(buf) for s, buf in self.stage_buffers.items()
        }
        self.cycle_count += 1
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
        """Scroll the viewed stage buffer up."""
        buf = self.get_active_view_buffer()
        buf.scroll_line = max(0, buf.scroll_line - lines)
        self._notify_redraw()

    def scroll_down(self, lines: int = 3) -> None:
        """Scroll the viewed stage buffer down."""
        buf = self.get_active_view_buffer()
        buf.scroll_line += lines
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
