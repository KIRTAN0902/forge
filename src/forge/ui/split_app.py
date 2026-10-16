"""Full-Screen prompt_toolkit Split-Pane Blacksmithing Workspace ("Anvil -> Hammer -> Temper")."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import (
    ANSI,
    AnyFormattedText,
    StyleAndTextTuples,
    to_formatted_text,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.mouse_events import MouseEvent
from prompt_toolkit.output import DummyOutput, Output
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.styles import Style

from forge.ui.state import ForgeStage, ForgeTUIState, format_tokens


def create_safe_output(output: Output | None = None) -> Output:
    """Create a safe Output instance, falling back to DummyOutput in headless/test environments."""
    if output is not None:
        return output
    try:
        return create_output()
    except Exception:
        return DummyOutput()


def render_sidebar_formatted_text(state: ForgeTUIState) -> StyleAndTextTuples:
    """Generate prompt_toolkit formatted text fragments for the left sidebar."""
    fragments: StyleAndTextTuples = []

    # Title
    fragments.append(("class:sidebar_header", "     FORGE BLACKSMITH     \n"))
    fragments.append(("class:border", "────────────────────────────\n\n"))

    # Stages: Anvil, Hammer, Temper
    stages = [ForgeStage.ANVIL, ForgeStage.HAMMER, ForgeStage.TEMPER]

    def make_click_handler(stg: ForgeStage) -> Callable[[MouseEvent], None]:
        def _handler(mouse_event: MouseEvent) -> None:
            state.select_view_stage(stg)

        return _handler

    for stg in stages:
        click_fn = make_click_handler(stg)
        stage_name = stg.value
        style_name = f"class:{stg.name.lower()}"

        is_active_worker = state.active_worker_stage == stg
        is_viewed = state.selected_view_stage == stg

        if is_active_worker:
            # Enclosed in ASCII box with left arrow
            # +----------+
            # |  Stage   | <--
            # +----------+
            pad_left = (10 - len(stage_name)) // 2
            pad_right = 10 - len(stage_name) - pad_left
            padded = f"{' ' * pad_left}{stage_name}{' ' * pad_right}"

            fragments.append(("class:active_box", "+----------+\n", click_fn))
            fragments.append(("class:active_box", "|", click_fn))
            fragments.append((style_name, padded, click_fn))
            fragments.append(("class:active_box", "| <--", click_fn))

            if is_viewed and not state.auto_follow:
                fragments.append(("class:viewing_badge", " [👁 VIEWING]\n", click_fn))
            else:
                fragments.append(("", "\n", click_fn))

            fragments.append(("class:active_box", "+----------+\n", click_fn))
        else:
            fragments.append(("", "   ", click_fn))
            fragments.append((style_name, f"{stage_name:<8}", click_fn))
            if is_viewed:
                fragments.append(("class:viewing_badge", " [👁 VIEWING]\n", click_fn))
            else:
                fragments.append(("", "\n", click_fn))

        fragments.append(("", "\n"))

    # Cycle Counter
    fragments.append(("class:border", "────────────────────────────\n"))
    fragments.append(("class:cycle_counter", f"Cycle :- {state.cycle_count}\n\n"))

    # Dual-Tier Token Counter
    fragments.append(("class:token_header", "Tokens :-\n"))
    active_label = (
        state.active_worker_stage.value
        if state.active_worker_stage
        else state.selected_view_stage.value
    )
    fragments.append(
        (
            "class:token_stage",
            f"  Stage: {format_tokens(state.active_stage_tokens)} ({active_label})\n",
        )
    )

    if (
        state.active_worker_stage is not None
        and state.selected_view_stage != state.active_worker_stage
    ):
        viewed_tokens = state.get_viewed_stage_tokens()
        fragments.append(
            (
                "class:token_viewed",
                f"  Viewed ({state.selected_view_stage.value}): {format_tokens(viewed_tokens)}\n",
            )
        )

    fragments.append(
        ("class:token_total", f"  Total: {format_tokens(state.total_session_tokens)}\n\n")
    )

    # Shortcuts Legend
    fragments.append(("class:border", "────────────────────────────\n"))
    fragments.append(("class:hint_title", "Shortcuts:\n"))
    fragments.append(("class:hint", " [Alt+1] Anvil (Orch)\n"))
    fragments.append(("class:hint", " [Alt+2] Hammer (Exec)\n"))
    fragments.append(("class:hint", " [Alt+3] Temper (QA)\n"))
    fragments.append(("class:hint", " [Alt+0] Auto-Follow\n"))
    fragments.append(("class:hint", " [Alt+←/→] Cycle History\n"))
    fragments.append(("class:hint", " [PgUp/Dn] Scroll View\n"))
    fragments.append(("class:hint", " [Ctrl+C] Cancel Task\n"))
    fragments.append(("class:hint", " [Ctrl+Q] Exit Forge\n"))

    return fragments


def render_sidebar_plain_text(state: ForgeTUIState) -> str:
    """Helper for unit tests to inspect plain text representation of the sidebar."""
    fragments = render_sidebar_formatted_text(state)
    return "".join(part[1] for part in to_formatted_text(fragments))


class SidebarControl(FormattedTextControl):
    """FormattedTextControl displaying Anvil, Hammer, Temper, Cycle, and Token counters."""

    def __init__(self, state: ForgeTUIState) -> None:
        self.state = state
        super().__init__(text=lambda: render_sidebar_formatted_text(self.state), show_cursor=False)


class ForgeSplitPaneApp:
    """Persistent, full-screen split-pane blacksmithing workspace application."""

    def __init__(
        self,
        state: ForgeTUIState | None = None,
        controller: Any = None,
        path: Path | None = None,
        output: Output | None = None,
    ) -> None:
        self.state = state or ForgeTUIState()
        self.path = (path or Path(".")).resolve()
        self.controller = controller
        self.output = create_safe_output(output)
        self._background_task: asyncio.Task[Any] | None = None

        # Input buffer
        self.input_buffer = Buffer(multiline=False)

        # Build prompt_toolkit UI layout
        self.sidebar_control = SidebarControl(self.state)
        self.sidebar_window = Window(
            content=self.sidebar_control,
            width=28,
            style="class:sidebar",
        )

        self.divider_vertical = Window(
            width=1,
            char="│",
            style="class:border",
        )

        # Header Control
        self.header_control = FormattedTextControl(
            text=self._render_header,
            show_cursor=False,
        )
        self.header_window = Window(
            content=self.header_control,
            height=3,
            style="class:header",
        )

        self.divider_horizontal_top = Window(
            height=1,
            char="─",
            style="class:border",
        )

        # Center Viewport Window
        self.viewport_control = FormattedTextControl(
            text=self._render_viewport,
            show_cursor=False,
        )
        self.viewport_window = Window(
            content=self.viewport_control,
            wrap_lines=True,
            style="class:viewport",
        )

        self.divider_horizontal_bottom = Window(
            height=1,
            char="─",
            style="class:border",
        )

        # Input prompt control
        self.input_control = BufferControl(
            buffer=self.input_buffer,
        )
        self.input_prompt_control = FormattedTextControl(
            text=[("class:prompt", "> ")],
            show_cursor=False,
        )
        self.input_window = Window(
            content=self.input_control,
            height=2,
            style="class:input_box",
        )
        self.input_box_row = VSplit(
            [
                Window(content=self.input_prompt_control, width=3, style="class:input_box"),
                self.input_window,
            ],
            height=2,
        )

        # Right pane container
        self.right_pane = HSplit(
            [
                self.header_window,
                self.divider_horizontal_top,
                self.viewport_window,
                self.divider_horizontal_bottom,
                self.input_box_row,
            ]
        )

        # Root split container
        self.root_container = VSplit(
            [
                self.sidebar_window,
                self.divider_vertical,
                self.right_pane,
            ]
        )

        self.layout = Layout(self.root_container, focused_element=self.input_window)

        # Keybindings
        self.kb = KeyBindings()
        self._setup_keybindings()

        # Styles
        self.style = Style.from_dict(
            {
                "anvil": "#3b82f6 bold",
                "hammer": "#ef4444 bold",
                "temper": "#22c55e bold",
                "cycle_counter": "#a855f7 bold",
                "token_header": "#f97316 bold",
                "token_stage": "#f97316",
                "token_viewed": "#fdba74 italic",
                "token_total": "#ea580c bold",
                "viewing_badge": "#eab308 bold",
                "active_box": "#38bdf8 bold",
                "border": "#334155",
                "sidebar_header": "#3b82f6 bold",
                "header_title": "#3b82f6 bold",
                "header_telemetry": "#94a3b8",
                "prompt": "#38bdf8 bold",
                "hint_title": "#64748b bold",
                "hint": "#64748b",
            }
        )

        # Application instance
        self.application: Application[None] = Application(
            layout=self.layout,
            key_bindings=self.kb,
            style=self.style,
            full_screen=True,
            mouse_support=True,
            output=self.output,
        )

        # Register state redraw notification
        self.state._on_redraw = self._handle_state_redraw

    def _handle_state_redraw(self) -> None:
        """Invalidate the prompt_toolkit application so it redraws on state mutations."""
        with contextlib.suppress(Exception):
            self.application.invalidate()

    def _render_header(self) -> AnyFormattedText:
        """Render the top header banner with centered FORGE title and live telemetry bar."""
        worker_str = (
            self.state.active_worker_stage.value if self.state.active_worker_stage else "IDLE"
        )
        viewport_str = self.state.selected_view_stage.value
        follow_str = "AUTO-FOLLOW" if self.state.auto_follow else "PINNED (Alt+0 to follow)"
        cycle_str = f"Cycle #{self.state.cycle_count}"
        if self.state.viewing_history_cycle is not None:
            cycle_str += f" [VIEWING HISTORY: Cycle {self.state.viewing_history_cycle}]"

        telemetry = (
            f" {cycle_str} | Worker: {worker_str} | Viewport: {viewport_str} [{follow_str}] | "
            f"Stage Tokens: {self.state.active_stage_tokens:,} | "
            f"Session Total: {self.state.total_session_tokens:,}"
        )

        # Dynamic terminal width for title centering
        cols = shutil.get_terminal_size((100, 30)).columns
        right_width = max(30, cols - 29)
        title = "FORGE"
        pad = max(0, (right_width - len(title)) // 2)
        centered_title = f"{' ' * pad}{title}\n"

        return [
            ("class:header_title", centered_title),
            ("class:header_telemetry", telemetry),
        ]

    def _render_viewport(self) -> AnyFormattedText:
        """Render the exclusive stage viewport buffer using ANSI conversion."""
        cols = shutil.get_terminal_size((100, 30)).columns
        viewport_width = max(30, cols - 32)
        stage_buffer = self.state.get_active_view_buffer()
        ansi_output = stage_buffer.render_to_ansi(width=viewport_width)
        return ANSI(ansi_output)

    def _setup_keybindings(self) -> None:
        """Wire all required keyboard shortcuts for non-blocking stage switching and navigation."""

        @self.kb.add("escape", "1")
        @self.kb.add("f1")
        def _switch_anvil(event: KeyPressEvent) -> None:
            self.state.select_view_stage(ForgeStage.ANVIL)

        @self.kb.add("escape", "2")
        @self.kb.add("f2")
        def _switch_hammer(event: KeyPressEvent) -> None:
            self.state.select_view_stage(ForgeStage.HAMMER)

        @self.kb.add("escape", "3")
        @self.kb.add("f3")
        def _switch_temper(event: KeyPressEvent) -> None:
            self.state.select_view_stage(ForgeStage.TEMPER)

        @self.kb.add("escape", "0")
        @self.kb.add("f4")
        def _auto_follow(event: KeyPressEvent) -> None:
            self.state.resume_auto_follow()

        @self.kb.add("escape", "left")
        def _history_prev(event: KeyPressEvent) -> None:
            self.state.view_cycle_history(-1)

        @self.kb.add("escape", "right")
        def _history_next(event: KeyPressEvent) -> None:
            self.state.view_cycle_history(1)

        @self.kb.add("pageup")
        def _scroll_up(event: KeyPressEvent) -> None:
            self.state.scroll_up(5)

        @self.kb.add("pagedown")
        def _scroll_down(event: KeyPressEvent) -> None:
            self.state.scroll_down(5)

        @self.kb.add("enter")
        def _handle_enter(event: KeyPressEvent) -> None:
            user_text = self.input_buffer.text.strip()
            if not user_text:
                return

            self.input_buffer.text = ""

            # If user types /exit or /quit
            if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
                event.app.exit()
                return

            # Submit prompt to background cycle controller
            if self.controller is not None:
                self._background_task = self.application.create_background_task(
                    self.controller.run_cycle_loop(user_text)
                )
            else:
                # Standalone simulation or test invocation
                self.state.append_thought(ForgeStage.ANVIL, f"User Goal: {user_text}")

        @self.kb.add("c-c")
        def _handle_ctrl_c(event: KeyPressEvent) -> None:
            if self._background_task and not self._background_task.done():
                self._background_task.cancel()
                if self.controller and hasattr(self.controller, "cancel"):
                    self.controller.cancel()
                self.state.append_thought(
                    self.state.selected_view_stage,
                    "[Cancelled by user (Ctrl+C)]",
                )
                self.state.set_active_stage(None)
            else:
                event.app.exit()

        @self.kb.add("c-q")
        def _handle_ctrl_q(event: KeyPressEvent) -> None:
            event.app.exit()

    async def run_async(self) -> None:
        """Run the full-screen TUI application asynchronously."""
        await self.application.run_async()

    def run(self) -> None:
        """Run the full-screen TUI application synchronously."""
        self.application.run()
