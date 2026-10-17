"""Full-Screen prompt_toolkit Split-Pane Blacksmithing Workspace ("Anvil -> Hammer -> Temper")."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import (
    ANSI,
    AnyFormattedText,
    StyleAndTextTuples,
    to_formatted_text,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.containers import (
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEvent
from prompt_toolkit.output import DummyOutput, Output
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.styles import Style

from forge.ui.state import ForgeStage, ForgeTUIState, format_tokens

# Animated braille spinner frames shown next to the actively running stage.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Bold ASCII block-art FORGE banner rendered in the top header when there is room.
_FORGE_ART: tuple[str, ...] = (
    " ███████╗ ██████╗ ██████╗  ██████╗ ███████╗",
    " ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝",
    " █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  ",
    " ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ",
    " ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗",
    " ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
)

_STAGE_DOT: dict[ForgeStage, str] = {
    ForgeStage.ANVIL: "🟦",
    ForgeStage.HAMMER: "🟥",
    ForgeStage.TEMPER: "🟩",
}


# Slash command catalogue used for both the popup completion menu and `/help`.
SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show the command reference",
    "/status": "Provider/model, connection & token diagnostics",
    "/model": "List or switch a stage's model",
    "/mode": "Set execution mode: auto | chat | cycle",
    "/cycle": "Force a full Anvil -> Hammer -> Temper cycle",
    "/research": "Run the STORM deep research pipeline",
    "/clear": "Clear the active stage viewport",
    "/reset": "Reset the whole session",
    "/exit": "Exit the Forge workspace",
}


class SlashCommandCompleter(Completer):
    """Pops a command menu whenever the input buffer starts with ``/``."""

    def __init__(self, commands: dict[str, str] | None = None) -> None:
        self.commands = commands or SLASH_COMMANDS

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for name, description in self.commands.items():
            if name.startswith(text):
                yield Completion(
                    name,
                    start_position=-len(text),
                    display=name,
                    display_meta=description,
                )


def create_safe_output(output: Output | None = None) -> Output:
    """Create a safe Output instance, falling back to DummyOutput in headless/test environments."""
    if output is not None:
        return output
    try:
        return create_output()
    except Exception:
        return DummyOutput()


def render_sidebar_formatted_text(
    state: ForgeTUIState, spinner_frame: str = ""
) -> StyleAndTextTuples:
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
        dot = _STAGE_DOT.get(stg, "⬜")

        is_active_worker = state.active_worker_stage == stg
        is_viewed = state.selected_view_stage == stg

        if is_active_worker:
            # Rounded box with a running indicator; the middle line keeps the classic
            # `|  Stage  | <--` shape while gaining a live spinner when work is in flight.
            pad_left = (10 - len(stage_name)) // 2
            pad_right = 10 - len(stage_name) - pad_left
            padded = f"{' ' * pad_left}{stage_name}{' ' * pad_right}"

            fragments.append(("class:active_box", "╭──────────╮\n", click_fn))
            fragments.append(("class:active_box", "|", click_fn))
            fragments.append((style_name, padded, click_fn))
            fragments.append(("class:active_box", "| <--", click_fn))

            if is_viewed and not state.auto_follow:
                fragments.append(("class:viewing_badge", " [👁 VIEWING]", click_fn))
            elif spinner_frame:
                fragments.append(("class:spinner", f" {spinner_frame}", click_fn))
            fragments.append(("", "\n", click_fn))

            fragments.append(("class:active_box", "╰──────────╯\n", click_fn))
        else:
            fragments.append(("", f" {dot} ", click_fn))
            fragments.append((style_name, f"{stage_name:<8}", click_fn))
            if is_viewed:
                fragments.append(("class:viewing_badge", " [👁 VIEWING]\n", click_fn))
            else:
                fragments.append(("", "\n", click_fn))

        fragments.append(("", "\n"))

    # Cycle Counter & Dynamic Mode
    fragments.append(("class:border", "────────────────────────────\n"))
    fragments.append(("class:cycle_counter", f"Cycle :- {state.cycle_count}\n"))
    fragments.append(("class:cycle_mode", f"Mode  :- {state.cycle_mode.value}\n\n"))

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
    fragments.append(("class:hint", " [F5] Toggle Mode\n"))
    fragments.append(("class:hint", " [Alt+←/→] History\n"))
    fragments.append(("class:hint", " [PgUp/Dn] Scroll View\n"))
    fragments.append(("class:hint", " [Tab] / Commands\n"))
    fragments.append(("class:hint", " [Ctrl+C] Cancel Task\n"))
    fragments.append(("class:hint", " [Ctrl+Q] Exit Forge\n"))

    return fragments


def render_sidebar_plain_text(state: ForgeTUIState) -> str:
    """Helper for unit tests to inspect plain text representation of the sidebar."""
    fragments = render_sidebar_formatted_text(state)
    return "".join(part[1] for part in to_formatted_text(fragments))


class SidebarControl(FormattedTextControl):
    """FormattedTextControl displaying Anvil, Hammer, Temper, Cycle, and Token counters."""

    def __init__(
        self,
        state: ForgeTUIState,
        spinner_getter: Callable[[], str] | None = None,
    ) -> None:
        self.state = state
        self._spinner_getter = spinner_getter
        super().__init__(text=self._render, show_cursor=False)

    def _render(self) -> StyleAndTextTuples:
        frame = self._spinner_getter() if self._spinner_getter is not None else ""
        return render_sidebar_formatted_text(self.state, frame)


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
        self._spinner_task: asyncio.Task[Any] | None = None
        self._spinner_index = 0
        self.bootstrap: Callable[[], Awaitable[None]] | None = None
        self.close_session: Callable[[], Awaitable[None]] | None = None

        # Input buffer with live `/` command autocompletion
        self.completer = SlashCommandCompleter()
        self.input_buffer = Buffer(
            multiline=False,
            completer=self.completer,
            complete_while_typing=True,
        )

        # Build prompt_toolkit UI layout
        self.sidebar_control = SidebarControl(self.state, self._current_spinner)
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
            height=self._header_height,
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

        # Bottom command bar: top frame, [Stage →] prompt + input, and a hint line
        self.input_control = BufferControl(
            buffer=self.input_buffer,
        )
        self.input_prompt_control = FormattedTextControl(
            text=self._render_input_prompt,
            show_cursor=False,
        )
        self.input_window = Window(
            content=self.input_control,
            height=1,
            style="class:input_box",
        )
        self.input_box_row = VSplit(
            [
                Window(
                    content=self.input_prompt_control,
                    dont_extend_width=True,
                    style="class:input_box",
                ),
                self.input_window,
            ],
            height=1,
        )
        self.input_top_border = Window(height=1, char="─", style="class:input_border")
        self.input_hint_control = FormattedTextControl(
            text=self._render_input_hint,
            show_cursor=False,
        )
        self.input_hint_window = Window(
            content=self.input_hint_control,
            height=1,
            style="class:hint",
        )
        self.input_area = HSplit(
            [
                self.input_top_border,
                self.input_box_row,
                self.input_hint_window,
            ]
        )

        # Right pane container
        self.right_pane = HSplit(
            [
                self.header_window,
                self.divider_horizontal_top,
                self.viewport_window,
                self.divider_horizontal_bottom,
                self.input_area,
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

        # Floating completion menu (pops above the input bar on `/`)
        self.layout = Layout(
            FloatContainer(
                content=self.root_container,
                floats=[
                    Float(
                        xcursor=True,
                        ycursor=True,
                        content=CompletionsMenu(max_height=8, scroll_offset=1),
                    )
                ],
            ),
            focused_element=self.input_window,
        )

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
                "cycle_mode": "#c084fc italic",
                "token_header": "#f97316 bold",
                "token_stage": "#f97316",
                "token_viewed": "#fdba74 italic",
                "token_total": "#ea580c bold",
                "viewing_badge": "#eab308 bold",
                "active_box": "#38bdf8 bold",
                "spinner": "#facc15 bold",
                "border": "#334155",
                "sidebar_header": "#3b82f6 bold",
                "header_title": "#3b82f6 bold",
                "header_art": "#22d3ee bold",
                "header_telemetry": "#94a3b8",
                "prompt": "#38bdf8 bold",
                "input_stage": "#eab308 bold",
                "input_border": "#1e293b",
                "question_prompt": "#00e5ff bold",
                "hint_title": "#64748b bold",
                "hint": "#64748b",
                "completion-menu.completion": "bg:#0f172a #cbd5e1",
                "completion-menu.completion.current": "bg:#38bdf8 #0f172a bold",
                "completion-menu.meta.completion": "bg:#0f172a #64748b",
                "completion-menu.meta.completion.current": "bg:#38bdf8 #0f172a",
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

    # -- helpers ------------------------------------------------------------------------

    def _handle_state_redraw(self) -> None:
        """Invalidate the prompt_toolkit application so it redraws on state mutations."""
        with contextlib.suppress(Exception):
            self.application.invalidate()

    def _current_spinner(self) -> str:
        """Return the animated spinner frame only while a stage is actively running."""
        if self.state.active_worker_stage is None:
            return ""
        return _SPINNER_FRAMES[self._spinner_index % len(_SPINNER_FRAMES)]

    def _header_height(self) -> int:
        return 2 if _is_compact_terminal() else len(_FORGE_ART)

    def _model_label(self) -> str:
        getter = getattr(self.controller, "current_model_label", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                return str(getter())
        return "(no model)"

    def _render_input_prompt(self) -> AnyFormattedText:
        """Render dynamic prompt: switches to question option indicator when question is pending."""
        if self.state.pending_question is not None:
            num = len(self.state.pending_question.options)
            prefix = f"[1-{num}]" if num > 0 else "[?]"
            return [("class:question_prompt", f"? {prefix} > ")]
        stage = self.state.selected_view_stage.value
        return [
            ("class:prompt", "> "),
            ("class:input_stage", f"[{stage} →] "),
        ]

    def _render_input_hint(self) -> AnyFormattedText:
        stage = self.state.selected_view_stage.value
        return [
            (
                "class:hint",
                f" [{stage} →] Type a message, /command (/help, /model, /cycle, "
                f"/research), or !shell...",
            )
        ]

    def _render_header(self) -> AnyFormattedText:
        """Render the top header: ASCII FORGE art + live telemetry (or a compact fallback)."""
        worker_str = (
            self.state.active_worker_stage.value if self.state.active_worker_stage else "IDLE"
        )
        viewport_str = self.state.selected_view_stage.value
        follow_str = "AUTO-FOLLOW" if self.state.auto_follow else "PINNED"
        cycle_str = f"Cycle #{self.state.cycle_count} [{self.state.cycle_mode.value}]"
        if self.state.viewing_history_cycle is not None:
            cycle_str += f" [HIST:{self.state.viewing_history_cycle}]"

        row_harness = "⚡ Autonomous Cybernetic Coding Harness"
        row_status = (
            f"{cycle_str} │ Worker: {worker_str} │ View: {viewport_str} [{follow_str}]"
        )
        row_tokens = (
            f"Stage Tokens: {self.state.active_stage_tokens:,} │ "
            f"Session Total: {self.state.total_session_tokens:,} │ "
            f"Model: {self._model_label()}"
        )

        if _is_compact_terminal():
            return [
                ("class:header_art", " FORGE ⚡ Cybernetic Coding Harness\n"),
                ("class:header_telemetry", f" {row_status}"),
            ]

        telemetry_by_row = {1: row_harness, 2: row_status, 3: row_tokens}
        fragments: StyleAndTextTuples = []
        for index, art_line in enumerate(_FORGE_ART):
            fragments.append(("class:header_art", art_line))
            overlay = telemetry_by_row.get(index)
            if overlay:
                fragments.append(("class:header_telemetry", f"   {overlay}"))
            fragments.append(("", "\n" if index < len(_FORGE_ART) - 1 else ""))
        return fragments

    def _render_viewport(self) -> AnyFormattedText:
        """Render the exclusive stage viewport buffer using ANSI conversion."""
        cols = shutil.get_terminal_size((100, 30)).columns
        viewport_width = max(30, cols - 32)
        stage_buffer = self.state.get_active_view_buffer()
        ansi_output = stage_buffer.render_to_ansi(
            width=viewport_width, pending_question=self.state.pending_question
        )
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

        @self.kb.add("f5")
        @self.kb.add("escape", "m")
        def _toggle_cycle_mode(event: KeyPressEvent) -> None:
            new_mode = self.state.toggle_cycle_mode()
            if self.controller and hasattr(self.controller, "set_cycle_mode"):
                self.controller.set_cycle_mode(new_mode)

        @self.kb.add("pageup")
        def _scroll_up(event: KeyPressEvent) -> None:
            self.state.scroll_up(5)

        @self.kb.add("pagedown")
        def _scroll_down(event: KeyPressEvent) -> None:
            self.state.scroll_down(5)

        @self.kb.add("tab")
        def _complete(event: KeyPressEvent) -> None:
            """Cycle through the `/` completion popup, opening it on first Tab."""
            buffer = event.current_buffer
            if buffer.complete_state:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=True)

        @self.kb.add("enter")
        def _handle_enter(event: KeyPressEvent) -> None:
            user_text = self.input_buffer.text
            self.input_buffer.text = ""

            # 1. Answer pending question if Orchestrator is awaiting user response
            if self.state.pending_question is not None:
                self.state.resolve_pending_question(user_text)
                return

            stripped = user_text.strip()
            if not stripped:
                return

            # 2. Explicit exit
            if stripped.lower() in {"/exit", "/quit", "exit", "quit"}:
                event.app.exit()
                return

            # 3. Everything else (including `/commands` and `!shell`) goes straight to the
            #    controller's deterministic dispatcher on a background task.
            async def _run_submission() -> None:
                await self.submit_text(stripped)

            self._background_task = self.application.create_background_task(_run_submission())

        @self.kb.add("c-c")
        def _handle_ctrl_c(event: KeyPressEvent) -> None:
            if self.state.pending_question is not None:
                self.state.cancel_pending_question()
                return

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

    async def submit_text(self, user_text: str) -> str:
        """Route a submitted input to the adaptive controller, resolving the pinned stage.

        Explicit ``/`` and ``!`` syntax is dispatched immediately by the controller's slash
        handler; free-form text is routed through intent classification with the pinned stage
        (when the developer has turned auto-follow off).
        """
        if self.controller is None:
            self.state.append_thought(ForgeStage.ANVIL, f"User Goal: {user_text}")
            return ""

        stripped = user_text.strip()
        if stripped.startswith("/") or stripped.startswith("!"):
            handler = getattr(self.controller, "handle_slash_command", None)
            if callable(handler):
                result = await handler(stripped)
                return result if isinstance(result, str) else ""

        stage: ForgeStage | None = None
        if not self.state.auto_follow:
            stage = self.state.selected_view_stage

        result = await self.controller.run_cycle_loop(user_text, target_stage=stage)
        return result if isinstance(result, str) else ""

    async def _spin(self) -> None:
        """Animate the running-stage spinner while the app is live."""
        while True:
            await asyncio.sleep(0.12)
            self._spinner_index += 1
            if self.state.active_worker_stage is not None:
                self._handle_state_redraw()

    async def run_async(self) -> None:
        """Run the full-screen TUI application asynchronously.

        The optional ``bootstrap`` coroutine builds the live multi-agent session inside this
        same event loop (so database/browser resources are not bound to a foreign loop), and
        ``close_session`` tears it down once the UI exits.
        """
        if self.bootstrap is not None:
            await self.bootstrap()
        self._spinner_task = asyncio.ensure_future(self._spin())
        try:
            await self.application.run_async()
        finally:
            if self._spinner_task is not None:
                self._spinner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._spinner_task
            if self.close_session is not None:
                await self.close_session()

    def run(self) -> None:
        """Run the full-screen TUI application synchronously."""
        self.application.run()


def _is_compact_terminal() -> bool:
    """True when the terminal is too short for the multi-line ASCII banner."""
    try:
        lines = shutil.get_terminal_size((100, 30)).lines
    except Exception:  # pragma: no cover - defensive
        return False
    return lines < 24
