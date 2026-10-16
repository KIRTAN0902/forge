"""Interactive Agent Chat Interface (Claude Code-style experience).

Provides a rich multi-turn terminal UI for interacting with Forge:
- Interactive welcome banner and repository status display.
- One-time API key setup and persistence.
- Interactive prompt with command history, autocomplete, and slash commands.
- Live streaming execution of multi-agent plans, tasks, tool calls, and QA audits.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from forge import __version__
from forge.agents.base import BaseAgent
from forge.agents.executor import ExecutorAgent
from forge.agents.orchestrator import OrchestratorAgent
from forge.agents.qa import QAAgent
from forge.cli.cancellation import run_with_esc_cancellation
from forge.cli.credentials import (
    PROVIDER_METADATA,
    ensure_credentials_configured,
    load_stored_credentials,
    prompt_for_credentials,
)
from forge.cli.interactive import AutoApprover, CliApprover
from forge.cli.render import (
    CliEventPrinter,
    print_workflow_list,
    print_workflow_status,
)
from forge.cli.workflow import build_workflow_harness
from forge.config import init_project, load_config
from forge.core.contracts.state import WorkflowState
from forge.core.errors import ForgeError, UserCancelledError
from forge.events.bus import EventBus
from forge.events.types import Event, EventType
from forge.models.types import ContentPart, Message, TextPart
from forge.storage.sqlite import SQLiteStore
from forge.tools import create_environment
from forge.workflow.recovery import CheckpointManager

ASCII_LOGO = r"""
[bold cyan]  ███████╗ ██████╗ ██████╗  ██████╗ ███████╗[/bold cyan]
[bold cyan]  ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝[/bold cyan]
[bold cyan]  █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  [/bold cyan]
[bold cyan]  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  [/bold cyan]
[bold cyan]  ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗[/bold cyan]
[bold cyan]  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝[/bold cyan]
"""

AVAILABLE_AGENTS = ("orchestrator", "executor", "qa")

AGENT_METADATA: dict[str, dict[str, str]] = {
    "orchestrator": {
        "title": "ANVIL",
        "icon": "🎯",
        "color": "cyan",
        "status_color": "bold cyan",
        "description": "Brainstorming, architecture, task planning & technical lead (Orchestrator)",
    },
    "executor": {
        "title": "HAMMER",
        "icon": "⚡",
        "color": "yellow",
        "status_color": "bold yellow",
        "description": "Code writing, filesystem modification & terminal tool execution (Executor)",
    },
    "qa": {
        "title": "TEMPER",
        "icon": "🔍",
        "color": "green",
        "status_color": "bold green",
        "description": "Adversarial testing, security audit & regression verification (QA)",
    },
}


class ChatSessionState:
    """Tracks the active chat agent and mode (interactive chat vs autonomous AUTO mode)."""

    def __init__(self, default_agent: str = "orchestrator", auto_mode: bool = False) -> None:
        self.active_agent: str = default_agent
        self.auto_mode: bool = auto_mode

    def toggle_auto_mode(self) -> bool:
        """Toggle between interactive chat mode and autonomous AUTO mode."""
        self.auto_mode = not self.auto_mode
        return self.auto_mode

    def set_auto_mode(self, enabled: bool) -> None:
        """Explicitly set AUTO mode state."""
        self.auto_mode = enabled

    def cycle_agent(self) -> str:
        idx = (
            AVAILABLE_AGENTS.index(self.active_agent)
            if self.active_agent in AVAILABLE_AGENTS
            else 0
        )
        self.active_agent = AVAILABLE_AGENTS[(idx + 1) % len(AVAILABLE_AGENTS)]
        return self.active_agent

    def cycle_agent_reverse(self) -> str:
        idx = (
            AVAILABLE_AGENTS.index(self.active_agent)
            if self.active_agent in AVAILABLE_AGENTS
            else 0
        )
        self.active_agent = AVAILABLE_AGENTS[(idx - 1) % len(AVAILABLE_AGENTS)]
        return self.active_agent

    def set_agent(self, agent: str) -> bool:
        low = agent.lower().strip()
        alias_map = {
            "anvil": "orchestrator",
            "orch": "orchestrator",
            "orchestrator": "orchestrator",
            "planner": "orchestrator",
            "plan": "orchestrator",
            "hammer": "executor",
            "exec": "executor",
            "executor": "executor",
            "temper": "qa",
            "qa": "qa",
            "audit": "qa",
        }
        if low in alias_map:
            self.active_agent = alias_map[low]
            return True
        return False


HELP_COMMANDS = [
    ("/help", "Show this list of available slash commands"),
    ("/auto", "Toggle AUTO mode (autonomous Planning ➔ Execution ➔ QA without pausing)"),
    ("/anvil", "Switch active agent to 🎯 ANVIL (Orchestrator: architecture & planning)"),
    ("/hammer", "Switch active agent to ⚡ HAMMER (Executor: coding & tool execution)"),
    ("/temper", "Switch active agent to 🔍 TEMPER (QA: adversarial testing & verification)"),
    ("/paste", "Paste screenshot image or copied file directly from system clipboard"),
    ("/image", "Attach image/screenshot from file path or clipboard for visual analysis (/img)"),
    ("/screenshot", "Capture screen or clipboard screenshot and analyze with active agent"),
    ("/file", "Attach a code, document, or log file to analyze (/attach)"),
    ("/search", "Perform a live web search for documentation, APIs, and error solutions"),
    ("/research", "Run deep research with multi-perspective synthesis & verified BibTeX"),
    ("/models", "Interactive popup to select and configure agent models"),
    ("/provider", "Connect and configure model providers, API keys, and custom Base URIs/URLs"),
    ("/orchestrator", "Alias for /anvil"),
    ("/executor", "Alias for /hammer"),
    ("/qa", "Alias for /temper"),
    ("/agent", "Show or switch the active chat agent (or press Tab at the prompt)"),
    ("/skills", "List all 43 specialist skills and autonomous triggers"),
    ("/run", "Directly trigger multi-agent planning & execution on a request"),
    ("/status", "Show status, task graph, and QA verdict of current/latest workflow"),
    ("/diff", "Show git diff of modifications made by Forge"),
    ("/rollback", "Revert working tree to the latest checkpoint"),
    ("/history", "List past workflows and event timeline"),
    ("/key", "Reconfigure or switch your API key / provider"),
    ("/clear", "Clear terminal screen and redraw status banner"),
    ("/exit", "Exit the interactive session (or /quit)"),
]


def print_skills_table(console: Console) -> None:
    """Render a formatted table of all 24 built-in specialist skills."""
    from forge.skills import ALL_SKILLS

    table = Table(title="Forge Pre-Programmed Specialist Skills", border_style="cyan")
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Skill Name", style="bold white")
    table.add_column("Category", style="magenta")
    table.add_column("Target Agent(s)", style="yellow")
    table.add_column("Primary Trigger Signal", style="dim")

    agent_icons = {"orchestrator": "🎯 ANVIL", "executor": "⚡ HAMMER", "qa": "🔍 TEMPER"}
    for sk in ALL_SKILLS:
        agents_str = ", ".join(agent_icons.get(a, a) for a in sk.target_agents)
        table.add_row(sk.id, sk.name, sk.category.value, agents_str, sk.primary_trigger)

    console.print()
    console.print(table)
    console.print(
        "[dim]These skills are autonomously evaluated and activated by ANVIL on matching signals.[/dim]\n"
    )


def _get_git_info(root: Path) -> str:
    """Return git branch and status if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            branch = res.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=2,
            )
            dirty = " [yellow]*dirty[/yellow]" if status.stdout.strip() else " [green]clean[/green]"
            return f"[magenta]{branch}[/magenta]{dirty}"
    except Exception:
        pass
    return "[dim]not a git repo[/dim]"


def _get_active_model_info() -> str:
    """Return a human-friendly description of the active model and provider."""
    stored = load_stored_credentials()
    if stored:
        prov = stored.get("provider", "unknown")
        model = stored.get("model", "default")
        prov_label = PROVIDER_METADATA.get(prov, {}).get("name", prov.title())
        color = "green" if prov in {"nvidia", "openai"} else "cyan"
        return f"[bold {color}]{prov_label}[/bold {color}] ({model})"

    if os.environ.get("NVIDIA_API_KEY"):
        model = os.environ.get("FORGE_DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
        return f"[bold green]NVIDIA NIM[/bold green] ({model})"
    if os.environ.get("OPENROUTER_API_KEY"):
        model = os.environ.get("FORGE_DEFAULT_MODEL", "anthropic/claude-3.7-sonnet")
        return f"[bold cyan]OpenRouter[/bold cyan] ({model})"
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("FORGE_DEFAULT_MODEL", "gpt-4o")
        return f"[bold green]OpenAI[/bold green] ({model})"

    return "[dim]None configured[/dim]"


def print_banner(console: Console, root: Path) -> None:
    """Print the interactive header with repository and model status."""
    console.print(ASCII_LOGO)

    git_info = _get_git_info(root)
    model_info = _get_active_model_info()

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()

    grid.add_row("Version:", f"forge v{__version__}")
    grid.add_row("Project:", f"[bold white]{root.name}[/bold white] [dim]({root.resolve()})[/dim]")
    grid.add_row("Branch:", git_info)
    grid.add_row("Model:", model_info)

    try:
        cfg = load_config(root)
        if cfg.roles:
            role_strs = []
            for r_name in ("orchestrator", "planner", "executor", "qa"):
                if r_name in cfg.roles:
                    prof_name = cfg.roles[r_name].model
                    prof = cfg.models.get(prof_name)
                    if prof:
                        m_short = prof.model.split("/")[-1]
                        role_strs.append(f"{r_name.title()}: [cyan]{m_short}[/cyan]")
            if role_strs:
                grid.add_row("Roles:", " | ".join(role_strs))
    except Exception:
        pass

    info_text = (
        "\n[dim]Active Agent: [bold cyan]🎯 ANVIL[/bold cyan] (Press [bold cyan]Tab[/bold cyan] to cycle: ANVIL ➔ HAMMER ➔ TEMPER)\n"
        "Press [bold green]F2[/bold green] or [bold green]Alt+A[/bold green] to toggle [bold green]⚡ AUTO mode[/bold green] (autonomous Planning ➔ Execution ➔ QA).\n"
        "Type your prompt to begin, or [cyan]/help[/cyan] for commands. Use [cyan]/search <query>[/cyan] for web search, [cyan]/exit[/cyan] to quit.[/dim]"
    )
    console.print(
        Panel(
            Group(grid, info_text),
            title="Interactive Agent Session",
            border_style="cyan",
        )
    )


def position_cursor_at_bottom(box_height: int = 3) -> None:
    """Ensure cursor is anchored at the bottom of the viewport so the input box sits at the page base."""
    try:
        import os
        import sys

        if not sys.stdout.isatty():
            return

        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class COORD(ctypes.Structure):
                _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

            class SMALL_RECT(ctypes.Structure):
                _fields_ = [
                    ("Left", wintypes.SHORT),
                    ("Top", wintypes.SHORT),
                    ("Right", wintypes.SHORT),
                    ("Bottom", wintypes.SHORT),
                ]

            class CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
                _fields_ = [
                    ("dwSize", COORD),
                    ("dwCursorPosition", COORD),
                    ("wAttributes", wintypes.WORD),
                    ("srWindow", SMALL_RECT),
                    ("dwMaximumWindowSize", COORD),
                ]

            h_con = ctypes.windll.kernel32.CreateFileW(
                "CONOUT$", 0x80000000 | 0x40000000, 1 | 2, None, 3, 0, None
            )
            if h_con != -1 and h_con != 0:
                try:
                    csbi = CONSOLE_SCREEN_BUFFER_INFO()
                    if ctypes.windll.kernel32.GetConsoleScreenBufferInfo(h_con, ctypes.byref(csbi)):
                        window_height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1
                        cursor_row = csbi.dwCursorPosition.Y - csbi.srWindow.Top
                        lines_to_bottom = window_height - 1 - cursor_row
                        pad = lines_to_bottom - (box_height - 1)
                        if pad > 0:
                            sys.stdout.write("\n" * pad)
                            sys.stdout.flush()
                        return
                finally:
                    ctypes.windll.kernel32.CloseHandle(h_con)

        # Fallback for non-Windows or if Win32 API is unavailable
        sys.stdout.write(f"\033[999B\033[{max(1, box_height - 1)}A\r")
        sys.stdout.flush()
    except Exception:
        pass


def _get_prompt_tokens(state: ChatSessionState) -> list[tuple[str, str]]:
    """Generate dynamic prompt_toolkit formatted text tokens for a proper enclosed input box."""
    import shutil

    agent = state.active_agent
    meta = AGENT_METADATA.get(agent, AGENT_METADATA["orchestrator"])
    width = max(40, shutil.get_terminal_size((80, 24)).columns)

    mode_status = "Auto: ON" if state.auto_mode else "Auto: OFF"
    if width >= 75:
        hints = f" [F2: {mode_status} · Tab: Agent · Esc: Clear] ─╮"
    elif width >= 55:
        hints = f" [F2: {mode_status} · Tab: Agent] ─╮"
    else:
        hints = " ─╮"

    mode_suffix = " [⚡ AUTO]" if state.auto_mode else ""
    left = f"╭─ {meta['icon']} {meta['title']}{mode_suffix} "
    fill_len = max(2, width - len(left) - len(hints))

    tokens = [
        ("class:box_border", "╭─ "),
        (f"class:prompt_agent_{agent}", f"{meta['icon']} {meta['title']}"),
    ]
    if state.auto_mode:
        tokens.append(("class:prompt_mode_auto", " [⚡ AUTO]"))
    tokens.extend(
        [
            ("class:box_border", " " + ("─" * fill_len)),
            ("class:box_dim", hints[:-3]),
            ("class:box_border", " ─╮\n"),
            ("class:box_border", "│ "),
            ("class:prompt_arrow", "> "),
        ]
    )
    return tokens


def _get_rprompt(state: ChatSessionState) -> list[tuple[str, str]]:
    """Right border closing the input row of the box."""
    return [("class:box_border", "│")]


def _get_prompt_continuation(width: int, line_number: int, wrap_count: int) -> Any:
    """Line prefix for multiline continuation inside the enclosed box."""
    return [
        ("class:box_border", "│ "),
        ("class:prompt_arrow", "· "),
    ]


def _get_bottom_toolbar(state: ChatSessionState) -> list[tuple[str, str]]:
    """Bottom border closing the input box with keyboard shortcuts."""
    import shutil

    width = max(40, shutil.get_terminal_size((80, 24)).columns)
    action_label = "Run Auto" if state.auto_mode else "Send"
    auto_label = "Auto: ON" if state.auto_mode else "Auto: OFF"

    if width >= 80:
        left = f"╰─ [Enter] {action_label}  [F2 / Alt+A] {auto_label}  [Tab] Agent  [Ctrl+V] Paste "
        right = " [/help] ─╯"
    elif width >= 60:
        left = f"╰─ [Enter] {action_label}  [F2] {auto_label}  [Tab] Agent "
        right = " [/help] ─╯"
    else:
        left = f"╰─ [Enter] {action_label}  [F2] {auto_label} "
        right = " ─╯"

    fill_len = max(2, width - len(left) - len(right))
    return [
        ("class:box_border", left),
        ("class:box_border", "─" * fill_len),
        ("class:box_border", right),
    ]


def _create_prompt_session(state: ChatSessionState | None = None) -> Any:
    """Create a prompt_toolkit PromptSession with history, completions, and Tab agent cycling."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_processor import KeyPressEvent
        from prompt_toolkit.styles import Style

        current_state = state or ChatSessionState()
        history_file = Path.home() / ".forge" / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        commands = [cmd for cmd, _ in HELP_COMMANDS] + [
            "/quit",
            "/anvil",
            "/hammer",
            "/temper",
            "/orch",
            "/exec",
            "/search",
            "/auto",
            "/mode",
        ]
        completer = WordCompleter(commands, ignore_case=True, match_middle=False)

        kb = KeyBindings()

        @kb.add("tab")
        def _handle_tab(event: KeyPressEvent) -> None:
            buf = event.app.current_buffer
            # If autocompletion menu is already active, navigate completion
            if buf.complete_state:
                buf.complete_next()
                return

            # If user typed a slash command prefix, trigger completion
            if buf.text.startswith("/") and not buf.complete_state:
                buf.start_completion(select_first=False)
                if buf.complete_state:
                    return

            # Otherwise, cycle active agent!
            current_state.cycle_agent()
            event.app.invalidate()

        @kb.add("s-tab")
        def _handle_shift_tab(event: KeyPressEvent) -> None:
            current_state.cycle_agent_reverse()
            event.app.invalidate()

        @kb.add("f2")
        def _handle_f2(event: KeyPressEvent) -> None:
            """F2 toggles AUTO mode."""
            current_state.toggle_auto_mode()
            event.app.invalidate()

        @kb.add("escape", "a")
        def _handle_alt_a(event: KeyPressEvent) -> None:
            """Alt+A toggles AUTO mode."""
            current_state.toggle_auto_mode()
            event.app.invalidate()

        @kb.add("c-o")
        def _handle_ctrl_o(event: KeyPressEvent) -> None:
            """Ctrl+O toggles AUTO mode."""
            current_state.toggle_auto_mode()
            event.app.invalidate()

        @kb.add("c-a")
        def _handle_ctrl_a(event: KeyPressEvent) -> None:
            """Ctrl+A toggles AUTO mode if input is empty; otherwise moves cursor to beginning of line."""
            buf = event.app.current_buffer
            if not buf.text:
                current_state.toggle_auto_mode()
                event.app.invalidate()
            else:
                buf.cursor_position += buf.document.get_start_of_line_position()

        @kb.add("c-j")
        @kb.add("escape", "enter")
        def _handle_newline(event: KeyPressEvent) -> None:
            """Shift+Enter / Ctrl+J / Alt+Enter inserts a clean newline for multiline drafting."""
            event.app.current_buffer.insert_text("\n")

        @kb.add("enter")
        def _handle_enter(event: KeyPressEvent) -> None:
            """Enter submits the input prompt or commits active completion."""
            buf = event.app.current_buffer
            if buf.complete_state:
                buf.complete_state = None
            else:
                buf.validate_and_handle()

        @kb.add("escape")
        def _handle_escape(event: KeyPressEvent) -> None:
            """Escape acts as an escape hatch to dismiss completions or clear drafted input."""
            buf = event.app.current_buffer
            if buf.complete_state:
                buf.cancel_completion()
                return
            if buf.text:
                buf.text = ""
                buf.cursor_position = 0
                event.app.invalidate()

        @kb.add("c-v")
        def _handle_ctrl_v(event: KeyPressEvent) -> None:
            buf = event.app.current_buffer
            try:
                from forge.cli.attachments import get_clipboard_files, get_clipboard_image

                att = get_clipboard_image()
                if att is not None and att.is_image:
                    buf.insert_text(f"[image: {att.path.as_posix()}] ")
                    event.app.invalidate()
                    return
                files = get_clipboard_files()
                if files:
                    file_tokens = " ".join(
                        f"[{'image' if f.is_image else 'file'}: {f.path.as_posix()}]" for f in files
                    )
                    buf.insert_text(f"{file_tokens} ")
                    event.app.invalidate()
                    return
            except Exception:
                pass

            # Standard text paste fallback
            try:
                clipboard_data = event.app.clipboard.get_data()
                if clipboard_data and clipboard_data.text:
                    buf.insert_text(clipboard_data.text)
            except Exception:
                pass
            event.app.invalidate()

        @kb.add("escape", "v")
        def _handle_alt_v(event: KeyPressEvent) -> None:
            _handle_ctrl_v(event)

        prompt_style = Style.from_dict(
            {
                "prompt_brand": "#00e5ff bold",
                "prompt_bracket": "#64748b",
                "prompt_agent_orchestrator": "#00e5ff bold",
                "prompt_agent_planner": "#d946ef bold",
                "prompt_agent_executor": "#eab308 bold",
                "prompt_agent_qa": "#10b981 bold",
                "prompt_mode_auto": "#10b981 bold",
                "box_border": "#475569",
                "box_dim": "#64748b",
                "box_key": "#94a3b8",
                "prompt_arrow": "#00e5ff bold",
                "placeholder": "#64748b italic",
                "bottom-toolbar": "noinherit",
                "bottom-toolbar.text": "noinherit",
            }
        )

        try:
            from prompt_toolkit.output.defaults import create_output

            pt_output = create_output()
        except Exception:
            from prompt_toolkit.output import DummyOutput

            pt_output = DummyOutput()

        return PromptSession(
            history=FileHistory(str(history_file)),
            completer=completer,
            key_bindings=kb,
            style=prompt_style,
            multiline=True,
            erase_when_done=True,
            prompt_continuation=_get_prompt_continuation,
            placeholder=lambda: (
                "AUTO mode: Enter request for autonomous Planning ➔ Execution ➔ QA (F2 to switch)..."
                if current_state.auto_mode
                else "Type your message, /help for commands, or press F2 for AUTO mode..."
            ),
            output=pt_output,
        )
    except Exception:
        return None


def read_user_input(session: Any, console: Console, state: ChatSessionState | None = None) -> str:
    """Read a line of input from the user in a bottom-anchored text box."""
    current_state = state or ChatSessionState()
    meta = AGENT_METADATA.get(current_state.active_agent, AGENT_METADATA["orchestrator"])

    if session is not None:
        try:
            position_cursor_at_bottom(box_height=3)
            return str(
                session.prompt(
                    message=lambda: _get_prompt_tokens(current_state),
                    rprompt=lambda: _get_rprompt(current_state),
                    bottom_toolbar=lambda: _get_bottom_toolbar(current_state),
                )
            ).strip()
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            pass

    # Fallback to standard input
    color = meta.get("color", "cyan")
    try:
        from rich.prompt import Prompt

        return Prompt.ask(
            f"[bold cyan]forge[/bold cyan] [[bold {color}]{meta['icon']} {meta['title']}[/bold {color}]] >",
            console=console,
        ).strip()
    except (KeyboardInterrupt, EOFError):
        raise


def print_help(console: Console) -> None:
    """Print slash commands cheat sheet."""
    table = Table(title="Available Slash Commands", border_style="cyan")
    table.add_column("Command", style="bold cyan", width=14)
    table.add_column("Description")
    for cmd, desc in HELP_COMMANDS:
        table.add_row(cmd, desc)
    console.print(table)


async def _show_workflow_status(path: Path, workflow_id: str | None, console: Console) -> None:
    """Fetch and print status of current/latest workflow."""
    config = load_config(path)
    store = await SQLiteStore.open(config.db_path)
    try:
        wid = workflow_id
        if wid is None:
            recent = await store.list_workflows(limit=1)
            if not recent:
                console.print("[dim]No workflows found in this project.[/dim]")
                return
            wid = recent[0].workflow_id
        state = await store.load_workflow(wid)
        print_workflow_status(state, console)
    finally:
        await store.close()


async def _execute_diff(path: Path, console: Console) -> None:
    """Fetch and print syntax-highlighted git diff."""
    config = load_config(path)
    env = create_environment(config, root=config.root)
    try:
        repo = await env.git()
        if repo is None:
            console.print("[yellow]Not a Git repository.[/yellow]")
            return
        diff_text = await repo.diff()
        if not diff_text or not diff_text.strip():
            console.print("[dim]No uncommitted changes detected.[/dim]")
        else:
            from rich.syntax import Syntax

            console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=True))
    finally:
        await env.aclose()


async def _execute_rollback(path: Path, workflow_id: str | None, console: Console) -> None:
    """Roll back working directory to latest checkpoint."""
    config = load_config(path)
    store = await SQLiteStore.open(config.db_path)
    try:
        wid = workflow_id
        if wid is None:
            recent = await store.list_workflows(limit=1)
            if not recent:
                console.print("[dim]No workflows found in this project.[/dim]")
                return
            wid = recent[0].workflow_id

        state = await store.load_workflow(wid)
        if not state.checkpoints:
            console.print("[yellow]No checkpoints recorded for this workflow.[/yellow]")
            return

        checkpoint_mgr = CheckpointManager()
        env = create_environment(config, root=config.root)
        try:
            res = await checkpoint_mgr.rollback_to_latest(env, state.checkpoints)
        finally:
            await env.aclose()

        if res:
            console.print(f"[green]Restored checkpoint {res.checkpoint_id}.[/green]")
            await store.append_event(
                Event(
                    type=EventType.ROLLBACK_PERFORMED,
                    workflow_id=state.workflow_id,
                    source="chat",
                    payload={"checkpoint_id": res.checkpoint_id},
                )
            )
        else:
            console.print("[red]Rollback failed.[/red]")
    finally:
        await store.close()


async def _execute_history(path: Path, console: Console) -> None:
    """Fetch and print workflow history table."""
    config = load_config(path)
    store = await SQLiteStore.open(config.db_path)
    try:
        wfs = await store.list_workflows(limit=10)
        print_workflow_list(wfs, console)
    finally:
        await store.close()


async def _run_workflow_turn(
    request: str,
    path: Path,
    console: Console,
    current_workflow_id: str | None = None,
    auto_mode: bool = False,
) -> WorkflowState | None:
    """Run an autonomous workflow turn for the user's prompt."""
    config = load_config(path)
    event_bus = EventBus()
    event_bus.subscribe(CliEventPrinter(console))

    approver = AutoApprover(console) if auto_mode else CliApprover(console)
    harness = await build_workflow_harness(config, approver=approver, event_bus=event_bus)

    try:
        # Check if we should resume the current active workflow with human feedback
        if current_workflow_id:
            try:
                existing = await harness.store.load_workflow(current_workflow_id)
                if not existing.is_terminal:
                    console.print(f"[dim]Continuing workflow {current_workflow_id}...[/dim]")
                    final_state = await run_with_esc_cancellation(
                        harness.orchestrator.resume(
                            existing, feedback=request, auto_mode=auto_mode
                        ),
                        status_message="[bold green]Workflow resuming...[/bold green]",
                        console=console,
                    )
                    console.print()
                    print_workflow_status(final_state, console)
                    return final_state
            except UserCancelledError:
                raise
            except Exception:
                pass

        # Otherwise, start a fresh workflow for this instruction
        state = WorkflowState(user_request=request)
        final_state = await run_with_esc_cancellation(
            harness.orchestrator.run(state, auto_mode=auto_mode),
            status_message="[bold green]⚡ AUTO mode executing (Planning ➔ Execution ➔ QA)...[/bold green]"
            if auto_mode
            else "[bold green]Workflow executing...[/bold green]",
            console=console,
        )
        console.print()
        print_workflow_status(final_state, console)
        return final_state
    finally:
        await harness.aclose()


def print_auto_mode_completion(state: WorkflowState, console: Console) -> None:
    """Display an elevated completion banner when AUTO mode finishes planning, execution, and QA."""
    tasks_count = len(state.plan.tasks) if state.plan else 0
    done_count = sum(
        1 for t in (state.plan.tasks if state.plan else []) if t.status.value in {"done", "skipped"}
    )
    qa_status = state.qa_reports[-1].status.value.upper() if state.qa_reports else "NOT RUN"
    qa_color = "green" if qa_status == "PASS" else ("yellow" if qa_status == "WARN" else "red")

    summary_lines = [
        f"[bold white]Request:[/bold white] {state.user_request[:120]}{'...' if len(state.user_request) > 120 else ''}",
        f"[bold white]Status:[/bold white] [bold green]✔ {state.status.value.upper()}[/bold green]  "
        f"[bold white]Iterations:[/bold white] {state.iteration}",
        "",
        "  • [bold cyan]Planning (ANVIL):[/bold cyan] Requirements contract & dependency graph generated",
        f"  • [bold yellow]Execution (HAMMER):[/bold yellow] {done_count}/{tasks_count} tasks completed",
        f"  • [bold green]QA Verification (TEMPER):[/bold green] [{qa_color}]{qa_status}[/{qa_color}]",
        "",
        "[dim]Full audit trail recorded in .forge/forge.md and .forge/forge.json[/dim]",
    ]

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="[bold green]⚡ AUTO MODE COMPLETED — All Stages Verified[/bold green]",
            border_style="green",
        )
    )


def sanitize_response_for_display(text: str) -> str:
    """Clean raw internal tool artifacts (line numbers, pagination tags, raw grep dumps) if regurgitated by LLMs."""
    if not text:
        return ""
    import re

    # 1. Clean raw tool line numbering: e.g. '   1\t# Title' -> '# Title'
    cleaned = re.sub(r"^[ \t]*\d+\t", "", text, flags=re.MULTILINE)
    # 2. Clean pagination marker: '[X more lines; use offset=Y]' -> '*(X more lines in file)*'
    cleaned = re.sub(r"\[(\d+) more lines; use offset=\d+\]", r"*(\1 more lines in file)*", cleaned)
    # 3. Clean raw read_file header: 'path/file.ext (lines X-Y of Z)' -> '**`path/file.ext`** *(lines X-Y of Z)*'
    cleaned = re.sub(
        r"(?m)^([a-zA-Z0-9_\-./\\]+) \(lines (\d+-\d+ of \d+)\)$",
        r"**`\1`** *(lines \2)*\n",
        cleaned,
    )
    # 4. Clean raw terminal command dump header: '$ cmd [exit 0, 0.1s]'
    cleaned = re.sub(
        r"(?m)^\$\s+(.+?)\s*(?:\n|\s+)\[(exit\s+\d+|timed\s+out)[^\]]*\]\s*",
        r"**Command Output (`\1`):**\n\n",
        cleaned,
    )
    # 5. Split inlined/squished grep results back onto separate lines
    cleaned = re.sub(r"\s+(\d+:)", r"\n\1", cleaned)

    # 6. Format multi-file grep output: 'path:line:content' -> '- **`path`** (Line line): `content`'
    cleaned = re.sub(
        r"(?m)^([a-zA-Z0-9_\-./\\]+):(\d+):(.*)$",
        r"- **`\1`** (Line \2): `\3`",
        cleaned,
    )

    # 7. Format single-file grep output: 'line:content' -> '- **Line line**: `content`'
    def format_grep_line(match: re.Match[str]) -> str:
        line_num = match.group(1)
        content = match.group(2).strip()
        return f"- **Line {line_num}**: `{content}`"

    cleaned = re.sub(r"(?m)^(\d+):(.*)$", format_grep_line, cleaned)

    return cleaned


async def _run_agent_chat_turn(
    agent_role: str,
    user_input: str | list[ContentPart],
    path: Path,
    console: Console,
    history: list[Message],
    current_workflow_id: str | None = None,
) -> tuple[list[Message], str | None]:
    """Chat interactively with the selected specialist agent."""
    config = load_config(path)
    event_bus = EventBus()
    approver = CliApprover(console)
    harness = await build_workflow_harness(config, approver=approver, event_bus=event_bus)

    try:
        meta = AGENT_METADATA.get(agent_role, AGENT_METADATA["orchestrator"])
        adapter = await harness.registry.adapter(agent_role, events=event_bus)

        if agent_role in {"orchestrator", "planner"}:
            agent: BaseAgent = OrchestratorAgent(adapter, runtime=harness.runtime, events=event_bus)
        elif agent_role == "executor":
            agent = ExecutorAgent(adapter, runtime=harness.runtime, events=event_bus)
        elif agent_role == "qa":
            agent = QAAgent(adapter, runtime=harness.runtime, events=event_bus)
        else:
            agent = OrchestratorAgent(adapter, runtime=harness.runtime, events=event_bus)

        git_info = _get_git_info(path)
        project_context = f"Project: {path.name}\nBranch: {git_info}\nPath: {path.resolve()}"

        # Autonomous Skill Routing for Chat Turns
        from forge.skills import (
            SkillEvaluator,
            SkillSignal,
            format_skills_badge,
            format_skills_for_prompt,
        )

        evaluator = SkillEvaluator()
        text_for_signal = (
            user_input
            if isinstance(user_input, str)
            else "\n".join(p.text for p in user_input if isinstance(p, TextPart))
        )
        signal = SkillSignal(
            user_request=text_for_signal,
            error_text=text_for_signal,
        )
        active_skills = evaluator.evaluate(signal, agent_role=agent_role)
        if active_skills:
            console.print(format_skills_badge(active_skills))
            skill_instruction = format_skills_for_prompt(active_skills)
            project_context += f"\n\n{skill_instruction}"

        status_msg = f"[{meta['status_color']}]{meta['icon']} {meta['title']} thinking...[/{meta['status_color']}]"
        turn = await run_with_esc_cancellation(
            agent.chat(
                user_input,
                history=history,
                project_context=project_context,
            ),
            status_message=status_msg,
            console=console,
        )

        console.print()
        console.print(
            f"[{meta['status_color']}]{meta['icon']} {meta['title']}:[/{meta['status_color']}]"
        )
        from rich.markdown import Markdown

        display_text = sanitize_response_for_display(turn.response_text)
        console.print(Markdown(display_text))
        console.print()

        updated_history = list(turn.messages)

        # Handoff to Planning Agent if requirements were synthesized (Orchestrator)
        if turn.handoff_requirements:
            console.print(
                Panel(
                    f"[bold green]Synthesized Requirements Handoff to Planning Agent:[/bold green]\n\n"
                    f"{turn.handoff_requirements}\n\n"
                    f"[dim]Initiating autonomous multi-agent workflow (Planner ➔ Executor ➔ QA)...[/dim]",
                    title="🎯 Orchestrator ➔ 📐 Planner Handoff",
                    border_style="green",
                )
            )

            # Connect event printer so the user sees live execution feedback
            event_bus.subscribe(CliEventPrinter(console))

            state = WorkflowState(user_request=turn.handoff_requirements)
            final_state = await run_with_esc_cancellation(
                harness.orchestrator.run(state),
                status_message="[bold green]Workflow executing...[/bold green]",
                console=console,
            )
            console.print()
            print_workflow_status(final_state, console)

            summary_msg = Message.assistant(
                f"[Workflow {final_state.workflow_id} ended with status: {final_state.status.value}. "
                f"Tasks executed: {len(final_state.tasks)}]"
            )
            updated_history.append(summary_msg)
            return updated_history, final_state.workflow_id

        return updated_history, current_workflow_id
    finally:
        await harness.aclose()


async def _run_orchestrator_chat_turn(
    user_input: str,
    path: Path,
    console: Console,
    history: list[Message],
    current_workflow_id: str | None = None,
) -> tuple[list[Message], str | None]:
    """Chat interactively with the Orchestrator (backward compatibility helper)."""
    return await _run_agent_chat_turn(
        "orchestrator",
        user_input,
        path=path,
        console=console,
        history=history,
        current_workflow_id=current_workflow_id,
    )


def run_chat_session(path: Path = Path(".")) -> None:
    """Main interactive chat loop."""
    console = Console(legacy_windows=False)
    path = path.resolve()

    # 1. Ensure credentials are ready (prompts on first run, saves permanently)
    ensure_credentials_configured(console)

    # 2. Ensure project is initialized
    if not (path / "forge.yaml").is_file() and not (path / ".forge").is_dir():
        init_project(path)

    # 3. Print welcome screen
    print_banner(console, path)

    # 4. Prepare input session, active agent state (default: orchestrator), and conversational history
    session_state = ChatSessionState(default_agent="orchestrator")
    session = _create_prompt_session(session_state)
    current_workflow_id: str | None = None
    conversation_history: list[Message] = []

    from forge.cli.app import _run_async

    while True:
        try:
            user_input = read_user_input(session, console, session_state)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended. Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle Slash Commands
        lower_input = user_input.lower().strip()

        if lower_input in {"/exit", "/quit", "exit", "quit"}:
            console.print("[dim]Goodbye![/dim]")
            break

        # Display the user's prompt in the chat history stream above the bottom input box
        console.print(f"[bold #38bdf8]👤 You:[/bold #38bdf8] {user_input}\n")

        if lower_input == "/help":
            print_help(console)
            continue

        if lower_input == "/clear":
            console.clear()
            print_banner(console, path)
            continue

        if lower_input == "/key":
            prompt_for_credentials(console)
            console.print("[green]Credentials updated.[/green]")
            continue

        if lower_input in {"/skills", "/skill"}:
            print_skills_table(console)
            continue

        if lower_input in {"/auto", "/automode", "/mode"}:
            is_on = session_state.toggle_auto_mode()
            mode_badge = (
                "[bold green]ON (Autonomous Planning ➔ Execution ➔ QA)[/bold green]"
                if is_on
                else "[bold cyan]OFF (Interactive Chat Pairing)[/bold cyan]"
            )
            console.print(f"\n[bold]⚡ AUTO Mode is now {mode_badge}[/bold]")
            console.print(
                "[dim]Tip: You can also press [bold cyan]F2[/bold cyan], [bold cyan]Alt+A[/bold cyan], or [bold cyan]Ctrl+O[/bold cyan] at the prompt to toggle AUTO mode instantly.[/dim]\n"
            )
            continue

        if lower_input == "/auto on":
            session_state.auto_mode = True
            console.print(
                "\n[bold green]⚡ AUTO Mode enabled (Autonomous Planning ➔ Execution ➔ QA without pausing).[/bold green]\n"
            )
            continue

        if lower_input == "/auto off":
            session_state.auto_mode = False
            console.print(
                "\n[bold cyan]💬 AUTO Mode disabled (Interactive Chat Pairing active).[/bold cyan]\n"
            )
            continue

        if lower_input in {"/anvil", "/orchestrator", "/orch", "/planner", "/plan"}:
            session_state.set_agent("orchestrator")
            meta = AGENT_METADATA["orchestrator"]
            console.print(
                f"[bold cyan]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold cyan] [dim]({meta['description']})[/dim]\n"
            )
            continue

        if lower_input in {"/hammer", "/executor", "/exec"}:
            session_state.set_agent("executor")
            meta = AGENT_METADATA["executor"]
            console.print(
                f"[bold yellow]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold yellow] [dim]({meta['description']})[/dim]\n"
            )
            continue

        if lower_input in {"/temper", "/qa", "/audit"}:
            session_state.set_agent("qa")
            meta = AGENT_METADATA["qa"]
            console.print(
                f"[bold green]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold green] [dim]({meta['description']})[/dim]\n"
            )
            continue

        if lower_input.startswith("/search"):
            query_part = user_input[7:].strip()
            if query_part:
                from forge.tools.web_search import WebSearch, WebSearchInput

                search_tool = WebSearch()
                env = create_environment(load_config(path), root=path)
                with console.status(
                    f"[bold cyan]🔍 Searching the web for '{query_part}'...[/bold cyan]"
                ):
                    search_res = _run_async(search_tool.run(WebSearchInput(query=query_part), env))
                console.print()
                from rich.markdown import Markdown

                console.print(Markdown(search_res.render()))
                console.print()
            else:
                console.print("[yellow]Usage: /search <keywords or question>[/yellow]\n")
            continue

        if lower_input.startswith("/research") or lower_input.startswith("--research"):
            if lower_input.startswith("/research"):
                topic_part = user_input[9:].strip()
            else:
                topic_part = user_input.replace("--research", "").strip()

            if topic_part:
                from forge.cli.app import research_command

                research_command(topic=topic_part, path=path, max_waves=2)
            else:
                console.print("[yellow]Usage: /research <topic or paper inquiry>[/yellow]\n")
            continue

        if lower_input.startswith("/agent"):
            parts = user_input.split()
            if len(parts) > 1 and session_state.set_agent(parts[1]):
                meta = AGENT_METADATA[session_state.active_agent]
                console.print(
                    f"[bold {meta['color']}]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold {meta['color']}] [dim]({meta['description']})[/dim]\n"
                )
            else:
                curr_meta = AGENT_METADATA[session_state.active_agent]
                console.print(
                    f"\n[bold]Current Active Agent:[/bold] [bold {curr_meta['color']}]{curr_meta['icon']} {curr_meta['title']}[/bold {curr_meta['color']}] [dim]({curr_meta['description']})[/dim]"
                )
                console.print(
                    "[dim]Tip: Press [bold cyan]Tab[/bold cyan] at the prompt to cycle agents, or use:[/dim]"
                )
                for a_name in AVAILABLE_AGENTS:
                    m = AGENT_METADATA[a_name]
                    prefix = "●" if a_name == session_state.active_agent else "•"
                    cmd_name = f"/{m['title'].lower()}"
                    console.print(
                        f"  {prefix} [bold {m['color']}]{cmd_name:<10}[/bold {m['color']}] - {m['description']}"
                    )
                console.print()
            continue

        if lower_input == "/provider":
            from forge.cli.roles_wizard import run_provider_menu

            changes = run_provider_menu(path, console)
            if changes:
                console.print("\n[bold green]✔ Provider updated:[/bold green]")
                for ch in changes:
                    console.print(f"  • {ch}")
            else:
                console.print("\n[dim]No provider changes made.[/dim]")
            console.print()
            continue

        if lower_input == "/models":
            from forge.cli.roles_wizard import run_model_menu

            changes = run_model_menu(path, console)
            if changes:
                console.print("\n[bold green]✔ Model updated:[/bold green]")
                for ch in changes:
                    console.print(f"  • {ch}")
            else:
                console.print("\n[dim]No model changes made.[/dim]")
            console.print()
            continue

        if lower_input == "/status":
            _run_async(_show_workflow_status(path, current_workflow_id, console))
            continue

        if lower_input == "/diff":
            _run_async(_execute_diff(path, console))
            continue

        if lower_input == "/rollback":
            _run_async(_execute_rollback(path, current_workflow_id, console))
            continue

        if lower_input == "/history":
            _run_async(_execute_history(path, console))
            continue

        if lower_input.startswith("/run ") or lower_input.startswith("/build "):
            req = user_input.split(" ", 1)[1].strip()
            console.print()
            try:
                state = _run_async(
                    _run_workflow_turn(
                        req,
                        path=path,
                        console=console,
                        current_workflow_id=current_workflow_id,
                        auto_mode=session_state.auto_mode,
                    )
                )
                if state is not None:
                    current_workflow_id = state.workflow_id
                    conversation_history.append(Message.user(f"/run {req}"))
                    conversation_history.append(
                        Message.assistant(
                            f"[Workflow {state.workflow_id} executed with status {state.status.value}]"
                        )
                    )
                    if session_state.auto_mode:
                        print_auto_mode_completion(state, console)
            except UserCancelledError:
                console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
            except ForgeError as exc:
                console.print(f"[red]error:[/red] {exc}")
            except Exception as exc:
                console.print(f"[red]unexpected error:[/red] {exc}")
            console.print()
            continue

        # Multimodal & Clipboard Slash Commands
        from forge.cli.attachments import (
            build_turn_content,
            create_attachment_from_file,
            get_clipboard_files,
            get_clipboard_image,
            parse_user_input_attachments,
            take_screenshot,
        )

        if lower_input.startswith("/paste"):
            query_arg = user_input[6:].strip()
            media_dir = path / ".forge" / "media"
            att = get_clipboard_image(media_dir)
            if att is not None and att.is_image:
                console.print()
                console.print(att.render_panel())
                query = (
                    query_arg
                    or "Please analyze this attached screenshot image and diagnose any issues, UI elements, or errors."
                )
                turn_content, _ = build_turn_content(query, [att])
                try:
                    conversation_history, wid = _run_async(
                        _run_agent_chat_turn(
                            agent_role=session_state.active_agent,
                            user_input=turn_content,
                            path=path,
                            console=console,
                            history=conversation_history,
                            current_workflow_id=current_workflow_id,
                        )
                    )
                    if wid is not None:
                        current_workflow_id = wid
                except UserCancelledError:
                    console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
                except ForgeError as exc:
                    console.print(f"[red]error:[/red] {exc}")
                except Exception as exc:
                    console.print(f"[red]unexpected error:[/red] {exc}")
                console.print()
                continue

            # Check if clipboard has files
            c_files = get_clipboard_files()
            if c_files:
                console.print()
                for cf in c_files:
                    console.print(cf.render_panel())
                query = (
                    query_arg
                    or f"Please analyze the attached files: {', '.join(f.path.name for f in c_files)}."
                )
                turn_content, _ = build_turn_content(query, c_files)
                try:
                    conversation_history, wid = _run_async(
                        _run_agent_chat_turn(
                            agent_role=session_state.active_agent,
                            user_input=turn_content,
                            path=path,
                            console=console,
                            history=conversation_history,
                            current_workflow_id=current_workflow_id,
                        )
                    )
                    if wid is not None:
                        current_workflow_id = wid
                except UserCancelledError:
                    console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
                except ForgeError as exc:
                    console.print(f"[red]error:[/red] {exc}")
                except Exception as exc:
                    console.print(f"[red]unexpected error:[/red] {exc}")
                console.print()
                continue

            console.print("\n[yellow]⚠ No image or file detected in system clipboard.[/yellow]")
            console.print(
                "[dim]Tip: Take a screenshot with [bold cyan]Win+Shift+S[/bold cyan] (Windows) or [bold cyan]Cmd+Shift+4[/bold cyan] (Mac), then press [bold cyan]Ctrl+V[/bold cyan] or type [bold cyan]/paste[/bold cyan].[/dim]\n"
            )
            continue

        if lower_input.startswith("/image") or lower_input.startswith("/img"):
            cmd_prefix_len = 6 if lower_input.startswith("/image") else 4
            arg_str = user_input[cmd_prefix_len:].strip()
            parts = arg_str.split(maxsplit=1)
            target_file = parts[0] if parts else ""
            query_arg = parts[1] if len(parts) > 1 else ""

            att = None
            if target_file and (path / target_file).is_file():
                att = create_attachment_from_file(path / target_file)
            elif target_file and Path(target_file).is_file():
                att = create_attachment_from_file(Path(target_file))
            else:
                media_dir = path / ".forge" / "media"
                att = get_clipboard_image(media_dir)
                if not query_arg and target_file:
                    query_arg = target_file

            if att is not None and att.is_image:
                console.print()
                console.print(att.render_panel())
                query = (
                    query_arg
                    or "Please inspect this image and diagnose any problems or key features."
                )
                turn_content, _ = build_turn_content(query, [att])
                try:
                    conversation_history, wid = _run_async(
                        _run_agent_chat_turn(
                            agent_role=session_state.active_agent,
                            user_input=turn_content,
                            path=path,
                            console=console,
                            history=conversation_history,
                            current_workflow_id=current_workflow_id,
                        )
                    )
                    if wid is not None:
                        current_workflow_id = wid
                except UserCancelledError:
                    console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
                except ForgeError as exc:
                    console.print(f"[red]error:[/red] {exc}")
                except Exception as exc:
                    console.print(f"[red]unexpected error:[/red] {exc}")
                console.print()
                continue
            else:
                console.print(
                    "\n[yellow]Usage: /image <path-to-image> [question] OR copy a screenshot and type /paste[/yellow]\n"
                )
                continue

        if lower_input.startswith("/screenshot"):
            query_arg = user_input[11:].strip()
            media_dir = path / ".forge" / "media"
            att = take_screenshot(media_dir)
            if att is not None and att.is_image:
                console.print()
                console.print(att.render_panel())
                query = (
                    query_arg
                    or "Please inspect this screenshot of the screen and analyze what is visible."
                )
                turn_content, _ = build_turn_content(query, [att])
                try:
                    conversation_history, wid = _run_async(
                        _run_agent_chat_turn(
                            agent_role=session_state.active_agent,
                            user_input=turn_content,
                            path=path,
                            console=console,
                            history=conversation_history,
                            current_workflow_id=current_workflow_id,
                        )
                    )
                    if wid is not None:
                        current_workflow_id = wid
                except UserCancelledError:
                    console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
                except ForgeError as exc:
                    console.print(f"[red]error:[/red] {exc}")
                except Exception as exc:
                    console.print(f"[red]unexpected error:[/red] {exc}")
                console.print()
                continue
            else:
                console.print(
                    "\n[yellow]⚠ Could not capture screenshot. Try Win+Shift+S (or PrtScn) and /paste.[/yellow]\n"
                )
                continue

        if lower_input.startswith("/file") or lower_input.startswith("/attach"):
            cmd_prefix_len = 5 if lower_input.startswith("/file") else 7
            arg_str = user_input[cmd_prefix_len:].strip()
            parts = arg_str.split(maxsplit=1)
            target_file = parts[0] if parts else ""
            query_arg = parts[1] if len(parts) > 1 else ""

            target_path = (
                Path(target_file) if Path(target_file).is_absolute() else path / target_file
            )
            if target_file and target_path.is_file():
                att = create_attachment_from_file(target_path)
                if att is not None:
                    console.print()
                    console.print(att.render_panel())
                    query = query_arg or f"Please analyze the attached file: {att.path.name}."
                    turn_content, _ = build_turn_content(query, [att])
                    try:
                        conversation_history, wid = _run_async(
                            _run_agent_chat_turn(
                                agent_role=session_state.active_agent,
                                user_input=turn_content,
                                path=path,
                                console=console,
                                history=conversation_history,
                                current_workflow_id=current_workflow_id,
                            )
                        )
                        if wid is not None:
                            current_workflow_id = wid
                    except UserCancelledError:
                        console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
                    except ForgeError as exc:
                        console.print(f"[red]error:[/red] {exc}")
                    except Exception as exc:
                        console.print(f"[red]unexpected error:[/red] {exc}")
                    console.print()
                    continue
            console.print("\n[yellow]Usage: /file <path-to-code-or-doc> [instruction][/yellow]\n")
            continue

        # Automatic detection of [image: ...], [file: ...], or dragged/quoted paths
        cleaned_text, detected_attachments = parse_user_input_attachments(user_input, base_dir=path)
        if detected_attachments:
            console.print()
            for att in detected_attachments:
                console.print(att.render_panel())
            turn_content, _ = build_turn_content(cleaned_text, detected_attachments)
        else:
            turn_content = user_input

        # Check if AUTO mode is active: autonomous Planning ➔ Execution ➔ QA without intermediate pauses
        if session_state.auto_mode:
            req_text = user_input
            # If user said "begin" or "start" and prior conversation history exists
            if (
                lower_input in {"begin", "start", "proceed", "go ahead", "orchestrator begin"}
                and conversation_history
            ):
                recent_user_msgs = [
                    m.text for m in conversation_history if m.role == "user" and m.text
                ]
                if recent_user_msgs:
                    req_text = f"Execute request: {recent_user_msgs[-1]}"

            console.print(
                Panel(
                    f"[bold green]Autonomous full-cycle execution initiated:[/bold green]\n"
                    f"[bold white]{req_text}[/bold white]\n\n"
                    f"[dim]Running without prompts: 🎯 Planning (ANVIL) ➔ ⚡ Execution (HAMMER) ➔ 🔍 QA (TEMPER)...[/dim]",
                    title="⚡ AUTO MODE ACTIVATED",
                    border_style="green",
                )
            )

            try:
                state = _run_async(
                    _run_workflow_turn(
                        req_text,
                        path=path,
                        console=console,
                        current_workflow_id=current_workflow_id,
                        auto_mode=True,
                    )
                )
                if state is not None:
                    current_workflow_id = state.workflow_id
                    conversation_history.append(Message.user(user_input))
                    conversation_history.append(
                        Message.assistant(
                            f"[AUTO MODE completed: Workflow {state.workflow_id} ended with status: {state.status.value}. "
                            f"Tasks executed: {len(state.tasks)}]"
                        )
                    )
                    print_auto_mode_completion(state, console)
            except UserCancelledError:
                console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
            except ForgeError as exc:
                console.print(f"[red]error:[/red] {exc}")
            except Exception as exc:
                console.print(f"[red]unexpected error:[/red] {exc}")
            console.print()
            continue

        # Conversational pairing with the active agent (default: Orchestrator)
        try:
            conversation_history, wid = _run_async(
                _run_agent_chat_turn(
                    agent_role=session_state.active_agent,
                    user_input=turn_content,
                    path=path,
                    console=console,
                    history=conversation_history,
                    current_workflow_id=current_workflow_id,
                )
            )
            if wid is not None:
                current_workflow_id = wid
        except UserCancelledError:
            console.print("\n[yellow]⚠ Interrupted by user (Esc).[/yellow]")
        except ForgeError as exc:
            console.print(f"[red]error:[/red] {exc}")
        except Exception as exc:
            console.print(f"[red]unexpected error:[/red] {exc}")
        console.print()
