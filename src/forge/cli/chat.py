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
from forge.cli.credentials import (
    PROVIDER_METADATA,
    ensure_credentials_configured,
    load_stored_credentials,
    prompt_for_credentials,
)
from forge.cli.interactive import CliApprover
from forge.cli.render import (
    CliEventPrinter,
    print_workflow_list,
    print_workflow_status,
)
from forge.cli.workflow import build_workflow_harness
from forge.config import init_project, load_config
from forge.core.contracts.state import WorkflowState
from forge.core.errors import ForgeError
from forge.events.bus import EventBus
from forge.events.types import Event, EventType
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

HELP_COMMANDS = [
    ("/help", "Show this list of available slash commands"),
    ("/roles", "Inspect or reconfigure agent role assignments (Planner, Executor, QA)"),
    ("/status", "Show status, task graph, and QA verdict of current/latest workflow"),
    ("/diff", "Show git diff of modifications made by Forge"),
    ("/rollback", "Revert working tree to the latest checkpoint"),
    ("/history", "List past workflows and event timeline"),
    ("/models", "Show active model configuration and provider"),
    ("/key", "Reconfigure or switch your API key / provider"),
    ("/clear", "Clear terminal screen and redraw status banner"),
    ("/exit", "Exit the interactive session (or /quit)"),
]


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
            for r_name in ("planner", "executor", "qa"):
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
        "\n[dim]Type your coding prompt to begin, or [cyan]/help[/cyan] for commands. "
        "Use [cyan]/exit[/cyan] to quit.[/dim]"
    )
    console.print(
        Panel(
            Group(grid, info_text),
            title="Interactive Agent Session",
            border_style="cyan",
        )
    )


def _create_prompt_session() -> Any:
    """Create a prompt_toolkit PromptSession with history and completions, or None."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory

        history_file = Path.home() / ".forge" / "chat_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        commands = [cmd for cmd, _ in HELP_COMMANDS] + ["/quit"]
        completer = WordCompleter(commands, ignore_case=True, match_middle=False)

        return PromptSession(
            history=FileHistory(str(history_file)),
            completer=completer,
        )
    except Exception:
        return None


def read_user_input(session: Any, console: Console) -> str:
    """Read a line of input from the user with fallback."""
    prompt_str = "forge > "
    if session is not None:
        try:
            return str(session.prompt(prompt_str)).strip()
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            pass

    # Fallback to standard input
    try:
        from rich.prompt import Prompt

        return Prompt.ask(f"[bold cyan]{prompt_str}[/bold cyan]", console=console).strip()
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
) -> WorkflowState | None:
    """Run an autonomous workflow turn for the user's prompt."""
    config = load_config(path)
    event_bus = EventBus()
    event_bus.subscribe(CliEventPrinter(console))

    approver = CliApprover(console)
    harness = await build_workflow_harness(config, approver=approver, event_bus=event_bus)

    try:
        # Check if we should resume the current active workflow with human feedback
        if current_workflow_id:
            try:
                existing = await harness.store.load_workflow(current_workflow_id)
                if not existing.is_terminal:
                    console.print(f"[dim]Continuing workflow {current_workflow_id}...[/dim]")
                    final_state = await harness.orchestrator.resume(existing, feedback=request)
                    console.print()
                    print_workflow_status(final_state, console)
                    return final_state
            except Exception:
                pass

        # Otherwise, start a fresh workflow for this instruction
        state = WorkflowState(user_request=request)
        final_state = await harness.orchestrator.run(state)
        console.print()
        print_workflow_status(final_state, console)
        return final_state
    finally:
        await harness.aclose()


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

    # 4. Prepare input session
    session = _create_prompt_session()
    current_workflow_id: str | None = None

    from forge.cli.app import _run_async

    while True:
        try:
            user_input = read_user_input(session, console)
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

        if lower_input in {"/roles", "/role"}:
            from rich.prompt import Confirm

            from forge.cli.roles_wizard import print_roles_table, run_roles_wizard

            config = load_config(path)
            print_roles_table(config, console)
            console.print()
            reconf = Confirm.ask(
                "Would you like to reconfigure agent models or roles?",
                default=False,
                console=console,
            )
            if reconf:
                run_roles_wizard(path, console)
            continue

        if lower_input == "/models":
            config = load_config(path)
            console.print(f"[bold]Configured Models for {config.project.name}:[/bold]")
            if config.models:
                for name, prof in config.models.items():
                    console.print(f"  • [cyan]{name}[/cyan]: {prof.provider}/{prof.model}")
            else:
                console.print(f"  • Default: {_get_active_model_info()}")
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

        # Regular engineering request or conversational prompt
        console.print()
        try:
            state = _run_async(
                _run_workflow_turn(
                    user_input,
                    path=path,
                    console=console,
                    current_workflow_id=current_workflow_id,
                )
            )
            if state is not None:
                current_workflow_id = state.workflow_id
        except ForgeError as exc:
            console.print(f"[red]error:[/red] {exc}")
        except Exception as exc:
            console.print(f"[red]unexpected error:[/red] {exc}")
        console.print()
