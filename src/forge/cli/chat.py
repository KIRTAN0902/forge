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
from forge.cli.interactive import CliApprover
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
from forge.models.types import Message
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
    """Tracks the active chat agent and handles cycling and switching."""

    def __init__(self, default_agent: str = "orchestrator") -> None:
        self.active_agent: str = default_agent

    def cycle_agent(self) -> str:
        idx = AVAILABLE_AGENTS.index(self.active_agent) if self.active_agent in AVAILABLE_AGENTS else 0
        self.active_agent = AVAILABLE_AGENTS[(idx + 1) % len(AVAILABLE_AGENTS)]
        return self.active_agent

    def cycle_agent_reverse(self) -> str:
        idx = AVAILABLE_AGENTS.index(self.active_agent) if self.active_agent in AVAILABLE_AGENTS else 0
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
    ("/anvil", "Switch active agent to 🎯 ANVIL (Orchestrator: architecture & planning)"),
    ("/hammer", "Switch active agent to ⚡ HAMMER (Executor: coding & tool execution)"),
    ("/temper", "Switch active agent to 🔍 TEMPER (QA: adversarial testing & verification)"),
    ("/search", "Perform a live web search for documentation, APIs, and error solutions"),
    ("/models", "Interactive popup to select and configure agent models"),
    ("/provider", "Connect and configure model providers, API keys, and custom Base URIs/URLs"),
    ("/orchestrator", "Alias for /anvil"),
    ("/executor", "Alias for /hammer"),
    ("/qa", "Alias for /temper"),
    ("/agent", "Show or switch the active chat agent (or press Tab at the prompt)"),
    ("/skills", "List all 24 pre-programmed specialist skills and autonomous triggers"),
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
    console.print("[dim]These skills are autonomously evaluated and activated by ANVIL on matching signals.[/dim]\n")


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
        "Type your prompt to begin, or [cyan]/help[/cyan] for commands. Use [cyan]/search <query>[/cyan] for web search, [cyan]/exit[/cyan] to quit.[/dim]"
    )
    console.print(
        Panel(
            Group(grid, info_text),
            title="Interactive Agent Session",
            border_style="cyan",
        )
    )


def _get_prompt_tokens(state: ChatSessionState) -> list[tuple[str, str]]:
    """Generate dynamic prompt_toolkit formatted text tokens reflecting the active agent."""
    agent = state.active_agent
    meta = AGENT_METADATA.get(agent, AGENT_METADATA["orchestrator"])
    return [
        ("class:prompt_brand", "forge"),
        ("class:prompt_bracket", " ["),
        (f"class:prompt_agent_{agent}", f"{meta['icon']} {meta['title']}"),
        ("class:prompt_bracket", "] > "),
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

        prompt_style = Style.from_dict({
            "prompt_brand": "#00e5ff bold",
            "prompt_bracket": "#64748b",
            "prompt_agent_orchestrator": "#00e5ff bold",
            "prompt_agent_planner": "#d946ef bold",
            "prompt_agent_executor": "#eab308 bold",
            "prompt_agent_qa": "#10b981 bold",
        })

        return PromptSession(
            history=FileHistory(str(history_file)),
            completer=completer,
            key_bindings=kb,
            style=prompt_style,
        )
    except Exception:
        return None


def read_user_input(
    session: Any, console: Console, state: ChatSessionState | None = None
) -> str:
    """Read a line of input from the user with dynamic agent prompt badge."""
    current_state = state or ChatSessionState()
    meta = AGENT_METADATA.get(current_state.active_agent, AGENT_METADATA["orchestrator"])

    if session is not None:
        try:
            return str(session.prompt(lambda: _get_prompt_tokens(current_state))).strip()
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
                    final_state = await run_with_esc_cancellation(
                        harness.orchestrator.resume(existing, feedback=request),
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
            harness.orchestrator.run(state),
            status_message="[bold green]Workflow executing...[/bold green]",
            console=console,
        )
        console.print()
        print_workflow_status(final_state, console)
        return final_state
    finally:
        await harness.aclose()


async def _run_agent_chat_turn(
    agent_role: str,
    user_input: str,
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
        signal = SkillSignal(
            user_request=user_input,
            error_text=user_input,
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
        console.print(f"[{meta['status_color']}]{meta['icon']} {meta['title']}:[/{meta['status_color']}]")
        from rich.markdown import Markdown

        console.print(Markdown(turn.response_text))
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

        if lower_input in {"/anvil", "/orchestrator", "/orch", "/planner", "/plan"}:
            session_state.set_agent("orchestrator")
            meta = AGENT_METADATA["orchestrator"]
            console.print(f"[bold cyan]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold cyan] [dim]({meta['description']})[/dim]\n")
            continue

        if lower_input in {"/hammer", "/executor", "/exec"}:
            session_state.set_agent("executor")
            meta = AGENT_METADATA["executor"]
            console.print(f"[bold yellow]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold yellow] [dim]({meta['description']})[/dim]\n")
            continue

        if lower_input in {"/temper", "/qa", "/audit"}:
            session_state.set_agent("qa")
            meta = AGENT_METADATA["qa"]
            console.print(f"[bold green]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold green] [dim]({meta['description']})[/dim]\n")
            continue

        if lower_input.startswith("/search"):
            query_part = user_input[7:].strip()
            if query_part:
                from forge.tools.web_search import WebSearch, WebSearchInput
                search_tool = WebSearch()
                env = create_environment(load_config(path), root=path)
                with console.status(f"[bold cyan]🔍 Searching the web for '{query_part}'...[/bold cyan]"):
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
                console.print(f"[bold {meta['color']}]✔ Switched active agent to: {meta['icon']} {meta['title']}[/bold {meta['color']}] [dim]({meta['description']})[/dim]\n")
            else:
                curr_meta = AGENT_METADATA[session_state.active_agent]
                console.print(f"\n[bold]Current Active Agent:[/bold] [bold {curr_meta['color']}]{curr_meta['icon']} {curr_meta['title']}[/bold {curr_meta['color']}] [dim]({curr_meta['description']})[/dim]")
                console.print("[dim]Tip: Press [bold cyan]Tab[/bold cyan] at the prompt to cycle agents, or use:[/dim]")
                for a_name in AVAILABLE_AGENTS:
                    m = AGENT_METADATA[a_name]
                    prefix = "●" if a_name == session_state.active_agent else "•"
                    cmd_name = f"/{m['title'].lower()}"
                    console.print(f"  {prefix} [bold {m['color']}]{cmd_name:<10}[/bold {m['color']}] - {m['description']}")
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
                    user_input=user_input,
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
