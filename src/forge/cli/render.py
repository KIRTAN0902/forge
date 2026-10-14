"""Terminal rendering helpers for Forge CLI using Rich (spec §18, §19)."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forge.core.contracts.plan import TaskStatus
from forge.core.contracts.qa import QAStatus
from forge.core.contracts.state import WorkflowState, WorkflowStatus
from forge.events.types import Event, EventType
from forge.storage.base import WorkflowSummary


def status_style(status: WorkflowStatus) -> str:
    match status:
        case WorkflowStatus.COMPLETED:
            return "green"
        case WorkflowStatus.RUNNING:
            return "blue"
        case WorkflowStatus.AWAITING_APPROVAL:
            return "yellow"
        case WorkflowStatus.BLOCKED:
            return "red"
        case WorkflowStatus.FAILED:
            return "bold red"
        case _:
            return "white"


def task_style(status: TaskStatus) -> str:
    match status:
        case TaskStatus.DONE:
            return "green"
        case TaskStatus.IN_PROGRESS:
            return "cyan"
        case TaskStatus.FAILED:
            return "red"
        case TaskStatus.SKIPPED:
            return "dim"
        case _:
            return "white"


class CliEventPrinter:
    """Subscribes to EventBus and renders live workflow events to the console."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console(legacy_windows=False)

    def __call__(self, event: Event) -> None:
        p = event.payload
        match event.type:
            case EventType.WORKFLOW_STARTED:
                req = p.get("user_request", "")
                self.console.print(f"[bold blue]▶ Workflow Started:[/bold blue] {req}")
            case EventType.WORKFLOW_RESUMED:
                self.console.print(
                    f"[bold blue]▶ Workflow Resumed[/bold blue] (Iter {p.get('iteration', 0)})"
                )
            case EventType.DECISION_MADE:
                action = p.get("action", "")
                rationale = p.get("rationale", "")
                iter_num = p.get("iteration", "")
                self.console.print(
                    f"[dim]Step {iter_num}:[/dim] [bold cyan]{action}[/bold cyan] — {rationale}"
                )
            case EventType.PLAN_CREATED:
                count = p.get("tasks_count", "")
                self.console.print(f"  [green]✓ Plan Created:[/green] {count} tasks planned")
            case EventType.TASK_STARTED:
                tid = p.get("task_id", "")
                self.console.print(f"  [blue]⚙ Task Started:[/blue] [bold]{tid}[/bold]")
            case EventType.TASK_COMPLETED:
                tid = p.get("task_id", "")
                self.console.print(f"  [green]✓ Task Completed:[/green] [bold]{tid}[/bold]")
            case EventType.TASK_FAILED:
                tid = p.get("task_id", "")
                self.console.print(f"  [red]✗ Task Failed:[/red] [bold]{tid}[/bold]")
            case EventType.QA_STARTED:
                scope = p.get("scope", "full")
                self.console.print(f"  [magenta]🔍 QA Audit Started[/magenta] ({scope})")
            case EventType.QA_COMPLETED:
                status = p.get("status", "")
                st = "green" if status == "PASS" else "red"
                self.console.print(
                    f"  [magenta]🔍 QA Audit Completed:[/magenta] [{st}]{status}[/{st}]"
                )
            case EventType.REWORK_REQUESTED:
                findings = p.get("findings", [])
                self.console.print(
                    f"  [yellow]⚠ Rework Requested:[/yellow] blocking findings: {findings}"
                )
            case EventType.CHECKPOINT_CREATED:
                label = p.get("label", "")
                self.console.print(f"  [dim]💾 Checkpoint saved: {label}[/dim]")
            case EventType.ROLLBACK_PERFORMED:
                cid = p.get("checkpoint_id", "")
                self.console.print(f"  [yellow]↺ Restored checkpoint {cid}[/yellow]")
            case EventType.APPROVAL_REQUESTED:
                rationale = p.get("rationale", "")
                self.console.print(
                    f"[bold yellow]⏸ Human Review Required:[/bold yellow] {rationale}"
                )
            case EventType.APPROVAL_RESOLVED:
                self.console.print("[bold green]▶ Human Review Approved[/bold green]")
            case EventType.WORKFLOW_COMPLETED:
                iters = p.get("iterations", "")
                self.console.print(
                    f"[bold green]✔ Workflow Completed Successfully![/bold green] "
                    f"({iters} iterations)"
                )
            case EventType.WORKFLOW_BLOCKED:
                self.console.print("[bold red]✖ Workflow Blocked[/bold red]")
            case EventType.WORKFLOW_FAILED:
                self.console.print("[bold red]✖ Workflow Failed[/bold red]")


def print_workflow_status(state: WorkflowState, console: Console) -> None:
    """Print detailed status of a workflow."""
    style = status_style(state.status)
    header = (
        f"[bold]Workflow ID:[/bold] {state.workflow_id}\n"
        f"[bold]Status:[/bold] [{style}]{state.status.value.upper()}[/{style}]\n"
        f"[bold]Iterations:[/bold] {state.iteration}\n"
        f"[bold]Request:[/bold] {state.user_request}"
    )
    console.print(Panel(header, title="Workflow Overview", border_style=style))

    if state.requirements:
        console.print(
            f"[bold]Requirements:[/bold] {state.requirements.summary} "
            f"({len(state.requirements.acceptance_criteria)} criteria)"
        )

    if state.plan and state.plan.tasks:
        table = Table(title="Task Graph", expand=True)
        table.add_column("Task ID", style="bold", width=12)
        table.add_column("Status", width=12)
        table.add_column("Attempts", width=10)
        table.add_column("Description")
        for t in state.plan.tasks:
            t_style = task_style(t.status)
            table.add_row(
                t.id,
                f"[{t_style}]{t.status.value.upper()}[/{t_style}]",
                str(t.attempts),
                t.description,
            )
        console.print(table)

    if state.latest_qa:
        qa = state.latest_qa
        qa_st = "green" if qa.status == QAStatus.PASS else "red"
        findings_info = (
            f"\n[red]Blocking findings ({len(qa.blocking_findings)}):[/red]\n"
            + "\n".join(f"  • [{f.severity}] {f.description}" for f in qa.blocking_findings)
            if qa.blocking_findings
            else ""
        )
        console.print(
            Panel(
                f"[bold]Latest QA Verdict:[/bold] [{qa_st}]{qa.status.value}[/{qa_st}]\n"
                f"Summary: {qa.summary}{findings_info}",
                title="QA Verification",
                border_style=qa_st,
            )
        )

    if state.checkpoints:
        latest_ckpt = state.checkpoints[-1]
        console.print(
            f"[dim]Checkpoints: {len(state.checkpoints)} taken "
            f"(latest: {latest_ckpt.label} [{latest_ckpt.id[:8]}])[/dim]"
        )


def print_workflow_list(workflows: list[WorkflowSummary], console: Console) -> None:
    """Print summary list of recent workflows."""
    if not workflows:
        console.print("[dim]No workflows found.[/dim]")
        return

    table = Table(title="Workflows")
    table.add_column("ID", style="bold", no_wrap=True, min_width=26)
    table.add_column("Status")
    table.add_column("Iterations", justify="right")
    table.add_column("Updated At")
    table.add_column("Request", overflow="fold")

    for w in workflows:
        st_style = status_style(w.status)
        table.add_row(
            w.workflow_id,
            f"[{st_style}]{w.status.value.upper()}[/{st_style}]",
            str(w.iteration),
            w.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            w.user_request[:60] + ("..." if len(w.user_request) > 60 else ""),
        )
    console.print(table)


def print_event_timeline(events: list[Event], console: Console) -> None:
    """Print chronological timeline of workflow events."""
    if not events:
        console.print("[dim]No events recorded for this workflow.[/dim]")
        return

    table = Table(title="Event Timeline", expand=True)
    table.add_column("Timestamp", width=12)
    table.add_column("Type", style="bold", width=22)
    table.add_column("Source", width=12)
    table.add_column("Details", overflow="fold")

    for e in events:
        time_str = e.at.strftime("%H:%M:%S")
        details = ", ".join(f"{k}={v}" for k, v in e.payload.items() if v is not None)
        table.add_row(time_str, str(e.type), e.source, details)
    console.print(table)
