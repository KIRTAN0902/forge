"""Interactive terminal approver for Forge CLI (spec §17)."""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel

from forge.policies.types import PolicyDecision, ProposedAction, RiskLevel


def _risk_style(risk: RiskLevel) -> str:
    match risk:
        case RiskLevel.LOW:
            return "green"
        case RiskLevel.MEDIUM:
            return "yellow"
        case RiskLevel.HIGH:
            return "red"
        case RiskLevel.CRITICAL:
            return "bold red"
        case _:
            return "white"


class CliApprover:
    """Prompts the user on the terminal when a policy returns REQUIRE_APPROVAL."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    async def request_approval(self, action: ProposedAction, decision: PolicyDecision) -> bool:
        if not sys.stdin.isatty():
            return False

        style = _risk_style(action.risk)
        args_repr = str(action.arguments)
        if len(args_repr) > 120:
            args_repr = args_repr[:117] + "..."

        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Action:[/bold] {action.name}\n"
                f"[bold]Risk:[/bold] [{style}]{action.risk.value.upper()}[/{style}]\n"
                f"[bold]Reason:[/bold] {decision.reason}\n"
                f"[bold]Arguments:[/bold] {args_repr}",
                title="[bold yellow]Human Approval Gate[/bold yellow]",
                border_style="yellow",
            )
        )

        def _prompt() -> bool:
            try:
                response = input("Approve this action? [y/N]: ").strip().lower()
                return response in {"y", "yes"}
            except (EOFError, KeyboardInterrupt):
                return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _prompt)
