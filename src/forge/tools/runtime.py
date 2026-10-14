"""The tool runtime: the single gate between a model's tool calls and the workspace.

For every call: look up the tool, validate arguments, let the tool assess the concrete
risk, ask the deterministic policy engine, ask a human when required, run with a
timeout, redact and truncate the output, and record audit events.

Failures never raise into the agent loop. They come back as ``ToolResult(ok=False)``
with an error the model can read and correct, because a model that sees why its call
failed can recover; a crashed loop cannot.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from forge.core.errors import ToolError
from forge.core.logging import get_logger
from forge.events.bus import EventBus
from forge.events.types import EventType
from forge.models.types import Message, ToolCall, ToolDefinition
from forge.policies.approval import Approver
from forge.policies.engine import PolicyEngine
from forge.policies.types import PolicyDecision, ProposedAction, Verdict
from forge.tools.base import Tool, ToolResult
from forge.tools.environment import ToolEnvironment

log = get_logger("tools.runtime")

AnyTool = Tool[Any, Any]


class ToolRuntime:
    def __init__(
        self,
        tools: Iterable[AnyTool],
        env: ToolEnvironment,
        policy: PolicyEngine,
        *,
        approver: Approver | None = None,
        events: EventBus | None = None,
        workflow_id: str | None = None,
        max_output_chars: int = 30_000,
    ) -> None:
        self.tools: dict[str, AnyTool] = {}
        for tool in tools:
            if tool.name in self.tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            self.tools[tool.name] = tool
        self.env = env
        self.policy = policy
        self.approver = approver
        self.events = events
        self.workflow_id = workflow_id
        self.max_output_chars = max_output_chars
        self.audit_log: list[ToolResult] = []

    # -- discovery ---------------------------------------------------------------------

    def definitions(self, names: Iterable[str] | None = None) -> list[ToolDefinition]:
        selected = self._select(names)
        return [tool.definition(self.env) for tool in selected]

    def _select(self, names: Iterable[str] | None) -> list[AnyTool]:
        if names is None:
            return list(self.tools.values())
        missing = [n for n in names if n not in self.tools]
        if missing:
            raise KeyError(f"unknown tools: {missing}")
        return [self.tools[n] for n in names]

    # -- execution ---------------------------------------------------------------------

    async def execute(
        self, call: ToolCall, *, actor: str = "agent", allowed: Iterable[str] | None = None
    ) -> ToolResult:
        """Run one call. ``allowed`` restricts which tools this actor may use."""
        started = time.perf_counter()
        allowed_set = set(allowed) if allowed is not None else None

        tool = self.tools.get(call.name)
        if tool is None or (allowed_set is not None and call.name not in allowed_set):
            available = sorted(allowed_set if allowed_set is not None else self.tools)
            return self._fail(
                call,
                f"unknown tool {call.name!r}. Available tools: {', '.join(available)}",
                started,
            )
        if call.parse_error:
            return self._fail(
                call,
                f"could not parse the arguments ({call.parse_error}). "
                "Send the arguments as a single JSON object.",
                started,
            )
        try:
            args = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return self._fail(call, "invalid arguments: " + _validation_summary(exc), started)

        try:
            assessment = tool.assess(args, self.env)
        except ToolError as exc:
            return self._fail(call, str(exc), started)

        action = ProposedAction(
            kind="tool",
            name=tool.name,
            actor=actor,
            risk=assessment.risk,
            permissions=sorted(assessment.permissions),
            paths=assessment.paths,
            command=assessment.command,
            arguments=_summarize_args(call.arguments),
            description=assessment.description,
        )
        decision = self.policy.evaluate(action)
        await self._emit(
            EventType.POLICY_EVALUATED,
            actor,
            tool=tool.name,
            call_id=call.id,
            risk=str(action.risk),
            verdict=str(decision.verdict),
            policy=decision.policy,
            reason=decision.reason,
        )
        if decision.verdict == Verdict.DENY:
            return self._fail(
                call, f"denied by policy: {decision.reason}", started, action, decision
            )
        if decision.verdict == Verdict.REQUIRE_APPROVAL:
            approved = await self._request_approval(action, decision, actor)
            if not approved:
                return self._fail(
                    call,
                    f"not approved by a human ({decision.reason}). Do not retry the same "
                    "action; choose a safer approach or explain why it is needed.",
                    started,
                    action,
                    decision,
                )

        await self._emit(
            EventType.TOOL_CALLED,
            actor,
            tool=tool.name,
            call_id=call.id,
            arguments=action.arguments,
            risk=str(action.risk),
        )
        timeout = tool.timeout_for(args, self.env)
        try:
            output = await asyncio.wait_for(tool.run(args, self.env), timeout=timeout)
        except TimeoutError:
            return await self._finish_fail(
                call, f"timed out after {timeout:.0f}s", started, action, decision, actor
            )
        except ToolError as exc:
            return await self._finish_fail(call, str(exc), started, action, decision, actor)
        except Exception as exc:  # a tool bug must not take down the agent loop
            log.exception("tool %s crashed", tool.name)
            return await self._finish_fail(
                call, f"internal tool error: {type(exc).__name__}: {exc}", started, action,
                decision, actor,
            )  # fmt: skip

        content, truncated = self._prepare(output.render())
        result = ToolResult(
            call_id=call.id,
            tool=tool.name,
            ok=True,
            content=content,
            data=output.model_dump(mode="json"),
            truncated=truncated,
            duration_s=time.perf_counter() - started,
            risk=action.risk,
            verdict=str(decision.verdict),
            changed_paths=output.changed_paths(),
        )
        self.audit_log.append(result)
        await self._emit_completion(result, actor)
        return result

    async def execute_many(
        self,
        calls: Sequence[ToolCall],
        *,
        actor: str = "agent",
        allowed: Iterable[str] | None = None,
    ) -> list[ToolResult]:
        """Run several calls, preserving order. Consecutive read-only calls run
        concurrently; anything that mutates runs alone, in order."""
        allowed_list = list(allowed) if allowed is not None else None
        results: list[ToolResult | None] = [None] * len(calls)
        batch: list[int] = []

        async def flush() -> None:
            if not batch:
                return
            outcomes = await asyncio.gather(
                *(self.execute(calls[i], actor=actor, allowed=allowed_list) for i in batch)
            )
            for index, outcome in zip(batch, outcomes, strict=True):
                results[index] = outcome
            batch.clear()

        for index, call in enumerate(calls):
            tool = self.tools.get(call.name)
            if tool is not None and tool.read_only:
                batch.append(index)
                continue
            await flush()
            results[index] = await self.execute(call, actor=actor, allowed=allowed_list)
        await flush()
        return [r for r in results if r is not None]

    @staticmethod
    def to_messages(calls: Sequence[ToolCall], results: Sequence[ToolResult]) -> list[Message]:
        return [Message.tool_result(c, r.content) for c, r in zip(calls, results, strict=True)]

    async def aclose(self) -> None:
        await self.env.aclose()

    # -- internals ---------------------------------------------------------------------

    async def _request_approval(
        self, action: ProposedAction, decision: PolicyDecision, actor: str
    ) -> bool:
        await self._emit(
            EventType.APPROVAL_REQUESTED,
            actor,
            tool=action.name,
            risk=str(action.risk),
            reason=decision.reason,
            command=action.command,
            paths=action.paths,
        )
        approved = False
        if self.approver is not None:
            approved = await self.approver.request_approval(action, decision)
        await self._emit(
            EventType.APPROVAL_RESOLVED,
            actor,
            tool=action.name,
            approved=approved,
            approver="none" if self.approver is None else type(self.approver).__name__,
        )
        return approved

    def _prepare(self, text: str) -> tuple[str, bool]:
        text = self.env.shell.redact(text)
        if len(text) <= self.max_output_chars:
            return text, False
        half = self.max_output_chars // 2
        omitted = len(text) - 2 * half
        return (
            f"{text[:half]}\n\n... [{omitted} characters omitted; narrow the request to see "
            f"them] ...\n\n{text[-half:]}",
            True,
        )

    def _fail(
        self,
        call: ToolCall,
        error: str,
        started: float,
        action: ProposedAction | None = None,
        decision: PolicyDecision | None = None,
    ) -> ToolResult:
        error = self.env.shell.redact(error)
        result = ToolResult(
            call_id=call.id,
            tool=call.name,
            ok=False,
            content=f"ERROR: {error}",
            error=error,
            duration_s=time.perf_counter() - started,
            risk=action.risk if action else None,
            verdict=str(decision.verdict) if decision else None,
        )
        self.audit_log.append(result)
        return result

    async def _finish_fail(
        self,
        call: ToolCall,
        error: str,
        started: float,
        action: ProposedAction,
        decision: PolicyDecision,
        actor: str,
    ) -> ToolResult:
        result = self._fail(call, error, started, action, decision)
        await self._emit_completion(result, actor)
        return result

    async def _emit_completion(self, result: ToolResult, actor: str) -> None:
        await self._emit(
            EventType.TOOL_COMPLETED,
            actor,
            tool=result.tool,
            call_id=result.call_id,
            ok=result.ok,
            error=result.error,
            duration_s=round(result.duration_s, 4),
            truncated=result.truncated,
            risk=str(result.risk) if result.risk else None,
            verdict=result.verdict,
            changed_paths=result.changed_paths,
        )
        for path in result.changed_paths:
            await self._emit(EventType.FILE_MODIFIED, actor, path=path, tool=result.tool)

    async def _emit(self, type: EventType, actor: str, **payload: Any) -> None:
        if self.events is not None:
            await self.events.emit(type, workflow_id=self.workflow_id, source=actor, **payload)


def _validation_summary(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:8]:
        location = ".".join(str(p) for p in err["loc"]) or "(arguments)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def _summarize_args(arguments: dict[str, Any], limit: int = 400) -> dict[str, Any]:
    """Arguments for audit records, with large values (file contents) shortened."""
    summary: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > limit:
            summary[key] = f"{value[:limit]}... [{len(value)} chars]"
        else:
            summary[key] = value
    return summary
