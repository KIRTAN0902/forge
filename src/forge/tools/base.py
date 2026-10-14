"""The tool contract.

Every tool declares (spec §14): an input schema, an output schema, a risk level, the
permissions it needs, a timeout, and how it is audited. Tools may refine their risk per
call in :meth:`Tool.assess` (``run_command`` classifies the actual command; ``write_file``
reports the path it touches) — the policy gate then decides with that information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from forge.models.types import ToolDefinition
from forge.policies.types import Permission, RiskLevel

if TYPE_CHECKING:
    from forge.tools.environment import ToolEnvironment


class ToolInput(BaseModel):
    """Base for tool arguments. Unknown arguments are rejected so typos surface to the
    model as a correctable error instead of being silently ignored."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class ToolOutput(BaseModel):
    """Base for tool results. ``render`` is what the model reads."""

    model_config = ConfigDict(extra="forbid")

    def render(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)

    def changed_paths(self) -> list[str]:
        """Workspace files this call modified (drives FileModified events)."""
        return []


InputT = TypeVar("InputT", bound=ToolInput)
OutputT = TypeVar("OutputT", bound=ToolOutput)


@dataclass
class ToolAssessment:
    risk: RiskLevel
    permissions: frozenset[Permission]
    paths: list[str] = field(default_factory=list)
    command: str | None = None
    description: str = ""


class Tool(ABC, Generic[InputT, OutputT]):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: type[InputT]
    output_model: type[OutputT]
    risk: ClassVar[RiskLevel] = RiskLevel.LOW
    permissions: ClassVar[frozenset[Permission]] = frozenset({Permission.READ})
    timeout_s: ClassVar[float] = 60.0

    @property
    def read_only(self) -> bool:
        return self.permissions <= {Permission.READ}

    def assess(self, args: InputT, env: ToolEnvironment) -> ToolAssessment:
        return ToolAssessment(risk=self.risk, permissions=self.permissions)

    def timeout_for(self, args: InputT, env: ToolEnvironment) -> float:
        return self.timeout_s

    @abstractmethod
    async def run(self, args: InputT, env: ToolEnvironment) -> OutputT: ...

    def definition(self, env: ToolEnvironment | None = None) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.describe(env),
            parameters=clean_schema(self.input_model.model_json_schema()),
        )

    def describe(self, env: ToolEnvironment | None) -> str:
        return self.description

    def output_schema(self) -> dict[str, Any]:
        return clean_schema(self.output_model.model_json_schema())


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's ``title`` noise; models read schemas, so keep them compact."""

    def walk(node: Any, parent_key: str | None = None) -> Any:
        if isinstance(node, dict):
            return {
                k: walk(v, k)
                for k, v in node.items()
                # 'title' is noise except where it is a property *named* title
                if not (k == "title" and parent_key != "properties" and isinstance(v, str))
            }
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    cleaned = walk(schema)
    assert isinstance(cleaned, dict)
    return cleaned


class ToolResult(BaseModel):
    """The outcome of one tool call, as audited and as returned to the model."""

    call_id: str
    tool: str
    ok: bool
    content: str
    data: dict[str, Any] | None = None
    error: str | None = None
    truncated: bool = False
    duration_s: float = 0.0
    risk: RiskLevel | None = None
    verdict: str | None = None
    changed_paths: list[str] = []
