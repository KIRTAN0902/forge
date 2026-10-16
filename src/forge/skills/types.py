"""Types and contract definitions for Forge's autonomous skill engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SkillCategory(StrEnum):
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DATABASE = "database"
    API = "api"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass(frozen=True)
class Skill:
    """Pre-programmed specialist skill specification."""

    id: str
    name: str
    category: SkillCategory
    target_agents: tuple[str, ...]
    primary_trigger: str
    protocol: str
    failure_mode_prevented: str


@dataclass
class SkillSignal:
    """Telemetry and environment signals evaluated to trigger skills."""

    user_request: str = ""
    touched_files: list[str] = field(default_factory=list)
    diff_text: str = ""
    test_exit_code: int | None = None
    test_output: str = ""
    consecutive_failures: int = 0
    error_text: str = ""
    ast_symbols: list[str] = field(default_factory=list)
    command_executed: str = ""


@dataclass
class SkillActivation:
    """An activated skill instance dispatched to an agent."""

    skill: Skill
    target_agent: str
    rationale: str
