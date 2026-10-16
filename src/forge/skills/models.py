"""Strict Pydantic v2 schemas and models for Forge's Cybernetic Harness and Autonomous Skill Router."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelTier(StrEnum):
    """Execution model topology tier."""

    PROSTHETIC_SLM = "prosthetic_slm"
    AMPLIFIER_FRONTIER = "amplifier_frontier"


class OperationType(StrEnum):
    """Discrete developer operation phase."""

    LOCATE_SYMBOL = "locate_symbol"
    WRITE_PURE_FUNCTION = "write_pure_function"
    EDIT_ORM_QUERY = "edit_orm_query"
    API_ENDPOINT = "api_endpoint"
    DB_MIGRATION = "db_migration"
    REFACTOR_SIGNATURE = "refactor_signature"
    WRITE_UNIT_TEST = "write_unit_test"
    DEBUG_FAILURE = "debug_failure"
    SECURITY_AUDIT = "security_audit"


class SkillSpec(BaseModel):
    """Specification of an autonomous cybernetic skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_id: str = Field(description="Unique skill identifier, e.g. DBG-01, SLM-01, HRN-04")
    name: str = Field(description="Descriptive name of the skill")
    target_agent: Literal["orchestrator", "executor", "qa"] = Field(
        description="Designated agent role to execute this skill"
    )
    applicable_tiers: set[ModelTier] = Field(
        description="Model topology tiers where this skill is active"
    )
    operation_types: set[OperationType] = Field(
        description="Operation categories for which this skill may be triggered"
    )
    trigger_file_globs: list[str] = Field(
        default_factory=list,
        description="File path globs that trigger activation (e.g. ['*migration*/*', 'alembic/*'])"
    )
    trigger_ast_nodes: list[str] = Field(
        default_factory=list,
        description="AST node types or import names that trigger activation"
    )
    trigger_error_regexes: list[str] = Field(
        default_factory=list,
        description="Error regex patterns in stderr/output that trigger activation"
    )
    execution_rules: list[str] = Field(
        default_factory=list,
        description="Compact, numbered imperative rules injected into the operational prompt"
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Single-tool or restricted tool schemas permitted during this skill phase"
    )


class RepoSignals(BaseModel):
    """Telemetry and environment signals capturing live repository state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    modified_files: list[str] = Field(
        default_factory=list,
        description="List of repo-relative paths of modified or touched files"
    )
    ast_imports: list[str] = Field(
        default_factory=list,
        description="Import symbols identified in touched files or AST analysis"
    )
    ast_call_radius: int = Field(
        default=0,
        description="AST depth or number of dependent call sites affected by diff"
    )
    cyclomatic_complexity: int = Field(
        default=0,
        description="Estimated cyclomatic complexity of target symbol or file"
    )
    consecutive_failures: int = Field(
        default=0,
        description="Consecutive verification/execution failures on current node"
    )
    last_stderr: str = Field(
        default="",
        description="Standard error or exception traceback from last command execution"
    )
    has_migration_files: bool = Field(
        default=False,
        description="Whether migration or schema definitions are modified"
    )
    model_tier: ModelTier = Field(
        default=ModelTier.AMPLIFIER_FRONTIER,
        description="Topology tier of the currently executing model"
    )
