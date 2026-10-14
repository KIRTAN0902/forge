from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.models.profile import (
    BUILTIN_PROVIDERS,
    ROLES,
    ModelProfile,
    ProviderConfig,
    RoleBinding,
)
from forge.policies.commands import DEFAULT_FORBIDDEN_COMMANDS
from forge.policies.engine import PolicyEngine
from forge.policies.rules import (
    DEFAULT_PROTECTED_PATHS,
    DEFAULT_SECRET_PATHS,
    standard_policy_engine,
)
from forge.policies.types import Permission, RiskLevel


class _Section(BaseModel):
    # Config is written by humans: reject unknown keys so typos surface immediately.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectConfig(_Section):
    name: str | None = None
    data_dir: str = Field(default=".forge", description="Relative to the project root.")


class LimitsConfig(_Section):
    """Hard boundaries on autonomous loops (spec §16)."""

    max_iterations: int = Field(default=10, ge=1)
    max_same_failure: int = Field(
        default=3, ge=1, description="Block after the same failure signature repeats this often."
    )
    max_wall_time_s: int = Field(default=3600, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_tool_calls_per_task: int = Field(default=150, ge=1)


class PolicyConfig(_Section):
    approval_threshold: RiskLevel = RiskLevel.HIGH
    permissions: list[Permission] = Field(
        default_factory=lambda: [
            Permission.READ,
            Permission.WRITE,
            Permission.EXECUTE,
            Permission.VCS,
        ],
        description="Granted permissions. NETWORK is off by default.",
    )
    protected_paths: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PROTECTED_PATHS),
        description="Gitignore-style patterns agents may not modify without approval.",
    )
    secret_paths: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SECRET_PATHS),
        description="Patterns agents may not read without approval ('!' re-includes).",
    )
    forbidden_commands: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FORBIDDEN_COMMANDS),
        description="Regexes; matching shell commands are always rejected.",
    )

    def engine(self) -> PolicyEngine:
        return standard_policy_engine(
            approval_threshold=self.approval_threshold,
            permissions=self.permissions,
            protected_paths=self.protected_paths,
            secret_paths=self.secret_paths,
            forbidden_commands=self.forbidden_commands,
        )


class ToolsConfig(_Section):
    shell: str = Field(
        default="auto",
        description="'auto', 'bash', 'sh', 'powershell', 'cmd', or a path to a shell.",
    )
    command_timeout_s: float = Field(default=300.0, gt=0, le=7200)
    max_output_chars: int = Field(
        default=30_000, ge=1_000, description="Per tool result, as seen by the model."
    )
    env_passthrough: list[str] = Field(
        default_factory=list,
        description="Secret-looking environment variables to expose to commands anyway.",
    )
    test_command: str | None = Field(
        default=None, description="Override the auto-detected test command."
    )
    ignore: list[str] = Field(
        default_factory=list, description="Extra patterns excluded from listings and search."
    )


class LoggingConfig(_Section):
    level: str = "INFO"
    json_file: bool = Field(default=True, description="Write JSON-lines logs to <data_dir>/logs.")


class ForgeConfig(_Section):
    version: int = 1
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description=f"Extra/overridden providers. Built in: {', '.join(BUILTIN_PROVIDERS)}.",
    )
    models: dict[str, ModelProfile] = Field(default_factory=dict)
    roles: dict[str, RoleBinding] = Field(default_factory=dict)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Set by the loader, not by the file.
    root: Path = Field(default_factory=Path.cwd, exclude=True)
    source: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_model_references(self) -> ForgeConfig:
        known_providers = {*BUILTIN_PROVIDERS, *self.providers}
        for name, profile in self.models.items():
            if profile.provider not in known_providers:
                raise ValueError(
                    f"models.{name}: unknown provider {profile.provider!r} "
                    f"(known: {', '.join(sorted(known_providers))})"
                )
        for role, binding in self.roles.items():
            if role not in ROLES:
                raise ValueError(f"roles.{role}: unknown role (expected one of {', '.join(ROLES)})")
            if binding.model not in self.models:
                raise ValueError(f"roles.{role}: unknown model profile {binding.model!r}")
        return self

    @property
    def data_path(self) -> Path:
        return self.root / self.project.data_dir

    @property
    def db_path(self) -> Path:
        return self.data_path / "forge.db"

    @property
    def log_dir(self) -> Path:
        return self.data_path / "logs"
