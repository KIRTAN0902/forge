"""Path and command policies, plus the standard policy-engine assembly."""

from __future__ import annotations

from collections.abc import Iterable

from forge.core.globs import PathMatcher
from forge.policies.commands import DEFAULT_FORBIDDEN_COMMANDS, forbidden_match
from forge.policies.engine import PermissionPolicy, PolicyEngine, RiskThresholdPolicy
from forge.policies.types import Permission, PolicyDecision, ProposedAction, RiskLevel, Verdict

DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    ".git/**",
    ".env",
    ".env.*",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa*",
)
DEFAULT_SECRET_PATHS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "!.env.example",
    "!.env.sample",
    "!.env.template",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/.npmrc",
    "**/.pypirc",
    "**/.netrc",
    "**/credentials.json",
    "**/secrets.yaml",
    "**/secrets.yml",
)


class PathPolicy:
    """Modifying a protected path, or reading a secret, needs a human."""

    name = "paths"

    def __init__(self, protected: Iterable[str], secrets: Iterable[str]) -> None:
        self.protected = PathMatcher(protected)
        self.secrets = PathMatcher(secrets)

    def evaluate(self, action: ProposedAction) -> PolicyDecision | None:
        writes = Permission.WRITE in action.permissions or Permission.VCS in action.permissions
        for path in action.paths:
            if writes and self.protected.matches(path):
                return PolicyDecision(
                    verdict=Verdict.REQUIRE_APPROVAL,
                    reason=f"modifies protected path {path!r}",
                    policy=self.name,
                )
            if Permission.READ in action.permissions and self.secrets.matches(path):
                return PolicyDecision(
                    verdict=Verdict.REQUIRE_APPROVAL,
                    reason=f"reads secret file {path!r}",
                    policy=self.name,
                )
        return None


class CommandPolicy:
    """Commands matching a forbidden pattern are rejected outright."""

    name = "commands"

    def __init__(self, forbidden: Iterable[str] = DEFAULT_FORBIDDEN_COMMANDS) -> None:
        self.forbidden = list(forbidden)

    def evaluate(self, action: ProposedAction) -> PolicyDecision | None:
        if action.command is None:
            return None
        pattern = forbidden_match(action.command, self.forbidden)
        if pattern is not None:
            return PolicyDecision(
                verdict=Verdict.DENY,
                reason=f"command matches forbidden pattern {pattern!r}",
                policy=self.name,
            )
        return None


def standard_policy_engine(
    *,
    approval_threshold: RiskLevel = RiskLevel.HIGH,
    permissions: Iterable[Permission] = (
        Permission.READ,
        Permission.WRITE,
        Permission.EXECUTE,
        Permission.VCS,
    ),
    protected_paths: Iterable[str] = DEFAULT_PROTECTED_PATHS,
    secret_paths: Iterable[str] = DEFAULT_SECRET_PATHS,
    forbidden_commands: Iterable[str] = DEFAULT_FORBIDDEN_COMMANDS,
) -> PolicyEngine:
    return PolicyEngine(
        [
            RiskThresholdPolicy(approval_threshold),
            PermissionPolicy(permissions),
            PathPolicy(protected_paths, secret_paths),
            CommandPolicy(forbidden_commands),
        ]
    )
