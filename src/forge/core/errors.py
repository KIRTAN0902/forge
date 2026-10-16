"""Exception hierarchy shared by every Forge package.

Every error raised deliberately by Forge derives from :class:`ForgeError` so callers
(the orchestrator, the CLI) can distinguish harness failures from unexpected bugs.
"""

from __future__ import annotations


class ForgeError(Exception):
    """Base class for all deliberate Forge errors."""


class ConfigError(ForgeError):
    """Configuration is missing, malformed, or inconsistent."""


class StorageError(ForgeError):
    """Persistent storage failed or holds unexpected data."""


class NotFoundError(StorageError):
    """A requested record does not exist."""


class PolicyViolation(ForgeError):
    """A proposed action was rejected by the policy gate."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason or message


class ModelError(ForgeError):
    """A model provider call failed."""


class ModelCapabilityError(ModelError):
    """A model was asked to do something its profile says it cannot do."""


class ModelOutputError(ModelError):
    """A model produced output that could not be parsed or validated."""


class ToolError(ForgeError):
    """A tool failed to execute."""


class WorkflowError(ForgeError):
    """The workflow reached an invalid or unrecoverable state."""


class UserCancelledError(ForgeError):
    """An operation or agent thinking turn was cancelled by the user (e.g. Esc or Ctrl+C)."""
