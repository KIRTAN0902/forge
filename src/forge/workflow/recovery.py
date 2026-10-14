"""Recovery engine: Git checkpoints and automated rollback (spec §15, §22)."""

from __future__ import annotations

from forge.core.contracts.state import Checkpoint
from forge.core.logging import get_logger
from forge.tools.environment import ToolEnvironment
from forge.tools.git import RestoreResult

log = get_logger("workflow.recovery")


class CheckpointManager:
    """Manages Git checkpoints and rollbacks for workflow recovery."""

    async def create_checkpoint(
        self, env: ToolEnvironment, label: str, iteration: int
    ) -> Checkpoint | None:
        """Create a checkpoint of the working tree without touching branch or index."""
        repo = await env.git()
        if repo is None:
            return None

        try:
            info = await repo.checkpoints.create(label)
            return Checkpoint(
                id=info.id,
                ref=info.commit,
                label=info.label,
                iteration=iteration,
            )
        except Exception as exc:
            log.warning("failed to create checkpoint '%s': %s", label, exc)
            return None

    async def rollback(self, env: ToolEnvironment, checkpoint_id: str) -> RestoreResult | None:
        """Restore working tree to checkpoint_id, saving pre-restore state."""
        repo = await env.git()
        if repo is None:
            return None

        try:
            return await repo.checkpoints.restore(checkpoint_id)
        except Exception as exc:
            log.error("rollback to checkpoint '%s' failed: %s", checkpoint_id, exc)
            return None

    async def rollback_to_latest(
        self, env: ToolEnvironment, checkpoints: list[Checkpoint]
    ) -> RestoreResult | None:
        """Roll back to the most recent checkpoint."""
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        return await self.rollback(env, latest.id)
