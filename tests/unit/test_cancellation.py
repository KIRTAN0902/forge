"""Unit tests for Esc key cancellation handling."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from forge.cli.cancellation import _flush_input_buffer, _is_tty, run_with_esc_cancellation
from forge.core.errors import UserCancelledError


@pytest.mark.asyncio
async def test_run_with_esc_cancellation_normal_completion() -> None:
    """Coroutine completes and returns value when no cancellation occurs."""
    async def fast_task() -> str:
        await asyncio.sleep(0.01)
        return "done"

    result = await run_with_esc_cancellation(fast_task())
    assert result == "done"


@pytest.mark.asyncio
async def test_run_with_esc_cancellation_inner_error_propagates() -> None:
    """Exceptions raised inside the coroutine propagate normally."""
    async def failing_task() -> None:
        await asyncio.sleep(0.01)
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        await run_with_esc_cancellation(failing_task())


@pytest.mark.asyncio
async def test_run_with_esc_cancellation_triggers_on_esc() -> None:
    """When Esc key is detected, task is cancelled and UserCancelledError is raised."""
    task_cancelled = False

    async def long_running_task() -> None:
        nonlocal task_cancelled
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    # Mock _esc_listener to simulate user pressing Esc
    async def mock_esc_listener(cancel_event: asyncio.Event) -> None:
        await asyncio.sleep(0.02)
        cancel_event.set()

    with (
        patch("forge.cli.cancellation._esc_listener", side_effect=mock_esc_listener),
        pytest.raises(UserCancelledError, match="Interrupted by user"),
    ):
        await run_with_esc_cancellation(long_running_task())

    assert task_cancelled is True


def test_is_tty_safe() -> None:
    """_is_tty returns a boolean safely."""
    res = _is_tty()
    assert isinstance(res, bool)


def test_flush_input_buffer_safe() -> None:
    """_flush_input_buffer executes safely without error."""
    _flush_input_buffer()
