"""User cancellation listener for interactive CLI sessions.

Allows interrupting agent thinking or long-running workflow steps with the Escape key
(or Ctrl+C) without terminating the overall CLI chat process.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

from rich.console import Console

from forge.core.errors import UserCancelledError

T = TypeVar("T")


def _is_tty() -> bool:
    """Check if standard input is attached to an interactive terminal."""
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _flush_input_buffer() -> None:
    """Flush pending keys from input buffer so they don't leak into subsequent prompts."""
    if sys.platform == "win32":
        try:
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getch()
        except Exception:
            pass


async def _esc_listener(cancel_event: asyncio.Event) -> None:
    """Watch for Escape (b'\\x1b') or Ctrl+C (b'\\x03') asynchronously."""
    if not _is_tty():
        # In non-interactive or redirected environments, don't poll keys
        await cancel_event.wait()
        return

    if sys.platform == "win32":
        import msvcrt

        while not cancel_event.is_set():
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b"\x1b", b"\x03"):
                        _flush_input_buffer()
                        cancel_event.set()
                        return
            except Exception:
                pass
            await asyncio.sleep(0.04)
    else:
        # Unix / macOS
        import select
        import termios
        import tty

        old_settings = None
        fd = None
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            pass

        try:
            while not cancel_event.is_set():
                try:
                    if fd is not None:
                        r, _, _ = select.select([sys.stdin], [], [], 0)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch in ("\x1b", "\x03"):
                                cancel_event.set()
                                return
                except Exception:
                    pass
                await asyncio.sleep(0.04)
        finally:
            if fd is not None and old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass


async def run_with_esc_cancellation(
    coro: Coroutine[Any, Any, T],
    *,
    status_message: str = "",
    console: Console | None = None,
    spinner: str = "dots",
) -> T:
    """Run an async coroutine while listening for Escape key cancellation.

    If the user presses Escape (or Ctrl+C) while `coro` is executing:
    - The underlying task is cancelled immediately.
    - Sockets / HTTP connections are aborted.
    - A `UserCancelledError` is raised.
    """
    cancel_event = asyncio.Event()
    main_task = asyncio.create_task(coro)
    listener_task = asyncio.create_task(_esc_listener(cancel_event))

    async def _wait_for_cancellation() -> None:
        await cancel_event.wait()

    cancel_waiter = asyncio.create_task(_wait_for_cancellation())

    try:
        if status_message and console is not None:
            hint_msg = f"{status_message} [dim](Press Esc to cancel)[/dim]"
            with console.status(hint_msg, spinner=spinner):
                done, pending = await asyncio.wait(
                    [main_task, cancel_waiter],
                    return_when=asyncio.FIRST_COMPLETED,
                )
        else:
            done, pending = await asyncio.wait(
                [main_task, cancel_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )

        if cancel_event.is_set():
            main_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await main_task
            _flush_input_buffer()
            raise UserCancelledError("Interrupted by user (Esc).")

        # Main task finished normally
        return await main_task
    finally:
        cancel_event.set()
        listener_task.cancel()
        cancel_waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await listener_task
        with contextlib.suppress(asyncio.CancelledError):
            await cancel_waiter
