"""Identifier generation.

IDs are ``<prefix>_<time-ordered hex>``: human-readable in logs, sortable by creation
time, and unique enough for a local single-user engine.
"""

from __future__ import annotations

import os
import time


def new_id(prefix: str) -> str:
    """Return a new time-ordered identifier such as ``wf_0192f3c1a2b4_9f3e1c7a``."""
    if not prefix or not prefix.isidentifier():
        raise ValueError(f"invalid id prefix: {prefix!r}")
    millis = time.time_ns() // 1_000_000
    return f"{prefix}_{millis:012x}_{os.urandom(4).hex()}"
