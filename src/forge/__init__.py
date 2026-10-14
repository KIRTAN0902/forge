"""Forge: a model-agnostic, multi-agent software-engineering harness.

The engine is headless. Interfaces (CLI today, web/IDE later) are thin clients of
the packages below; nothing in the engine imports from ``forge.cli``.
"""

import sys

if sys.platform == "win32":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

__version__ = "0.1.0"

__all__ = ["__version__"]
