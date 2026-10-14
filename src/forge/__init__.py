"""Forge: a model-agnostic, multi-agent software-engineering harness.

The engine is headless. Interfaces (CLI today, web/IDE later) are thin clients of
the packages below; nothing in the engine imports from ``forge.cli``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
