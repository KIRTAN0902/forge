"""Logging setup.

The engine logs through the standard library under the ``forge`` logger namespace.
Interfaces decide where logs go: the CLI attaches a console handler, and every run can
also write JSON lines to a file for later inspection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from forge.core.time import utcnow

ROOT_LOGGER = "forge"

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``forge`` namespace."""
    if name == ROOT_LOGGER or name.startswith(ROOT_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


class JsonLineFormatter(logging.Formatter):
    """Formats records as one JSON object per line, including ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str | int = "INFO",
    *,
    json_file: Path | None = None,
    console: logging.Handler | None = None,
) -> logging.Logger:
    """Configure the ``forge`` logger. Safe to call repeatedly (replaces handlers)."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    if console is not None:
        logger.addHandler(console)
    if json_file is not None:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(json_file, encoding="utf-8")
        file_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(file_handler)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger
