from __future__ import annotations

import os
from pathlib import Path

import pytest

from forge.core.atomic import atomic_write_text


def test_atomic_write_replaces_unicode_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"
    atomic_write_text(target, '{"status": "café"}')
    atomic_write_text(target, '{"status": "naïve"}')

    assert target.read_text(encoding="utf-8") == '{"status": "naïve"}'
    assert not list(target.parent.glob("*.tmp"))


def test_atomic_write_preserves_original_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_text("original", encoding="utf-8")

    def fail_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "replacement")

    assert target.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob("*.tmp"))
