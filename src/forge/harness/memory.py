"""Forge Episodic Repository Memory and SFT Problem-Solution Trace Store ([HRN-09])."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RepoHeuristic(BaseModel):
    """Architectural heuristic or convention learned about the repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    value: str
    category: str = "general"
    associated_paths: list[str] = Field(default_factory=list)


class SFTTrace(BaseModel):
    """Problem-solution trace successfully verified by compiler and test gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str
    prompt: str
    initial_code: str
    sliced_context: str
    patch_diff: str
    verification_evidence: str


class EpisodicRepoMemory:
    """Manages persistent project-local heuristics and SFT training traces in .forge/memory/."""

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.memory_dir = self.repo_root / ".forge" / "memory"
        self.heuristics_file = self.memory_dir / "repo_heuristics.json"
        self.sft_traces_file = self.memory_dir / "sft_traces.jsonl"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.heuristics_file.exists():
            self.heuristics_file.write_text("{}", encoding="utf-8")
        if not self.sft_traces_file.exists():
            self.sft_traces_file.touch()

    def record_heuristic(
        self,
        key: str,
        value: str,
        category: str = "general",
        associated_paths: list[str] | None = None,
    ) -> None:
        """Store an architectural heuristic or learned repository pattern."""
        self._ensure_storage()
        paths = associated_paths or []
        heuristic = RepoHeuristic(
            key=key,
            value=value,
            category=category,
            associated_paths=paths,
        )

        try:
            content = self.heuristics_file.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(content) if content.strip() else {}
        except Exception:
            data = {}

        data[key] = heuristic.model_dump()
        self.heuristics_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def retrieve_relevant_heuristics(self, file_paths: list[str]) -> list[str]:
        """Retrieve stored heuristics matching the active file paths."""
        self._ensure_storage()
        try:
            content = self.heuristics_file.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(content) if content.strip() else {}
        except Exception:
            return []

        matched: list[str] = []
        normalized_targets = [p.replace("\\", "/").lower() for p in file_paths]

        for item in data.values():
            val = item.get("value", "")
            key = item.get("key", "")
            item_paths = item.get("associated_paths", [])

            # Check if any associated path matches
            path_match = False
            for p in item_paths:
                norm_p = p.replace("\\", "/").lower()
                if any(norm_p in target or target in norm_p for target in normalized_targets):
                    path_match = True
                    break

            if path_match or not item_paths:
                matched.append(f"[{key}] {val}")

        return matched

    def record_sft_trace(
        self,
        timestamp: str,
        prompt: str,
        initial_code: str,
        sliced_context: str,
        patch_diff: str,
        verification_evidence: str,
    ) -> None:
        """Append a successfully verified problem-solution trace into sft_traces.jsonl."""
        self._ensure_storage()
        trace = SFTTrace(
            timestamp=timestamp,
            prompt=prompt,
            initial_code=initial_code,
            sliced_context=sliced_context,
            patch_diff=patch_diff,
            verification_evidence=verification_evidence,
        )

        line = trace.model_dump_json() + "\n"
        with self.sft_traces_file.open("a", encoding="utf-8") as f:
            f.write(line)

    def load_all_traces(self) -> list[SFTTrace]:
        """Load all recorded problem-solution traces from sft_traces.jsonl."""
        self._ensure_storage()
        traces: list[SFTTrace] = []
        with self.sft_traces_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        traces.append(SFTTrace.model_validate_json(stripped))
                    except Exception:
                        continue
        return traces
