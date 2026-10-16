"""Heuristic signal evaluator for automatic skill activation."""

from __future__ import annotations

import re

from forge.skills.registry import SkillRegistry
from forge.skills.types import SkillActivation, SkillSignal

# High-entropy secret patterns
SECRET_PATTERNS = (
    re.compile(r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}"),  # AWS Access Key
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),  # OpenAI/OpenRouter Key
    re.compile(r"gh[pousr]_[0-9a-zA-Z]{36}"),  # GitHub Token
    re.compile(r"ey[a-zA-Z0-9_-]{10,}\.ey[a-zA-Z0-9_-]{10,}"),  # JWT Token
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),  # Private Key
)

# Vulnerable sinks in diffs
SINK_PATTERNS = (
    re.compile(r"\bexec\s*\("),
    re.compile(r"\beval\s*\("),
    re.compile(r"subprocess\.(?:run|Popen|call)\([^)]*shell\s*=\s*True"),
    re.compile(r"\.execute\s*\(\s*f[\"']"),  # Raw SQL f-string
)

# Blocking I/O patterns in async contexts
BLOCKING_PATTERNS = (
    re.compile(r"time\.sleep\s*\("),
    re.compile(r"requests\.(?:get|post|put|delete|patch)\s*\("),
    re.compile(r"urllib\.request\."),
)


class SkillEvaluator:
    """Evaluates telemetry signals to determine which skills to activate."""

    def __init__(self, registry: SkillRegistry | None = None) -> None:
        self.registry = registry if registry is not None else SkillRegistry.with_anthropic()

    def evaluate(self, signal: SkillSignal, agent_role: str | None = None) -> list[SkillActivation]:
        """Evaluate signals and return matching active skills."""
        activations: list[SkillActivation] = []

        # DBG-01: Stratified Fault Localization & Repro-First
        if (signal.test_exit_code is not None and signal.test_exit_code != 0) or any(
            w in signal.user_request.lower() or w in signal.error_text.lower()
            for w in ("bug", "fix", "error", "fails", "failed", "crash", "traceback", "exception")
        ):
            skill = self.registry.get("DBG-01")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="executor",
                        rationale="Active bug or test failure detected; mandate failing repro test before edits.",
                    )
                )

        # DBG-02: Automated Git Bisect & Delta Forensics
        if any(
            w in signal.user_request.lower()
            for w in ("regression", "used to work", "worked before", "broke in commit", "bisect")
        ):
            skill = self.registry.get("DBG-02")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="executor",
                        rationale="Regression reported in previously working feature; engage git bisect forensics.",
                    )
                )

        # DBG-03: Loop-Break & Backtracking State Machine
        if signal.consecutive_failures >= 3:
            skill = self.registry.get("DBG-03")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="orchestrator",
                        rationale=f"{signal.consecutive_failures} consecutive failures detected on current task; trigger replan or rollback.",
                    )
                )

        # REF-01: Mikado Graph & Strangler-Fig Engine
        if len(signal.touched_files) >= 5 or any(
            w in signal.user_request.lower() for w in ("refactor", "rename interface", "strangler")
        ):
            skill = self.registry.get("REF-01")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="orchestrator",
                        rationale="Broad interface changes or multi-file refactoring in progress.",
                    )
                )

        # REF-02: Tarjan SCC Circular Dependency Untangler
        if "cycle" in signal.error_text.lower() or "importerror" in signal.error_text.lower():
            skill = self.registry.get("REF-02")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="executor",
                        rationale="Import cycle or circular dependency encountered in traceback.",
                    )
                )

        # DB-01: Three-Phase Expand-Contract Migration Guard
        if any(
            any(m in f.lower() for m in ("migration", "alembic", "models", "schema.sql"))
            for f in signal.touched_files
        ):
            skill = self.registry.get("DB-01")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="qa",
                        rationale="Database migration or ORM schema files modified; enforce non-destructive expand/contract.",
                    )
                )

        # API-01: Bi-Directional Schema & Contract Guard
        if any(
            any(
                a in f.lower()
                for a in ("route", "api", "controller", "endpoint", "openapi", "pydantic")
            )
            for f in signal.touched_files
        ):
            skill = self.registry.get("API-01")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="qa",
                        rationale="API routes or serialization schemas modified; check backwards compatibility.",
                    )
                )

        # SEC-01: AST Taint-Flow Source-to-Sink Analyzer
        if any(pat.search(signal.diff_text) for pat in SINK_PATTERNS):
            skill = self.registry.get("SEC-01")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="qa",
                        rationale="Potentially dangerous sink (subprocess/exec/raw SQL) detected in diff.",
                    )
                )

        # SEC-02: Shannon-Entropy Secret Scrubber
        if any(pat.search(signal.diff_text) for pat in SECRET_PATTERNS):
            skill = self.registry.get("SEC-02")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="qa",
                        rationale="Potential high-entropy secret or API key pattern detected in diff.",
                    )
                )

        # SEC-03: Anti-Slopsquatting & Supply-Chain Auditor
        if any(
            w in signal.command_executed.lower()
            for w in ("pip install", "uv add", "npm install", "cargo add")
        ) or any(
            any(m in f.lower() for m in ("requirements.txt", "pyproject.toml", "package.json"))
            for f in signal.touched_files
        ):
            skill = self.registry.get("SEC-03")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="orchestrator",
                        rationale="Package installation or manifest change detected; audit dependency legitimacy.",
                    )
                )

        # PERF-01: Asyncio Event-Loop Blocking Call Detector
        if "async def" in signal.diff_text and any(
            pat.search(signal.diff_text) for pat in BLOCKING_PATTERNS
        ):
            skill = self.registry.get("PERF-01")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="qa",
                        rationale="Synchronous blocking call (sleep/requests) detected inside async coroutine.",
                    )
                )

        # TST-03: Mock-Integrity & Anti-Mock-Tautology Guard
        if any(f.startswith("tests/") or "test_" in f for f in signal.touched_files) and any(
            m in signal.diff_text for m in ("mock", "patch(", "MagicMock", "assert_called")
        ):
            skill = self.registry.get("TST-03")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="qa",
                        rationale="Mocks or patches detected in test diff; verify SUT itself is not mocked.",
                    )
                )

        # DOC-03: Semantic AST-Diff Changelog & SemVer Classifier
        if any(w in signal.user_request.lower() for w in ("commit", "release", "changelog", "pr ")):
            skill = self.registry.get("DOC-03")
            if skill:
                activations.append(
                    SkillActivation(
                        skill=skill,
                        target_agent="orchestrator",
                        rationale="Release or commit activity; classify changes and generate Conventional Commit message.",
                    )
                )

        # Anthropic Domain Skills (ANT-01 to ANT-19)
        try:
            from pathlib import Path

            from forge.skills.anthropic_loader import parse_anthropic_skills

            req_lower = signal.user_request.lower()
            for spec, skill_item, keywords in parse_anthropic_skills():
                kw_match = any(re.search(rf"\b{re.escape(kw)}\b", req_lower) for kw in keywords)
                file_match = (
                    any(
                        any(Path(f).match(glob) for glob in spec.trigger_file_globs)
                        for f in signal.touched_files
                    )
                    if spec.trigger_file_globs
                    else False
                )
                if kw_match or file_match:
                    activations.append(
                        SkillActivation(
                            skill=skill_item,
                            target_agent=spec.target_agent,
                            rationale=f"Triggered by Anthropic domain heuristics for {spec.name}.",
                        )
                    )
        except Exception:
            pass

        # Frontend & Design Engineering Skills (DSN-01 to DSN-04)
        try:
            from pathlib import Path

            from forge.skills.design_loader import parse_design_skills

            req_lower = signal.user_request.lower()
            for spec, skill_item, keywords in parse_design_skills():
                kw_match = any(re.search(rf"\b{re.escape(kw)}\b", req_lower) for kw in keywords)
                file_match = (
                    any(
                        any(Path(f).match(glob) for glob in spec.trigger_file_globs)
                        for f in signal.touched_files
                    )
                    if spec.trigger_file_globs
                    else False
                )
                if kw_match or file_match:
                    activations.append(
                        SkillActivation(
                            skill=skill_item,
                            target_agent=spec.target_agent,
                            rationale=f"Triggered by design engineering heuristics for {spec.name}.",
                        )
                    )
        except Exception:
            pass

        # Filter by agent role if specified
        if agent_role:
            target_role = agent_role.lower().strip()
            activations = [act for act in activations if target_role in act.skill.target_agents]

        return activations
