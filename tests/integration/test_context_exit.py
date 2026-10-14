"""Phase 3 exit condition: each agent can receive targeted repository context.

The same billing task is prepared for the Planner, Executor and QA, for two model
profiles with very different windows. Each context must fit its model's budget,
contain the files that matter, and leave out the ones that do not.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from forge.context import ContextBuilder, RepositoryAnalyzer
from forge.core.contracts import (
    AcceptanceCriterion,
    FindingType,
    FunctionalRequirement,
    Plan,
    QAFinding,
    RequirementContract,
    Severity,
    TestRun,
)
from forge.models import (
    Message,
    ModelCapabilities,
    ModelProfile,
    ModelRegistry,
)
from forge.models.providers.scripted import ScriptedProvider
from forge.tools import GitRepo, Workspace

REQUEST = "Discounts on invoices must never exceed 50%"
CONTRACT = RequirementContract(
    summary="Cap invoice discounts at 50%",
    functional=[
        FunctionalRequirement(id="FR-001", description="Discount percent is clamped to 50")
    ],
    acceptance_criteria=[
        AcceptanceCriterion(
            id="AC-001",
            description="apply_discount(80) returns half the subtotal",
            requirement_ids=["FR-001"],
        ),
        AcceptanceCriterion(id="AC-002", description="Existing invoice tests still pass"),
    ],
)
PLAN = Plan.model_validate(
    {
        "goal": "Cap discounts at 50%",
        "tasks": [
            {
                "id": "T1",
                "description": "Clamp percent in Invoice.apply_discount",
                "files_hint": ["app/billing/invoice.py"],
                "acceptance_criteria": ["AC-001"],
            },
            {
                "id": "T2",
                "description": "Add a test for the clamp",
                "dependencies": ["T1"],
                "files_hint": ["tests/test_invoice.py"],
                "acceptance_criteria": ["AC-001"],
            },
        ],
    }
)
IRRELEVANT = {"app/utils/strings.py", "app/users/models.py", "web/src/client.ts"}

PROFILES = {
    "local-8k": ModelCapabilities(context_window=8192, max_output_tokens=2048),
    "frontier-200k": ModelCapabilities(
        context_window=200_000,
        max_output_tokens=16_000,
        tool_calling=True,
        structured_output=True,
    ),
}


async def registry_with_profiles() -> ModelRegistry:
    registry = ModelRegistry(
        profiles={
            name: ModelProfile(
                provider="p", model=name, capabilities=caps, detect_capabilities=False
            )
            for name, caps in PROFILES.items()
        }
    )
    registry.register_provider("p", ScriptedProvider())
    return registry


@pytest.mark.parametrize("profile", list(PROFILES))
async def test_each_role_gets_targeted_context(multi_repo: Path, profile: str) -> None:
    # The executor has already edited the invoice module.
    invoice = multi_repo / "app/billing/invoice.py"
    invoice.write_text(
        invoice.read_text().replace(
            "return self.subtotal() * (1 - percent / 100)",
            "percent = min(percent, 50)\n        return self.subtotal() * (1 - percent / 100)",
        )
    )
    repo = await GitRepo.discover(multi_repo)
    assert repo is not None
    analysis = await RepositoryAnalyzer(Workspace(multi_repo)).analyze(repo)
    diff = await repo.diff()
    builder = ContextBuilder(analysis)

    registry = await registry_with_profiles()
    contexts = {}
    for role in ("planner", "executor", "qa"):
        adapter = await registry.adapter_for_profile(profile, role=role)
        caps, opt = adapter.capabilities, adapter.optimization
        if role == "planner":
            contexts[role] = builder.planner(
                request=REQUEST, requirements=CONTRACT, capabilities=caps, optimization=opt
            )
        elif role == "executor":
            contexts[role] = builder.executor(
                task=PLAN.task("T1"),
                plan=PLAN,
                requirements=CONTRACT,
                diff=diff,
                changed_files=["app/billing/invoice.py"],
                capabilities=caps,
                optimization=opt,
            )
        else:
            contexts[role] = builder.qa(
                requirements=CONTRACT,
                plan=PLAN,
                changed_files=["app/billing/invoice.py"],
                diff=diff,
                capabilities=caps,
                optimization=opt,
                test_results=[TestRun(command="pytest -q", exit_code=0, passed=3, failed=0)],
                previous_findings=[
                    QAFinding(
                        id="QA-001",
                        severity=Severity.P2,
                        type=FindingType.TEST_GAP,
                        description="No test covers the 50% cap",
                        affected_files=["tests/test_invoice.py"],
                    )
                ],
            )

    for role, ctx in contexts.items():
        # Within the budget derived from this model's real window.
        assert ctx.tokens <= ctx.budget.context_tokens + 50, (role, ctx.tokens, ctx.budget)
        assert ctx.budget.window == PROFILES[profile].context_window
        # Targeted: the billing implementation is in, unrelated modules are out.
        assert ctx.files and ctx.files[0] == "app/billing/invoice.py", (role, ctx.files)
        assert not set(ctx.files) & IRRELEVANT, (role, ctx.files)
        # Never the whole repository.
        assert len(ctx.files) < len(analysis.files)
        assert "MULTI_FILES" not in ctx.render()

    assert "Repository structure" in [t for t, _, _ in contexts["planner"].sections]
    assert "min(percent, 50)" in (contexts["executor"].section("Current changes (git diff)") or "")
    assert "QA-001" in (contexts["qa"].section("Previous findings") or "")
    assert "AC-001" in (contexts["qa"].section("Requirements and acceptance criteria") or "")
    if profile == "frontier-200k":
        # A big window gets full file contents rather than outlines.
        assert all(
            mode == "full"
            for t, _, mode in contexts["executor"].sections
            if t.startswith(("File", "Changed file"))
        )


class FilesToChange(BaseModel):
    files: list[str]
    reason: str


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("FORGE_LIVE_TESTS") != "1", reason="FORGE_LIVE_TESTS!=1")
async def test_live_model_uses_planner_context(multi_repo: Path) -> None:
    """With only the planner context (no tools), a small local model can name the right file."""
    try:
        up = httpx.get("http://localhost:11434/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        up = False
    if not up:
        pytest.skip("ollama not running")
    registry = ModelRegistry(
        profiles={
            "q": ModelProfile(
                provider="ollama",
                model="qwen2.5:7b",
                params={"temperature": 0},
                capabilities=ModelCapabilities(context_window=8192),
            )
        }
    )
    try:
        adapter = await registry.adapter("planner")
        analysis = await RepositoryAnalyzer(Workspace(multi_repo)).analyze(
            await GitRepo.discover(multi_repo)
        )
        ctx = ContextBuilder(analysis).planner(
            request=REQUEST, capabilities=adapter.capabilities, optimization=adapter.optimization
        )
        answer = await adapter.structured(
            [
                Message.system("You plan code changes. Answer only from the context given."),
                ctx.to_message("Which existing file(s) must change to implement the request?"),
            ],
            FilesToChange,
        )
        print(f"\ncontext {ctx.tokens}/{ctx.budget.context_tokens} tokens -> {answer}")
        assert any(f.endswith("app/billing/invoice.py") for f in answer.files)
    finally:
        await registry.aclose()
