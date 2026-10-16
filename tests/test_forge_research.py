"""Comprehensive test suite for Forge's Publication-Grade Deep Research Subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from forge.research.compiler import ManuscriptCompiler
from forge.research.harvester import (
    AcademicHarvester,
    STORMInquiryEngine,
    build_cite_key,
    generate_bibtex_entry,
)
from forge.research.models import (
    ContradictionEntry,
    EmpiricalExperimentResult,
    EvidenceNode,
    PerspectivePersona,
    ResearchLedger,
    SourceTier,
)
from forge.research.verifier import (
    EmpiricalLabRunner,
    ZeroTrustCitationAuditor,
)


def test_deterministic_cite_key_and_bibtex_formatting() -> None:
    """[RES-02] Test deterministic cite key generation and BibTeX formatting."""
    key1 = build_cite_key(
        ["Noah Shinn", "Federico Cassano"],
        2023,
        "Reflexion: Language Agents with Verbal Reinforcement Learning",
    )
    assert key1 == "shinn2023reflexion"

    key2 = build_cite_key(
        ["Vaswani, Ashish", "Shazeer, Noam"],
        2017,
        "Attention Is All You Need",
    )
    assert key2 == "vaswani2017attention"

    node = EvidenceNode(
        cite_key="shinn2023reflexion",
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        authors=["Noah Shinn", "Federico Cassano"],
        year=2023,
        venue_or_source="NeurIPS 2023",
        doi_or_arxiv_id="2303.11366",
        url="https://arxiv.org/abs/2303.11366",
        source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
        abstract_or_summary="We introduce Reflexion, an architecture for reinforcement learning via verbal feedback.",
    )

    bibtex = generate_bibtex_entry(node)
    assert "@article{shinn2023reflexion," in bibtex
    assert "title = {Reflexion: Language Agents with Verbal Reinforcement Learning}," in bibtex
    assert "author = {Noah Shinn and Federico Cassano}," in bibtex
    assert "year = {2023}," in bibtex
    assert "journal = {NeurIPS 2023}," in bibtex
    assert "eprint = {2303.11366}," in bibtex
    assert "archivePrefix = {arXiv}," in bibtex


def test_zero_trust_citation_auditor_ghost_and_numeric_drift() -> None:
    """[RES-03] Test detection of ghost citations and numeric drift."""
    auditor = ZeroTrustCitationAuditor()

    node = EvidenceNode(
        cite_key="shinn2023reflexion",
        title="Reflexion",
        authors=["Noah Shinn"],
        year=2023,
        venue_or_source="NeurIPS",
        doi_or_arxiv_id="2303.11366",
        url="https://arxiv.org/abs/2303.11366",
        source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
        abstract_or_summary="Achieves 72% pass@1 on HumanEval benchmark with verbal feedback.",
        verbatim_quotes=["Reflexion improves pass@1 on HumanEval to 72%."],
    )

    ledger = ResearchLedger(
        topic="Reflexion Architectures",
        evidence_map={"shinn2023reflexion": node},
    )

    # 1. Ghost citation test
    ghost_text = "Recent studies [@fake2026ghost] claim unbounded reasoning loops."
    violations_ghost = auditor.audit_manuscript_claims(ghost_text, ledger)
    assert any("Ghost citation detected: [@fake2026ghost]" in v for v in violations_ghost)

    # 2. Numeric drift test: Claim asserts 95% while evidence states 72%
    drift_text = (
        "Reflexion demonstrates an impressive pass@1 of 95% on HumanEval [@shinn2023reflexion]."
    )
    violations_drift = auditor.audit_manuscript_claims(drift_text, ledger)
    assert any(
        "Numeric drift detected in citation [@shinn2023reflexion]" in v and "95%" in v
        for v in violations_drift
    )

    # 3. Grounded statement without drift
    valid_text = "Reflexion demonstrates a pass@1 of 72% on HumanEval [@shinn2023reflexion]."
    violations_valid = auditor.audit_manuscript_claims(valid_text, ledger)
    assert not any("Numeric drift" in v for v in violations_valid)
    assert not any("Ghost citation" in v for v in violations_valid)


@pytest.mark.asyncio
async def test_storm_inquiry_engine_saturation_logic() -> None:
    """[RES-01] Test STORM multi-perspective inquiry early termination on saturation."""
    mock_harvester = AsyncMock(spec=AcademicHarvester)

    node1 = EvidenceNode(
        cite_key="shinn2023reflexion",
        title="Reflexion",
        authors=["Shinn"],
        year=2023,
        source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
    )
    node2 = EvidenceNode(
        cite_key="vaswani2017attention",
        title="Attention Is All You Need",
        authors=["Vaswani"],
        year=2017,
        source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
    )

    # Wave 1 returns new nodes; Wave 2 returns nothing new (saturation)
    mock_harvester.search_arxiv.side_effect = [
        [node1],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    mock_harvester.search_semantic_scholar_and_snowball.side_effect = [
        [node2],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    mock_harvester.snowball_citations.return_value = []

    storm = STORMInquiryEngine(mock_harvester)
    personas = [
        PerspectivePersona(
            name="Theoretical Formalist",
            role_description="Theory",
            focus_questions=["Q1"],
            search_queries=["query1"],
        )
    ]

    evidence, gains = await storm.execute_inquiry_waves(
        topic="Self-Reflective Agents",
        personas=personas,
        max_waves=3,
        saturation_threshold=0.10,
    )

    assert "shinn2023reflexion" in evidence
    assert "vaswani2017attention" in evidence
    # Wave 1 gave gain 1.0; Wave 2 gave 0.0, causing early saturation break before Wave 3
    assert len(gains) == 2
    assert gains[0] == 1.0
    assert gains[1] == 0.0


@pytest.mark.asyncio
async def test_empirical_lab_runner_sandbox_execution(tmp_path: Path) -> None:
    """[RES-05] Test EmpiricalLabRunner benchmark execution and table extraction."""
    runner = EmpiricalLabRunner(tmp_path)

    hypothesis = "Linear accumulation scales with sub-millisecond execution"
    code = (
        "import time\n"
        "t0 = time.perf_counter()\n"
        "s = sum(range(1000))\n"
        "elapsed_ms = (time.perf_counter() - t0) * 1000\n"
        "print('| Metric | Value | Unit |')\n"
        "print('|---|---|---|')\n"
        "print(f'| Elapsed Time | {elapsed_ms:.4f} | ms |')\n"
    )

    result = await runner.run_empirical_verification(hypothesis, code)

    assert result.exit_code == 0
    assert result.hypothesis == hypothesis
    assert "| Metric | Value | Unit |" in result.metrics_table_markdown
    assert "| Elapsed Time |" in result.metrics_table_markdown
    assert Path(tmp_path / result.script_path).exists()


def test_manuscript_compiler_end_to_end_artifacts(tmp_path: Path) -> None:
    """[RES-06] Test end-to-end manuscript compilation to Markdown, LaTeX, and BibTeX."""
    compiler = ManuscriptCompiler(tmp_path)

    node1 = EvidenceNode(
        cite_key="shinn2023reflexion",
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        authors=["Noah Shinn", "Federico Cassano"],
        year=2023,
        venue_or_source="NeurIPS 2023",
        doi_or_arxiv_id="2303.11366",
        url="https://arxiv.org/abs/2303.11366",
        source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
        abstract_or_summary="Introduces verbal reinforcement learning architecture for autonomous reasoning.",
        verbatim_quotes=["Improves HumanEval pass@1 to 72%."],
    )

    node2 = EvidenceNode(
        cite_key="vaswani2017attention",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        year=2017,
        venue_or_source="NeurIPS 2017",
        doi_or_arxiv_id="1706.03762",
        url="https://arxiv.org/abs/1706.03762",
        source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
        abstract_or_summary="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    )

    ledger = ResearchLedger(
        topic="Autonomous Reasoning Architectures",
        perspectives=[
            PerspectivePersona(
                name="Systems Architect",
                role_description="Evaluates systems",
                focus_questions=["How does it scale?"],
                search_queries=["autonomous scaling"],
            )
        ],
        evidence_map={
            "shinn2023reflexion": node1,
            "vaswani2017attention": node2,
        },
        contradictions=[
            ContradictionEntry(
                topic="Verbal Reflection vs Transformer Retraining",
                source_a_key="shinn2023reflexion",
                claim_a="Verbal reflection avoids gradient updates.",
                source_b_key="vaswani2017attention",
                claim_b="Self-attention parameters store linguistic invariants.",
                confounder_analysis="Difference between in-context learning and weight adaptation.",
            )
        ],
        experiments=[
            EmpiricalExperimentResult(
                hypothesis="Verbal feedback loop stabilizes tool calling",
                script_path=".forge/research/experiments/exp_123.py",
                exit_code=0,
                metrics_table_markdown="| Metric | Value |\n|---|---|\n| Accuracy | 91.2% |",
                raw_stdout="Simulation finished.",
            )
        ],
    )

    artifacts = compiler.export_artifacts(ledger)

    assert "paper_md" in artifacts
    assert "paper_tex" in artifacts
    assert "references_bib" in artifacts
    assert "evidence_ledger" in artifacts

    # 1. Verify paper.md
    md_content = artifacts["paper_md"].read_text(encoding="utf-8")
    assert (
        "# Autonomous Cybernetic Synthesis: An Empirical Study of Autonomous Reasoning Architectures"
        in md_content
    )
    assert "## Abstract" in md_content
    assert "## 1. Introduction & Formal Problem Definition" in md_content
    assert "## 2. Taxonomy & Multi-Perspective Literature Review" in md_content
    assert "## 3. Deep Technical Analysis" in md_content
    assert "## 4. Empirical Sandbox Verification & Ablations" in md_content
    assert "[@shinn2023reflexion]" in md_content

    # 2. Verify paper.tex
    tex_content = artifacts["paper_tex"].read_text(encoding="utf-8")
    assert r"\documentclass[conference]{IEEEtran}" in tex_content
    assert r"\cite{shinn2023reflexion}" in tex_content
    assert r"\bibliography{references}" in tex_content

    # 3. Verify references.bib
    bib_content = artifacts["references_bib"].read_text(encoding="utf-8")
    assert "@article{shinn2023reflexion," in bib_content
    assert "@article{vaswani2017attention," in bib_content

    # 4. Verify evidence_ledger.json
    ledger_data = json.loads(artifacts["evidence_ledger"].read_text(encoding="utf-8"))
    assert ledger_data["topic"] == "Autonomous Reasoning Architectures"
    assert "shinn2023reflexion" in ledger_data["evidence_map"]
