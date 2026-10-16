"""Publication-grade Autonomous Deep Research data models and schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SourceTier(StrEnum):
    """Hierarchy of scientific & empirical evidence authority."""

    TIER_1_PRIMARY_PAPER = (
        "tier_1_primary_paper"  # Peer-reviewed journal/conference or arXiv preprint
    )
    TIER_2_ENGINEERING_ARTIFACT = (
        "tier_2_engineering_artifact"  # Official docs, GitHub repo, benchmark logs
    )
    TIER_3_SECONDARY_WEB = "tier_3_secondary_web"  # Technical blogs, community discussions


class EvidenceNode(BaseModel):
    """Cryptographically trackable node of academic or empirical evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cite_key: str = Field(
        description="Deterministic citation key, e.g. 'shinn2023reflexion' or 'vaswani2017attention'."
    )
    title: str = Field(description="Full paper or publication title.")
    authors: list[str] = Field(default_factory=list, description="List of author names.")
    year: int = Field(description="Publication year.")
    venue_or_source: str = Field(
        default="",
        description="Conference, journal, arXiv category, or publisher venue.",
    )
    doi_or_arxiv_id: str | None = Field(
        default=None,
        description="DOI identifier or arXiv ID (e.g. '2303.11366').",
    )
    url: str = Field(default="", description="Permanent URL to abstract or paper.")
    citation_count: int = Field(default=0, description="Global citation count from academic graph.")
    source_tier: SourceTier = Field(
        default=SourceTier.TIER_1_PRIMARY_PAPER,
        description="Authority tier of the source.",
    )
    abstract_or_summary: str = Field(
        default="",
        description="Abstract, executive summary, or technical excerpt.",
    )
    verbatim_quotes: list[str] = Field(
        default_factory=list,
        description="Exact text spans extracted verbatim to back up specific factual claims.",
    )
    experimental_setup: dict[str, str] = Field(
        default_factory=dict,
        description="Key experimental metadata: dataset, metrics, parameter count, hardware constraints.",
    )
    verified_in_registry: bool = Field(
        default=False,
        description="Whether this source was cryptographically verified via Crossref/OpenAlex/arXiv.",
    )


class PerspectivePersona(BaseModel):
    """STORM-style perspective persona to direct multi-hop literature inquiry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Persona name, e.g. 'Theoretical Formalist'.")
    role_description: str = Field(
        description="Detailed background, priorities, and skepticism level of this persona."
    )
    focus_questions: list[str] = Field(
        default_factory=list,
        description="Orthogonal research questions explored by this persona.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Concrete academic search queries formulated by this persona.",
    )


class ContradictionEntry(BaseModel):
    """Structured divergence analysis between competing publications or claims."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(description="Sub-topic or phenomenon exhibiting conflicting findings.")
    source_a_key: str = Field(description="cite_key of first source.")
    claim_a: str = Field(description="Concrete claim or reported empirical metric from source A.")
    source_b_key: str = Field(description="cite_key of second source.")
    claim_b: str = Field(description="Concrete claim or reported empirical metric from source B.")
    confounder_analysis: str = Field(
        description="Confounder explanation resolving why they diverge (dataset, compute scale, assumptions)."
    )


class EmpiricalExperimentResult(BaseModel):
    """Reproducible sandbox verification experiment outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis: str = Field(description="Empirical or algorithmic hypothesis being tested.")
    script_path: str = Field(
        description="Relative path to benchmark script inside .forge/research/experiments/."
    )
    exit_code: int = Field(description="Process exit code (0 indicates clean execution).")
    metrics_table_markdown: str = Field(
        description="Markdown table summarizing measured runtimes, memory, or metric distributions."
    )
    raw_stdout: str = Field(default="", description="Sanitized stdout from the benchmark process.")


class ResearchLedger(BaseModel):
    """Immutable ledger of evidence, perspectives, and verified claims for a research manuscript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(description="Primary research topic or inquiry prompt.")
    perspectives: list[PerspectivePersona] = Field(
        default_factory=list,
        description="Active STORM inquiry perspectives.",
    )
    evidence_map: dict[str, EvidenceNode] = Field(
        default_factory=dict,
        description="Map of cite_key -> EvidenceNode for all harvested evidence.",
    )
    contradictions: list[ContradictionEntry] = Field(
        default_factory=list,
        description="Identified contradictions and confounders.",
    )
    experiments: list[EmpiricalExperimentResult] = Field(
        default_factory=list,
        description="Empirical sandbox benchmarks executed during research.",
    )
