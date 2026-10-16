"""Forge Publication-Grade Autonomous Deep Research Subsystem."""

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
    ContradictionAnalyzer,
    EmpiricalLabRunner,
    ZeroTrustCitationAuditor,
)

__all__ = [
    "AcademicHarvester",
    "ContradictionAnalyzer",
    "ContradictionEntry",
    "EmpiricalExperimentResult",
    "EmpiricalLabRunner",
    "EvidenceNode",
    "ManuscriptCompiler",
    "PerspectivePersona",
    "ResearchLedger",
    "STORMInquiryEngine",
    "SourceTier",
    "ZeroTrustCitationAuditor",
    "build_cite_key",
    "generate_bibtex_entry",
]
