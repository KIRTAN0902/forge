"""Context engine: role-specific, budgeted repository context (spec §13).

Never send the whole repository to every agent. The analyzer builds one picture of the
codebase; the selector picks the files relevant to a request; the budget manager sizes
the context to each model's real window; the builder assembles the Planner, Executor and
QA contexts.
"""

from forge.context.analyzer import (
    FileRecord,
    GitInfo,
    RepositoryAnalysis,
    RepositoryAnalyzer,
    TestSetup,
)
from forge.context.budget import (
    ContextBudget,
    ContextBudgetManager,
    PackedSection,
    PackResult,
    Section,
    pack,
)
from forge.context.builder import BuiltContext, ContextBuilder
from forge.context.history import compact_history
from forge.context.security import SecurityHit, is_sensitive_path, scan_files, scan_text
from forge.context.selector import ContextSelector, FileCandidate, query_terms
from forge.context.tokens import estimate_message_tokens, estimate_tokens

__all__ = [
    "BuiltContext",
    "ContextBudget",
    "ContextBudgetManager",
    "ContextBuilder",
    "ContextSelector",
    "FileCandidate",
    "FileRecord",
    "GitInfo",
    "PackResult",
    "PackedSection",
    "RepositoryAnalysis",
    "RepositoryAnalyzer",
    "Section",
    "SecurityHit",
    "TestSetup",
    "compact_history",
    "estimate_message_tokens",
    "estimate_tokens",
    "is_sensitive_path",
    "pack",
    "query_terms",
    "scan_files",
    "scan_text",
]
