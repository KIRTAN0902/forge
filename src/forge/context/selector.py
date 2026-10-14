"""Choosing which files belong in a role's context.

Signals, strongest first:

1. **Explicit seeds**: files the plan, a QA finding or the user named.
2. **Mentions**: paths that appear in error output, stack traces or QA evidence.
3. **Changed files**: what this workflow has already touched.
4. **Relevance to the request**: query terms in the path, in the file's symbol names,
   and in its content (TF-IDF, so common words count less).
5. **Structure**: import neighbours of strong candidates, and the tests for them.

Test files mention the domain vocabulary as heavily as the code they test, so their
relevance is down-weighted: they should accompany the implementation, not outrank it.
Structural boosts are computed from a snapshot of the direct scores, so two files that
import each other cannot inflate each other in a loop.

Every candidate carries the reasons it was chosen, so a context can explain itself.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from forge.context.analyzer import RepositoryAnalysis
from forge.tools.analysis import is_test_path, split_identifier

_STOPWORDS = frozenset(
    """
    the and for with that this from into when then than should must would could have has
    are was were will not all any its our your their use using make add new fix bug issue
    error test tests code file files function implement support allow ensure able can get
    set data also only each every via per but does doesn don which what there here them
    """.split()
    # Programming keywords: requirements say "returns X", but every file contains `return`.
    + """
    return returns class def import self none null true false const let var async await
    print public private static void int str string bool list dict type value values
    """.split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_PATHISH = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,8}")
MAX_CONTENT_SCAN_BYTES = 300_000
TEST_RELEVANCE_WEIGHT = 0.5
# A large budget is no reason to include weak matches: candidates scoring below this
# fraction of the best candidate are dropped.
RELATIVE_FLOOR = 0.2

_Bump = Callable[[str, float, str], None]


def query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in _WORD.findall(text):
        parts = split_identifier(word) or [word.lower()]
        for part in [*parts, word.lower()] if len(parts) > 1 else parts:
            term = _stem(part)
            if len(term) >= 3 and term not in _STOPWORDS and term not in terms:
                terms.append(term)
    return terms


def _stem(word: str) -> str:
    """A deliberately small stemmer: plural and verb endings only.

    ``invoices`` and ``invoice`` must map to the same term, ``classes`` to ``class``.
    """
    word = word.lower()
    if len(word) <= 4:
        return word
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    if word.endswith("ed") and len(word) > 5:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


@dataclass
class FileCandidate:
    path: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, score: float, reason: str) -> None:
        self.score += score
        if reason not in self.reasons:
            self.reasons.append(reason)


class ContextSelector:
    def __init__(self, analysis: RepositoryAnalysis) -> None:
        self.analysis = analysis

    def select(
        self,
        query: str,
        *,
        seeds: Iterable[str] = (),
        mentions: Iterable[str] = (),
        changed: Iterable[str] = (),
        limit: int = 12,
        include_tests: bool = True,
        min_score: float = 1.5,
    ) -> list[FileCandidate]:
        analysis = self.analysis
        eligible = {
            f.path
            for f in analysis.files
            if not f.is_generated
            and not analysis.workspace.is_secret(f.path)
            and (f.is_code or f.language in {"markdown", "yaml", "toml", "json", "sql", "html"})
        }
        candidates: dict[str, FileCandidate] = {}

        def bump(path: str, score: float, reason: str) -> None:
            candidates.setdefault(path, FileCandidate(path)).add(score, reason)

        for path in seeds:
            normalized = _match_path(path, eligible | set(analysis.paths))
            if normalized:
                bump(normalized, 12.0, "explicitly referenced")
        for path in _mentioned_paths(mentions, eligible):
            bump(path, 9.0, "mentioned in errors/findings")
        for path in changed:
            normalized = _match_path(path, eligible)
            if normalized:
                bump(normalized, 6.0, "changed in this workflow")

        terms = query_terms(query)
        if terms:
            self._score_relevance(terms, eligible, bump)

        self._expand_structure(candidates, eligible, bump, include_tests)

        ranked = sorted(candidates.values(), key=lambda c: (-c.score, c.path))
        if not include_tests:
            ranked = [c for c in ranked if not is_test_path(c.path) or c.score >= 12.0]
        floor = max(min_score, ranked[0].score * RELATIVE_FLOOR) if ranked else min_score
        return [c for c in ranked if c.score >= floor][:limit]

    def _score_relevance(self, terms: list[str], eligible: set[str], bump: _Bump) -> None:
        analysis = self.analysis
        symbol_tokens: dict[str, set[str]] = defaultdict(set)
        for symbol in analysis.symbols.all():
            if symbol.path in eligible:
                symbol_tokens[symbol.path].update(_stem(t) for t in split_identifier(symbol.name))
                symbol_tokens[symbol.path].add(symbol.name.lower())

        contents: dict[str, str] = {}
        for record in analysis.files:
            if record.path in eligible and record.size <= MAX_CONTENT_SCAN_BYTES:
                text = analysis.read(record.path)
                if text is not None:
                    contents[record.path] = text.lower()
        doc_count = max(len(contents), 1)
        document_frequency = {t: sum(1 for c in contents.values() if t in c) for t in terms}

        for path in eligible:
            weight = TEST_RELEVANCE_WEIGHT if is_test_path(path) else 1.0
            pure = PurePosixPath(path)
            dir_tokens = {_stem(t) for part in pure.parts[:-1] for t in split_identifier(part)}
            name_tokens = {_stem(t) for t in split_identifier(pure.stem)}
            in_name = [t for t in terms if t in name_tokens]
            in_dirs = [t for t in terms if t in dir_tokens and t not in name_tokens]
            if in_name or in_dirs:
                # A file *named* after the concept beats one that merely lives in a
                # directory named after it (every file under tools/ "matches" tool).
                score = (4.0 * len(in_name) + 1.5 * len(in_dirs)) * weight
                bump(path, score, f"path matches {', '.join([*in_name, *in_dirs])}")
            matched_symbols = [t for t in terms if t in symbol_tokens.get(path, ())]
            if matched_symbols:
                score = min(2.5 * len(matched_symbols), 7.5) * weight
                bump(path, score, f"defines {', '.join(matched_symbols)}")
            text = contents.get(path)
            if text:
                score = 0.0
                hits = []
                for term in terms:
                    tf = text.count(term)
                    if tf:
                        idf = math.log((doc_count + 1) / (document_frequency[term] + 0.5))
                        score += (1 + math.log(min(tf, 30))) * max(idf, 0.1) * 0.8
                        hits.append(term)
                if score > 0:
                    bump(path, min(score, 8.0) * weight, f"mentions {', '.join(hits[:4])}")

    def _expand_structure(
        self,
        candidates: dict[str, FileCandidate],
        eligible: set[str],
        bump: _Bump,
        include_tests: bool,
    ) -> None:
        analysis = self.analysis
        base = {path: c.score for path, c in candidates.items()}
        strongest = sorted(base.items(), key=lambda kv: -kv[1])[:6]
        # Each file keeps only its single strongest structural link; summing them would
        # let hub modules (imported everywhere) outrank the files that matter.
        best: dict[str, tuple[float, str]] = {}

        def offer(target: str, score: float, reason: str) -> None:
            if score > best.get(target, (0.0, ""))[0]:
                best[target] = (score, reason)

        for path, score in strongest:
            if score < 3 or is_test_path(path):
                continue  # tests importing code is not evidence the code is relevant
            name = PurePosixPath(path).name
            for neighbor in analysis.imports.imports.get(path, set()):
                if neighbor in eligible and not neighbor.endswith("__init__.py"):
                    offer(neighbor, score * 0.25, f"imported by {name}")
            for neighbor in analysis.imports.importers.get(path, set()):
                if neighbor in eligible and not is_test_path(neighbor):
                    offer(neighbor, score * 0.2, f"imports {name}")
            if include_tests:
                for test in analysis.tests_for(path):
                    if test in eligible:
                        offer(test, score * 0.3, f"tests {name}")
        for target, (score, reason) in best.items():
            bump(target, score, reason)


def _match_path(path: str, known: set[str]) -> str | None:
    normalized = path.replace("\\", "/").strip().lstrip("./")
    if normalized in known:
        return normalized
    matches = [k for k in known if k.endswith("/" + normalized)]
    return matches[0] if len(matches) == 1 else None


def _mentioned_paths(texts: Iterable[str], known: set[str]) -> list[str]:
    found: list[str] = []
    for text in texts:
        for token in _PATHISH.findall(text):
            match = _match_path(token, known)
            if match and match not in found:
                found.append(match)
    return found
