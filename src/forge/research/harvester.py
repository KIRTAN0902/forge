"""Multi-source academic harvester and citation snowballing engine."""

from __future__ import annotations

import html
import re
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from forge.research.models import EvidenceNode, PerspectivePersona, SourceTier

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "using",
    "towards",
    "scaling",
    "evaluating",
    "improving",
    "based",
}

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def build_cite_key(authors: list[str], year: int, title: str) -> str:
    """Generate deterministic citation key: firstauthor_year_firstword."""
    first_author = "anonymous"
    if authors:
        raw = authors[0].strip()
        if "," in raw:
            first_author = raw.split(",")[0].strip()
        else:
            parts = raw.split()
            first_author = parts[-1] if parts else "anonymous"

    author_clean = re.sub(r"[^a-zA-Z0-9]", "", first_author).lower() or "anonymous"

    words = re.findall(r"[a-zA-Z0-9]+", title.lower())
    keyword = "paper"
    for w in words:
        if w not in STOPWORDS and len(w) >= 3:
            keyword = w
            break

    return f"{author_clean}{year}{keyword}"


def generate_bibtex_entry(node: EvidenceNode) -> str:
    """Deterministically format valid BibTeX block directly from EvidenceNode metadata."""
    authors_str = " and ".join(node.authors) if node.authors else "Unknown Author"
    entry_type = "article" if node.venue_or_source else "misc"
    lines = [
        f"@{entry_type}{{{node.cite_key},",
        f"  title = {{{node.title}}},",
        f"  author = {{{authors_str}}},",
        f"  year = {{{node.year}}},",
    ]
    if node.venue_or_source:
        lines.append(f"  journal = {{{node.venue_or_source}}},")
    if node.doi_or_arxiv_id:
        if "10." in node.doi_or_arxiv_id:
            lines.append(f"  doi = {{{node.doi_or_arxiv_id}}},")
        else:
            lines.append(f"  eprint = {{{node.doi_or_arxiv_id}}},")
            lines.append("  archivePrefix = {arXiv},")
    if node.url:
        lines.append(f"  url = {{{node.url}}},")
    lines.append("}")
    return "\n".join(lines)


class AcademicHarvester:
    """Multi-source harvester querying arXiv, Semantic Scholar, OpenAlex, and Crossref."""

    def __init__(self, timeout_s: float = 15.0) -> None:
        self.timeout_s = timeout_s

    async def search_arxiv(self, query: str, max_results: int = 10) -> list[EvidenceNode]:
        """Query arXiv Atom XML API and build verified EvidenceNodes."""
        encoded_query = urllib.parse.quote(query)
        url = (
            f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}"
            f"&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return []
                xml_text = resp.text
        except Exception:
            return []

        nodes: list[EvidenceNode] = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ATOM_NS):
                title_elem = entry.find("atom:title", ATOM_NS)
                title = (
                    re.sub(r"\s+", " ", title_elem.text or "").strip()
                    if title_elem is not None
                    else "Untitled"
                )

                summary_elem = entry.find("atom:summary", ATOM_NS)
                summary = (
                    re.sub(r"\s+", " ", summary_elem.text or "").strip()
                    if summary_elem is not None
                    else ""
                )

                authors: list[str] = []
                for auth in entry.findall("atom:author", ATOM_NS):
                    name_elem = auth.find("atom:name", ATOM_NS)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                id_elem = entry.find("atom:id", ATOM_NS)
                arxiv_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else arxiv_url

                pub_elem = entry.find("atom:published", ATOM_NS)
                pub_year = 2024
                if pub_elem is not None and pub_elem.text and len(pub_elem.text) >= 4:
                    try:
                        pub_year = int(pub_elem.text[:4])
                    except ValueError:
                        pub_year = 2024

                cite_key = build_cite_key(authors, pub_year, title)

                # Extract initial numeric verbatim quotes from summary
                numeric_quotes: list[str] = []
                for sent in re.split(r"[.!?]\s+", summary):
                    if re.search(r"\b\d+(?:\.\d+)?%?", sent):
                        numeric_quotes.append(sent.strip() + ".")

                node = EvidenceNode(
                    cite_key=cite_key,
                    title=title,
                    authors=authors,
                    year=pub_year,
                    venue_or_source="arXiv preprint",
                    doi_or_arxiv_id=arxiv_id,
                    url=arxiv_url or f"https://arxiv.org/abs/{arxiv_id}",
                    citation_count=0,
                    source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
                    abstract_or_summary=summary,
                    verbatim_quotes=numeric_quotes[:5],
                    verified_in_registry=True,
                )
                nodes.append(node)
        except Exception:
            return []

        return nodes

    async def search_semantic_scholar_and_snowball(
        self, query: str, limit: int = 10
    ) -> list[EvidenceNode]:
        """Query Semantic Scholar Graph API with automatic OpenAlex fallback."""
        encoded_query = urllib.parse.quote(query)
        s2_url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}"
            f"&limit={limit}&fields=title,authors,year,venue,citationCount,externalIds,abstract,tldr"
        )
        nodes: list[EvidenceNode] = []
        s2_succeeded = False

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.get(s2_url)
                if resp.status_code == 200:
                    data = resp.json()
                    for paper in data.get("data", []):
                        title = paper.get("title", "").strip() or "Untitled"
                        year = paper.get("year") or 2024
                        authors = [
                            a.get("name", "") for a in paper.get("authors", []) if a.get("name")
                        ]
                        venue = paper.get("venue") or ""
                        citations = paper.get("citationCount") or 0
                        abstract = paper.get("abstract") or ""
                        tldr = (paper.get("tldr") or {}).get("text", "")
                        combined_summary = tldr or abstract
                        ext_ids = paper.get("externalIds") or {}
                        doi = ext_ids.get("DOI") or ext_ids.get("ArXiv") or ""

                        cite_key = build_cite_key(authors, year, title)
                        quotes: list[str] = []
                        if combined_summary:
                            for sent in re.split(r"[.!?]\s+", combined_summary):
                                if re.search(r"\b\d+(?:\.\d+)?%?", sent):
                                    quotes.append(sent.strip() + ".")

                        node = EvidenceNode(
                            cite_key=cite_key,
                            title=title,
                            authors=authors,
                            year=year,
                            venue_or_source=venue or "Academic Publication",
                            doi_or_arxiv_id=doi or None,
                            url=f"https://doi.org/{doi}"
                            if doi and "10." in doi
                            else (f"https://arxiv.org/abs/{doi}" if doi else ""),
                            citation_count=citations,
                            source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
                            abstract_or_summary=combined_summary,
                            verbatim_quotes=quotes[:5],
                            verified_in_registry=bool(doi),
                        )
                        nodes.append(node)
                    if nodes:
                        s2_succeeded = True
        except Exception:
            pass

        if not s2_succeeded:
            # Fallback to OpenAlex
            nodes = await self._search_openalex(query, limit)

        return nodes

    async def _search_openalex(self, query: str, limit: int = 10) -> list[EvidenceNode]:
        """Query OpenAlex Works API (zero-authentication, generous rate limits)."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.openalex.org/works?search={encoded_query}&per-page={limit}"
        nodes: list[EvidenceNode] = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return []
                data = resp.json()
                for work in data.get("results", []):
                    title = work.get("title") or "Untitled"
                    year = work.get("publication_year") or 2024
                    authors: list[str] = []
                    for auth in work.get("authorships", []):
                        raw_name = (auth.get("author") or {}).get("display_name")
                        if raw_name:
                            authors.append(raw_name)

                    venue = (work.get("primary_location") or {}).get("source", {}).get(
                        "display_name"
                    ) or ""
                    doi = work.get("doi") or ""
                    doi_clean = doi.replace("https://doi.org/", "") if doi else None
                    citations = work.get("cited_by_count") or 0

                    # Inverted index abstract to text
                    abstract_index = work.get("abstract_inverted_index") or {}
                    abstract = self._reconstruct_abstract(abstract_index)

                    cite_key = build_cite_key(authors, year, title)
                    quotes: list[str] = []
                    if abstract:
                        for sent in re.split(r"[.!?]\s+", abstract):
                            if re.search(r"\b\d+(?:\.\d+)?%?", sent):
                                quotes.append(sent.strip() + ".")

                    node = EvidenceNode(
                        cite_key=cite_key,
                        title=title,
                        authors=authors,
                        year=year,
                        venue_or_source=venue or "OpenAlex Index",
                        doi_or_arxiv_id=doi_clean,
                        url=doi or (work.get("id") or ""),
                        citation_count=citations,
                        source_tier=SourceTier.TIER_1_PRIMARY_PAPER,
                        abstract_or_summary=abstract,
                        verbatim_quotes=quotes[:5],
                        verified_in_registry=bool(doi_clean),
                    )
                    nodes.append(node)
        except Exception:
            return []
        return nodes

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
        """Reconstruct abstract from OpenAlex inverted index dictionary."""
        if not inverted_index:
            return ""
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)

    async def snowball_citations(
        self, seed_nodes: list[EvidenceNode], top_k: int = 5
    ) -> list[EvidenceNode]:
        """Perform backward (foundational references) and forward (recent citing works) snowballing."""
        if not seed_nodes:
            return []

        snowballed: dict[str, EvidenceNode] = {}
        existing_keys = {n.cite_key for n in seed_nodes}

        # 1. Backward snowballing: search foundational terms from top seed titles
        for seed in seed_nodes[:3]:
            # Extract main technical keywords from seed title
            keywords = [
                w for w in re.findall(r"[a-zA-Z]{4,}", seed.title) if w.lower() not in STOPWORDS
            ]
            if len(keywords) >= 2:
                sub_query = f"{keywords[0]} {keywords[1]} foundational architecture"
                foundational = await self.search_semantic_scholar_and_snowball(
                    sub_query, limit=top_k
                )
                for node in foundational:
                    if node.cite_key not in existing_keys and node.cite_key not in snowballed:
                        snowballed[node.cite_key] = node

        # 2. Forward snowballing: search recent 2025/2026 works citing or extending seed themes
        for seed in seed_nodes[:2]:
            sub_query = f"extending {seed.title[:40]} 2025 2026"
            recent = await self.search_arxiv(sub_query, max_results=top_k)
            for node in recent:
                if node.cite_key not in existing_keys and node.cite_key not in snowballed:
                    snowballed[node.cite_key] = node

        return list(snowballed.values())[:top_k]

    async def fetch_fulltext_sections(self, node: EvidenceNode) -> EvidenceNode:
        """Fetch HTML or extended text sections for arXiv papers to extract verbatim quotes."""
        if not node.doi_or_arxiv_id or "10." in node.doi_or_arxiv_id:
            return node

        arxiv_id = node.doi_or_arxiv_id.replace("abs/", "").replace("pdf/", "")
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        quotes = list(node.verbatim_quotes)

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(html_url)
                if resp.status_code == 200:
                    text = resp.text
                    clean_text = re.sub(r"<[^>]+>", " ", text)
                    clean_text = html.unescape(clean_text)
                    clean_text = re.sub(r"\s+", " ", clean_text)

                    # Extract numeric sentences with metrics
                    metric_matches = re.findall(
                        r"([^.\n]*?(?:accuracy|latency|f1|bleu|rouge|speedup|pass@1|overhead|reduction)\s*(?:of|is|=|:)?\s*\d+(?:\.\d+)?%?[^.\n]*?\.)",
                        clean_text,
                        re.IGNORECASE,
                    )
                    for match in metric_matches[:5]:
                        clean_m = match.strip()
                        if len(clean_m) > 15 and clean_m not in quotes:
                            quotes.append(clean_m)
        except Exception:
            pass

        return EvidenceNode(
            cite_key=node.cite_key,
            title=node.title,
            authors=node.authors,
            year=node.year,
            venue_or_source=node.venue_or_source,
            doi_or_arxiv_id=node.doi_or_arxiv_id,
            url=node.url,
            citation_count=node.citation_count,
            source_tier=node.source_tier,
            abstract_or_summary=node.abstract_or_summary,
            verbatim_quotes=quotes,
            experimental_setup=node.experimental_setup,
            verified_in_registry=node.verified_in_registry,
        )


class STORMInquiryEngine:
    """STORM-style multi-perspective inquiry and iterative saturation controller."""

    def __init__(self, harvester: AcademicHarvester | None = None) -> None:
        self.harvester = harvester or AcademicHarvester()

    @staticmethod
    def generate_orthogonal_personas(topic: str) -> list[PerspectivePersona]:
        """Synthesize 4 orthogonal domain specialist perspectives for the topic."""
        return [
            PerspectivePersona(
                name="Theoretical Formalist",
                role_description=(
                    "Analyzes algorithmic convergence, mathematical representations, state spaces, "
                    "complexity bounds, and proof assumptions."
                ),
                focus_questions=[
                    f"What formal mathematical guarantees or bounds underpin {topic}?",
                    "What are the asymptotic time and space complexities of the core abstractions?",
                ],
                search_queries=[
                    f"{topic} formal proofs convergence bounds",
                    f"{topic} computational complexity state space analysis",
                ],
            ),
            PerspectivePersona(
                name="Systems & Runtime Engineer",
                role_description=(
                    "Scrutinizes runtime efficiency, memory hierarchy bottlenecks, cache locality, "
                    "token budgets, concurrency, async IO, and execution overhead."
                ),
                focus_questions=[
                    f"What is the empirical latency and token overhead of {topic} at scale?",
                    "How does the system perform under resource-constrained local deployment?",
                ],
                search_queries=[
                    f"{topic} systems runtime overhead latency tokens",
                    f"{topic} high throughput local inference memory footprint",
                ],
            ),
            PerspectivePersona(
                name="Empirical Benchmark Auditor",
                role_description=(
                    "Investigates dataset leakage, test decontamination, metric robustness, "
                    "ablation completeness, and variance across stochastic seeds."
                ),
                focus_questions=[
                    f"Are benchmark improvements in {topic} statistically significant across seeds?",
                    "What ablation studies isolate the genuine source of empirical gains?",
                ],
                search_queries=[
                    f"{topic} benchmark evaluation decontamination ablation",
                    f"{topic} empirical reproduction variance statistical significance",
                ],
            ),
            PerspectivePersona(
                name="Adversarial Failure-Mode Critic",
                role_description=(
                    "Probes catastrophic edge cases, jailbreaks, distribution drift, infinite loops, "
                    "reward hacking, and silent failure propagation."
                ),
                focus_questions=[
                    f"Under what adverse distribution shifts does {topic} catastrophic fail?",
                    "Can autonomous feedback loops produce runaway degeneration or infinite retries?",
                ],
                search_queries=[
                    f"{topic} adversarial vulnerability failure modes bugs",
                    f"{topic} runaway loop hallucination drift degradation",
                ],
            ),
        ]

    async def execute_inquiry_waves(
        self,
        topic: str,
        personas: list[PerspectivePersona],
        max_waves: int = 3,
        saturation_threshold: float = 0.10,
    ) -> tuple[dict[str, EvidenceNode], list[float]]:
        """Run iterative retrieval waves, monitoring Information Gain Ratio until saturation."""
        collective_evidence: dict[str, EvidenceNode] = {}
        wave_gains: list[float] = []

        for wave_idx in range(1, max_waves + 1):
            initial_count = len(collective_evidence)
            new_keys_in_wave = 0

            for persona in personas:
                # Query top queries for this persona
                for query in persona.search_queries:
                    arxiv_papers = await self.harvester.search_arxiv(query, max_results=3)
                    for paper in arxiv_papers:
                        if paper.cite_key not in collective_evidence:
                            collective_evidence[paper.cite_key] = paper
                            new_keys_in_wave += 1

                    s2_papers = await self.harvester.search_semantic_scholar_and_snowball(
                        query, limit=3
                    )
                    for paper in s2_papers:
                        if paper.cite_key not in collective_evidence:
                            collective_evidence[paper.cite_key] = paper
                            new_keys_in_wave += 1

            # Perform snowballing on top 3 nodes if available
            if collective_evidence:
                snowballed = await self.harvester.snowball_citations(
                    list(collective_evidence.values())[:3], top_k=3
                )
                for node in snowballed:
                    if node.cite_key not in collective_evidence:
                        collective_evidence[node.cite_key] = node
                        new_keys_in_wave += 1

            total_after_wave = len(collective_evidence)
            gain_ratio = (new_keys_in_wave / total_after_wave) if total_after_wave > 0 else 0.0
            wave_gains.append(gain_ratio)

            # Check for saturation after wave 1
            if wave_idx >= 1 and gain_ratio < saturation_threshold and initial_count > 0:
                break

        return collective_evidence, wave_gains
