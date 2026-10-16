"""Web search and URL retrieval tools for Forge agents."""

from __future__ import annotations

import html
import os
import re
import urllib.parse
from html.parser import HTMLParser
from typing import ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from forge.policies.types import Permission, RiskLevel
from forge.tools.base import Tool, ToolInput, ToolOutput
from forge.tools.environment import ToolEnvironment

# ==============================================================================
# HTML Result Parser for DuckDuckGo
# ==============================================================================


class DuckDuckGoHTMLParser(HTMLParser):
    """Parses search result titles, links, and snippets from DuckDuckGo HTML pages."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current_result: dict[str, str] = {}
        self._in_title = False
        self._in_snippet = False
        self._title_buffer: list[str] = []
        self._snippet_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v or "" for k, v in attrs}
        classes = attr_dict.get("class", "").split()

        # Check for result title link
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            href = attr_dict.get("href", "")
            # Unwrap DuckDuckGo redirect: //duckduckgo.com/l/?uddg=<actual_url>&...
            if "uddg=" in href:
                match = re.search(r"uddg=([^&]+)", href)
                if match:
                    href = urllib.parse.unquote(match.group(1))
            self._current_result = {"url": href, "title": "", "snippet": ""}
            self._in_title = True
            self._title_buffer = []

        # Check for result snippet
        elif "result__snippet" in classes or "result-snippet" in classes:
            self._in_snippet = True
            self._snippet_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            raw_title = "".join(self._title_buffer).strip()
            self._current_result["title"] = html.unescape(raw_title)

        elif self._in_snippet and tag in {"a", "td", "div", "span"}:
            self._in_snippet = False
            raw_snip = "".join(self._snippet_buffer).strip()
            self._current_result["snippet"] = html.unescape(raw_snip)
            if self._current_result.get("url") and self._current_result.get("title"):
                self.results.append(dict(self._current_result))
                self._current_result = {}

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buffer.append(data)
        elif self._in_snippet:
            self._snippet_buffer.append(data)


# ==============================================================================
# Web Search Tool
# ==============================================================================


class WebSearchResultItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    url: str
    snippet: str


class WebSearchInput(ToolInput):
    query: str = Field(description="Search keywords or technical question to look up on the web.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to return.",
    )


class WebSearchOutput(ToolOutput):
    query: str
    results: list[WebSearchResultItem] = Field(default_factory=list)
    source: str = "web"

    def render(self) -> str:
        if not self.results:
            return f"No search results found for query: '{self.query}'."

        lines = [f"## Web Search Results for '{self.query}' (via {self.source}):\n"]
        for i, item in enumerate(self.results, start=1):
            lines.append(f"{i}. [{item.title}]({item.url})")
            lines.append(f"   {item.snippet}\n")
        return "\n".join(lines)


class WebSearch(Tool[WebSearchInput, WebSearchOutput]):
    """Searches the web for documentation, API references, library usage, and troubleshooting."""

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web for up-to-date technical documentation, API specifications, "
        "library examples, bug workarounds, and error solutions."
    )
    input_model = WebSearchInput
    output_model = WebSearchOutput
    risk: ClassVar[RiskLevel] = RiskLevel.LOW
    permissions: ClassVar[frozenset[Permission]] = frozenset({Permission.READ})
    timeout_s: ClassVar[float] = 30.0

    async def _search_tavily(
        self, query: str, max_results: int, api_key: str
    ) -> list[WebSearchResultItem]:
        url = "https://api.tavily.com/search"
        payload = {"api_key": api_key, "query": query, "max_results": max_results}
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            items = []
            for r in data.get("results", []):
                items.append(
                    WebSearchResultItem(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                    )
                )
            return items

    async def _search_brave(
        self, query: str, max_results: int, api_key: str
    ) -> list[WebSearchResultItem]:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
        params = {"q": query, "count": str(max_results)}
        async with httpx.AsyncClient(headers=headers, timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            items = []
            for r in data.get("web", {}).get("results", []):
                items.append(
                    WebSearchResultItem(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("description", ""),
                    )
                )
            return items

    async def _search_duckduckgo(self, query: str, max_results: int) -> list[WebSearchResultItem]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        async with httpx.AsyncClient(
            headers=headers, timeout=15.0, follow_redirects=True
        ) as client:
            # Query DuckDuckGo Lite endpoint
            resp = await client.post("https://lite.duckduckgo.com/lite/", data={"q": query})
            if resp.status_code != 200:
                resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})

            if resp.status_code != 200:
                return []

            parser = DuckDuckGoHTMLParser()
            parser.feed(resp.text)

            items: list[WebSearchResultItem] = []
            for r in parser.results[:max_results]:
                title = r.get("title", "").strip()
                url = r.get("url", "").strip()
                snippet = r.get("snippet", "").strip()
                if title and url:
                    items.append(WebSearchResultItem(title=title, url=url, snippet=snippet))
            return items

    async def run(self, args: WebSearchInput, env: ToolEnvironment) -> WebSearchOutput:
        query = args.query.strip()
        max_results = args.max_results

        # 1. Check for Tavily API Key
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            try:
                results = await self._search_tavily(query, max_results, tavily_key)
                if results:
                    return WebSearchOutput(query=query, results=results, source="Tavily")
            except Exception:
                pass

        # 2. Check for Brave Search API Key
        brave_key = os.environ.get("BRAVE_API_KEY")
        if brave_key:
            try:
                results = await self._search_brave(query, max_results, brave_key)
                if results:
                    return WebSearchOutput(query=query, results=results, source="Brave Search")
            except Exception:
                pass

        # 3. Default zero-config fallback: DuckDuckGo
        try:
            results = await self._search_duckduckgo(query, max_results)
            return WebSearchOutput(query=query, results=results, source="DuckDuckGo")
        except Exception as e:
            return WebSearchOutput(
                query=query,
                results=[],
                source=f"Error: {e}",
            )


# ==============================================================================
# Fetch Web Page Tool
# ==============================================================================


class FetchWebPageInput(ToolInput):
    url: str = Field(
        description="The full HTTP/HTTPS URL of the documentation or web page to fetch."
    )
    max_chars: int = Field(
        default=6000,
        ge=200,
        le=50000,
        description="Maximum characters of text to extract from the page.",
    )


class FetchWebPageOutput(ToolOutput):
    url: str
    status_code: int
    title: str = ""
    content: str = ""

    def render(self) -> str:
        header = f"## {self.title or self.url} (Status: {self.status_code})\n\n"
        return header + self.content


class FetchWebPage(Tool[FetchWebPageInput, FetchWebPageOutput]):
    """Fetches a web page or online documentation and extracts readable markdown text."""

    name: ClassVar[str] = "fetch_web_page"
    description: ClassVar[str] = (
        "Fetch and extract clean readable text from an online documentation page, "
        "API reference, or GitHub repository URL."
    )
    input_model = FetchWebPageInput
    output_model = FetchWebPageOutput
    risk: ClassVar[RiskLevel] = RiskLevel.LOW
    permissions: ClassVar[frozenset[Permission]] = frozenset({Permission.READ})
    timeout_s: ClassVar[float] = 30.0

    @staticmethod
    def _clean_html(raw_html: str, max_chars: int) -> tuple[str, str]:
        """Extract page title and strip scripts/styles/tags to plain text."""
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
        title = html.unescape(title_match.group(1)).strip() if title_match else ""

        # Remove scripts, styles, head
        cleaned = re.sub(
            r"<(script|style|head|noscript)[^>]*>[\s\S]*?</\1>", "", raw_html, flags=re.IGNORECASE
        )
        # Convert links to markdown
        cleaned = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            r"[\2](\1)",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Convert headers and paragraph breaks
        cleaned = re.sub(r"<(h[1-6]|p|div|br|li)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
        # Strip remaining tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Unescape HTML entities
        text = html.unescape(cleaned)
        # Normalize whitespace and blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated to {max_chars} characters]"

        return title, text

    async def run(self, args: FetchWebPageInput, env: ToolEnvironment) -> FetchWebPageOutput:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
        }
        try:
            async with httpx.AsyncClient(
                headers=headers, timeout=20.0, follow_redirects=True
            ) as client:
                resp = await client.get(args.url)
                title, content = self._clean_html(resp.text, args.max_chars)
                return FetchWebPageOutput(
                    url=str(resp.url),
                    status_code=resp.status_code,
                    title=title,
                    content=content,
                )
        except Exception as e:
            return FetchWebPageOutput(
                url=args.url,
                status_code=500,
                title="Error Fetching URL",
                content=f"Failed to fetch {args.url}: {e}",
            )
