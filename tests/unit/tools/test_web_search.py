"""Tests for WebSearch and FetchWebPage tools."""

from __future__ import annotations

import pytest

from forge.policies.types import Permission, RiskLevel
from forge.tools import READ_ONLY_TOOLS, ROLE_TOOLSETS, default_tools
from forge.tools.web_search import (
    DuckDuckGoHTMLParser,
    FetchWebPage,
    WebSearch,
    WebSearchOutput,
    WebSearchResultItem,
)


def test_web_search_metadata_and_registration() -> None:
    tool = WebSearch()
    assert tool.name == "web_search"
    assert tool.risk == RiskLevel.LOW
    assert tool.permissions == frozenset({Permission.READ})

    fetch_tool = FetchWebPage()
    assert fetch_tool.name == "fetch_web_page"
    assert fetch_tool.risk == RiskLevel.LOW
    assert fetch_tool.permissions == frozenset({Permission.READ})

    all_tool_names = [t.name for t in default_tools()]
    assert "web_search" in all_tool_names
    assert "fetch_web_page" in all_tool_names

    # Check read-only and role toolsets
    assert "web_search" in READ_ONLY_TOOLS
    assert "fetch_web_page" in READ_ONLY_TOOLS
    assert "web_search" in ROLE_TOOLSETS["orchestrator"]
    assert "web_search" in ROLE_TOOLSETS["executor"]
    assert "web_search" in ROLE_TOOLSETS["qa"]


def test_web_search_output_render() -> None:
    output = WebSearchOutput(
        query="python sys.monitoring",
        source="DuckDuckGo",
        results=[
            WebSearchResultItem(
                title="sys.monitoring — Execution event monitoring",
                url="https://docs.python.org/3/library/sys.monitoring.html",
                snippet="sys.monitoring is a namespace providing event monitoring.",
            )
        ],
    )
    rendered = output.render()
    assert "## Web Search Results for 'python sys.monitoring'" in rendered
    assert "[sys.monitoring — Execution event monitoring](https://docs.python.org/3/library/sys.monitoring.html)" in rendered
    assert "sys.monitoring is a namespace" in rendered


def test_duckduckgo_html_parser() -> None:
    sample_html = """
    <!DOCTYPE html>
    <html>
    <body>
    <div class="result">
        <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&rut=1">Python 3 Documentation</a>
        </h2>
        <a class="result__snippet">Official Python 3 language and library reference documentation.</a>
    </div>
    </body>
    </html>
    """
    parser = DuckDuckGoHTMLParser()
    parser.feed(sample_html)
    assert len(parser.results) == 1
    assert parser.results[0]["title"] == "Python 3 Documentation"
    assert parser.results[0]["url"] == "https://docs.python.org/3/"
    assert "Official Python 3" in parser.results[0]["snippet"]


@pytest.mark.asyncio
async def test_fetch_web_page_html_cleaning() -> None:
    sample_html = """
    <html>
    <head><title>Test Documentation Page</title><script>alert('skip');</script></head>
    <body>
        <h1>API Guide</h1>
        <p>This is the first paragraph with a <a href="https://example.com/spec">link</a>.</p>
        <p>Second paragraph with details.</p>
    </body>
    </html>
    """
    title, text = FetchWebPage._clean_html(sample_html, max_chars=1000)
    assert title == "Test Documentation Page"
    assert "API Guide" in text
    assert "[link](https://example.com/spec)" in text
    assert "alert('skip')" not in text
