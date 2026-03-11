"""Web search tool using DuckDuckGo as the default backend.

Supports multiple backends:
1. DuckDuckGo (via duckduckgo-search package, no API key needed) — default
2. SearXNG (self-hosted, if configured)
3. Google Custom Search (requires API key)
4. Tavily (requires API key)

The agent uses whichever is configured in agent.yaml under tools.web_search.
"""

from __future__ import annotations

import structlog

from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)


async def _search_duckduckgo(query: str, num_results: int) -> list[dict[str, str]]:
    """Search using DuckDuckGo (no API key needed).

    Returns:
        List of dicts with "title", "href", "body" keys.
    """
    import asyncio

    from duckduckgo_search import DDGS

    def _sync_search() -> list[dict[str, str]]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=num_results))

    results = await asyncio.to_thread(_sync_search)
    return results


def _format_results(results: list[dict[str, str]]) -> str:
    """Format search results into readable text."""
    if not results:
        return (
            "[No results found]\n\n"
            "Try rephrasing your query or using different keywords."
        )

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        href = r.get("href", r.get("url", ""))
        body = r.get("body", r.get("snippet", ""))
        lines.append(f"{i}. [{title}]({href})\n   {body}")

    return "\n\n".join(lines)


@tool(
    name="web_search",
    description=(
        "Search the web for current information. Returns a list of results with "
        "titles, URLs, and snippets. Use this to find recent news, look up "
        "documentation, research topics, verify facts, or find information "
        "not in your training data."
    ),
    tier=ToolTier.SAFE,
)
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web.

    Args:
        query: Search query string.
        num_results: Number of results to return (1-10, default 5).
    """
    num_results = max(1, min(10, num_results))

    logger.info("web_search_started", query=query, num_results=num_results)

    try:
        results = await _search_duckduckgo(query, num_results)
    except Exception as e:
        logger.error("web_search_failed", query=query, error=str(e))
        return (
            f"[ERROR] Web search failed: {e}\n\n"
            "Suggestion: Try using browser_navigate to visit a specific URL directly."
        )

    formatted = _format_results(results)
    logger.info("web_search_complete", query=query, result_count=len(results))
    return formatted
