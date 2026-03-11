"""Tests for web search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


class TestWebSearch:
    """Tests for the web_search tool."""

    @patch("agent.tools.builtins.web_search._search_duckduckgo")
    async def test_search_returns_formatted_results(
        self, mock_ddg: AsyncMock
    ) -> None:
        from agent.tools.builtins.web_search import web_search

        mock_ddg.return_value = [
            {
                "title": "Python Tutorial",
                "href": "https://python.org/tutorial",
                "body": "Learn Python programming step by step.",
            },
            {
                "title": "Real Python",
                "href": "https://realpython.com",
                "body": "Python tutorials, articles, and resources.",
            },
        ]

        result = await web_search(query="python tutorial")

        assert "Python Tutorial" in result
        assert "https://python.org/tutorial" in result
        assert "Real Python" in result
        assert "1." in result
        assert "2." in result

    @patch("agent.tools.builtins.web_search._search_duckduckgo")
    async def test_search_empty_results(self, mock_ddg: AsyncMock) -> None:
        from agent.tools.builtins.web_search import web_search

        mock_ddg.return_value = []

        result = await web_search(query="xyznonexistentquery123")

        assert "No results found" in result

    @patch("agent.tools.builtins.web_search._search_duckduckgo")
    async def test_search_backend_failure(self, mock_ddg: AsyncMock) -> None:
        from agent.tools.builtins.web_search import web_search

        mock_ddg.side_effect = Exception("Network error")

        result = await web_search(query="test query")

        assert "ERROR" in result
        assert "Network error" in result
        assert "browser_navigate" in result  # Suggests fallback

    @patch("agent.tools.builtins.web_search._search_duckduckgo")
    async def test_search_respects_num_results(self, mock_ddg: AsyncMock) -> None:
        from agent.tools.builtins.web_search import web_search

        mock_ddg.return_value = [
            {"title": f"Result {i}", "href": f"https://example.com/{i}", "body": f"Snippet {i}"}
            for i in range(3)
        ]

        await web_search(query="test", num_results=3)

        mock_ddg.assert_called_once_with("test", 3)

    @patch("agent.tools.builtins.web_search._search_duckduckgo")
    async def test_search_clamps_num_results(self, mock_ddg: AsyncMock) -> None:
        from agent.tools.builtins.web_search import web_search

        mock_ddg.return_value = []

        # num_results > 10 should be clamped to 10
        await web_search(query="test", num_results=50)
        mock_ddg.assert_called_with("test", 10)

        # num_results < 1 should be clamped to 1
        await web_search(query="test", num_results=-5)
        mock_ddg.assert_called_with("test", 1)


class TestFormatResults:
    """Tests for result formatting."""

    def test_format_empty(self) -> None:
        from agent.tools.builtins.web_search import _format_results

        result = _format_results([])
        assert "No results found" in result

    def test_format_multiple_results(self) -> None:
        from agent.tools.builtins.web_search import _format_results

        results = [
            {"title": "Title A", "href": "https://a.com", "body": "Snippet A"},
            {"title": "Title B", "href": "https://b.com", "body": "Snippet B"},
        ]

        formatted = _format_results(results)
        assert "1. [Title A](https://a.com)" in formatted
        assert "2. [Title B](https://b.com)" in formatted
        assert "Snippet A" in formatted
        assert "Snippet B" in formatted
