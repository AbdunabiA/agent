"""Tests for browser tools (Playwright-based)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


def _mock_page() -> AsyncMock:
    """Create a mock Playwright page with common methods."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.is_closed.return_value = False
    page.title = AsyncMock(return_value="Example Domain")
    page.goto = AsyncMock()
    page.evaluate = AsyncMock(return_value="Example text content on the page.")
    page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n\x1a\nfake_image_data")
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.close = AsyncMock()
    return page


def _mock_element(text: str = "Hello", attr_val: str = "https://example.com") -> AsyncMock:
    """Create a mock Playwright element."""
    el = AsyncMock()
    el.inner_text = AsyncMock(return_value=text)
    el.get_attribute = AsyncMock(return_value=attr_val)
    return el


class TestBrowserNavigate:
    """Tests for browser_navigate tool."""

    @patch("agent.tools.builtins.browser._get_page")
    async def test_navigate_returns_title_and_content(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_navigate

        page = _mock_page()
        mock_get_page.return_value = page

        result = await browser_navigate(url="https://example.com")

        page.goto.assert_called_once_with(
            "https://example.com", wait_until="load", timeout=30000
        )
        assert "Title: Example Domain" in result
        assert "URL: https://example.com" in result
        assert "Example text content" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_navigate_handles_error(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_navigate

        page = _mock_page()
        page.goto.side_effect = Exception("Connection refused")
        mock_get_page.return_value = page

        result = await browser_navigate(url="https://invalid.example.com")

        assert "[ERROR]" in result
        assert "Connection refused" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_navigate_custom_wait(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_navigate

        page = _mock_page()
        mock_get_page.return_value = page

        await browser_navigate(url="https://example.com", wait_for="networkidle")

        page.goto.assert_called_once_with(
            "https://example.com", wait_until="networkidle", timeout=30000
        )


class TestBrowserScreenshot:
    """Tests for browser_screenshot tool."""

    @patch("agent.tools.builtins.browser._get_page")
    async def test_screenshot_returns_metadata(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_screenshot

        page = _mock_page()
        mock_get_page.return_value = page

        result = await browser_screenshot()

        page.screenshot.assert_called_once_with(full_page=False, type="png")
        assert "Screenshot captured" in result
        assert "Example Domain" in result
        assert "base64 PNG" in result.lower() or "Base64" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_screenshot_full_page(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_screenshot

        page = _mock_page()
        mock_get_page.return_value = page

        await browser_screenshot(full_page=True)

        page.screenshot.assert_called_once_with(full_page=True, type="png")


class TestBrowserClick:
    """Tests for browser_click tool."""

    @patch("agent.tools.builtins.browser._get_page")
    async def test_click_element(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_click

        page = _mock_page()
        mock_get_page.return_value = page

        result = await browser_click(selector="#submit-btn")

        page.click.assert_called_once_with("#submit-btn", timeout=5000)
        assert "Clicked" in result
        assert "#submit-btn" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_click_failure(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_click

        page = _mock_page()
        page.click.side_effect = Exception("Element not found")
        mock_get_page.return_value = page

        result = await browser_click(selector="#nonexistent")

        assert "[ERROR]" in result
        assert "Element not found" in result


class TestBrowserFill:
    """Tests for browser_fill tool."""

    @patch("agent.tools.builtins.browser._get_page")
    async def test_fill_input(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_fill

        page = _mock_page()
        mock_get_page.return_value = page

        result = await browser_fill(selector="#email", value="test@example.com")

        page.fill.assert_called_once_with("#email", "test@example.com", timeout=5000)
        assert "Filled" in result
        assert "#email" in result
        assert "test@example.com" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_fill_truncates_long_value(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_fill

        page = _mock_page()
        mock_get_page.return_value = page

        long_value = "x" * 100
        result = await browser_fill(selector="#input", value=long_value)

        assert "..." in result  # Value display is truncated

    @patch("agent.tools.builtins.browser._get_page")
    async def test_fill_failure(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_fill

        page = _mock_page()
        page.fill.side_effect = Exception("Not an input element")
        mock_get_page.return_value = page

        result = await browser_fill(selector="#div", value="text")

        assert "[ERROR]" in result


class TestBrowserExtract:
    """Tests for browser_extract tool."""

    @patch("agent.tools.builtins.browser._get_page")
    async def test_extract_text(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_extract

        page = _mock_page()
        page.query_selector_all = AsyncMock(return_value=[
            _mock_element("Item 1"),
            _mock_element("Item 2"),
        ])
        mock_get_page.return_value = page

        result = await browser_extract(selector="li")

        assert "Found 2 elements" in result
        assert "Item 1" in result
        assert "Item 2" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_extract_attribute(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_extract

        page = _mock_page()
        page.query_selector_all = AsyncMock(return_value=[
            _mock_element(attr_val="https://link1.com"),
            _mock_element(attr_val="https://link2.com"),
        ])
        mock_get_page.return_value = page

        result = await browser_extract(selector="a", attribute="href")

        assert "https://link1.com" in result
        assert "https://link2.com" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_extract_no_elements(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_extract

        page = _mock_page()
        page.query_selector_all = AsyncMock(return_value=[])
        mock_get_page.return_value = page

        result = await browser_extract(selector=".nonexistent")

        assert "No elements found" in result

    @patch("agent.tools.builtins.browser._get_page")
    async def test_extract_failure(self, mock_get_page: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_extract

        page = _mock_page()
        page.query_selector_all = AsyncMock(side_effect=Exception("Page crashed"))
        mock_get_page.return_value = page

        result = await browser_extract(selector="div")

        assert "[ERROR]" in result


class TestBrowserClose:
    """Tests for browser_close tool."""

    @patch("agent.tools.builtins.browser.cleanup_browser")
    async def test_close_browser(self, mock_cleanup: AsyncMock) -> None:
        from agent.tools.builtins.browser import browser_close

        result = await browser_close()

        mock_cleanup.assert_called_once()
        assert "closed" in result.lower()


class TestBrowserLazyInit:
    """Tests for lazy browser initialization."""

    @patch("agent.tools.builtins.browser.async_playwright")
    async def test_get_page_creates_browser(self, mock_pw: MagicMock) -> None:
        import agent.tools.builtins.browser as browser_mod

        # Reset module state
        browser_mod._browser = None
        browser_mod._context = None
        browser_mod._page = None
        browser_mod._pw_instance = None

        mock_pw_instance = AsyncMock()
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        mock_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)
        mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        page = await browser_mod._get_page()

        assert page is mock_page
        mock_pw_instance.chromium.launch.assert_called_once()
        mock_browser.new_context.assert_called_once()
        mock_context.new_page.assert_called_once()

        # Cleanup module state
        browser_mod._browser = None
        browser_mod._context = None
        browser_mod._page = None
        browser_mod._pw_instance = None
