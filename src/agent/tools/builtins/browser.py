"""Browser control tools using Playwright.

The agent can browse the web, interact with pages, take screenshots,
and extract content. A single browser context is shared across tool calls
within a session for continuity (login sessions, multi-step workflows).
"""

from __future__ import annotations

import base64

import structlog
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)

# Module-level browser state (shared across calls within a session)
_browser: Browser | None = None
_context: BrowserContext | None = None
_page: Page | None = None
_pw_instance: object | None = None  # Playwright instance


async def _get_page() -> Page:
    """Get or create the browser page. Lazy initialization."""
    global _browser, _context, _page, _pw_instance

    if _page is None or _page.is_closed():
        if _browser is None:
            headless = True
            try:
                from agent.config import get_config
                headless = get_config().tools.browser.headless
            except Exception:
                pass
            _pw_instance = await async_playwright().start()
            _browser = await _pw_instance.chromium.launch(  # type: ignore[union-attr]
                headless=headless,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            logger.info("browser_launched", headless=headless)
        if _context is None:
            _context = await _browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Agent/1.0 (Autonomous AI Assistant)",
            )
        _page = await _context.new_page()

    return _page


async def cleanup_browser() -> None:
    """Close browser resources. Called on agent shutdown."""
    global _browser, _context, _page, _pw_instance

    if _page and not _page.is_closed():
        await _page.close()
    if _context:
        await _context.close()
    if _browser:
        await _browser.close()
    if _pw_instance:
        await _pw_instance.stop()  # type: ignore[union-attr]

    _browser = _context = _page = _pw_instance = None
    logger.info("browser_cleaned_up")


@tool(
    name="browser_navigate",
    description=(
        "Navigate a HEADLESS browser to a URL and return the page text content. "
        "The user CANNOT see this browser — it is invisible. Use this ONLY for "
        "web scraping, data extraction, or reading page content programmatically. "
        "Do NOT use this to open websites, play music/videos, or show anything to the user. "
        "To open a URL visibly for the user, use shell_exec with the system's open command "
        "(Windows: 'start URL', macOS: 'open URL', Linux: 'xdg-open URL')."
    ),
    tier=ToolTier.MODERATE,
)
async def browser_navigate(url: str, wait_for: str = "load") -> str:
    """Navigate to a URL.

    Args:
        url: The URL to navigate to.
        wait_for: Wait condition — "load" (default), "domcontentloaded", or "networkidle".
    """
    page = await _get_page()

    try:
        await page.goto(url, wait_until=wait_for, timeout=30000)  # type: ignore[arg-type]
    except Exception as e:
        logger.error("browser_navigate_failed", url=url, error=str(e))
        return f"[ERROR] Failed to navigate to {url}: {e}"

    title = await page.title()

    # Extract readable text content
    text_content: str = await page.evaluate("""
        () => {
            const body = document.body;
            if (!body) return '[Empty page]';

            const clone = body.cloneNode(true);
            clone.querySelectorAll('script, style, noscript').forEach(el => el.remove());

            return clone.innerText.replace(/\\n{3,}/g, '\\n\\n').trim().substring(0, 10000);
        }
    """)

    current_url = page.url
    logger.info("browser_navigated", url=current_url, title=title)
    return f"Title: {title}\nURL: {current_url}\n\nContent:\n{text_content}"


@tool(
    name="browser_screenshot",
    description=(
        "Take a screenshot of the current browser page. Returns metadata about the "
        "captured image. Use this to see what a page looks like visually, verify UI "
        "state, or capture information."
    ),
    tier=ToolTier.SAFE,
)
async def browser_screenshot(full_page: bool = False) -> str:
    """Take a screenshot.

    Args:
        full_page: If True, capture the entire scrollable page. If False, capture viewport only.
    """
    page = await _get_page()
    screenshot_bytes = await page.screenshot(full_page=full_page, type="png")
    b64 = base64.b64encode(screenshot_bytes).decode()

    title = await page.title()
    url = page.url

    logger.info("browser_screenshot_taken", url=url, size_bytes=len(screenshot_bytes))

    return (
        f"Screenshot captured ({len(screenshot_bytes)} bytes)\n"
        f"Page: {title}\n"
        f"URL: {url}\n"
        f"Base64 length: {len(b64)}\n"
        f"[Image data available as base64 PNG]"
    )


@tool(
    name="browser_click",
    description=(
        "Click an element on the current page identified by CSS selector or text content. "
        "Use this to interact with buttons, links, and clickable elements."
    ),
    tier=ToolTier.MODERATE,
)
async def browser_click(selector: str, timeout: int = 5000) -> str:  # noqa: ASYNC109
    """Click an element.

    Args:
        selector: CSS selector or text= selector (e.g., "text=Submit", "#login-btn").
        timeout: Max time to wait for element in ms.
    """
    page = await _get_page()

    try:
        await page.click(selector, timeout=timeout)
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
    except Exception as e:
        logger.error("browser_click_failed", selector=selector, error=str(e))
        return f"[ERROR] Failed to click '{selector}': {e}"

    title = await page.title()
    url = page.url
    logger.info("browser_clicked", selector=selector, url=url)
    return f"Clicked '{selector}'. Page now: {title} ({url})"


@tool(
    name="browser_fill",
    description=(
        "Fill a form input field with text. Use this to type into text fields, "
        "search boxes, login forms, etc."
    ),
    tier=ToolTier.MODERATE,
)
async def browser_fill(selector: str, value: str) -> str:
    """Fill a form field.

    Args:
        selector: CSS selector for the input element (e.g., "#email", "input[name=username]").
        value: Text to type into the field.
    """
    page = await _get_page()

    try:
        await page.fill(selector, value, timeout=5000)
    except Exception as e:
        logger.error("browser_fill_failed", selector=selector, error=str(e))
        return f"[ERROR] Failed to fill '{selector}': {e}"

    display_value = f"{value[:50]}..." if len(value) > 50 else value
    logger.info("browser_filled", selector=selector)
    return f"Filled '{selector}' with '{display_value}'"


@tool(
    name="browser_extract",
    description=(
        "Extract specific content from the current page using CSS selectors. "
        "Use this to get text from specific elements, table data, lists, etc."
    ),
    tier=ToolTier.SAFE,
)
async def browser_extract(selector: str, attribute: str | None = None) -> str:
    """Extract content from elements.

    Args:
        selector: CSS selector (e.g., "h1", ".price", "table tr").
        attribute: If set, extract this attribute instead of text (e.g., "href", "src").
    """
    page = await _get_page()

    try:
        elements = await page.query_selector_all(selector)

        if not elements:
            return f"[No elements found matching '{selector}']"

        results: list[str] = []
        for i, el in enumerate(elements[:50]):  # Limit to 50 elements
            if attribute:
                val = await el.get_attribute(attribute)
                results.append(f"{i + 1}. {val}")
            else:
                text = await el.inner_text()
                results.append(f"{i + 1}. {text.strip()[:200]}")

        logger.info("browser_extracted", selector=selector, count=len(elements))
        return f"Found {len(elements)} elements matching '{selector}':\n" + "\n".join(results)

    except Exception as e:
        logger.error("browser_extract_failed", selector=selector, error=str(e))
        return f"[ERROR] Failed to extract '{selector}': {e}"


@tool(
    name="browser_close",
    description=(
        "Close the browser session. A new browser will be created "
        "on the next browser tool call."
    ),
    tier=ToolTier.SAFE,
)
async def browser_close() -> str:
    """Close the browser."""
    await cleanup_browser()
    return "Browser closed."
