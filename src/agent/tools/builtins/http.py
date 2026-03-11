"""HTTP request tool."""

from __future__ import annotations

import httpx

from agent.tools.registry import ToolTier, tool

MAX_BODY_SIZE = 50 * 1024  # 50KB


@tool(
    name="http_request",
    description=(
        "Make an HTTP request to a URL. Supports GET, POST, PUT, DELETE. "
        "Use this to interact with APIs, fetch web pages, download data, "
        "or test endpoints."
    ),
    tier=ToolTier.MODERATE,
)
async def http_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: str | None = None,
    timeout: int = 30,  # noqa: ASYNC109
) -> str:
    """Make an HTTP request.

    Args:
        url: The URL to request.
        method: HTTP method (GET, POST, PUT, DELETE, PATCH).
        headers: Optional headers dict.
        body: Optional request body (for POST/PUT).
        timeout: Request timeout in seconds.

    Returns:
        Formatted response with status, headers, and body.
    """
    # Validate URL
    if not url.startswith(("http://", "https://")):
        return f"[ERROR] Invalid URL: {url}. Must start with http:// or https://"

    # Validate method
    method = method.upper()
    valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
    if method not in valid_methods:
        return f"[ERROR] Invalid HTTP method: {method}"

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body if body else None,
            )

        # Format result
        parts = [f"Status: {response.status_code} {response.reason_phrase}"]

        # Selected response headers
        header_lines = []
        for key, value in response.headers.items():
            header_lines.append(f"  {key}: {value}")
        if header_lines:
            parts.append("Headers:\n" + "\n".join(header_lines[:20]))

        # Body
        response_body = response.text
        if len(response_body) > MAX_BODY_SIZE:
            response_body = (
                response_body[:MAX_BODY_SIZE]
                + f"\n[Body truncated: {len(response.text)} chars -> {MAX_BODY_SIZE} chars]"
            )
        parts.append(f"Body:\n{response_body}")

        return "\n\n".join(parts)

    except httpx.ConnectError as e:
        return f"[ERROR] Connection failed: {e}"
    except httpx.TimeoutException:
        return f"[ERROR] Request timed out after {timeout}s"
    except httpx.HTTPError as e:
        return f"[ERROR] HTTP error: {e}"
    except Exception as e:
        return f"[ERROR] Request failed: {e}"
