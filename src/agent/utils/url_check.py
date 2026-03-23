"""Shared URL verification utility.

Used by both the LiteLLM agent loop and Claude SDK paths to verify
URLs in tool outputs before returning them to the user.
"""

from __future__ import annotations

import re

import httpx
import structlog

logger = structlog.get_logger(__name__)

_URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+")
_MAX_URLS = 5
_TIMEOUT = 5


async def check_urls_in_output(output: str) -> str:
    """Verify URLs in tool/agent output, marking broken ones.

    Sends HEAD requests to up to 5 URLs found in the output.
    Annotates broken URLs with [BROKEN:<status>] or [UNREACHABLE].

    Args:
        output: Text that may contain URLs.

    Returns:
        The output with broken URLs annotated.
    """
    urls = _URL_PATTERN.findall(output)
    if not urls:
        return output

    for url in urls[:_MAX_URLS]:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.head(url, follow_redirects=True)
                if resp.status_code >= 400:
                    output = output.replace(
                        url,
                        f"{url} [BROKEN:{resp.status_code}]",
                    )
        except Exception:
            output = output.replace(url, f"{url} [UNREACHABLE]")
    return output
