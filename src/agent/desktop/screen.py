"""Screen capture with optional region support.

Takes screenshots using pyautogui/Pillow.
Returns image as bytes or base64 for LLM analysis.
"""

from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass
from pathlib import Path

import structlog

from agent.desktop.platform_utils import get_platform

logger = structlog.get_logger(__name__)


@dataclass
class Screenshot:
    """Captured screenshot data."""

    image_bytes: bytes  # PNG bytes
    base64: str  # Base64 encoded for LLM
    width: int
    height: int
    region: tuple[int, int, int, int] | None = None  # (x, y, w, h) if partial


async def capture_screen(
    region: tuple[int, int, int, int] | None = None,
    scale: float = 1.0,
) -> Screenshot:
    """Capture the screen or a region.

    Args:
        region: Optional (x, y, width, height) for partial capture.
        scale: Scale factor for output (0.5 = half resolution, saves LLM tokens).

    Returns:
        Screenshot with PNG bytes, base64 string, and dimensions.

    Raises:
        RuntimeError: If no display is available.
        ImportError: If pyautogui is not installed.
    """
    info = get_platform()

    if not info.has_display:
        raise RuntimeError("No display available. Desktop tools require a GUI environment.")

    if not info.has_pyautogui:
        raise ImportError(
            "pyautogui is required for screen capture. "
            "Install with: pip install 'agent-ai[desktop]'"
        )

    loop = asyncio.get_event_loop()

    def _capture() -> tuple[bytes, int, int]:
        import pyautogui
        from PIL import Image

        img = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()

        # Scale down if requested
        if scale != 1.0 and scale > 0:
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Convert to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()

        return png_bytes, img.width, img.height

    png_bytes, width, height = await loop.run_in_executor(None, _capture)

    b64 = base64.b64encode(png_bytes).decode()

    logger.info(
        "screen_captured",
        width=width,
        height=height,
        size_kb=len(png_bytes) // 1024,
        region=region,
    )

    return Screenshot(
        image_bytes=png_bytes,
        base64=b64,
        width=width,
        height=height,
        region=region,
    )


async def save_screenshot(
    path: str,
    region: tuple[int, int, int, int] | None = None,
    scale: float = 1.0,
) -> str:
    """Capture and save screenshot to a file.

    Args:
        path: Output file path.
        region: Optional capture region.
        scale: Scale factor.

    Returns:
        Absolute path of saved file.
    """
    screenshot = await capture_screen(region=region, scale=scale)

    output_path = Path(path).expanduser().resolve()  # noqa: ASYNC240
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(screenshot.image_bytes)

    logger.info("screenshot_saved", path=str(output_path))
    return str(output_path)
