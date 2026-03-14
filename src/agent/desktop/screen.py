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
    monitor: int | None = None,
) -> Screenshot:
    """Capture the screen or a region.

    Args:
        region: Optional (x, y, width, height) for partial capture.
        scale: Scale factor for output (0.5 = half resolution, saves LLM tokens).
        monitor: Optional monitor index for multi-monitor setups.

    Returns:
        Screenshot with PNG bytes, base64 string, and dimensions.

    Raises:
        RuntimeError: If no display is available.
        ImportError: If no screenshot tool is available.
    """
    info = get_platform()

    if not info.has_display:
        raise RuntimeError("No display available. Desktop tools require a GUI environment.")

    # Auto-scale for HiDPI to save tokens (apply before capture)
    if info.scale_factor > 1.0 and scale == 1.0:
        scale = 1.0 / info.scale_factor

    # Use Wayland-native tools if available
    if info.display_server == "wayland" and info.has_grim:
        png_bytes, width, height = await _capture_wayland(region, scale, monitor)
    elif info.has_pyautogui:
        loop = asyncio.get_event_loop()
        png_bytes, width, height = await loop.run_in_executor(
            None, _capture_pyautogui, region, scale
        )
    else:
        raise ImportError(
            "No screenshot tool available. "
            "Install pyautogui (pip install 'agent-ai[desktop]') "
            "or grim (Wayland)."
        )

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


def _capture_pyautogui(
    region: tuple[int, int, int, int] | None,
    scale: float,
) -> tuple[bytes, int, int]:
    """Capture using pyautogui (blocking, run in executor)."""
    import pyautogui
    from PIL import Image

    img = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()

    # Scale down if requested
    if scale != 1.0 and scale > 0:
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()

    return png_bytes, img.width, img.height


async def _get_wayland_outputs() -> list[str]:
    """Get list of Wayland output names via grim or wlr-randr."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "wlr-randr", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            import json

            data = json.loads(stdout.decode())
            return [o["name"] for o in data if "name" in o]
    except FileNotFoundError:
        pass

    # Fallback: parse grim error output which lists outputs
    return []


async def _capture_wayland(
    region: tuple[int, int, int, int] | None,
    scale: float,
    monitor: int | None = None,
) -> tuple[bytes, int, int]:
    """Capture using grim (Wayland-native screenshot tool)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = ["grim"]

    # Region selection
    if region:
        x, y, w, h = region
        cmd.extend(["-g", f"{x},{y} {w}x{h}"])

    # Monitor selection — grim expects output names like "eDP-1", not indices
    if monitor is not None:
        outputs = await _get_wayland_outputs()
        if 0 <= monitor < len(outputs):
            cmd.extend(["-o", outputs[monitor]])
        else:
            logger.warning(
                "wayland_monitor_not_found",
                index=monitor,
                available=outputs,
            )

    # Scale
    if scale != 1.0 and scale > 0:
        cmd.extend(["-s", str(scale)])

    cmd.append(tmp_path)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        error = stderr.decode(errors="ignore")
        raise RuntimeError(f"grim failed: {error}")

    import contextlib
    import os

    from PIL import Image

    with Image.open(tmp_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        width, height = img.width, img.height

    with contextlib.suppress(OSError):
        os.unlink(tmp_path)

    return png_bytes, width, height
