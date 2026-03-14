"""Cross-platform mouse control using pyautogui.

All mouse operations include safety features:
- Fail-safe: moving mouse to corner (0,0) raises pyautogui.FailSafeException
- Coordinates are validated against screen bounds
- All operations run in executor to avoid blocking the event loop
"""

from __future__ import annotations

import asyncio

import structlog

from agent.desktop.errors import desktop_op
from agent.desktop.platform_utils import get_platform

logger = structlog.get_logger(__name__)


def _validate_coords(x: int, y: int) -> tuple[int, int]:
    """Clamp coordinates to within screen bounds."""
    info = get_platform()
    x = max(0, min(x, info.screen_width - 1))
    y = max(0, min(y, info.screen_height - 1))
    return x, y


def _require_mouse() -> None:
    """Raise ImportError if no mouse tool is available."""
    info = get_platform()
    if not info.has_pyautogui and not info.has_ydotool:
        raise ImportError(
            "No mouse tool available. "
            "Install pyautogui (pip install 'agent-ai[desktop]') "
            "or ydotool (Wayland)."
        )


@desktop_op("mouse_move")
async def move_to(x: int, y: int, duration: float = 0.3) -> tuple[int, int]:
    """Move mouse cursor to absolute position.

    Args:
        x: X coordinate (0 = left edge).
        y: Y coordinate (0 = top edge).
        duration: Time in seconds for the movement animation.

    Returns:
        The actual (x, y) coordinates after clamping.
    """
    _require_mouse()
    info = get_platform()
    x, y = _validate_coords(x, y)

    # Apply scale factor for HiDPI
    scaled_x = int(x * info.scale_factor) if info.scale_factor > 1.0 else x
    scaled_y = int(y * info.scale_factor) if info.scale_factor > 1.0 else y

    if info.display_server == "wayland" and info.has_ydotool and not info.has_pyautogui:
        proc = await asyncio.create_subprocess_exec(
            "ydotool", "mousemove", "--absolute", "-x", str(scaled_x), "-y", str(scaled_y),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    else:
        loop = asyncio.get_event_loop()

        def _move() -> None:
            import pyautogui

            pyautogui.moveTo(x, y, duration=duration)

        await loop.run_in_executor(None, _move)

    logger.info("mouse_moved", x=x, y=y)
    return x, y


@desktop_op("mouse_click")
async def click(
    x: int | None = None,
    y: int | None = None,
    button: str = "left",
    clicks: int = 1,
) -> str:
    """Click at position (or current position if x,y not specified).

    Args:
        x: X coordinate. None = current position.
        y: Y coordinate. None = current position.
        button: "left", "right", or "middle".
        clicks: Number of clicks (1=single, 2=double, 3=triple).

    Returns:
        Description of the click action.
    """
    _require_mouse()
    info = get_platform()

    if x is not None and y is not None:
        x, y = _validate_coords(x, y)

    if info.display_server == "wayland" and info.has_ydotool and not info.has_pyautogui:
        # ydotool path: move first if coords given, then click
        if x is not None and y is not None:
            scaled_x = int(x * info.scale_factor) if info.scale_factor > 1.0 else x
            scaled_y = int(y * info.scale_factor) if info.scale_factor > 1.0 else y
            move_proc = await asyncio.create_subprocess_exec(
                "ydotool", "mousemove", "--absolute",
                "-x", str(scaled_x), "-y", str(scaled_y),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await move_proc.communicate()

        # ydotool click: button codes 0x40=left, 0x41=right, 0x42=middle
        btn_code = {"left": "0x40", "right": "0x41", "middle": "0x42"}.get(button, "0x40")
        for _ in range(clicks):
            click_proc = await asyncio.create_subprocess_exec(
                "ydotool", "click", btn_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await click_proc.communicate()

        actual_x, actual_y = x or 0, y or 0
    else:
        loop = asyncio.get_event_loop()

        def _click() -> tuple[int, int]:
            import pyautogui

            pyautogui.click(x=x, y=y, button=button, clicks=clicks)
            pos = pyautogui.position()
            return pos.x, pos.y

        actual_x, actual_y = await loop.run_in_executor(None, _click)

    logger.info("mouse_clicked", x=actual_x, y=actual_y, button=button, clicks=clicks)

    suffix = f" x{clicks}" if clicks > 1 else ""
    return f"Clicked {button} at ({actual_x}, {actual_y}){suffix}"


@desktop_op("mouse_drag")
async def drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
    button: str = "left",
) -> str:
    """Drag from one position to another.

    Args:
        start_x, start_y: Starting position.
        end_x, end_y: Ending position.
        duration: Time for the drag operation.
        button: Mouse button to hold during drag.

    Returns:
        Description of the drag action.
    """
    _require_mouse()
    info = get_platform()

    start_x, start_y = _validate_coords(start_x, start_y)
    end_x, end_y = _validate_coords(end_x, end_y)

    if info.display_server == "wayland" and info.has_ydotool and not info.has_pyautogui:
        scale = info.scale_factor if info.scale_factor > 1.0 else 1.0
        sx, sy = int(start_x * scale), int(start_y * scale)
        ex, ey = int(end_x * scale), int(end_y * scale)

        # Move to start, press, move to end, release
        btn_code = {"left": "0x40", "right": "0x41", "middle": "0x42"}.get(button, "0x40")
        for cmd in [
            ["ydotool", "mousemove", "--absolute", "-x", str(sx), "-y", str(sy)],
            ["ydotool", "mousedown", btn_code],
            ["ydotool", "mousemove", "--absolute", "-x", str(ex), "-y", str(ey)],
            ["ydotool", "mouseup", btn_code],
        ]:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
    else:
        loop = asyncio.get_event_loop()

        def _drag() -> None:
            import pyautogui

            pyautogui.moveTo(start_x, start_y, duration=0.1)
            pyautogui.drag(
                end_x - start_x,
                end_y - start_y,
                duration=duration,
                button=button,
            )

        await loop.run_in_executor(None, _drag)

    logger.info("mouse_dragged", from_x=start_x, from_y=start_y, to_x=end_x, to_y=end_y)
    return f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"


@desktop_op("mouse_scroll")
async def scroll(amount: int, x: int | None = None, y: int | None = None) -> str:
    """Scroll the mouse wheel.

    Args:
        amount: Positive = scroll up, negative = scroll down.
        x, y: Position to scroll at (optional, uses current position).

    Returns:
        Description of the scroll action.
    """
    _require_mouse()
    info = get_platform()

    if x is not None and y is not None:
        x, y = _validate_coords(x, y)

    if info.display_server == "wayland" and info.has_ydotool and not info.has_pyautogui:
        # Move to position first if specified
        if x is not None and y is not None:
            scaled_x = int(x * info.scale_factor) if info.scale_factor > 1.0 else x
            scaled_y = int(y * info.scale_factor) if info.scale_factor > 1.0 else y
            move_proc = await asyncio.create_subprocess_exec(
                "ydotool", "mousemove", "--absolute",
                "-x", str(scaled_x), "-y", str(scaled_y),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await move_proc.communicate()

        # ydotool mousemove with --wheel: positive=up, negative=down
        scroll_proc = await asyncio.create_subprocess_exec(
            "ydotool", "mousemove", "--wheel", "--",
            "-x", "0", "-y", str(amount),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await scroll_proc.communicate()
    else:
        loop = asyncio.get_event_loop()

        if x is not None and y is not None:
            def _scroll_at() -> None:
                import pyautogui

                pyautogui.scroll(amount, x=x, y=y)

            await loop.run_in_executor(None, _scroll_at)
        else:
            def _scroll() -> None:
                import pyautogui

                pyautogui.scroll(amount)

            await loop.run_in_executor(None, _scroll)

    direction = "up" if amount > 0 else "down"
    logger.info("mouse_scrolled", amount=amount, direction=direction)
    return f"Scrolled {direction} by {abs(amount)}"


@desktop_op("mouse_position")
async def get_position() -> tuple[int, int]:
    """Get current mouse cursor position.

    Returns:
        Tuple of (x, y) coordinates.
    """
    info = get_platform()

    if not info.has_pyautogui:
        raise ImportError(
            "pyautogui is required for get_position. "
            "Install with: pip install 'agent-ai[desktop]'"
        )

    import pyautogui

    pos = pyautogui.position()
    return pos.x, pos.y
