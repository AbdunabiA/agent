"""Cross-platform mouse control using pyautogui.

All mouse operations include safety features:
- Fail-safe: moving mouse to corner (0,0) raises pyautogui.FailSafeException
- Coordinates are validated against screen bounds
- All operations run in executor to avoid blocking the event loop
"""

from __future__ import annotations

import asyncio

import structlog

from agent.desktop.platform_utils import get_platform

logger = structlog.get_logger(__name__)


def _validate_coords(x: int, y: int) -> tuple[int, int]:
    """Clamp coordinates to within screen bounds."""
    info = get_platform()
    x = max(0, min(x, info.screen_width - 1))
    y = max(0, min(y, info.screen_height - 1))
    return x, y


def _require_pyautogui() -> None:
    """Raise ImportError if pyautogui is not available."""
    info = get_platform()
    if not info.has_pyautogui:
        raise ImportError(
            "pyautogui is required for mouse control. "
            "Install with: pip install 'agent-ai[desktop]'"
        )


async def move_to(x: int, y: int, duration: float = 0.3) -> tuple[int, int]:
    """Move mouse cursor to absolute position.

    Args:
        x: X coordinate (0 = left edge).
        y: Y coordinate (0 = top edge).
        duration: Time in seconds for the movement animation.

    Returns:
        The actual (x, y) coordinates after clamping.
    """
    _require_pyautogui()
    x, y = _validate_coords(x, y)

    loop = asyncio.get_event_loop()

    def _move() -> None:
        import pyautogui

        pyautogui.moveTo(x, y, duration=duration)

    await loop.run_in_executor(None, _move)

    logger.info("mouse_moved", x=x, y=y)
    return x, y


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
    _require_pyautogui()

    if x is not None and y is not None:
        x, y = _validate_coords(x, y)

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
    _require_pyautogui()

    start_x, start_y = _validate_coords(start_x, start_y)
    end_x, end_y = _validate_coords(end_x, end_y)

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


async def scroll(amount: int, x: int | None = None, y: int | None = None) -> str:
    """Scroll the mouse wheel.

    Args:
        amount: Positive = scroll up, negative = scroll down.
        x, y: Position to scroll at (optional, uses current position).

    Returns:
        Description of the scroll action.
    """
    _require_pyautogui()

    loop = asyncio.get_event_loop()

    if x is not None and y is not None:
        x, y = _validate_coords(x, y)

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


async def get_position() -> tuple[int, int]:
    """Get current mouse cursor position.

    Returns:
        Tuple of (x, y) coordinates.
    """
    _require_pyautogui()

    import pyautogui

    pos = pyautogui.position()
    return pos.x, pos.y
