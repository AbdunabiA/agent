"""Desktop control tools -- screen, mouse, keyboard, apps, windows.

These tools give the agent the ability to see and interact with the
desktop GUI, just like a human user would.

Requirements:
- A display/GUI environment (not a headless server)
- pyautogui, Pillow installed
- For vision features: a multimodal LLM (Claude, GPT-4o, Gemini)
- Linux: optionally wmctrl, xdotool for window management
- Windows: optionally pygetwindow for window management
"""

from __future__ import annotations

from agent.tools.registry import ToolTier, tool  # noqa: I001

# --- Screen ---


@tool(
    name="screen_capture",
    description=(
        "Take a screenshot of the entire screen or a specific region. "
        "Use this to SEE what's currently displayed. Returns screenshot metadata. "
        "Combine with find_on_screen to locate specific UI elements."
    ),
    tier=ToolTier.SAFE,
)
async def screen_capture(
    region_x: int | None = None,
    region_y: int | None = None,
    region_width: int | None = None,
    region_height: int | None = None,
    scale: float = 0.75,
) -> str:
    """Capture the screen and return metadata.

    Args:
        region_x: Left edge of capture region (optional).
        region_y: Top edge of capture region (optional).
        region_width: Width of capture region (optional).
        region_height: Height of capture region (optional).
        scale: Scale factor (0.5 = half resolution, saves tokens). Default 0.75.
    """
    from agent.desktop.screen import capture_screen

    region = None
    if all(v is not None for v in [region_x, region_y, region_width, region_height]):
        region = (region_x, region_y, region_width, region_height)  # type: ignore[arg-type]

    screenshot = await capture_screen(region=region, scale=scale)

    return (
        f"Screenshot captured: {screenshot.width}x{screenshot.height} "
        f"({len(screenshot.image_bytes) // 1024} KB)\n"
        f"Use find_on_screen to locate specific UI elements, "
        f"or screen_describe to get a full description."
    )


@tool(
    name="screen_describe",
    description=(
        "Take a screenshot and describe what's on screen using a vision LLM. "
        "Returns a detailed description of the visible content, applications, "
        "and UI elements. This is the agent's 'eyes'."
    ),
    tier=ToolTier.SAFE,
)
async def screen_describe(
    scale: float = 0.75,
) -> str:
    """Capture and describe the screen.

    Args:
        scale: Scale factor for screenshot (default 0.75).
    """
    from agent.config import get_config
    from agent.desktop.screen import capture_screen
    from agent.desktop.vision import VisionAnalyzer
    from agent.llm.provider import LLMProvider

    screenshot = await capture_screen(scale=scale)

    config = get_config()
    llm = LLMProvider(config.models)
    vision_model = config.desktop.vision_model or None
    analyzer = VisionAnalyzer(llm, model=vision_model)

    description = await analyzer.describe_screen(screenshot)
    return (
        f"Screen ({screenshot.width}x{screenshot.height}):\n\n"
        f"{description}"
    )


@tool(
    name="find_on_screen",
    description=(
        "Find a UI element on screen by describing it in natural language. "
        "Returns the element's coordinates so you can click it. "
        "Examples: 'the search bar', 'the red Submit button', 'the X close button'."
    ),
    tier=ToolTier.SAFE,
)
async def find_on_screen(element_description: str, scale: float = 0.75) -> str:
    """Find a UI element by description.

    Args:
        element_description: Natural language description of what to find.
        scale: Scale factor for screenshot (default 0.75).
    """
    from agent.config import get_config
    from agent.desktop.screen import capture_screen
    from agent.desktop.vision import VisionAnalyzer
    from agent.llm.provider import LLMProvider

    screenshot = await capture_screen(scale=scale)

    config = get_config()
    llm = LLMProvider(config.models)
    vision_model = config.desktop.vision_model or None
    analyzer = VisionAnalyzer(llm, model=vision_model)

    result = await analyzer.find_element(screenshot, element_description)

    if result and result.get("found"):
        # Scale coordinates back to actual screen resolution if scaled
        sx = int(int(result["x"]) / scale) if scale != 1.0 else int(result["x"])  # type: ignore[arg-type]
        sy = int(int(result["y"]) / scale) if scale != 1.0 else int(result["y"])  # type: ignore[arg-type]
        return (
            f"Found: {result.get('description', element_description)}\n"
            f"Position: ({sx}, {sy})\n"
            f"Size: ~{result.get('width', '?')}x{result.get('height', '?')}\n"
            f"To click it: use mouse_click with x={sx}, y={sy}"
        )
    return f"Element not found: '{element_description}'"


# --- Mouse ---


@tool(
    name="mouse_click",
    description=(
        "Click the mouse at specific coordinates or the current position. "
        "Supports left/right/middle click and double-click."
    ),
    tier=ToolTier.MODERATE,
)
async def mouse_click(
    x: int | None = None,
    y: int | None = None,
    button: str = "left",
    clicks: int = 1,
) -> str:
    """Click the mouse.

    Args:
        x: X coordinate (0 = left edge). None = current position.
        y: Y coordinate (0 = top edge). None = current position.
        button: "left", "right", or "middle".
        clicks: 1 = single click, 2 = double click.
    """
    from agent.desktop.mouse import click

    return await click(x=x, y=y, button=button, clicks=clicks)


@tool(
    name="mouse_move",
    description="Move the mouse cursor to specific coordinates.",
    tier=ToolTier.MODERATE,
)
async def mouse_move(x: int, y: int) -> str:
    """Move mouse.

    Args:
        x: X coordinate.
        y: Y coordinate.
    """
    from agent.desktop.mouse import move_to

    await move_to(x, y)
    return f"Mouse moved to ({x}, {y})"


@tool(
    name="mouse_scroll",
    description="Scroll the mouse wheel up or down.",
    tier=ToolTier.MODERATE,
)
async def mouse_scroll(amount: int, x: int | None = None, y: int | None = None) -> str:
    """Scroll.

    Args:
        amount: Positive = up, negative = down.
        x: Optional X position to scroll at.
        y: Optional Y position to scroll at.
    """
    from agent.desktop.mouse import scroll

    return await scroll(amount, x, y)


@tool(
    name="mouse_drag",
    description="Click and drag from one position to another.",
    tier=ToolTier.MODERATE,
)
async def mouse_drag(start_x: int, start_y: int, end_x: int, end_y: int) -> str:
    """Drag.

    Args:
        start_x: Starting X position.
        start_y: Starting Y position.
        end_x: Ending X position.
        end_y: Ending Y position.
    """
    from agent.desktop.mouse import drag

    return await drag(start_x, start_y, end_x, end_y)


# --- Keyboard ---


@tool(
    name="keyboard_type",
    description=(
        "Type text as if from the keyboard. Supports unicode. "
        "Use for typing into focused text fields, editors, search bars, etc."
    ),
    tier=ToolTier.MODERATE,
)
async def keyboard_type(text: str) -> str:
    """Type text.

    Args:
        text: Text to type. Can include unicode characters.
    """
    from agent.desktop.keyboard import type_text

    return await type_text(text)


@tool(
    name="keyboard_press",
    description=(
        "Press a single key: enter, tab, escape, backspace, delete, "
        "arrow keys (up/down/left/right), home, end, pageup, pagedown, "
        "f1-f12, or any character."
    ),
    tier=ToolTier.MODERATE,
)
async def keyboard_press(key: str) -> str:
    """Press a key.

    Args:
        key: Key name (e.g., "enter", "tab", "escape", "f5").
    """
    from agent.desktop.keyboard import press_key

    return await press_key(key)


@tool(
    name="keyboard_hotkey",
    description=(
        "Press a keyboard shortcut. Automatically adapts Ctrl/Cmd for macOS. "
        "Smart shortcuts: 'copy', 'paste', 'save', 'undo', 'find', 'select_all'. "
        "Or specify keys: 'ctrl,c' or 'alt,tab' or 'ctrl,shift,t'."
    ),
    tier=ToolTier.MODERATE,
)
async def keyboard_hotkey(keys: str) -> str:
    """Press a key combination.

    Args:
        keys: Comma-separated keys (e.g., "ctrl,c" or "copy" or "alt,tab").
    """
    from agent.desktop.keyboard import hotkey

    key_list = [k.strip() for k in keys.split(",")]
    return await hotkey(*key_list)


# --- Applications ---


@tool(
    name="app_launch",
    description=(
        "Launch an application by name. Cross-platform: "
        "Linux: 'firefox', 'code', 'nautilus'. "
        "macOS: 'Safari', 'Visual Studio Code', 'Finder'. "
        "Windows: 'notepad', 'chrome', 'explorer'."
    ),
    tier=ToolTier.MODERATE,
)
async def app_launch(app_name: str, args: str = "") -> str:
    """Launch an app.

    Args:
        app_name: Application name.
        args: Optional space-separated arguments.
    """
    from agent.desktop.apps import launch_app

    arg_list = args.split() if args else None
    return await launch_app(app_name, args=arg_list)


@tool(
    name="app_list",
    description="List installed applications on this computer.",
    tier=ToolTier.SAFE,
)
async def app_list() -> str:
    """List installed apps."""
    from agent.desktop.apps import list_installed_apps

    return await list_installed_apps()


@tool(
    name="open_file",
    description=(
        "Open a file with its default application "
        "(e.g., .pdf with PDF viewer, .png with image viewer)."
    ),
    tier=ToolTier.MODERATE,
)
async def open_file_tool(file_path: str) -> str:
    """Open a file.

    Args:
        file_path: Path to the file.
    """
    from agent.desktop.apps import open_file

    return await open_file(file_path)


@tool(
    name="open_url",
    description="Open a URL in the default web browser.",
    tier=ToolTier.MODERATE,
)
async def open_url_tool(url: str) -> str:
    """Open URL.

    Args:
        url: URL to open.
    """
    from agent.desktop.apps import open_url

    return await open_url(url)


# --- Windows ---


@tool(
    name="window_list",
    description="List all open windows with their titles, positions, and sizes.",
    tier=ToolTier.SAFE,
)
async def window_list() -> str:
    """List open windows."""
    from agent.desktop.windows import list_windows

    windows = await list_windows()

    if not windows:
        return "No windows found (or window management tools not available)"

    lines = [f"Open Windows ({len(windows)}):"]
    for w in windows:
        active = " [ACTIVE]" if w.is_active else ""
        lines.append(f"  {w.app_name}: \"{w.title}\"{active}")
        lines.append(f"    Position: ({w.x}, {w.y}) Size: {w.width}x{w.height}")

    return "\n".join(lines)


@tool(
    name="window_focus",
    description="Bring a window to the foreground by its title (partial match).",
    tier=ToolTier.MODERATE,
)
async def window_focus(title: str) -> str:
    """Focus a window.

    Args:
        title: Window title or partial match.
    """
    from agent.desktop.windows import focus_window

    return await focus_window(title)


@tool(
    name="window_close",
    description="Close a window by its title.",
    tier=ToolTier.MODERATE,
)
async def window_close(title: str) -> str:
    """Close a window.

    Args:
        title: Window title to close.
    """
    from agent.desktop.windows import close_window

    return await close_window(title)
