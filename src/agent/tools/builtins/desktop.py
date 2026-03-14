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

import time

from agent.desktop.accessibility import UIElement
from agent.tools.executor import ImageContent, MultimodalToolOutput  # noqa: I001
from agent.tools.registry import ToolTier, tool

# --- Element cache for click_element ---

_element_cache: dict[int, UIElement] = {}
_element_cache_list: list[UIElement] = []
_element_cache_timestamp: float = 0.0
_CACHE_TTL_SECONDS: float = 30.0


def _update_element_cache(elements: list[UIElement]) -> None:
    """Replace the element cache with new elements."""
    global _element_cache, _element_cache_list, _element_cache_timestamp
    _element_cache = {el.id: el for el in elements}
    _element_cache_list = list(elements)
    _element_cache_timestamp = time.monotonic()


def _get_cached_element(element_id: int) -> UIElement | None:
    """Get an element from cache, or None if expired/missing."""
    if not _element_cache:
        return None
    elapsed = time.monotonic() - _element_cache_timestamp
    if elapsed > _CACHE_TTL_SECONDS:
        return None
    return _element_cache.get(element_id)


def _is_cache_valid() -> bool:
    """Check whether the element cache exists and hasn't expired."""
    if not _element_cache_list:
        return False
    return (time.monotonic() - _element_cache_timestamp) <= _CACHE_TTL_SECONDS


def _search_elements(
    name: str | None = None,
    role: str | None = None,
    limit: int = 10,
) -> list[UIElement]:
    """Search cached elements by name (case-insensitive substring) and/or role (exact).

    Args:
        name: Substring to match against element names (case-insensitive).
        role: Exact canonical role to match.
        limit: Maximum results to return.

    Returns:
        Matching elements from the cache list.
    """
    results: list[UIElement] = []
    name_lower = name.lower() if name else None

    for el in _element_cache_list:
        if name_lower and name_lower not in el.name.lower():
            continue
        if role and el.role != role:
            continue
        results.append(el)
        if len(results) >= limit:
            break

    return results


# --- Target parsing for interact() ---

_ROLE_WORD_MAP: dict[str, str] = {
    "button": "button",
    "btn": "button",
    "field": "text_field",
    "input": "text_field",
    "textbox": "text_field",
    "text field": "text_field",
    "text_field": "text_field",
    "menu": "menu_item",
    "menu item": "menu_item",
    "checkbox": "checkbox",
    "check box": "checkbox",
    "radio": "radio_button",
    "dropdown": "dropdown",
    "select": "dropdown",
    "combobox": "dropdown",
    "link": "link",
    "tab": "tab",
    "slider": "slider",
    "toggle": "toggle",
    "switch": "toggle",
    "list item": "list_item",
    "tree item": "tree_item",
    "spinner": "spinner",
}


def _parse_target(target: str) -> tuple[str | None, str | None]:
    """Parse a natural-language target into (name, role).

    Examples:
        "Save button"  → ("Save", "button")
        "Name field"   → ("Name", "text_field")
        "File menu"    → ("File", "menu_item")
        "OK"           → ("OK", None)
    """
    target = target.strip()
    if not target:
        return (None, None)

    # Check if the last word(s) match a known role
    words = target.split()

    # Try two-word role match first (e.g., "text field", "list item")
    if len(words) >= 3:
        two_word = f"{words[-2]} {words[-1]}".lower()
        if two_word in _ROLE_WORD_MAP:
            name_part = " ".join(words[:-2]).strip() or None
            return (name_part, _ROLE_WORD_MAP[two_word])

    # Try single-word role match
    if len(words) >= 2:
        last_word = words[-1].lower()
        if last_word in _ROLE_WORD_MAP:
            name_part = " ".join(words[:-1]).strip() or None
            return (name_part, _ROLE_WORD_MAP[last_word])

    # No role word found — entire string is the name
    return (target, None)

# --- Capabilities ---


@tool(
    name="desktop_capabilities",
    description=(
        "Check what desktop control capabilities are available on this system. "
        "Shows display type, screen resolution, available tools (screenshot, mouse, "
        "keyboard, window management), and platform-specific details."
    ),
    tier=ToolTier.SAFE,
)
async def desktop_capabilities() -> str:
    """Get desktop capability summary.

    Returns:
        Human-readable capabilities summary.
    """
    from agent.desktop.platform_utils import get_capabilities_summary

    return get_capabilities_summary()


# --- Screen ---


@tool(
    name="screen_capture",
    description=(
        "Take a screenshot of the entire screen or a specific region. "
        "Returns the actual screenshot image so you can see what's displayed. "
        "Set annotate=True to overlay numbered bounding boxes on interactive "
        "UI elements (buttons, fields, links) for precise clicking via click_element."
    ),
    tier=ToolTier.SAFE,
)
async def screen_capture(
    region_x: int | None = None,
    region_y: int | None = None,
    region_width: int | None = None,
    region_height: int | None = None,
    scale: float = 0.75,
    annotate: bool = False,
) -> str:
    """Capture the screen and return the screenshot image.

    Args:
        region_x: Left edge of capture region (optional).
        region_y: Top edge of capture region (optional).
        region_width: Width of capture region (optional).
        region_height: Height of capture region (optional).
        scale: Scale factor (0.5 = half resolution, saves tokens). Default 0.75.
        annotate: If True, overlay numbered bounding boxes on UI elements (SoM).
    """
    from agent.desktop.screen import capture_screen

    region: tuple[int, int, int, int] | None = None
    if all(v is not None for v in [region_x, region_y, region_width, region_height]):
        assert region_x is not None and region_y is not None
        assert region_width is not None and region_height is not None
        region = (region_x, region_y, region_width, region_height)

    screenshot = await capture_screen(region=region, scale=scale)

    if annotate:
        from agent.desktop.accessibility import get_ui_elements
        from agent.desktop.som import annotate_screenshot

        elements = await get_ui_elements()
        if elements:
            # Compute effective scale from actual screenshot dimensions
            # (capture_screen may auto-adjust scale on HiDPI)
            from agent.desktop.platform_utils import get_platform

            _info = get_platform()
            effective_scale = (
                screenshot.width / _info.screen_width
                if _info.screen_width > 0 and not region
                else scale
            )
            annotated = await annotate_screenshot(
                screenshot, elements, scale=effective_scale,
            )
            _update_element_cache(elements)

            # Build element list text
            el_lines = []
            for el in elements[:50]:
                parent = f" (in \"{el.parent_name}\")" if el.parent_name else ""
                el_lines.append(f"[{el.id}] {el.role}: \"{el.name}\"{parent}")
            el_text = "\n".join(el_lines)
            if len(elements) > 50:
                el_text += f"\n... and {len(elements) - 50} more elements"

            return MultimodalToolOutput(  # type: ignore[return-value]
                text=(
                    f"Annotated screenshot: {screenshot.width}x{screenshot.height} "
                    f"({len(annotated.annotated_bytes) // 1024} KB), "
                    f"{len(elements)} UI elements detected.\n\n"
                    f"Elements:\n{el_text}\n\n"
                    f"Use click_element(id=N) to click an element by its number."
                ),
                images=[ImageContent(base64_data=annotated.annotated_base64)],
            )

    return MultimodalToolOutput(  # type: ignore[return-value]
        text=(
            f"Screenshot captured: {screenshot.width}x{screenshot.height} "
            f"({len(screenshot.image_bytes) // 1024} KB)"
        ),
        images=[ImageContent(base64_data=screenshot.base64)],
    )


# --- UI Elements (Accessibility + SoM) ---


@tool(
    name="ui_elements",
    description=(
        "Detect interactive UI elements (buttons, text fields, menus, links, etc.) "
        "on the current screen using accessibility APIs. Returns a numbered list "
        "of elements with their roles and names. Use click_element(id=N) to click."
    ),
    tier=ToolTier.SAFE,
)
async def ui_elements(window_title: str | None = None) -> str:
    """List interactive UI elements from the accessibility tree.

    Args:
        window_title: Optional window title to scope detection.
            If None, uses the foreground window.
    """
    from agent.desktop.accessibility import get_ui_elements

    elements = await get_ui_elements(window_title=window_title)

    if not elements:
        return (
            "No UI elements detected. This may mean:\n"
            "- No accessibility API is available (check desktop_capabilities)\n"
            "- The current window has no interactive elements\n"
            "- Accessibility permissions are not granted (macOS: System Preferences "
            "→ Privacy & Security → Accessibility)"
        )

    _update_element_cache(elements)

    lines = [f"Interactive UI Elements ({len(elements)}):"]
    for el in elements:
        focused = " [FOCUSED]" if el.is_focused else ""
        disabled = " [DISABLED]" if not el.is_enabled else ""
        value = f" = \"{el.value}\"" if el.value else ""
        parent = f" (in \"{el.parent_name}\")" if el.parent_name else ""
        desc = f" — {el.description}" if el.description else ""
        lines.append(
            f"  [{el.id}] {el.role}: \"{el.name}\"{value}{parent}{desc}"
            f"{focused}{disabled}  @ ({el.x},{el.y}) {el.width}x{el.height}"
        )

    lines.append("")
    lines.append("Use click_element(id=N) to click an element by its number.")

    return "\n".join(lines)


@tool(
    name="click_element",
    description=(
        "Click a UI element by its ID number from a previous ui_elements or "
        "annotated screen_capture call. More accurate than clicking by coordinates."
    ),
    tier=ToolTier.MODERATE,
)
async def click_element(
    element_id: int,
    button: str = "left",
    clicks: int = 1,
) -> str:
    """Click a UI element by its numbered ID.

    Args:
        element_id: The element number from ui_elements or annotated screenshot.
        button: "left", "right", or "middle".
        clicks: 1 = single click, 2 = double click.
    """
    from agent.desktop.mouse import click

    if not _element_cache:
        return (
            "Error: No element cache available. "
            "Run ui_elements or screen_capture(annotate=True) first."
        )

    el = _get_cached_element(element_id)
    if el is None:
        elapsed = time.monotonic() - _element_cache_timestamp
        if elapsed > _CACHE_TTL_SECONDS:
            return (
                f"Error: Element cache expired ({elapsed:.0f}s old, "
                f"TTL={_CACHE_TTL_SECONDS:.0f}s). "
                "Run ui_elements or screen_capture(annotate=True) again."
            )
        valid_ids = sorted(_element_cache.keys())
        id_range = f"{valid_ids[0]}-{valid_ids[-1]}" if valid_ids else "none"
        return (
            f"Error: Element ID {element_id} not found. "
            f"Valid IDs: {id_range}. "
            "Run ui_elements to refresh the list."
        )

    # Click center of element bounding box
    center_x = el.x + el.width // 2
    center_y = el.y + el.height // 2

    result = await click(x=center_x, y=center_y, button=button, clicks=clicks)

    return (
        f"Clicked [{el.id}] {el.role}: \"{el.name}\" "
        f"at ({center_x}, {center_y}). {result}"
    )


# --- Accessibility-first tools ---


@tool(
    name="find_element",
    description=(
        "Search for UI elements by name and/or role without taking a screenshot. "
        "Much faster and cheaper than screen_capture. Returns matching elements "
        "with IDs for use with click_element or interact."
    ),
    tier=ToolTier.SAFE,
)
async def find_element(
    name: str | None = None,
    role: str | None = None,
) -> str:
    """Search for UI elements by name and/or role.

    Args:
        name: Substring to match element names (case-insensitive).
        role: Exact role to match (button, text_field, menu_item, etc.).
    """
    if name is None and role is None:
        return "Error: Provide at least one of 'name' or 'role' to search."

    # Auto-refresh cache if expired or empty
    if not _is_cache_valid():
        from agent.desktop.accessibility import get_ui_elements

        elements = await get_ui_elements()
        _update_element_cache(elements)

    matches = _search_elements(name=name, role=role)

    if not matches:
        search_desc = []
        if name:
            search_desc.append(f'name="{name}"')
        if role:
            search_desc.append(f"role={role}")
        return (
            f"No elements found matching {', '.join(search_desc)}.\n"
            f"Cache has {len(_element_cache_list)} elements.\n"
            "Try screen_capture(annotate=True) for visual fallback."
        )

    lines = [f"Found {len(matches)} matching element(s):"]
    for el in matches:
        parent = f' (in "{el.parent_name}")' if el.parent_name else ""
        desc = f" — {el.description}" if el.description else ""
        focused = " [FOCUSED]" if el.is_focused else ""
        disabled = " [DISABLED]" if not el.is_enabled else ""
        value = f' = "{el.value}"' if el.value else ""
        lines.append(
            f"  [{el.id}] {el.role}: \"{el.name}\"{value}{parent}{desc}"
            f"{focused}{disabled}  @ ({el.x},{el.y}) {el.width}x{el.height}"
        )

    lines.append("")
    lines.append(
        "Use click_element(id=N) to click, or interact(target, action) "
        "for one-step actions."
    )

    return "\n".join(lines)


@tool(
    name="interact",
    description=(
        "Find and interact with a UI element in one step. Specify a target like "
        "'Save button', 'Name field', 'File menu', or just 'OK'. Actions: "
        "'click' (default), 'type' (requires text param), 'focus', 'read'. "
        "Faster than screen_capture → click_element flow."
    ),
    tier=ToolTier.MODERATE,
)
async def interact(
    target: str,
    action: str = "click",
    text: str | None = None,
) -> str:
    """Find and interact with a UI element by name.

    Args:
        target: Natural-language element description (e.g. "Save button",
            "Name field", "File menu", "OK").
        action: "click", "type", "focus", or "read".
        text: Text to type (required when action="type").
    """
    if action == "type" and not text:
        return "Error: action='type' requires the 'text' parameter."

    name, role = _parse_target(target)

    # Search cache first
    matches = _search_elements(name=name, role=role) if _is_cache_valid() else []

    # If no match, refresh cache and search again
    if not matches:
        from agent.desktop.accessibility import get_ui_elements

        elements = await get_ui_elements()
        _update_element_cache(elements)
        matches = _search_elements(name=name, role=role)

    if not matches:
        return (
            f"No element found matching \"{target}\".\n"
            f"Cache has {len(_element_cache_list)} elements.\n"
            "Try screen_capture(annotate=True) for visual fallback, or "
            "find_element() to search with different terms."
        )

    el = matches[0]
    center_x = el.x + el.width // 2
    center_y = el.y + el.height // 2

    if action == "click":
        from agent.desktop.mouse import click

        result = await click(x=center_x, y=center_y, button="left", clicks=1)
        return (
            f"Clicked [{el.id}] {el.role}: \"{el.name}\" "
            f"at ({center_x}, {center_y}). {result}"
        )

    elif action == "type":
        from agent.desktop.keyboard import type_text
        from agent.desktop.mouse import click

        await click(x=center_x, y=center_y, button="left", clicks=1)
        assert text is not None
        result = await type_text(text)
        return (
            f"Focused [{el.id}] {el.role}: \"{el.name}\" "
            f"and typed {len(text)} characters. {result}"
        )

    elif action == "focus":
        from agent.desktop.mouse import click

        await click(x=center_x, y=center_y, button="left", clicks=1)
        return (
            f"Focused [{el.id}] {el.role}: \"{el.name}\" "
            f"at ({center_x}, {center_y})."
        )

    elif action == "read":
        props = [f'Name: "{el.name}"', f"Role: {el.role}"]
        if el.value:
            props.append(f'Value: "{el.value}"')
        props.append(f"Enabled: {el.is_enabled}")
        props.append(f"Focused: {el.is_focused}")
        if el.parent_name:
            props.append(f'Parent: "{el.parent_name}" ({el.parent_role})')
        if el.description:
            props.append(f'Description: "{el.description}"')
        props.append(f"Position: ({el.x},{el.y}) {el.width}x{el.height}")
        return f"Element [{el.id}] properties:\n  " + "\n  ".join(props)

    else:
        return (
            f"Unknown action '{action}'. "
            "Supported actions: click, type, focus, read."
        )


@tool(
    name="screen_read",
    description=(
        "Get a text-only summary of the current screen state using accessibility "
        "APIs. No screenshot, no vision tokens — much cheaper than screen_capture. "
        "Returns active window info, focused element, element counts by role, "
        "and the full element list."
    ),
    tier=ToolTier.SAFE,
)
async def screen_read(window_title: str | None = None) -> str:
    """Read the screen state as text using accessibility APIs.

    Args:
        window_title: Optional window title to scope the read.
    """
    from agent.desktop.accessibility import get_ui_elements
    from agent.desktop.windows import list_windows

    lines: list[str] = []

    # Window info
    try:
        windows = await list_windows()
        if isinstance(windows, list) and windows:
            active = [w for w in windows if w.is_active]
            if active:
                w = active[0]
                lines.append(f"Active window: \"{w.title}\" ({w.app_name})")
                lines.append(f"  Position: ({w.x},{w.y}) Size: {w.width}x{w.height}")
            lines.append(f"Open windows: {len(windows)}")
    except Exception:
        lines.append("Window info: unavailable")

    lines.append("")

    # UI elements
    elements = await get_ui_elements(window_title=window_title)
    _update_element_cache(elements)

    if not elements:
        lines.append("No UI elements detected.")
        return "\n".join(lines)

    # Focused element
    focused = [el for el in elements if el.is_focused]
    if focused:
        f = focused[0]
        value = f' = "{f.value}"' if f.value else ""
        lines.append(f"Focused element: [{f.id}] {f.role}: \"{f.name}\"{value}")
    else:
        lines.append("Focused element: none detected")

    # Role counts
    role_counts: dict[str, int] = {}
    for el in elements:
        role_counts[el.role] = role_counts.get(el.role, 0) + 1

    counts_str = ", ".join(f"{r}: {c}" for r, c in sorted(role_counts.items()))
    lines.append(f"Elements ({len(elements)}): {counts_str}")
    lines.append("")

    # Full element list
    lines.append("Element list:")
    for el in elements:
        parent = f' (in "{el.parent_name}")' if el.parent_name else ""
        desc = f" — {el.description}" if el.description else ""
        focused_tag = " [FOCUSED]" if el.is_focused else ""
        disabled = " [DISABLED]" if not el.is_enabled else ""
        value = f' = "{el.value}"' if el.value else ""
        lines.append(
            f"  [{el.id}] {el.role}: \"{el.name}\"{value}{parent}{desc}"
            f"{focused_tag}{disabled}"
        )

    return "\n".join(lines)


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

    # desktop_op decorator returns a string on error/unavailable
    if isinstance(windows, str):
        return windows

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
