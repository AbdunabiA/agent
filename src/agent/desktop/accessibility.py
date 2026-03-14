"""Cross-platform accessibility tree extraction.

Extracts UI elements (buttons, fields, menus) with exact bounding boxes
from OS accessibility APIs. Used for Set of Marks (SoM) annotation to
improve click accuracy.

Platform backends:
- Windows: uiautomation package (COM-based)
- macOS: AppleScript via osascript (System Events)
- Linux: pyatspi2 / AT-SPI2 D-Bus
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass

import structlog

from agent.desktop.platform_utils import OSType, get_platform

logger = structlog.get_logger(__name__)

# Maximum elements to collect before filtering
_MAX_RAW_ELEMENTS = 500

# Maximum elements after filtering
_MAX_FILTERED_ELEMENTS = 200

# Canonical interactive roles
INTERACTIVE_ROLES: frozenset[str] = frozenset({
    "button",
    "text_field",
    "menu_item",
    "checkbox",
    "radio_button",
    "dropdown",
    "link",
    "tab",
    "list_item",
    "slider",
    "toggle",
    "tree_item",
    "spinner",
    "menu",
    "toolbar_button",
})

# Maps platform-specific role names → canonical role
_ROLE_MAP: dict[str, str] = {
    # Windows UIAutomation ControlTypeName
    "button": "button",
    "ButtonControl": "button",
    "edit": "text_field",
    "EditControl": "text_field",
    "text": "text_field",
    "TextControl": "text_field",
    "menuitem": "menu_item",
    "MenuItemControl": "menu_item",
    "checkbox": "checkbox",
    "CheckBoxControl": "checkbox",
    "radiobutton": "radio_button",
    "RadioButtonControl": "radio_button",
    "combobox": "dropdown",
    "ComboBoxControl": "dropdown",
    "hyperlink": "link",
    "HyperlinkControl": "link",
    "tabitem": "tab",
    "TabItemControl": "tab",
    "listitem": "list_item",
    "ListItemControl": "list_item",
    "slider": "slider",
    "SliderControl": "slider",
    "treeitem": "tree_item",
    "TreeItemControl": "tree_item",
    "spinner": "spinner",
    "SpinnerControl": "spinner",
    "menu": "menu",
    "MenuControl": "menu",
    "toolbar": "toolbar_button",
    "ToolBarControl": "toolbar_button",
    # macOS AX roles
    "AXButton": "button",
    "AXTextField": "text_field",
    "AXTextArea": "text_field",
    "AXMenuItem": "menu_item",
    "AXCheckBox": "checkbox",
    "AXRadioButton": "radio_button",
    "AXPopUpButton": "dropdown",
    "AXComboBox": "dropdown",
    "AXLink": "link",
    "AXTab": "tab",
    "AXTabButton": "tab",
    "AXRow": "list_item",
    "AXCell": "list_item",
    "AXSlider": "slider",
    "AXOutlineRow": "tree_item",
    "AXIncrementor": "spinner",
    "AXMenu": "menu",
    "AXMenuBar": "menu",
    "AXToolbar": "toolbar_button",
    # Linux AT-SPI roles
    "push button": "button",
    "toggle button": "toggle",
    "password text": "text_field",
    "entry": "text_field",
    "menu item": "menu_item",
    "check box": "checkbox",
    "check menu item": "checkbox",
    "radio button": "radio_button",
    "radio menu item": "radio_button",
    "combo box": "dropdown",
    "link": "link",
    "page tab": "tab",
    "list item": "list_item",
    "table cell": "list_item",
    "spin button": "spinner",
    "menu bar": "menu",
    "tool bar": "toolbar_button",
    "tree item": "tree_item",
}


@dataclass
class UIElement:
    """A single interactive UI element with its bounding box."""

    id: int  # SoM number (1-based, assigned after sorting)
    role: str  # Normalized role name
    name: str  # Visible label / accessible name
    x: int  # Left edge
    y: int  # Top edge
    width: int  # Bounding box width
    height: int  # Bounding box height
    is_enabled: bool = True
    is_focused: bool = False
    value: str = ""  # Current value for inputs
    parent_name: str = ""  # Immediate parent's name
    parent_role: str = ""  # Immediate parent's role
    description: str = ""  # Accessible description/help text


def _normalize_role(raw: str) -> str:
    """Map a platform-specific role name to a canonical role.

    Args:
        raw: Raw role name from the accessibility API.

    Returns:
        Canonical role name, or the raw name lowercased if unknown.
    """
    if raw in _ROLE_MAP:
        return _ROLE_MAP[raw]
    # Try case-insensitive lookup
    lower = raw.lower().strip()
    if lower in _ROLE_MAP:
        return _ROLE_MAP[lower]
    return lower


def _filter_elements(
    elements: list[UIElement],
    screen_w: int,
    screen_h: int,
) -> list[UIElement]:
    """Remove non-interactive, zero-size, and off-screen elements.

    Args:
        elements: Raw elements from accessibility API.
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.

    Returns:
        Filtered list (not yet numbered).
    """
    filtered: list[UIElement] = []
    for el in elements:
        # Skip non-interactive roles
        if el.role not in INTERACTIVE_ROLES:
            continue
        # Skip zero-size
        if el.width <= 0 or el.height <= 0:
            continue
        # Skip off-screen (completely outside screen bounds)
        if el.x + el.width <= 0 or el.y + el.height <= 0:
            continue
        if el.x >= screen_w or el.y >= screen_h:
            continue
        # Skip disabled elements without names (likely decorative)
        if not el.is_enabled and not el.name:
            continue
        filtered.append(el)

    if len(filtered) > _MAX_FILTERED_ELEMENTS:
        logger.warning(
            "ui_elements_capped",
            raw_count=len(filtered),
            cap=_MAX_FILTERED_ELEMENTS,
        )
        filtered = filtered[:_MAX_FILTERED_ELEMENTS]

    return filtered


def _sort_and_number(elements: list[UIElement]) -> list[UIElement]:
    """Sort elements top-to-bottom, left-to-right and assign 1-based IDs.

    Args:
        elements: Filtered elements (IDs will be overwritten).

    Returns:
        Sorted and numbered list.
    """
    # Sort by y (top-to-bottom), then x (left-to-right)
    elements.sort(key=lambda e: (e.y, e.x))
    for i, el in enumerate(elements, start=1):
        el.id = i
    return elements


async def get_ui_elements(window_title: str | None = None) -> list[UIElement]:
    """Extract interactive UI elements from the accessibility tree.

    Cross-platform entry point. Detects the OS and delegates to the
    appropriate backend.

    Args:
        window_title: Optional window title to scope extraction.
            If None, uses the foreground/active window.

    Returns:
        List of UIElement with bounding boxes, sorted and numbered.
        Returns empty list if no accessibility API is available.
    """
    info = get_platform()

    if not info.has_display:
        logger.warning("accessibility_no_display", msg="No display available")
        return []

    raw_elements: list[UIElement] = []

    try:
        if info.os_type == OSType.WINDOWS:
            raw_elements = await _get_elements_windows(window_title)
        elif info.os_type == OSType.MACOS:
            raw_elements = await _get_elements_macos(window_title)
        elif info.os_type == OSType.LINUX:
            raw_elements = await _get_elements_linux(window_title)
        else:
            logger.warning("accessibility_unknown_os", os_type=info.os_type)
            return []
    except Exception as e:
        logger.warning("accessibility_extraction_failed", error=str(e))
        return []

    if len(raw_elements) > _MAX_RAW_ELEMENTS:
        logger.warning(
            "accessibility_raw_capped",
            raw_count=len(raw_elements),
            cap=_MAX_RAW_ELEMENTS,
        )
        raw_elements = raw_elements[:_MAX_RAW_ELEMENTS]

    filtered = _filter_elements(raw_elements, info.screen_width, info.screen_height)
    numbered = _sort_and_number(filtered)

    logger.info(
        "ui_elements_extracted",
        raw_count=len(raw_elements),
        filtered_count=len(numbered),
        platform=info.os_type.value,
    )

    return numbered


async def _get_elements_windows(window_title: str | None = None) -> list[UIElement]:
    """Extract UI elements using the uiautomation package (Windows COM).

    Runs in executor since uiautomation is blocking COM.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_elements_windows_sync, window_title)


def _get_elements_windows_sync(window_title: str | None = None) -> list[UIElement]:
    """Blocking Windows UI element extraction."""
    try:
        import uiautomation as auto  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "accessibility_missing_package",
            package="uiautomation",
            install="pip install uiautomation",
        )
        return []

    elements: list[UIElement] = []

    try:
        if window_title:
            control = auto.WindowControl(searchDepth=1, Name=window_title)
        else:
            control = auto.GetForegroundControl()

        if control is None:
            return []

        _walk_windows_tree(
            control, elements, depth=0, max_depth=15,
            parent_name="", parent_role="",
        )
    except Exception as e:
        logger.warning("windows_accessibility_error", error=str(e))

    return elements


def _walk_windows_tree(
    control: object,
    elements: list[UIElement],
    depth: int,
    max_depth: int,
    parent_name: str = "",
    parent_role: str = "",
) -> None:
    """Recursively walk the Windows UI automation tree."""
    if depth > max_depth or len(elements) >= _MAX_RAW_ELEMENTS:
        return

    try:
        # Access control properties
        name = getattr(control, "Name", "") or ""
        control_type = getattr(control, "ControlTypeName", "") or ""
        rect = getattr(control, "BoundingRectangle", None)

        if rect is not None:
            x = int(getattr(rect, "left", 0))
            y = int(getattr(rect, "top", 0))
            right = int(getattr(rect, "right", 0))
            bottom = int(getattr(rect, "bottom", 0))
            w = right - x
            h = bottom - y

            role = _normalize_role(control_type)

            is_enabled = True
            with contextlib.suppress(Exception):
                is_enabled = bool(getattr(control, "IsEnabled", True))

            is_focused = False
            with contextlib.suppress(Exception):
                is_focused = bool(getattr(control, "HasKeyboardFocus", False))

            value = ""
            try:
                val_pattern = getattr(control, "GetValuePattern", None)
                if val_pattern and callable(val_pattern):
                    vp = val_pattern()
                    if vp:
                        value = str(getattr(vp, "Value", ""))
            except Exception:
                pass

            description = ""
            with contextlib.suppress(Exception):
                description = str(getattr(control, "HelpText", "") or "")

            elements.append(UIElement(
                id=0,  # Will be assigned after sorting
                role=role,
                name=name,
                x=x,
                y=y,
                width=w,
                height=h,
                is_enabled=is_enabled,
                is_focused=is_focused,
                value=value,
                parent_name=parent_name,
                parent_role=parent_role,
                description=description,
            ))

        # Walk children — pass this element's name/role as parent info
        children_method = getattr(control, "GetChildren", None)
        if children_method and callable(children_method):
            children = children_method()
            if children:
                cur_name = name or parent_name
                cur_role = _normalize_role(control_type) if control_type else parent_role
                for child in children:
                    _walk_windows_tree(
                        child, elements, depth + 1, max_depth,
                        parent_name=cur_name, parent_role=cur_role,
                    )
    except Exception:
        pass


async def _get_elements_macos(window_title: str | None = None) -> list[UIElement]:
    """Extract UI elements via AppleScript (macOS System Events).

    Uses osascript to query the accessibility tree. No extra deps needed.
    """
    # Build AppleScript to get UI element info
    target = f'window "{window_title}"' if window_title else "front window"

    script = f"""
tell application "System Events"
    set frontApp to first application process whose frontmost is true
    tell frontApp
        set elemList to entire contents of {target}
        set output to ""
        repeat with elem in elemList
            try
                set elemRole to role of elem
                set elemName to name of elem
                set elemPos to position of elem
                set elemSize to size of elem
                set elemEnabled to enabled of elem
                set elemFocused to focused of elem
                set posX to item 1 of elemPos
                set posY to item 2 of elemPos
                set sizeW to item 1 of elemSize
                set sizeH to item 2 of elemSize
                set elemDesc to ""
                try
                    set elemDesc to description of elem
                end try
                set output to output & elemRole & "|||" & elemName & "|||" & ¬
                    posX & "|||" & posY & "|||" & sizeW & "|||" & sizeH & "|||" & ¬
                    elemEnabled & "|||" & elemFocused & "|||" & elemDesc & linefeed
            end try
        end repeat
        return output
    end tell
end tell
"""

    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except TimeoutError:
        logger.warning("macos_accessibility_timeout")
        return []
    except FileNotFoundError:
        logger.warning("macos_osascript_not_found")
        return []

    if proc.returncode != 0:
        error = stderr.decode(errors="ignore").strip()
        if "accessibility" in error.lower() or "not allowed" in error.lower():
            logger.warning(
                "macos_accessibility_denied",
                msg="Grant accessibility access in System Preferences → "
                    "Privacy & Security → Accessibility",
            )
        else:
            logger.warning("macos_osascript_error", error=error)
        return []

    output = stdout.decode(errors="ignore")
    return _parse_macos_output(output)


def _parse_macos_output(output: str) -> list[UIElement]:
    """Parse delimited output from AppleScript."""
    elements: list[UIElement] = []

    for line in output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = line.split("|||")
        if len(parts) < 6:
            continue

        try:
            role = _normalize_role(parts[0].strip())
            name = parts[1].strip()
            x = int(float(parts[2].strip()))
            y = int(float(parts[3].strip()))
            w = int(float(parts[4].strip()))
            h = int(float(parts[5].strip()))

            is_enabled = True
            if len(parts) > 6:
                is_enabled = parts[6].strip().lower() == "true"

            is_focused = False
            if len(parts) > 7:
                is_focused = parts[7].strip().lower() == "true"

            description = ""
            if len(parts) > 8:
                description = parts[8].strip()

            elements.append(UIElement(
                id=0,
                role=role,
                name=name,
                x=x,
                y=y,
                width=w,
                height=h,
                is_enabled=is_enabled,
                is_focused=is_focused,
                description=description,
            ))
        except (ValueError, IndexError):
            continue

    return elements


async def _get_elements_linux(window_title: str | None = None) -> list[UIElement]:
    """Extract UI elements using pyatspi2 or AT-SPI2 D-Bus (Linux).

    Tries pyatspi first, falls back to gdbus.
    """
    # Try pyatspi
    try:
        loop = asyncio.get_event_loop()
        elements = await loop.run_in_executor(
            None, _get_elements_pyatspi, window_title,
        )
        return elements
    except ImportError:
        logger.info("accessibility_pyatspi_unavailable", msg="Trying gdbus fallback")
    except Exception as e:
        logger.warning("pyatspi_error", error=str(e))

    # Fallback to gdbus
    try:
        return await _get_elements_gdbus(window_title)
    except Exception as e:
        logger.warning(
            "accessibility_linux_unavailable",
            error=str(e),
            msg="Install pyatspi2 (system package) for accessibility support",
        )
        return []


def _get_elements_pyatspi(window_title: str | None = None) -> list[UIElement]:
    """Blocking pyatspi extraction."""
    import pyatspi  # type: ignore[import-not-found]

    desktop = pyatspi.Registry.getDesktop(0)
    elements: list[UIElement] = []

    for app_idx in range(desktop.childCount):
        app = desktop.getChildAtIndex(app_idx)
        if app is None:
            continue

        for win_idx in range(app.childCount):
            win = app.getChildAtIndex(win_idx)
            if win is None:
                continue

            # Filter by window title if specified
            if window_title and window_title.lower() not in (win.name or "").lower():
                continue

            _walk_atspi_tree(
                win, elements, depth=0, max_depth=15,
                parent_name="", parent_role="",
            )

            # If we found elements for the specified window, stop
            if window_title and elements:
                break
        if window_title and elements:
            break

    return elements


def _walk_atspi_tree(
    node: object,
    elements: list[UIElement],
    depth: int,
    max_depth: int,
    parent_name: str = "",
    parent_role: str = "",
) -> None:
    """Recursively walk the AT-SPI tree."""
    if depth > max_depth or len(elements) >= _MAX_RAW_ELEMENTS:
        return

    try:
        role_name = str(node.getRoleName())  # type: ignore[attr-defined]
        name = str(node.name) if hasattr(node, "name") else ""

        # Get bounding box
        try:
            component = node.queryComponent()  # type: ignore[attr-defined]
            if component:
                extents = component.getExtents(0)  # 0 = DESKTOP_COORDS
                x = int(extents.x)
                y = int(extents.y)
                w = int(extents.width)
                h = int(extents.height)
            else:
                x = y = w = h = 0
        except Exception:
            x = y = w = h = 0

        role = _normalize_role(role_name)

        # Get state
        is_enabled = True
        is_focused = False
        try:
            state_set = node.getState()  # type: ignore[attr-defined]
            import pyatspi  # noqa: F811
            is_enabled = state_set.contains(pyatspi.STATE_ENABLED)
            is_focused = state_set.contains(pyatspi.STATE_FOCUSED)
        except Exception:
            pass

        value = ""
        try:
            val_iface = node.queryValue()  # type: ignore[attr-defined]
            if val_iface:
                value = str(val_iface.currentValue)
        except Exception:
            pass

        description = ""
        with contextlib.suppress(Exception):
            description = str(getattr(node, "description", "") or "")

        elements.append(UIElement(
            id=0,
            role=role,
            name=name,
            x=x,
            y=y,
            width=w,
            height=h,
            is_enabled=is_enabled,
            is_focused=is_focused,
            value=value,
            parent_name=parent_name,
            parent_role=parent_role,
            description=description,
        ))

        # Walk children — pass this element's name/role as parent info
        cur_name = name or parent_name
        cur_role = role if role_name else parent_role
        child_count = getattr(node, "childCount", 0)
        for i in range(child_count):
            try:
                child = node.getChildAtIndex(i)  # type: ignore[attr-defined]
                if child:
                    _walk_atspi_tree(
                        child, elements, depth + 1, max_depth,
                        parent_name=cur_name, parent_role=cur_role,
                    )
            except Exception:
                continue
    except Exception:
        pass


async def _get_elements_gdbus(window_title: str | None = None) -> list[UIElement]:
    """Fallback extraction using gdbus to query AT-SPI2 D-Bus.

    This is a best-effort fallback when pyatspi is not installed.
    It queries the accessibility bus directly.
    """
    # Check if AT-SPI2 bus is available
    try:
        proc = await asyncio.create_subprocess_exec(
            "gdbus", "call", "--session",
            "--dest", "org.a11y.Bus",
            "--object-path", "/org/a11y/bus",
            "--method", "org.a11y.Bus.GetAddress",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except (FileNotFoundError, TimeoutError):
        logger.warning("gdbus_not_available")
        return []

    if proc.returncode != 0:
        logger.warning("atspi_bus_unavailable")
        return []

    # AT-SPI2 D-Bus introspection is complex; return empty with a helpful message
    logger.info(
        "gdbus_atspi_limited",
        msg="gdbus AT-SPI2 fallback has limited support. "
            "Install python3-pyatspi for full accessibility tree access.",
    )
    return []
