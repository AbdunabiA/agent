"""Cross-platform utility detection.

Detects OS and available capabilities once at startup.
Provides platform-specific command builders.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)


class OSType(StrEnum):
    """Supported operating systems."""

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlatformInfo:
    """Immutable snapshot of detected platform capabilities."""

    os_type: OSType
    has_display: bool
    display_server: str  # "x11", "wayland", "quartz", "win32", ""
    has_pyautogui: bool
    has_wmctrl: bool  # Linux window management
    has_xdotool: bool  # Linux window management
    has_osascript: bool  # macOS AppleScript
    screen_width: int
    screen_height: int
    scale_factor: float  # HiDPI scaling
    # Wayland-specific tools
    has_wtype: bool = False  # Wayland keyboard input
    has_ydotool: bool = False  # Wayland mouse/keyboard
    has_grim: bool = False  # Wayland screenshot
    has_slurp: bool = False  # Wayland region selection
    # Windows-specific
    has_pygetwindow: bool = False
    # Accessibility APIs
    has_uiautomation: bool = False  # Windows UI Automation
    has_pyatspi: bool = False  # Linux AT-SPI
    # Multi-monitor
    monitor_count: int = 1


def detect_platform() -> PlatformInfo:
    """Detect current platform and capabilities.

    Called once at startup; result is cached via get_platform().
    """
    system = platform.system().lower()

    if system == "darwin":
        os_type = OSType.MACOS
    elif system == "windows" or "mingw" in system or "msys" in system:
        os_type = OSType.WINDOWS
    elif system == "linux":
        os_type = OSType.LINUX
    else:
        os_type = OSType.UNKNOWN

    # Check display availability
    has_display = False
    display_server = ""

    if os_type == OSType.LINUX:
        wayland = os.environ.get("WAYLAND_DISPLAY", "")
        display = os.environ.get("DISPLAY", "")
        if wayland:
            has_display = True
            display_server = "wayland"
        elif display:
            has_display = True
            display_server = "x11"
    elif os_type == OSType.MACOS:
        has_display = True
        display_server = "quartz"
    elif os_type == OSType.WINDOWS:
        has_display = True
        display_server = "win32"

    # Check pyautogui availability
    has_pyautogui = False
    try:
        import pyautogui  # noqa: F401

        has_pyautogui = True
    except ImportError:
        pass

    # Check platform-specific window management tools
    has_wmctrl = shutil.which("wmctrl") is not None
    has_xdotool = shutil.which("xdotool") is not None
    has_osascript = shutil.which("osascript") is not None

    # Wayland-specific tools
    has_wtype = shutil.which("wtype") is not None
    has_ydotool = shutil.which("ydotool") is not None
    has_grim = shutil.which("grim") is not None
    has_slurp = shutil.which("slurp") is not None

    # Windows-specific
    has_pygetwindow = False
    try:
        import pygetwindow  # noqa: F401

        has_pygetwindow = True
    except ImportError:
        pass

    # Accessibility APIs
    has_uiautomation = False
    if os_type == OSType.WINDOWS:
        try:
            import uiautomation  # noqa: F401

            has_uiautomation = True
        except ImportError:
            pass

    has_pyatspi = False
    if os_type == OSType.LINUX:
        try:
            import pyatspi  # noqa: F401

            has_pyatspi = True
        except ImportError:
            pass

    # Screen resolution
    screen_width, screen_height = 1920, 1080
    scale_factor = 1.0
    monitor_count = 1

    if has_display and has_pyautogui:
        try:
            import pyautogui

            size = pyautogui.size()
            screen_width, screen_height = size.width, size.height
        except Exception:
            pass

    # Try to detect scale factor and monitor count
    if os_type == OSType.WINDOWS:
        try:
            import ctypes

            awareness = ctypes.c_int()
            ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
            dpi = ctypes.windll.user32.GetDpiForSystem()
            scale_factor = dpi / 96.0
            monitor_count = ctypes.windll.user32.GetSystemMetrics(80) or 1  # SM_CMONITORS
        except Exception:
            pass
    elif os_type == OSType.MACOS:
        try:
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5,
            )
            # Count "Resolution:" lines
            lines = [ln for ln in result.stdout.split("\n") if "Resolution:" in ln]
            monitor_count = max(1, len(lines))
            # Check for "Retina" in output
            if "Retina" in result.stdout:
                scale_factor = 2.0
        except Exception:
            pass

    logger.info(
        "platform_detected",
        os_type=os_type.value,
        display_server=display_server,
        has_pyautogui=has_pyautogui,
        screen=f"{screen_width}x{screen_height}",
        monitors=monitor_count,
        scale=scale_factor,
    )

    return PlatformInfo(
        os_type=os_type,
        has_display=has_display,
        display_server=display_server,
        has_pyautogui=has_pyautogui,
        has_wmctrl=has_wmctrl,
        has_xdotool=has_xdotool,
        has_osascript=has_osascript,
        screen_width=screen_width,
        screen_height=screen_height,
        scale_factor=scale_factor,
        has_wtype=has_wtype,
        has_ydotool=has_ydotool,
        has_grim=has_grim,
        has_slurp=has_slurp,
        has_pygetwindow=has_pygetwindow,
        has_uiautomation=has_uiautomation,
        has_pyatspi=has_pyatspi,
        monitor_count=monitor_count,
    )


# Module-level cached platform info
_platform_info: PlatformInfo | None = None


def get_platform() -> PlatformInfo:
    """Get cached platform info (detects on first call)."""
    global _platform_info
    if _platform_info is None:
        _platform_info = detect_platform()
    return _platform_info


def reset_platform_cache() -> None:
    """Reset the cached platform info. Useful for testing."""
    global _platform_info
    _platform_info = None


def get_app_launch_command(app_name: str) -> list[str]:
    """Get the platform-specific command to launch an application.

    Args:
        app_name: Application name or path.

    Returns:
        Command list suitable for subprocess.
    """
    info = get_platform()

    if info.os_type == OSType.MACOS:
        return ["open", "-a", app_name]
    elif info.os_type == OSType.WINDOWS:
        return ["cmd", "/c", "start", "", app_name]
    else:  # Linux
        if shutil.which(app_name.lower()):
            return [app_name.lower()]
        return ["xdg-open", app_name]


def get_capabilities() -> dict[str, bool]:
    """Get a dict of available desktop capabilities.

    Returns:
        Dict mapping capability name to availability.
    """
    info = get_platform()
    caps: dict[str, bool] = {
        "display": info.has_display,
        "screenshots": info.has_pyautogui or info.has_grim,
        "mouse_control": info.has_pyautogui or info.has_ydotool,
        "keyboard_input": info.has_pyautogui or info.has_wtype,
        "window_management": (
            info.has_wmctrl
            or info.has_xdotool
            or info.has_osascript
            or info.has_pygetwindow
        ),
        "app_launching": True,  # Always available via subprocess
        "wayland_native": info.display_server == "wayland" and (info.has_grim or info.has_wtype),
        "accessibility_tree": (
            info.has_uiautomation
            or info.has_pyatspi
            or info.has_osascript  # macOS uses AppleScript, no extra dep
        ),
    }
    return caps


def get_capabilities_summary() -> str:
    """Get a human-readable summary of desktop capabilities.

    Suitable for injection into the LLM system prompt.

    Returns:
        Multi-line string describing available capabilities.
    """
    info = get_platform()
    caps = get_capabilities()

    lines = [
        f"OS: {info.os_type.value}, Display: {info.display_server or 'none'}",
        f"Screen: {info.screen_width}x{info.screen_height} "
        f"(scale: {info.scale_factor}x, monitors: {info.monitor_count})",
    ]

    available = [k for k, v in caps.items() if v]
    unavailable = [k for k, v in caps.items() if not v]

    if available:
        lines.append(f"Available: {', '.join(available)}")
    if unavailable:
        lines.append(f"Unavailable: {', '.join(unavailable)}")

    # Accessibility tree info
    if caps.get("accessibility_tree"):
        if info.os_type == OSType.WINDOWS:
            lines.append("Accessibility: uiautomation (UI element detection + SoM)")
        elif info.os_type == OSType.MACOS:
            lines.append("Accessibility: AppleScript (UI element detection + SoM)")
        elif info.os_type == OSType.LINUX:
            lines.append("Accessibility: pyatspi (UI element detection + SoM)")
        lines.append(
            "Preferred: interact(target, action) for one-step actions, "
            "screen_read() for text-only state, find_element(name) for search"
        )
    else:
        if info.os_type == OSType.WINDOWS:
            lines.append(
                "Accessibility: not available. "
                "Install uiautomation (pip install uiautomation) for UI element detection."
            )
        elif info.os_type == OSType.LINUX:
            lines.append(
                "Accessibility: not available. "
                "Install python3-pyatspi for UI element detection."
            )

    if info.display_server == "wayland":
        tools = []
        if info.has_grim:
            tools.append("grim (screenshots)")
        if info.has_wtype:
            tools.append("wtype (keyboard)")
        if info.has_ydotool:
            tools.append("ydotool (mouse)")
        if info.has_slurp:
            tools.append("slurp (region select)")
        if tools:
            lines.append(f"Wayland tools: {', '.join(tools)}")
        else:
            lines.append(
                "Note: Wayland detected but no native tools found. "
                "Install grim, wtype, ydotool for full support."
            )

    return "\n".join(lines)


def get_hotkey_modifier() -> str:
    """Get the platform's primary modifier key.

    Returns "command" on macOS, "ctrl" on Linux/Windows.
    """
    info = get_platform()
    return "command" if info.os_type == OSType.MACOS else "ctrl"
