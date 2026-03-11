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

    # Screen resolution
    screen_width, screen_height = 1920, 1080
    scale_factor = 1.0

    if has_display and has_pyautogui:
        try:
            import pyautogui

            size = pyautogui.size()
            screen_width, screen_height = size.width, size.height
        except Exception:
            pass

    logger.info(
        "platform_detected",
        os_type=os_type.value,
        display_server=display_server,
        has_pyautogui=has_pyautogui,
        screen=f"{screen_width}x{screen_height}",
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


def get_hotkey_modifier() -> str:
    """Get the platform's primary modifier key.

    Returns "command" on macOS, "ctrl" on Linux/Windows.
    """
    info = get_platform()
    return "command" if info.os_type == OSType.MACOS else "ctrl"
