"""Cross-platform window management.

List, focus, close, minimize, and maximize windows.
Uses platform-specific APIs for each OS.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from agent.desktop.errors import desktop_op
from agent.desktop.platform_utils import OSType, get_platform

logger = structlog.get_logger(__name__)


@dataclass
class WindowInfo:
    """Information about an open window."""

    id: str  # Window ID (platform-specific)
    title: str
    app_name: str
    x: int
    y: int
    width: int
    height: int
    is_active: bool
    is_minimized: bool


@desktop_op("window_list")
async def list_windows() -> list[WindowInfo]:
    """List all open windows.

    Returns:
        List of WindowInfo for each visible window.
    """
    info = get_platform()

    if info.os_type == OSType.LINUX:
        return await _list_windows_linux()
    elif info.os_type == OSType.MACOS:
        return await _list_windows_macos()
    elif info.os_type == OSType.WINDOWS:
        return await _list_windows_windows()

    return []


@desktop_op("window_focus")
async def focus_window(title_or_id: str) -> str:
    """Bring a window to the foreground.

    Args:
        title_or_id: Window title (partial match) or window ID.

    Returns:
        Status message.
    """
    info = get_platform()

    if info.os_type == OSType.LINUX:
        return await _focus_window_linux(title_or_id)
    elif info.os_type == OSType.MACOS:
        return await _focus_window_macos(title_or_id)
    elif info.os_type == OSType.WINDOWS:
        return await _focus_window_windows(title_or_id)

    return f"[ERROR] Window management not supported on {info.os_type.value}"


@desktop_op("window_close")
async def close_window(title_or_id: str) -> str:
    """Close a window by title.

    Args:
        title_or_id: Window title (partial match) or window ID.

    Returns:
        Status message.
    """
    info = get_platform()

    if info.os_type == OSType.LINUX:
        return await _close_window_linux(title_or_id)
    elif info.os_type == OSType.MACOS:
        return await _close_window_macos(title_or_id)
    elif info.os_type == OSType.WINDOWS:
        return await _close_window_windows(title_or_id)

    return f"[ERROR] Window management not supported on {info.os_type.value}"


@desktop_op("window_minimize")
async def minimize_window(title_or_id: str) -> str:
    """Minimize a window.

    Args:
        title_or_id: Window title (partial match).

    Returns:
        Status message.
    """
    info = get_platform()

    if info.os_type == OSType.WINDOWS:
        loop = asyncio.get_event_loop()

        def _minimize() -> str:
            try:
                import pygetwindow as gw

                windows = gw.getWindowsWithTitle(title_or_id)
                if windows:
                    windows[0].minimize()
                    return f"Minimized window '{title_or_id}'"
                return f"[ERROR] No window found matching '{title_or_id}'"
            except ImportError:
                return "[ERROR] pygetwindow required on Windows. pip install pygetwindow"

        return await loop.run_in_executor(None, _minimize)

    elif info.os_type == OSType.MACOS:
        script = (
            'tell application "System Events"\n'
            f'  set targetProc to first application process whose name contains "{title_or_id}"\n'
            "  click (first button of first window of targetProc whose "
            'subrole is "AXMinimizeButton")\n'
            "end tell"
        )
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode == 0:
            return f"Minimized window '{title_or_id}'"
        return f"[ERROR] Could not minimize window '{title_or_id}'"

    elif info.os_type == OSType.LINUX:
        if info.has_xdotool:
            proc = await asyncio.create_subprocess_exec(
                "xdotool", "search", "--name", title_or_id, "windowminimize",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            if proc.returncode == 0:
                return f"Minimized window '{title_or_id}'"
        return f"[ERROR] Could not minimize window '{title_or_id}'"

    return f"[ERROR] Minimize not supported on {info.os_type.value}"


@desktop_op("window_maximize")
async def maximize_window(title_or_id: str) -> str:
    """Maximize a window.

    Args:
        title_or_id: Window title (partial match).

    Returns:
        Status message.
    """
    info = get_platform()

    if info.os_type == OSType.WINDOWS:
        loop = asyncio.get_event_loop()

        def _maximize() -> str:
            try:
                import pygetwindow as gw

                windows = gw.getWindowsWithTitle(title_or_id)
                if windows:
                    windows[0].maximize()
                    return f"Maximized window '{title_or_id}'"
                return f"[ERROR] No window found matching '{title_or_id}'"
            except ImportError:
                return "[ERROR] pygetwindow required on Windows. pip install pygetwindow"

        return await loop.run_in_executor(None, _maximize)

    elif info.os_type == OSType.LINUX:
        if info.has_wmctrl:
            proc = await asyncio.create_subprocess_exec(
                "wmctrl", "-r", title_or_id, "-b", "add,maximized_vert,maximized_horz",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            if proc.returncode == 0:
                return f"Maximized window '{title_or_id}'"
        return f"[ERROR] Could not maximize window '{title_or_id}'"

    return f"[ERROR] Maximize not supported on {info.os_type.value}"


# --- Linux implementations ---


async def _list_windows_linux() -> list[WindowInfo]:
    """List windows on Linux using wmctrl or swaymsg (Wayland)."""
    info = get_platform()
    windows: list[WindowInfo] = []

    # Try swaymsg for Wayland/Sway
    if info.display_server == "wayland":
        import shutil

        if shutil.which("swaymsg"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "swaymsg", "-t", "get_tree", "-r",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    import json

                    tree = json.loads(stdout.decode(errors="ignore"))
                    windows = _parse_sway_tree(tree)
                    if windows:
                        return windows
            except Exception:
                pass

    if info.has_wmctrl:
        proc = await asyncio.create_subprocess_exec(
            "wmctrl", "-l", "-G",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        for line in stdout.decode(errors="ignore").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(None, 8)
            if len(parts) >= 9:
                windows.append(WindowInfo(
                    id=parts[0],
                    title=parts[8],
                    app_name=parts[8].rsplit(" - ", 1)[-1] if " - " in parts[8] else parts[8],
                    x=int(parts[2]),
                    y=int(parts[3]),
                    width=int(parts[4]),
                    height=int(parts[5]),
                    is_active=False,
                    is_minimized=False,
                ))

    return windows


async def _focus_window_linux(title_or_id: str) -> str:
    """Focus window on Linux."""
    info = get_platform()

    if info.has_xdotool:
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "search", "--name", title_or_id, "windowactivate",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode == 0:
            return f"Focused window matching '{title_or_id}'"

    if info.has_wmctrl:
        proc = await asyncio.create_subprocess_exec(
            "wmctrl", "-a", title_or_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode == 0:
            return f"Focused window matching '{title_or_id}'"

    return f"[ERROR] No window found matching '{title_or_id}' (install xdotool or wmctrl)"


async def _close_window_linux(title_or_id: str) -> str:
    """Close window on Linux."""
    info = get_platform()

    if info.has_xdotool:
        proc = await asyncio.create_subprocess_exec(
            "xdotool", "search", "--name", title_or_id, "windowclose",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode == 0:
            return f"Closed window matching '{title_or_id}'"

    if info.has_wmctrl:
        proc = await asyncio.create_subprocess_exec(
            "wmctrl", "-c", title_or_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode == 0:
            return f"Closed window matching '{title_or_id}'"

    return f"[ERROR] Could not close window '{title_or_id}'"


# --- macOS implementations ---


async def _list_windows_macos() -> list[WindowInfo]:
    """List windows on macOS using AppleScript."""
    script = (
        'tell application "System Events"\n'
        "  set windowList to {}\n"
        "  repeat with proc in (every application process whose visible is true)\n"
        "    set appName to name of proc\n"
        "    repeat with win in (every window of proc)\n"
        "      set winTitle to name of win\n"
        "      set {winX, winY} to position of win\n"
        "      set {winW, winH} to size of win\n"
        '      set end of windowList to appName & "|" & winTitle & "|" & '
        'winX & "|" & winY & "|" & winW & "|" & winH\n'
        "    end repeat\n"
        "  end repeat\n"
        "  return windowList\n"
        "end tell"
    )
    proc = await asyncio.create_subprocess_exec(
        "osascript", "-e", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    windows: list[WindowInfo] = []
    for line in stdout.decode(errors="ignore").strip().split(", "):
        parts = line.split("|")
        if len(parts) >= 6:
            windows.append(WindowInfo(
                id="",
                title=parts[1],
                app_name=parts[0],
                x=int(parts[2]) if parts[2].strip().lstrip("-").isdigit() else 0,
                y=int(parts[3]) if parts[3].strip().lstrip("-").isdigit() else 0,
                width=int(parts[4]) if parts[4].strip().isdigit() else 0,
                height=int(parts[5]) if parts[5].strip().isdigit() else 0,
                is_active=False,
                is_minimized=False,
            ))

    return windows


async def _focus_window_macos(title_or_id: str) -> str:
    """Focus window on macOS."""
    script = (
        'tell application "System Events"\n'
        "  set frontApp to first application process "
        f'whose name contains "{title_or_id}"\n'
        "  set frontmost of frontApp to true\n"
        "end tell"
    )
    proc = await asyncio.create_subprocess_exec(
        "osascript", "-e", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    if proc.returncode == 0:
        return f"Focused window matching '{title_or_id}'"
    return f"[ERROR] No window found matching '{title_or_id}'"


async def _close_window_macos(title_or_id: str) -> str:
    """Close window on macOS."""
    script = f'tell application "{title_or_id}" to quit'
    proc = await asyncio.create_subprocess_exec(
        "osascript", "-e", script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.wait()
    if proc.returncode == 0:
        return f"Closed window matching '{title_or_id}'"
    return f"[ERROR] Could not close '{title_or_id}'"


# --- Windows implementations ---


async def _list_windows_windows() -> list[WindowInfo]:
    """List windows on Windows using pygetwindow or ctypes fallback."""
    loop = asyncio.get_event_loop()

    def _list() -> list[WindowInfo]:
        try:
            import pygetwindow as gw

            windows: list[WindowInfo] = []
            for win in gw.getAllWindows():
                if win.title and win.visible:
                    windows.append(WindowInfo(
                        id=str(win._hWnd),
                        title=win.title,
                        app_name=(
                            win.title.rsplit(" - ", 1)[-1]
                            if " - " in win.title
                            else win.title
                        ),
                        x=win.left,
                        y=win.top,
                        width=win.width,
                        height=win.height,
                        is_active=win.isActive,
                        is_minimized=win.isMinimized,
                    ))
            return windows
        except ImportError:
            # ctypes fallback for Windows
            return _list_windows_ctypes()

    return await loop.run_in_executor(None, _list)


def _list_windows_ctypes() -> list[WindowInfo]:
    """List windows on Windows using ctypes (fallback when pygetwindow not installed)."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        windows: list[WindowInfo] = []

        def enum_callback(hwnd: int, _: int) -> bool:
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if not title:
                return True

            # Get window rect
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))

            windows.append(WindowInfo(
                id=str(hwnd),
                title=title,
                app_name=title.rsplit(" - ", 1)[-1] if " - " in title else title,
                x=rect.left,
                y=rect.top,
                width=rect.right - rect.left,
                height=rect.bottom - rect.top,
                is_active=(hwnd == user32.GetForegroundWindow()),
                is_minimized=bool(user32.IsIconic(hwnd)),
            ))
            return True

        enum_windows_proc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HWND, wintypes.LPARAM
        )
        user32.EnumWindows(enum_windows_proc(enum_callback), 0)
        return windows
    except Exception as e:
        logger.warning("ctypes_window_list_failed", error=str(e))
        return []


async def _focus_window_windows(title_or_id: str) -> str:
    """Focus window on Windows."""
    loop = asyncio.get_event_loop()

    def _focus() -> str:
        try:
            import pygetwindow as gw

            windows = gw.getWindowsWithTitle(title_or_id)
            if windows:
                windows[0].activate()
                return f"Focused window matching '{title_or_id}'"
            return f"[ERROR] No window found matching '{title_or_id}'"
        except ImportError:
            # ctypes fallback
            return _focus_window_ctypes(title_or_id)
        except Exception as e:
            return f"[ERROR] Could not focus window: {e}"

    return await loop.run_in_executor(None, _focus)


def _focus_window_ctypes(title_or_id: str) -> str:
    """Focus window using ctypes (fallback)."""
    try:
        import ctypes

        user32 = ctypes.windll.user32

        # Try as hwnd first
        try:
            hwnd = int(title_or_id)
            if user32.IsWindow(hwnd):
                user32.SetForegroundWindow(hwnd)
                # Restore if minimized
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                return f"Focused window {hwnd}"
        except ValueError:
            pass

        # Search by title
        windows = _list_windows_ctypes()
        for win in windows:
            if title_or_id.lower() in win.title.lower():
                hwnd = int(win.id)
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
                return f"Focused window matching '{title_or_id}'"

        return f"[ERROR] No window found matching '{title_or_id}'"
    except Exception as e:
        return f"[ERROR] Could not focus window: {e}"


async def _close_window_windows(title_or_id: str) -> str:
    """Close window on Windows."""
    loop = asyncio.get_event_loop()

    def _close() -> str:
        try:
            import pygetwindow as gw

            windows = gw.getWindowsWithTitle(title_or_id)
            if windows:
                windows[0].close()
                return f"Closed window matching '{title_or_id}'"
            return f"[ERROR] No window found matching '{title_or_id}'"
        except ImportError:
            return "[ERROR] pygetwindow required on Windows. pip install pygetwindow"
        except Exception as e:
            return f"[ERROR] Could not close window: {e}"

    return await loop.run_in_executor(None, _close)


def _parse_sway_tree(node: dict, windows: list[WindowInfo] | None = None) -> list[WindowInfo]:
    """Parse swaymsg get_tree output into WindowInfo list."""
    if windows is None:
        windows = []

    if node.get("type") == "con" and node.get("name"):
        rect = node.get("rect", {})
        windows.append(WindowInfo(
            id=str(node.get("id", "")),
            title=node.get("name", ""),
            app_name=node.get("app_id", "") or node.get("name", ""),
            x=rect.get("x", 0),
            y=rect.get("y", 0),
            width=rect.get("width", 0),
            height=rect.get("height", 0),
            is_active=node.get("focused", False),
            is_minimized=False,
        ))

    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        _parse_sway_tree(child, windows)

    return windows
