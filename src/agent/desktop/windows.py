"""Cross-platform window management.

List, focus, close, minimize, and maximize windows.
Uses platform-specific APIs for each OS.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

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
    """List windows on Linux using wmctrl."""
    info = get_platform()
    windows: list[WindowInfo] = []

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
    """List windows on Windows using pygetwindow."""
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
            logger.warning("pygetwindow_not_installed")
            return []

    return await loop.run_in_executor(None, _list)


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
            return "[ERROR] pygetwindow required on Windows. pip install pygetwindow"
        except Exception as e:
            return f"[ERROR] Could not focus window: {e}"

    return await loop.run_in_executor(None, _focus)


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
