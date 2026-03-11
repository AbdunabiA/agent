"""Cross-platform application launching and management."""

from __future__ import annotations

import asyncio
import subprocess
import webbrowser
from pathlib import Path

import structlog

from agent.desktop.platform_utils import OSType, get_app_launch_command, get_platform

logger = structlog.get_logger(__name__)


async def launch_app(
    app_name: str,
    args: list[str] | None = None,
    wait: bool = False,
) -> str:
    """Launch an application.

    Args:
        app_name: Application name or path.
            Linux: "firefox", "code", "nautilus"
            macOS: "Safari", "Visual Studio Code", "Finder"
            Windows: "notepad", "chrome", "explorer"
        args: Additional arguments to pass to the application.
        wait: If True, wait for the app to close before returning.

    Returns:
        Status message.
    """
    cmd = get_app_launch_command(app_name)
    if args:
        cmd.extend(args)

    try:
        if wait:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            logger.info("app_launched_and_closed", app=app_name, exit_code=proc.returncode)
            return f"App '{app_name}' launched and closed (exit code: {proc.returncode})"
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                ),
            )
            logger.info("app_launched", app=app_name)
            return f"App '{app_name}' launched"
    except FileNotFoundError:
        return f"[ERROR] App '{app_name}' not found. Check if it's installed and in PATH."
    except Exception as e:
        return f"[ERROR] Failed to launch '{app_name}': {e}"


async def open_file(file_path: str) -> str:
    """Open a file with its default application.

    Args:
        file_path: Path to the file.

    Returns:
        Status message.
    """
    path = Path(file_path).expanduser().resolve()  # noqa: ASYNC240
    if not path.exists():
        return f"[ERROR] File not found: {path}"

    info = get_platform()

    if info.os_type == OSType.MACOS:
        cmd = ["open", str(path)]
    elif info.os_type == OSType.WINDOWS:
        cmd = ["cmd", "/c", "start", "", str(path)]
    else:
        cmd = ["xdg-open", str(path)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        logger.info("file_opened", path=str(path))
        return f"Opened '{path.name}' with default application"
    except Exception as e:
        return f"[ERROR] Failed to open '{file_path}': {e}"


async def open_url(url: str) -> str:
    """Open a URL in the default web browser.

    Args:
        url: URL to open.

    Returns:
        Status message.
    """
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, lambda: webbrowser.open(url))

    if success:
        logger.info("url_opened", url=url)
        return f"Opened {url} in default browser"
    return f"[ERROR] Failed to open URL: {url}"


async def list_installed_apps() -> str:
    """List installed applications.

    Platform-specific:
    - Linux: scan /usr/share/applications/*.desktop
    - macOS: scan /Applications/ + ~/Applications/
    - Windows: scan Program Files directories

    Returns:
        Formatted list of installed applications.
    """
    info = get_platform()
    apps: list[str] = []

    if info.os_type == OSType.LINUX:
        desktop_dirs = [
            Path("/usr/share/applications"),
            Path.home() / ".local/share/applications",
        ]
        for dir_path in desktop_dirs:
            if dir_path.exists():
                for f in dir_path.glob("*.desktop"):
                    try:
                        name = None
                        for line in f.read_text(errors="ignore").split("\n"):
                            if line.startswith("Name=") and name is None:
                                name = line.split("=", 1)[1].strip()
                                break
                        if name:
                            apps.append(name)
                    except Exception:
                        pass

    elif info.os_type == OSType.MACOS:
        for apps_dir in [Path("/Applications"), Path.home() / "Applications"]:
            if apps_dir.exists():
                for f in apps_dir.iterdir():
                    if f.suffix == ".app":
                        apps.append(f.stem)

    elif info.os_type == OSType.WINDOWS:
        for prog_dir in [
            Path(r"C:\Program Files"),
            Path(r"C:\Program Files (x86)"),
        ]:
            if prog_dir.exists():
                try:
                    for f in prog_dir.iterdir():
                        if f.is_dir():
                            apps.append(f.name)
                except PermissionError:
                    pass

    apps = sorted(set(apps))

    lines = [f"Installed Applications ({len(apps)}):"]
    for app in apps[:100]:
        lines.append(f"  - {app}")
    if len(apps) > 100:
        lines.append(f"  ... and {len(apps) - 100} more")

    return "\n".join(lines)
