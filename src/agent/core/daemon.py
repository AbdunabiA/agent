"""Daemon service management — install/start/stop the agent as an OS service.

Supports:
- macOS: launchd (~/Library/LaunchAgents/)
- Linux: systemd user service (~/.config/systemd/user/)
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import structlog

from agent.config import get_agent_home

logger = structlog.get_logger(__name__)

# Service identifiers
_MACOS_LABEL = "com.agent.gateway"
_LINUX_SERVICE = "agent"


@dataclass
class DaemonStatus:
    """Current state of the daemon service."""

    installed: bool
    running: bool
    pid: int | None = None
    service_path: str | None = None
    log_path: str | None = None


def _get_agent_executable() -> str:
    """Find the agent executable path."""
    # Prefer the executable in the same venv as the running Python
    venv_bin = Path(sys.executable).parent / "agent"
    if venv_bin.exists():
        return str(venv_bin)

    # Fallback to PATH
    found = shutil.which("agent")
    if found:
        return found

    # Last resort: python -m agent
    return f"{sys.executable} -m agent"


def _log_dir() -> Path:
    """Return the directory for daemon logs."""
    log_dir = get_agent_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# -----------------------------------------------------------------------
# macOS (launchd)
# -----------------------------------------------------------------------

def _macos_plist_path() -> Path:
    """Return the launchd plist file path."""
    return Path.home() / "Library" / "LaunchAgents" / f"{_MACOS_LABEL}.plist"


def _macos_plist_content() -> str:
    """Generate the launchd plist XML."""
    agent_bin = _get_agent_executable()
    log_dir = _log_dir()
    working_dir = str(Path.home() / "Desktop")

    # If the executable has a space (python -m agent), split into args
    parts = agent_bin.split()
    if len(parts) == 1:
        program_args = f"""\
        <string>{agent_bin}</string>
        <string>start</string>"""
    else:
        args_xml = "\n".join(f"        <string>{p}</string>" for p in parts)
        program_args = f"""\
{args_xml}
        <string>start</string>"""

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{_MACOS_LABEL}</string>

            <key>ProgramArguments</key>
            <array>
        {program_args}
            </array>

            <key>WorkingDirectory</key>
            <string>{working_dir}</string>

            <key>RunAtLoad</key>
            <true/>

            <key>KeepAlive</key>
            <dict>
                <key>SuccessfulExit</key>
                <false/>
            </dict>

            <key>ThrottleInterval</key>
            <integer>10</integer>

            <key>StandardOutPath</key>
            <string>{log_dir / "agent-stdout.log"}</string>

            <key>StandardErrorPath</key>
            <string>{log_dir / "agent-stderr.log"}</string>

            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>{os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")}</string>
                <key>HOME</key>
                <string>{Path.home()}</string>
            </dict>
        </dict>
        </plist>
    """)


def _macos_install() -> str:
    """Install the launchd plist."""
    plist = _macos_plist_path()
    if plist.exists():
        return f"Already installed at {plist}. Use 'agent daemon uninstall' first to reinstall."

    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(_macos_plist_content())
    logger.info("daemon_installed", path=str(plist))
    return f"Service installed at {plist}\nRun 'agent daemon start' to start it."


def _macos_uninstall() -> str:
    """Uninstall the launchd plist."""
    plist = _macos_plist_path()
    if not plist.exists():
        return "Service is not installed."

    # Stop first if running
    subprocess.run(
        ["launchctl", "unload", str(plist)],
        capture_output=True,
    )
    plist.unlink()
    logger.info("daemon_uninstalled", path=str(plist))
    return "Service uninstalled."


def _macos_start() -> str:
    """Start the launchd service."""
    plist = _macos_plist_path()
    if not plist.exists():
        return "Service not installed. Run 'agent daemon install' first."

    result = subprocess.run(
        ["launchctl", "load", str(plist)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"Failed to start: {result.stderr.strip()}"

    logger.info("daemon_started")
    return "Agent daemon started. It will auto-restart on crash and start on login."


def _macos_stop() -> str:
    """Stop the launchd service."""
    plist = _macos_plist_path()
    if not plist.exists():
        return "Service not installed."

    result = subprocess.run(
        ["launchctl", "unload", str(plist)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"Failed to stop: {result.stderr.strip()}"

    logger.info("daemon_stopped")
    return "Agent daemon stopped."


def _macos_status() -> DaemonStatus:
    """Check launchd service status."""
    plist = _macos_plist_path()
    if not plist.exists():
        return DaemonStatus(
            installed=False, running=False,
            service_path=str(plist),
        )

    result = subprocess.run(
        ["launchctl", "list", _MACOS_LABEL],
        capture_output=True, text=True,
    )
    running = result.returncode == 0
    pid = None
    if running:
        for line in result.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 1 and parts[0].isdigit():
                pid = int(parts[0])
                break

    log_dir = _log_dir()
    return DaemonStatus(
        installed=True,
        running=running,
        pid=pid,
        service_path=str(plist),
        log_path=str(log_dir),
    )


# -----------------------------------------------------------------------
# Linux (systemd)
# -----------------------------------------------------------------------

def _linux_service_path() -> Path:
    """Return the systemd user service file path."""
    return Path.home() / ".config" / "systemd" / "user" / f"{_LINUX_SERVICE}.service"


def _linux_service_content() -> str:
    """Generate the systemd service unit file."""
    agent_bin = _get_agent_executable()
    log_dir = _log_dir()
    working_dir = str(Path.home() / "Desktop")

    return textwrap.dedent(f"""\
        [Unit]
        Description=Agent AI Assistant
        After=network.target

        [Service]
        Type=simple
        ExecStart={agent_bin} start
        WorkingDirectory={working_dir}
        Restart=on-failure
        RestartSec=10
        StandardOutput=append:{log_dir / "agent-stdout.log"}
        StandardError=append:{log_dir / "agent-stderr.log"}
        Environment=PATH={os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")}
        Environment=HOME={Path.home()}

        [Install]
        WantedBy=default.target
    """)


def _linux_install() -> str:
    """Install the systemd user service."""
    service = _linux_service_path()
    if service.exists():
        return f"Already installed at {service}. Use 'agent daemon uninstall' first to reinstall."

    service.parent.mkdir(parents=True, exist_ok=True)
    service.write_text(_linux_service_content())
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    subprocess.run(
        ["systemctl", "--user", "enable", _LINUX_SERVICE],
        capture_output=True,
    )
    logger.info("daemon_installed", path=str(service))
    return f"Service installed at {service}\nRun 'agent daemon start' to start it."


def _linux_uninstall() -> str:
    """Uninstall the systemd user service."""
    service = _linux_service_path()
    if not service.exists():
        return "Service is not installed."

    subprocess.run(
        ["systemctl", "--user", "stop", _LINUX_SERVICE],
        capture_output=True,
    )
    subprocess.run(
        ["systemctl", "--user", "disable", _LINUX_SERVICE],
        capture_output=True,
    )
    service.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    logger.info("daemon_uninstalled", path=str(service))
    return "Service uninstalled."


def _linux_start() -> str:
    """Start the systemd user service."""
    service = _linux_service_path()
    if not service.exists():
        return "Service not installed. Run 'agent daemon install' first."

    result = subprocess.run(
        ["systemctl", "--user", "start", _LINUX_SERVICE],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"Failed to start: {result.stderr.strip()}"

    logger.info("daemon_started")
    return "Agent daemon started."


def _linux_stop() -> str:
    """Stop the systemd user service."""
    result = subprocess.run(
        ["systemctl", "--user", "stop", _LINUX_SERVICE],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return f"Failed to stop: {result.stderr.strip()}"

    logger.info("daemon_stopped")
    return "Agent daemon stopped."


def _linux_status() -> DaemonStatus:
    """Check systemd user service status."""
    service = _linux_service_path()
    if not service.exists():
        return DaemonStatus(
            installed=False, running=False,
            service_path=str(service),
        )

    result = subprocess.run(
        ["systemctl", "--user", "is-active", _LINUX_SERVICE],
        capture_output=True, text=True,
    )
    running = result.stdout.strip() == "active"

    pid = None
    if running:
        pid_result = subprocess.run(
            ["systemctl", "--user", "show", _LINUX_SERVICE, "--property=MainPID"],
            capture_output=True, text=True,
        )
        for line in pid_result.stdout.strip().split("\n"):
            if line.startswith("MainPID="):
                val = line.split("=", 1)[1]
                if val.isdigit() and val != "0":
                    pid = int(val)

    log_dir = _log_dir()
    return DaemonStatus(
        installed=True,
        running=running,
        pid=pid,
        service_path=str(service),
        log_path=str(log_dir),
    )


# -----------------------------------------------------------------------
# Platform dispatcher
# -----------------------------------------------------------------------

def _get_platform() -> str:
    """Return 'macos', 'linux', or 'unsupported'."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    return "unsupported"


def daemon_install() -> str:
    """Install the agent as an OS service."""
    plat = _get_platform()
    if plat == "macos":
        return _macos_install()
    if plat == "linux":
        return _linux_install()
    return "Daemon service is not supported on this platform. Use 'nohup agent start &' instead."


def daemon_uninstall() -> str:
    """Uninstall the agent OS service."""
    plat = _get_platform()
    if plat == "macos":
        return _macos_uninstall()
    if plat == "linux":
        return _linux_uninstall()
    return "Daemon service is not supported on this platform."


def daemon_start() -> str:
    """Start the agent daemon service."""
    plat = _get_platform()
    if plat == "macos":
        return _macos_start()
    if plat == "linux":
        return _linux_start()
    return "Daemon service is not supported on this platform."


def daemon_stop() -> str:
    """Stop the agent daemon service."""
    plat = _get_platform()
    if plat == "macos":
        return _macos_stop()
    if plat == "linux":
        return _linux_stop()
    return "Daemon service is not supported on this platform."


def daemon_restart() -> str:
    """Restart the agent daemon service."""
    daemon_stop()
    return daemon_start()


def daemon_status() -> DaemonStatus:
    """Get the current daemon service status."""
    plat = _get_platform()
    if plat == "macos":
        return _macos_status()
    if plat == "linux":
        return _linux_status()
    return DaemonStatus(installed=False, running=False)
