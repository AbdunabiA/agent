"""Monitoring system — watches files, endpoints, git repos, and shell commands.

Periodically checks monitored targets and emits alerts on changes.
Notifies users via ProactiveMessenger.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from agent.core.events import Events

if TYPE_CHECKING:
    from agent.core.events import EventBus
    from agent.core.proactive import ProactiveMessenger

logger = structlog.get_logger(__name__)


class MonitorType(StrEnum):
    """Types of monitors."""

    FILE_CHANGE = "file_change"
    HTTP_ENDPOINT = "http_endpoint"
    GIT_REPO = "git_repo"
    SHELL_COMMAND = "shell_command"


@dataclass
class Monitor:
    """A single monitor definition."""

    id: str
    type: MonitorType
    target: str  # Path, URL, or command
    interval: int = 60  # Check interval in seconds
    channel: str | None = None
    user_id: str | None = None
    description: str = ""
    last_state: str = ""
    last_check: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)


class MonitorManager:
    """Manages periodic monitors that watch for changes.

    Supports: file changes, HTTP endpoints, git repos, shell commands.
    Uses APScheduler for periodic checks.
    """

    def __init__(
        self,
        event_bus: EventBus,
        proactive: ProactiveMessenger | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.proactive = proactive
        self._monitors: dict[str, Monitor] = {}
        self._scheduler: Any = None
        self._started = False

    def start(self) -> None:
        """Start the monitor scheduler."""
        try:
            from apscheduler.schedulers.asyncio import (  # type: ignore[import-untyped]
                AsyncIOScheduler,
            )

            self._scheduler = AsyncIOScheduler()
            self._scheduler.start()
            self._started = True
            logger.info("monitor_manager_started")
        except ImportError:
            logger.warning("monitor_manager_disabled", reason="apscheduler not installed")

    def stop(self) -> None:
        """Stop the monitor scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._started = False
            logger.info("monitor_manager_stopped")

    async def add_monitor(
        self,
        type: MonitorType,
        target: str,
        interval: int = 60,
        channel: str | None = None,
        user_id: str | None = None,
        description: str = "",
    ) -> Monitor:
        """Add a new monitor.

        Args:
            type: Monitor type (file_change, http_endpoint, etc.).
            target: What to monitor (path, URL, command).
            interval: Check interval in seconds.
            channel: Channel to notify on change.
            user_id: User to notify.
            description: Human description of what's monitored.

        Returns:
            The created Monitor.
        """
        monitor_id = str(uuid4())[:8]
        monitor = Monitor(
            id=monitor_id,
            type=type,
            target=target,
            interval=max(10, interval),  # Minimum 10s
            channel=channel,
            user_id=user_id,
            description=description or f"Monitor {type}: {target}",
        )

        self._monitors[monitor_id] = monitor

        # Schedule periodic check
        if self._scheduler and self._started:
            self._scheduler.add_job(
                self._check_monitor,
                "interval",
                seconds=monitor.interval,
                args=[monitor_id],
                id=f"monitor_{monitor_id}",
                replace_existing=True,
            )

        logger.info(
            "monitor_added",
            id=monitor_id,
            type=type,
            target=target,
            interval=interval,
        )
        return monitor

    def remove_monitor(self, monitor_id: str) -> bool:
        """Remove a monitor.

        Args:
            monitor_id: The monitor ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if monitor_id not in self._monitors:
            return False

        self._monitors.pop(monitor_id)

        if self._scheduler:
            import contextlib

            with contextlib.suppress(Exception):
                self._scheduler.remove_job(f"monitor_{monitor_id}")

        logger.info("monitor_removed", id=monitor_id)
        return True

    def list_monitors(self) -> list[dict[str, Any]]:
        """List all active monitors.

        Returns:
            List of monitor info dicts.
        """
        return [
            {
                "id": m.id,
                "type": m.type,
                "target": m.target,
                "interval": m.interval,
                "description": m.description,
                "last_check": m.last_check.isoformat() if m.last_check else None,
                "channel": m.channel,
                "user_id": m.user_id,
            }
            for m in self._monitors.values()
        ]

    async def _check_monitor(self, monitor_id: str) -> None:
        """Check a single monitor for changes."""
        monitor = self._monitors.get(monitor_id)
        if not monitor:
            return

        try:
            new_state = await self._get_state(monitor)
            monitor.last_check = datetime.now()

            if monitor.last_state and new_state != monitor.last_state:
                # Change detected!
                await self._notify_change(monitor, new_state)

            monitor.last_state = new_state

        except Exception as e:
            logger.error(
                "monitor_check_failed",
                id=monitor_id,
                type=monitor.type,
                error=str(e),
            )

    async def _get_state(self, monitor: Monitor) -> str:
        """Get the current state of a monitored target.

        Returns a string hash or state representation.
        """
        if monitor.type == MonitorType.FILE_CHANGE:
            return await self._check_file(monitor.target)
        elif monitor.type == MonitorType.HTTP_ENDPOINT:
            return await self._check_http(monitor.target)
        elif monitor.type == MonitorType.GIT_REPO:
            return await self._check_git(monitor.target)
        elif monitor.type == MonitorType.SHELL_COMMAND:
            return await self._check_shell(monitor.target)
        return ""

    async def _check_file(self, path: str) -> str:
        """Check file modification state."""
        import os

        loop = asyncio.get_event_loop()
        try:
            stat = await loop.run_in_executor(None, os.stat, path)
            return f"{stat.st_mtime}:{stat.st_size}"
        except FileNotFoundError:
            return "not_found"

    async def _check_http(self, url: str) -> str:
        """Check HTTP endpoint state.

        Transient errors (timeout, connection refused) re-raise so that
        _check_monitor logs them without triggering a false change alert.
        """
        import sys

        try:
            # Use -c with a script that reads the URL from argv to avoid injection
            script = (
                "import sys, urllib.request; "
                "r = urllib.request.urlopen(sys.argv[1], timeout=10); "
                "print(r.status, len(r.read()))"
            )
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script, url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = stdout.decode(errors="ignore").strip()
            if proc.returncode != 0:
                raise ConnectionError(f"HTTP check failed (exit {proc.returncode})")
            return output
        except (TimeoutError, ConnectionError, OSError) as e:
            # Re-raise transient errors so _check_monitor logs them
            # without treating them as a state change
            raise RuntimeError(f"HTTP check transient error: {e}") from e

    async def _check_git(self, path: str) -> str:
        """Check git repo HEAD state."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", path, "rev-parse", "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode(errors="ignore").strip()
        except Exception:
            return "error"

    async def _check_shell(self, command: str) -> str:
        """Run a shell command and hash its output.

        Uses explicit shell invocation via ``exec`` to avoid direct
        ``create_subprocess_shell``, keeping the command visible in
        process arguments for auditing.
        """
        import sys

        if sys.platform == "win32":
            shell_args = ["cmd", "/c", command]
        else:
            shell_args = ["sh", "-c", command]

        try:
            proc = await asyncio.create_subprocess_exec(
                *shell_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode(errors="ignore")
            return hashlib.md5(output.encode()).hexdigest()
        except Exception as e:
            return f"error:{e}"

    async def _notify_change(self, monitor: Monitor, new_state: str) -> None:
        """Notify user about a monitor change."""
        message = (
            f"🔔 **Monitor Alert**: {monitor.description}\n"
            f"Type: {monitor.type}\n"
            f"Target: {monitor.target}\n"
            f"Change detected at {datetime.now().strftime('%H:%M:%S')}"
        )

        await self.event_bus.emit(Events.MONITORING_ALERT, {
            "monitor_id": monitor.id,
            "type": monitor.type,
            "target": monitor.target,
            "description": monitor.description,
        })

        if self.proactive and monitor.user_id:
            await self.proactive.send_proactive(
                user_id=monitor.user_id,
                content=message,
                channel=monitor.channel,
            )

        logger.info(
            "monitor_change_detected",
            id=monitor.id,
            type=monitor.type,
            target=monitor.target,
        )
