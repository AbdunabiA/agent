"""Heartbeat daemon — periodic proactive agent actions.

Reads HEARTBEAT.md checklist and asks the LLM if any action is needed.
Uses APScheduler for scheduling with circuit breaker protection.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from agent.core.events import EventBus, Events
from agent.core.session import Session

if TYPE_CHECKING:
    from agent.config import AgentPersonaConfig
    from agent.core.agent_loop import AgentLoop

logger = structlog.get_logger(__name__)


class HeartbeatDaemon:
    """Periodic heartbeat that wakes the agent for proactive actions.

    Reads HEARTBEAT.md checklist and asks the LLM if any action is needed.
    Uses APScheduler for scheduling.
    """

    def __init__(
        self,
        agent_loop: AgentLoop,
        config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> None:
        self.agent_loop = agent_loop
        self.config = config
        self.event_bus = event_bus
        self._scheduler = None
        self._consecutive_failures: int = 0
        self._max_failures: int = 3
        self._enabled: bool = True
        self._heartbeat_session: Session | None = None
        self._last_tick: datetime | None = None

    async def start(self) -> None:
        """Start the heartbeat scheduler.

        Parses config.heartbeat_interval and schedules the heartbeat job.
        """
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
        except ImportError:
            logger.warning(
                "heartbeat_disabled",
                reason="apscheduler not installed. Install with: pip install apscheduler",
            )
            return

        interval = self._parse_interval(self.config.heartbeat_interval)
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self._tick,
            "interval",
            seconds=interval,
            id="heartbeat",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info("heartbeat_started", interval_seconds=interval)

    async def stop(self) -> None:
        """Stop the heartbeat."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("heartbeat_stopped")

    async def _tick(self) -> None:
        """Single heartbeat tick.

        1. Check circuit breaker
        2. Read HEARTBEAT.md
        3. Build heartbeat context message
        4. Send to agent loop
        5. Handle response
        6. Track success/failure
        """
        if not self._enabled:
            return

        self._last_tick = datetime.now()

        await self.event_bus.emit(Events.HEARTBEAT_TICK, {
            "timestamp": datetime.now().isoformat(),
        })

        try:
            checklist = await asyncio.to_thread(self._read_heartbeat_md)

            heartbeat_message = (
                f"[HEARTBEAT] The time is {datetime.now().strftime('%Y-%m-%d %H:%M')}.\n"
                f"Check the following items and take action ONLY if needed. "
                f"If nothing needs to be done, respond with exactly 'HEARTBEAT_OK'.\n\n"
                f"Checklist:\n{checklist}"
            )

            # Use a dedicated heartbeat session
            if self._heartbeat_session is None:
                self._heartbeat_session = Session(session_id="heartbeat")

            response = await self.agent_loop.process_message(
                heartbeat_message, self._heartbeat_session, trigger="heartbeat"
            )

            if "HEARTBEAT_OK" in response.content:
                logger.debug("heartbeat_ok", message="No action needed")
            else:
                await self.event_bus.emit(Events.HEARTBEAT_ACTION, {
                    "action": response.content,
                })
                logger.info("heartbeat_action", action=response.content[:200])

            self._consecutive_failures = 0

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(
                "heartbeat_failed",
                error=str(e),
                consecutive_failures=self._consecutive_failures,
            )

            if self._consecutive_failures >= self._max_failures:
                self._enabled = False
                logger.critical(
                    "heartbeat_circuit_breaker",
                    message="Heartbeat disabled after consecutive failures",
                    failures=self._consecutive_failures,
                )
                await self.event_bus.emit(Events.AGENT_ERROR, {
                    "type": "circuit_breaker",
                    "component": "heartbeat",
                    "message": "Heartbeat disabled after 3 consecutive failures",
                })

    def _read_heartbeat_md(self) -> str:
        """Read the HEARTBEAT.md file.

        Looks for HEARTBEAT.md in project root, then ~/.config/agent/.

        Returns:
            Content of HEARTBEAT.md or a default message.
        """
        search_paths = [
            Path("HEARTBEAT.md"),
            Path.home() / ".config" / "agent" / "HEARTBEAT.md",
        ]

        for path in search_paths:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8")
                except OSError:
                    continue

        return (
            "No HEARTBEAT.md found. Default checks:\n"
            "- Check if any scheduled reminders are due\n"
            "- Report any issues that need attention\n"
        )

    @staticmethod
    def _parse_interval(interval: str) -> int:
        """Parse interval string to seconds.

        Supports: '30m' -> 1800, '1h' -> 3600, '15m' -> 900, '30s' -> 30.

        Args:
            interval: Interval string like '30m', '1h'.

        Returns:
            Interval in seconds.
        """
        match = re.match(r"^(\d+)\s*([smh])?$", interval.strip().lower())
        if not match:
            logger.warning("invalid_interval", interval=interval, default="30m")
            return 1800  # Default 30 minutes

        value = int(match.group(1))
        unit = match.group(2) or "m"

        multipliers = {"s": 1, "m": 60, "h": 3600}
        return value * multipliers[unit]

    @property
    def is_enabled(self) -> bool:
        """Whether the heartbeat is enabled."""
        return self._enabled

    @property
    def last_tick(self) -> datetime | None:
        """Time of the last heartbeat tick."""
        return self._last_tick

    def enable(self) -> None:
        """Re-enable heartbeat (reset circuit breaker)."""
        self._enabled = True
        self._consecutive_failures = 0

    def disable(self) -> None:
        """Manually disable heartbeat."""
        self._enabled = False
