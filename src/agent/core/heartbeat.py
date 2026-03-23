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
    from agent.core.proactive import ProactiveMessenger
    from agent.core.scheduler import TaskScheduler
    from agent.llm.claude_sdk import ClaudeSDKService
    from agent.memory.store import FactStore

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
        fact_store: FactStore | None = None,
        scheduler: TaskScheduler | None = None,
        proactive: ProactiveMessenger | None = None,
        sdk_service: ClaudeSDKService | None = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.config = config
        self.event_bus = event_bus
        self.fact_store = fact_store
        self.scheduler = scheduler
        self.proactive = proactive
        self.sdk_service = sdk_service
        self._scheduler = None
        self._consecutive_failures: int = 0
        self._max_failures: int = 3
        self._enabled: bool = True
        self._heartbeat_session: Session | None = None
        self._last_tick: datetime | None = None
        self._tick_lock = asyncio.Lock()
        import time as _t

        self._last_maintenance: float = _t.time()  # avoid maintenance on first tick

    async def start(self) -> None:
        """Start the heartbeat scheduler.

        Parses config.heartbeat_interval and schedules the heartbeat job.
        """
        try:
            from apscheduler.schedulers.asyncio import (  # type: ignore[import-untyped]
                AsyncIOScheduler,
            )
        except ImportError:
            logger.warning(
                "heartbeat_disabled",
                reason="apscheduler not installed. Install with: pip install apscheduler",
            )
            return

        interval = self._parse_interval(self.config.heartbeat_interval)
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self._tick,
            "interval",
            seconds=interval,
            id="heartbeat",
            replace_existing=True,
        )
        scheduler.start()
        self._scheduler = scheduler
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

        # Skip if a tick is already in progress
        if self._tick_lock.locked():
            logger.debug("heartbeat_tick_skipped", reason="previous tick still running")
            return

        async with self._tick_lock:
            # Skip heartbeat if no LLM backend is available at all.
            if self.agent_loop.llm is None and self.sdk_service is None:
                return

            self._last_tick = datetime.now()

            await self.event_bus.emit(
                Events.HEARTBEAT_TICK,
                {
                    "timestamp": datetime.now().isoformat(),
                },
            )

            try:
                checklist = await asyncio.to_thread(self._read_heartbeat_md)

                # Build dynamic context
                context_parts = [
                    f"[HEARTBEAT] The time is {datetime.now().strftime('%Y-%m-%d %H:%M')}.\n"
                    f"Check the following items and take action ONLY if needed. "
                    f"If nothing needs to be done, respond with exactly 'HEARTBEAT_OK'.\n"
                    f"If you need to notify the user, prefix your response with [NOTIFY].\n",
                    f"Checklist:\n{checklist}",
                ]

                # Add pending scheduled tasks
                if self.scheduler:
                    pending = [t for t in self.scheduler.list_tasks() if t.status == "pending"]
                    if pending:
                        task_lines = [
                            f"  - {t.description} ({t.type}: {t.schedule})" for t in pending[:10]
                        ]
                        context_parts.append("\nPending Scheduled Tasks:\n" + "\n".join(task_lines))

                # Add recent memory facts
                if self.fact_store:
                    try:
                        facts = await self.fact_store.get_relevant(limit=5)
                        if facts:
                            context_parts.append(
                                "\nRecent Memory Facts:\n"
                                + "\n".join(f"  - {f.key}: {f.value}" for f in facts)
                            )
                    except Exception:
                        pass

                heartbeat_message = "\n".join(context_parts)

                # Choose LLM backend: prefer LiteLLM, fall back to SDK
                try:
                    if self.agent_loop.llm is not None:
                        response_content = await asyncio.wait_for(
                            self._tick_via_litellm(heartbeat_message),
                            timeout=120,
                        )
                    else:
                        response_content = await asyncio.wait_for(
                            self._tick_via_sdk(heartbeat_message),
                            timeout=120,
                        )
                except TimeoutError:
                    self._consecutive_failures += 1
                    logger.error(
                        "heartbeat_llm_timeout",
                        timeout_seconds=120,
                        consecutive_failures=self._consecutive_failures,
                    )
                    return

                if "HEARTBEAT_OK" in response_content:
                    logger.debug("heartbeat_ok", message="No action needed")
                else:
                    await self.event_bus.emit(
                        Events.HEARTBEAT_ACTION,
                        {
                            "action": response_content,
                        },
                    )
                    logger.info("heartbeat_action", action=response_content[:200])

                    # Check for [NOTIFY] prefix — send proactive notification
                    if "[NOTIFY]" in response_content and self.proactive:
                        notify_content = response_content.replace("[NOTIFY]", "").strip()
                        await self.proactive.send_to_all_known_users(
                            f"📋 **Agent Update**: {notify_content}"
                        )

                self._consecutive_failures = 0

                # Daily maintenance check
                import time as _time

                now = _time.time()
                if now - self._last_maintenance > 86400:
                    await self._run_maintenance()
                    self._last_maintenance = now

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
                    await self.event_bus.emit(
                        Events.AGENT_ERROR,
                        {
                            "type": "circuit_breaker",
                            "component": "heartbeat",
                            "message": "Heartbeat disabled after 3 consecutive failures",
                        },
                    )

    async def _run_maintenance(self) -> None:
        """Run daily maintenance tasks."""
        logger.info("heartbeat_maintenance_started")

        # Optimize database if available via the agent loop's session store
        try:
            session_store = getattr(self.agent_loop, "session_store", None)
            db = getattr(session_store, "_db", None) if session_store else None
            if db is not None:
                async with db.db.execute("PRAGMA optimize") as cursor:
                    await cursor.fetchone()
                logger.info("heartbeat_maintenance_db_optimized")
        except Exception as e:
            logger.warning("heartbeat_maintenance_db_failed", error=str(e))

        # Database backup
        try:
            session_store = getattr(self.agent_loop, "session_store", None)
            db = getattr(session_store, "_db", None) if session_store else None
            if db is not None and hasattr(db, "backup"):
                await db.backup()
                logger.info("heartbeat_maintenance_db_backup_complete")
        except Exception as e:
            logger.warning("heartbeat_maintenance_backup_failed", error=str(e))

        # WAL checkpoint
        try:
            session_store = getattr(self.agent_loop, "session_store", None)
            db = getattr(session_store, "_db", None) if session_store else None
            if db is not None and hasattr(db, "checkpoint"):
                await db.checkpoint()
                logger.info("heartbeat_maintenance_wal_checkpoint_complete")
        except Exception as e:
            logger.warning("heartbeat_maintenance_checkpoint_failed", error=str(e))

        await self.event_bus.emit(
            Events.HEARTBEAT_ACTION,
            {
                "action": "daily_maintenance",
            },
        )
        logger.info("heartbeat_maintenance_complete")

    async def _tick_via_litellm(self, heartbeat_message: str) -> str:
        """Run heartbeat via LiteLLM (original path)."""
        if self._heartbeat_session is None:
            self._heartbeat_session = Session(session_id="heartbeat")

        response = await self.agent_loop.process_message(
            heartbeat_message, self._heartbeat_session, trigger="heartbeat"
        )
        return response.content

    async def _tick_via_sdk(self, heartbeat_message: str) -> str:
        """Run heartbeat via Claude SDK when LiteLLM is unavailable."""
        assert self.sdk_service is not None

        result_parts: list[str] = []
        async for event in self.sdk_service.run_task_stream(
            heartbeat_message,
            task_id="heartbeat",
            channel="heartbeat",
        ):
            if event.type in ("text", "result") and event.content:
                result_parts.append(event.content)

        return "".join(result_parts) or "HEARTBEAT_OK"

    def _read_heartbeat_md(self) -> str:
        """Read the HEARTBEAT.md file.

        Looks for HEARTBEAT.md in project root, then ~/.config/agent/.

        Returns:
            Content of HEARTBEAT.md or a default message.
        """
        from agent.config import get_agent_home

        search_paths = [
            Path("HEARTBEAT.md"),
            get_agent_home() / "HEARTBEAT.md",
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
