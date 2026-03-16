"""Task scheduler — manages scheduled tasks, reminders, and cron jobs.

Uses APScheduler with an async event loop scheduler.
Supports persistent tasks via SQLite and natural language schedule parsing.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from agent.core.events import EventBus, Events

if TYPE_CHECKING:
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)

# Type for the delivery callback: (description, channel, user_id) -> None
DeliveryCallback = Callable[[str, str | None, str | None], Awaitable[None]]


@dataclass
class ScheduledTask:
    """A scheduled task (reminder or cron job)."""

    id: str
    description: str
    type: str  # "reminder", "cron"
    schedule: str  # ISO datetime or cron expression
    status: str = "pending"  # "pending", "running", "completed", "failed"
    channel: str | None = None
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    next_run: datetime | None = None
    last_run: datetime | None = None


# Natural language to cron expression mappings
_NaturalReplacement = str | Callable[[re.Match[str]], str]

_NATURAL_PATTERNS: list[tuple[str, _NaturalReplacement]] = [
    # "every morning at 8am" -> "0 8 * * *"
    (r"every\s+morning\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * *"),
    # "every evening at 6pm" / "every evening at 18"
    (r"every\s+evening\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * *"),
    # "every day at noon"
    (r"every\s+day\s+at\s+noon", "0 12 * * *"),
    # "every day at midnight"
    (r"every\s+day\s+at\s+midnight", "0 0 * * *"),
    # "daily at <hour>am"
    (r"daily\s+at\s+(\d{1,2})\s*am", r"0 \1 * * *"),
    # "daily at <hour>pm"
    (r"daily\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * *"),
    # "daily at <hour>" (24h)
    (r"daily\s+at\s+(\d{1,2}):(\d{2})", r"\2 \1 * * *"),
    # "every <N> minutes"
    (r"every\s+(\d+)\s+minutes?", r"*/\1 * * * *"),
    # "every <N> hours"
    (r"every\s+(\d+)\s+hours?", r"0 */\1 * * *"),
    # "every hour"
    (r"every\s+hour", "0 * * * *"),
    # Day-of-week patterns: pm-specific MUST come before am-optional to match correctly
    (r"every\s+monday\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * 1"),
    (r"every\s+monday\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * 1"),
    (r"every\s+tuesday\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * 2"),
    (r"every\s+tuesday\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * 2"),
    (
        r"every\s+wednesday\s+at\s+(\d{1,2})\s*pm",
        lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * 3",
    ),
    (r"every\s+wednesday\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * 3"),
    (r"every\s+thursday\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * 4"),
    (r"every\s+thursday\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * 4"),
    (r"every\s+friday\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * 5"),
    (r"every\s+friday\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * 5"),
    (r"every\s+saturday\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * 6"),
    (r"every\s+saturday\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * 6"),
    (r"every\s+sunday\s+at\s+(\d{1,2})\s*pm", lambda m: f"0 {(int(m.group(1)) % 12) + 12} * * 0"),
    (r"every\s+sunday\s+at\s+(\d{1,2})\s*(?:am)?", r"0 \1 * * 0"),
]


def parse_natural_schedule(text: str) -> str | None:
    """Parse natural language schedule to cron expression.

    Args:
        text: Natural language like "every morning at 8am".

    Returns:
        Cron expression string, or None if not parseable.
    """
    text_lower = text.strip().lower()

    for pattern, replacement in _NATURAL_PATTERNS:
        match = re.match(pattern, text_lower)
        if match:
            if callable(replacement):
                return replacement(match)
            return match.expand(replacement)

    # Check if it's already a cron expression (5 space-separated fields)
    parts = text_lower.split()
    if len(parts) == 5 and all(
        re.match(r"^[\d*/,-]+$", p) for p in parts
    ):
        return text_lower

    return None


class TaskScheduler:
    """Manages scheduled tasks: reminders, cron jobs, one-time events.

    Uses APScheduler with an async event loop scheduler.
    Tasks are persisted to SQLite for survival across restarts.
    """

    def __init__(
        self,
        event_bus: EventBus,
        database: Database | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.database = database
        self._scheduler = None
        self._tasks: dict[str, ScheduledTask] = {}
        self._delivery_callback: DeliveryCallback | None = None

    def set_delivery_callback(self, callback: DeliveryCallback) -> None:
        """Set the callback used to deliver reminders to users.

        Args:
            callback: Async function (description, channel, user_id) -> None.
        """
        self._delivery_callback = callback

    async def add_reminder(
        self,
        description: str,
        run_at: datetime,
        channel: str | None = None,
        user_id: str | None = None,
    ) -> ScheduledTask:
        """Schedule a one-time reminder.

        Args:
            description: What to remind about.
            run_at: When to trigger the reminder.
            channel: Optional channel to send to.
            user_id: Optional user ID to deliver to.

        Returns:
            The created ScheduledTask.
        """
        task_id = str(uuid4())[:8]
        task = ScheduledTask(
            id=task_id,
            description=description,
            type="reminder",
            schedule=run_at.isoformat(),
            channel=channel,
            user_id=user_id,
            next_run=run_at,
        )
        self._tasks[task_id] = task

        if self._scheduler:
            self._scheduler.add_job(
                self._execute_task,
                "date",
                run_date=run_at,
                args=[task_id],
                id=f"reminder_{task_id}",
                replace_existing=True,
            )

        # Persist
        await self._persist_task(task)

        logger.info("reminder_added", id=task_id, run_at=run_at.isoformat(), user_id=user_id)
        return task

    async def add_cron(
        self,
        description: str,
        cron_expression: str,
        channel: str | None = None,
        user_id: str | None = None,
    ) -> ScheduledTask:
        """Schedule a recurring task with cron expression.

        Also accepts natural language schedules like "every morning at 8am".

        Args:
            description: What the recurring task does.
            cron_expression: Cron expression or natural language schedule.
            channel: Optional channel to send to.
            user_id: Optional user ID to deliver to.

        Returns:
            The created ScheduledTask.
        """
        # Try natural language parsing first
        parsed = parse_natural_schedule(cron_expression)
        if parsed:
            cron_expression = parsed

        task_id = str(uuid4())[:8]
        task = ScheduledTask(
            id=task_id,
            description=description,
            type="cron",
            schedule=cron_expression,
            channel=channel,
            user_id=user_id,
        )
        self._tasks[task_id] = task

        if self._scheduler:
            parts = cron_expression.split()
            if len(parts) == 5:
                self._scheduler.add_job(
                    self._execute_task,
                    "cron",
                    minute=parts[0],
                    hour=parts[1],
                    day=parts[2],
                    month=parts[3],
                    day_of_week=parts[4],
                    args=[task_id],
                    id=f"cron_{task_id}",
                    replace_existing=True,
                )

        # Persist
        await self._persist_task(task)

        logger.info("cron_added", id=task_id, cron=cron_expression)
        return task

    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task.

        Args:
            task_id: The task ID to remove.

        Returns:
            True if removed, False if not found.
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks.pop(task_id)

        if self._scheduler:
            import contextlib

            job_id = f"{task.type}_{task_id}"
            with contextlib.suppress(Exception):
                self._scheduler.remove_job(job_id)

        logger.info("task_removed", id=task_id)
        return True

    def list_tasks(self) -> list[ScheduledTask]:
        """List all scheduled tasks.

        Returns:
            List of all ScheduledTask objects.
        """
        return list(self._tasks.values())

    def start(self) -> None:
        """Start the scheduler."""
        try:
            from apscheduler.schedulers.asyncio import (  # type: ignore[import-untyped]
                AsyncIOScheduler,
            )

            scheduler = AsyncIOScheduler()
            scheduler.start()
            self._scheduler = scheduler
            logger.info("scheduler_started")
        except ImportError:
            logger.warning(
                "scheduler_disabled",
                reason="apscheduler not installed",
            )

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("scheduler_stopped")

    async def load_persisted_tasks(self) -> int:
        """Load tasks from database and re-schedule them.

        Returns:
            Number of tasks loaded.
        """
        if not self.database:
            return 0

        try:
            async with self.database.db.execute(
                "SELECT id, description, type, schedule, status, channel, "
                "user_id, created_at, last_run, next_run "
                "FROM scheduled_tasks WHERE status IN ('pending', 'running')"
            ) as cursor:
                rows = await cursor.fetchall()

            count = 0
            for row in rows:
                task = ScheduledTask(
                    id=row["id"],
                    description=row["description"],
                    type=row["type"],
                    schedule=row["schedule"],
                    status=row["status"],
                    channel=row["channel"],
                    user_id=row["user_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                if row["last_run"]:
                    _lr = datetime.fromisoformat(row["last_run"])
                    task.last_run = (
                        _lr if _lr.tzinfo else _lr.replace(tzinfo=UTC)
                    )
                if row["next_run"]:
                    _nr = datetime.fromisoformat(row["next_run"])
                    task.next_run = (
                        _nr if _nr.tzinfo else _nr.replace(tzinfo=UTC)
                    )

                self._tasks[task.id] = task

                # Re-schedule
                if self._scheduler:
                    if task.type == "reminder" and task.next_run:
                        if task.next_run > datetime.now(tz=UTC):
                            self._scheduler.add_job(
                                self._execute_task,
                                "date",
                                run_date=task.next_run,
                                args=[task.id],
                                id=f"reminder_{task.id}",
                                replace_existing=True,
                            )
                    elif task.type == "cron":
                        parts = task.schedule.split()
                        if len(parts) == 5:
                            self._scheduler.add_job(
                                self._execute_task,
                                "cron",
                                minute=parts[0],
                                hour=parts[1],
                                day=parts[2],
                                month=parts[3],
                                day_of_week=parts[4],
                                args=[task.id],
                                id=f"cron_{task.id}",
                                replace_existing=True,
                            )

                count += 1

            if count:
                logger.info("persisted_tasks_loaded", count=count)
            return count

        except Exception as e:
            logger.warning("persisted_tasks_load_failed", error=str(e))
            return 0

    async def _persist_task(self, task: ScheduledTask) -> None:
        """Save a task to the database."""
        if not self.database:
            return

        try:
            await self.database.db.execute(
                "INSERT OR REPLACE INTO scheduled_tasks "
                "(id, description, type, schedule, status, channel, "
                "user_id, created_at, last_run, next_run) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task.id,
                    task.description,
                    task.type,
                    task.schedule,
                    task.status,
                    task.channel,
                    task.user_id,
                    task.created_at.isoformat(),
                    task.last_run.isoformat() if task.last_run else None,
                    task.next_run.isoformat() if task.next_run else None,
                ),
            )
            await self.database.db.commit()
        except Exception as e:
            logger.warning("task_persist_failed", id=task.id, error=str(e))

    async def _update_task_status(self, task: ScheduledTask) -> None:
        """Update a task's status in the database."""
        if not self.database:
            return

        try:
            await self.database.db.execute(
                "UPDATE scheduled_tasks SET status = ?, last_run = ? WHERE id = ?",
                (
                    task.status,
                    task.last_run.isoformat() if task.last_run else None,
                    task.id,
                ),
            )
            await self.database.db.commit()
        except Exception as e:
            logger.warning("task_status_update_failed", id=task.id, error=str(e))

    async def _execute_task(self, task_id: str) -> None:
        """Execute a scheduled task.

        Delivers the reminder to the user via the delivery callback,
        then emits a HEARTBEAT_ACTION event for dashboard/WebSocket.

        Args:
            task_id: The task to execute.
        """
        task = self._tasks.get(task_id)
        if not task:
            return

        task.status = "running"
        task.last_run = datetime.now(tz=UTC)

        try:
            # Deliver the reminder to the user via channel
            if self._delivery_callback:
                await self._delivery_callback(
                    task.description, task.channel, task.user_id
                )
                logger.info(
                    "reminder_delivered",
                    task_id=task_id,
                    channel=task.channel,
                    user_id=task.user_id,
                )
            else:
                logger.warning(
                    "reminder_no_delivery",
                    task_id=task_id,
                    reason="no delivery callback set",
                )

            await self.event_bus.emit(Events.HEARTBEAT_ACTION, {
                "type": "scheduled_task",
                "task_id": task_id,
                "description": task.description,
                "channel": task.channel,
                "user_id": task.user_id,
            })

            # Mark reminder as completed, keep cron pending for next run
            task.status = "completed" if task.type == "reminder" else "pending"

        except Exception as e:
            task.status = "failed"
            logger.error("task_execution_failed", task_id=task_id, error=str(e))

        # Persist updated status
        await self._update_task_status(task)
