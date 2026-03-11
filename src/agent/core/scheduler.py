"""Task scheduler — manages scheduled tasks, reminders, and cron jobs.

Uses APScheduler with an async event loop scheduler.
Tasks are stored in memory for Phase 2 (SQLite in Phase 4).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import structlog

from agent.core.events import EventBus, Events

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


class TaskScheduler:
    """Manages scheduled tasks: reminders, cron jobs, one-time events.

    Uses APScheduler with an async event loop scheduler.
    Tasks are stored in memory for Phase 2 (SQLite in Phase 4).
    """

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
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

        Args:
            description: What the recurring task does.
            cron_expression: Cron expression (e.g., "0 9 * * 1").
            channel: Optional channel to send to.

        Returns:
            The created ScheduledTask.
        """
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
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            self._scheduler = AsyncIOScheduler()
            self._scheduler.start()
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
        task.last_run = datetime.now()

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
