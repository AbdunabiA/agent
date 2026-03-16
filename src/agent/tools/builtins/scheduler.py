"""Scheduler tools — let the LLM set reminders and manage scheduled tasks."""

from __future__ import annotations

import contextvars
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.core.scheduler import TaskScheduler

_global_scheduler: TaskScheduler | None = None

# Per-task context vars — safe for concurrent asyncio tasks (GAP 15)
_channel_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "scheduler_channel", default=None,
)
_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "scheduler_user_id", default=None,
)


def set_scheduler(
    scheduler: TaskScheduler,
    channel: str | None = None,
    user_id: str | None = None,
) -> None:
    """Set the global TaskScheduler instance (called during agent startup).

    Args:
        scheduler: The initialized TaskScheduler.
        channel: Default channel for reminders.
        user_id: Default user ID for reminders.
    """
    global _global_scheduler
    _global_scheduler = scheduler
    _channel_var.set(channel)
    _user_id_var.set(user_id)


def set_context(channel: str | None = None, user_id: str | None = None) -> None:
    """Update the current channel/user context for reminder delivery.

    Args:
        channel: Channel name (e.g. "telegram").
        user_id: User ID within the channel.
    """
    _channel_var.set(channel)
    _user_id_var.set(user_id)


def get_scheduler() -> TaskScheduler:
    """Get the global TaskScheduler instance.

    Returns:
        The shared TaskScheduler.

    Raises:
        RuntimeError: If set_scheduler() hasn't been called yet.
    """
    if _global_scheduler is None:
        raise RuntimeError(
            "TaskScheduler not initialized. Call set_scheduler() during startup."
        )
    return _global_scheduler


def _parse_delay(delay_str: str) -> timedelta | None:
    """Parse a human-friendly delay string into a timedelta.

    Supports: '5m', '30m', '1h', '2h30m', '1d', '90s', '1h30m',
    and natural language like '5 minutes', '2 hours'.

    Args:
        delay_str: Human-friendly delay like "5m", "1h", "30 minutes".

    Returns:
        timedelta or None if unparseable.
    """
    delay_str = delay_str.strip().lower()

    # Try compact format: 5m, 1h, 30s, 1d, 2h30m
    total_seconds = 0
    compact = re.findall(r"(\d+)\s*([smhd])", delay_str)
    if compact:
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        for value, unit in compact:
            total_seconds += int(value) * multipliers[unit]
        return timedelta(seconds=total_seconds)

    # Try natural language: "5 minutes", "2 hours", "1 day"
    natural = re.findall(
        r"(\d+)\s*(seconds?|minutes?|mins?|hours?|hrs?|days?)", delay_str
    )
    if natural:
        multipliers = {
            "second": 1, "seconds": 1,
            "minute": 60, "minutes": 60, "min": 60, "mins": 60,
            "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
            "day": 86400, "days": 86400,
        }
        for value, unit in natural:
            total_seconds += int(value) * multipliers.get(unit, 60)
        return timedelta(seconds=total_seconds)

    # Try plain number (assume minutes)
    if delay_str.isdigit():
        return timedelta(minutes=int(delay_str))

    return None


@tool(
    name="set_reminder",
    description=(
        "Set a reminder that will be delivered to the user after a delay. "
        "Use a human-friendly delay like '5m', '1h', '30 minutes', '2h30m'. "
        "The reminder message will be sent to the user when the time is up."
    ),
    tier=ToolTier.SAFE,
)
async def set_reminder(
    description: str,
    delay: str,
) -> str:
    """Set a one-time reminder.

    Args:
        description: What to remind about (e.g. "Check the deployment").
        delay: When to fire, as a human-friendly delay (e.g. "5m", "1h", "30 minutes").

    Returns:
        Confirmation message with the scheduled time.
    """
    scheduler = get_scheduler()

    delta = _parse_delay(delay)
    if delta is None:
        return (
            f"Could not parse delay '{delay}'. "
            "Use formats like '5m', '1h', '30 minutes', '2h30m'."
        )

    if delta.total_seconds() < 10:
        return "Reminder delay must be at least 10 seconds."

    if delta.total_seconds() > 7 * 86400:
        return "Reminder delay must be less than 7 days."

    run_at = datetime.now() + delta
    task = await scheduler.add_reminder(
        description=description,
        run_at=run_at,
        channel=_channel_var.get(),
        user_id=_user_id_var.get(),
    )

    return (
        f"Reminder set (id={task.id}). "
        f"I'll remind you about \"{description}\" at {run_at.strftime('%H:%M:%S')} "
        f"(in {delay})."
    )


@tool(
    name="list_reminders",
    description="List all pending scheduled reminders and tasks.",
    tier=ToolTier.SAFE,
)
async def list_reminders() -> str:
    """List all scheduled tasks.

    Returns:
        Formatted list of tasks, or a message if none exist.
    """
    scheduler = get_scheduler()
    tasks = scheduler.list_tasks()

    if not tasks:
        return "No scheduled reminders or tasks."

    lines: list[str] = []
    for t in tasks:
        status_icon = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌"}
        icon = status_icon.get(t.status, "❓")
        time_info = t.next_run.strftime("%Y-%m-%d %H:%M:%S") if t.next_run else t.schedule
        lines.append(f"{icon} [{t.id}] {t.description} — {t.type} @ {time_info} ({t.status})")

    return f"Scheduled tasks ({len(tasks)}):\n" + "\n".join(lines)


@tool(
    name="cancel_reminder",
    description="Cancel a scheduled reminder by its ID.",
    tier=ToolTier.SAFE,
)
async def cancel_reminder(task_id: str) -> str:
    """Cancel a scheduled task.

    Args:
        task_id: The task ID to cancel (from list_reminders output).

    Returns:
        Confirmation or error message.
    """
    scheduler = get_scheduler()
    removed = scheduler.remove_task(task_id)
    if removed:
        return f"Reminder {task_id} cancelled."
    return f"No reminder found with ID '{task_id}'."
