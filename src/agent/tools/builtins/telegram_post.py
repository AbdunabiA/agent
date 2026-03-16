"""Telegram posting tools — post to channels, send messages to users.

Lets the agent publish content to Telegram channels/groups where the bot
is an admin, and send direct messages to users who have started the bot.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

from agent.core.events import EventBus, Events
from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)

# Global state — set during startup and before each message
_global_event_bus: EventBus | None = None
_global_scheduler: Any | None = None  # TaskScheduler

# Per-task context vars — safe for concurrent asyncio tasks (GAP 15)
_channel_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tg_post_channel", default=None,
)
_user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tg_post_user_id", default=None,
)


def set_telegram_post_bus(event_bus: EventBus) -> None:
    """Set the global EventBus for channel posting."""
    global _global_event_bus
    _global_event_bus = event_bus


def set_telegram_post_scheduler(scheduler: Any) -> None:
    """Set the global TaskScheduler for scheduled posts."""
    global _global_scheduler
    _global_scheduler = scheduler


def set_telegram_post_context(
    channel: str | None = None, user_id: str | None = None,
) -> None:
    """Update the current channel/user context."""
    _channel_var.set(channel)
    _user_id_var.set(user_id)


def _parse_delay(delay_str: str) -> timedelta | None:
    """Parse a human-friendly delay string into a timedelta.

    Supports: '5m', '30m', '1h', '2h30m', '1d', '90s',
    and natural language like '5 minutes', '2 hours'.
    """
    import re

    delay_str = delay_str.strip().lower()

    # Compact format: 5m, 1h, 30s, 1d, 2h30m
    compact = re.findall(r"(\d+)\s*([smhd])", delay_str)
    if compact:
        multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        total = sum(int(v) * multipliers[u] for v, u in compact)
        return timedelta(seconds=total)

    # Natural language: "5 minutes", "2 hours"
    natural = re.findall(
        r"(\d+)\s*(seconds?|minutes?|mins?|hours?|hrs?|days?)", delay_str,
    )
    if natural:
        mult = {
            "second": 1, "seconds": 1,
            "minute": 60, "minutes": 60, "min": 60, "mins": 60,
            "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
            "day": 86400, "days": 86400,
        }
        total = sum(int(v) * mult.get(u, 60) for v, u in natural)
        return timedelta(seconds=total)

    # Plain number → assume minutes
    if delay_str.isdigit():
        return timedelta(minutes=int(delay_str))

    return None


@tool(
    name="post_to_channel",
    description=(
        "Post a message to a Telegram channel or group chat. "
        "The bot must be an admin in the target channel. "
        "Provide the channel username (e.g. '@mychannel') or numeric chat ID. "
        "Supports Markdown formatting. Can optionally attach a photo and pin the post."
    ),
    tier=ToolTier.MODERATE,
)
async def post_to_channel(
    chat_id: str,
    text: str,
    photo_path: str = "",
    pin: bool = False,
    parse_mode: str = "Markdown",
) -> str:
    """Post a message to a Telegram channel or group.

    Args:
        chat_id: Channel username (@channel) or numeric chat ID.
        text: Message text (Markdown supported).
        photo_path: Optional path to an image to attach.
        pin: Whether to pin the message after posting.
        parse_mode: Parse mode — "Markdown", "HTML", or empty string.

    Returns:
        Confirmation message or error.
    """
    if _global_event_bus is None:
        return "[ERROR] Channel posting not available — event bus not initialized."

    if not text.strip():
        return "[ERROR] Message text cannot be empty."

    if not chat_id.strip():
        return "[ERROR] chat_id is required (e.g. '@mychannel' or numeric ID)."

    # Validate photo if provided
    if photo_path:
        resolved = await asyncio.to_thread(
            lambda: Path(photo_path).expanduser().resolve(),
        )
        if not await asyncio.to_thread(resolved.exists):
            return f"[ERROR] Photo not found: {resolved}"
        photo_path = str(resolved)

    await _global_event_bus.emit(
        Events.CHANNEL_POST,
        {
            "chat_id": chat_id.strip(),
            "text": text,
            "photo_path": photo_path,
            "pin": pin,
            "parse_mode": parse_mode,
            "channel": "telegram",
        },
    )

    logger.info(
        "channel_post_requested",
        chat_id=chat_id,
        text_len=len(text),
        has_photo=bool(photo_path),
        pin=pin,
    )

    pin_note = " (pinned)" if pin else ""
    photo_note = " with photo" if photo_path else ""
    return f"Posted to {chat_id}{photo_note}{pin_note}."


@tool(
    name="send_telegram_message",
    description=(
        "Send a direct message to a Telegram user who has previously started "
        "the bot. Provide the numeric user ID. Useful for notifications, "
        "alerts, or proactive outreach to users who have interacted with the bot."
    ),
    tier=ToolTier.MODERATE,
)
async def send_telegram_message(
    user_id: str,
    text: str,
    parse_mode: str = "Markdown",
) -> str:
    """Send a direct message to a Telegram user.

    Args:
        user_id: Numeric Telegram user ID.
        text: Message text.
        parse_mode: Parse mode — "Markdown", "HTML", or empty string.

    Returns:
        Confirmation message or error.
    """
    if _global_event_bus is None:
        return "[ERROR] Messaging not available — event bus not initialized."

    if not text.strip():
        return "[ERROR] Message text cannot be empty."

    if not user_id.strip():
        return "[ERROR] user_id is required."

    await _global_event_bus.emit(
        Events.CHANNEL_SEND_MESSAGE,
        {
            "user_id": user_id.strip(),
            "text": text,
            "parse_mode": parse_mode,
            "channel": "telegram",
        },
    )

    logger.info(
        "telegram_message_requested",
        user_id=user_id,
        text_len=len(text),
    )

    return f"Message sent to user {user_id}."


@tool(
    name="schedule_post",
    description=(
        "Schedule a message to be posted to a Telegram channel at a future time. "
        "Use a human-friendly delay like '5m', '1h', '30 minutes', '2h30m'. "
        "The post will be delivered after the specified delay."
    ),
    tier=ToolTier.MODERATE,
)
async def schedule_post(
    chat_id: str,
    text: str,
    delay: str,
    photo_path: str = "",
    pin: bool = False,
    parse_mode: str = "Markdown",
) -> str:
    """Schedule a post to a Telegram channel.

    Args:
        chat_id: Channel username (@channel) or numeric chat ID.
        text: Message text (Markdown supported).
        delay: When to post (e.g. '5m', '1h', '30 minutes').
        photo_path: Optional path to an image to attach.
        pin: Whether to pin the message after posting.
        parse_mode: Parse mode — "Markdown", "HTML", or empty string.

    Returns:
        Confirmation message or error.
    """
    if _global_event_bus is None:
        return "[ERROR] Scheduling not available — event bus not initialized."

    if not text.strip():
        return "[ERROR] Message text cannot be empty."

    if not chat_id.strip():
        return "[ERROR] chat_id is required."

    td = _parse_delay(delay)
    if td is None:
        return f"[ERROR] Could not parse delay: '{delay}'. Use formats like '5m', '1h', '30 minutes'."

    if td.total_seconds() < 10:
        return "[ERROR] Delay must be at least 10 seconds."

    # Validate photo if provided
    if photo_path:
        resolved = await asyncio.to_thread(
            lambda: Path(photo_path).expanduser().resolve(),
        )
        if not await asyncio.to_thread(resolved.exists):
            return f"[ERROR] Photo not found: {resolved}"
        photo_path = str(resolved)

    run_at = datetime.now(tz=timezone.utc) + td

    if _global_scheduler is None:
        return "[ERROR] Scheduler not available — cannot schedule posts."

    # Encode post metadata in the description so the delivery callback
    # can reconstruct the CHANNEL_POST event. The description is stored
    # in the scheduler's task list and is visible via list_reminders/
    # cancel_reminder tools.
    post_meta = json.dumps({
        "chat_id": chat_id.strip(),
        "photo_path": photo_path,
        "pin": pin,
        "parse_mode": parse_mode,
    })
    description = f"[scheduled_post:{post_meta}] {text}"

    task = await _global_scheduler.add_reminder(
        description=description,
        run_at=run_at,
        channel="telegram",
        user_id=_user_id_var.get(),
    )

    logger.info(
        "post_scheduled",
        task_id=task.id,
        chat_id=chat_id,
        run_at=run_at.isoformat(),
        delay=str(td),
    )

    return (
        f"Post scheduled for {chat_id} at "
        f"{run_at.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(in {delay}). Task ID: {task.id} — "
        f"use cancel_reminder to cancel it."
    )
