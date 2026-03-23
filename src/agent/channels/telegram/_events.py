"""Telegram event handlers — orchestration and controller status notifications."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.channels.telegram._core import TelegramChannel

logger = structlog.get_logger(__name__)


def _register_task_user(self: TelegramChannel, task_id: str, user_id: str) -> None:
    """Register which user spawned a task (for status notifications)."""
    self._task_user_map[task_id] = user_id


def _user_for_task(self: TelegramChannel, data: dict[str, Any]) -> str | None:
    """Resolve the Telegram user_id for an orchestration event."""
    task_id = data.get("task_id", "")
    # Direct lookup
    if task_id in self._task_user_map:
        return self._task_user_map[task_id]
    # Channel tasks use session_id = "telegram:<user_id>"
    if task_id.startswith("telegram:"):
        return task_id.split(":", 1)[1]
    # Try parent_session_id (subagents spawned within a channel task)
    parent = data.get("parent_session_id", "")
    if parent.startswith("telegram:"):
        user_id = parent.split(":", 1)[1]
        # Cache for future lookups (completed/failed events)
        if task_id:
            self._task_user_map[task_id] = user_id
        return user_id
    return None


async def _notify_user(self: TelegramChannel, user_id: str, text: str) -> None:
    """Send a status notification to a Telegram user."""
    if not self._bot:
        return
    with contextlib.suppress(Exception):
        await self._bot.send_message(
            chat_id=int(user_id),
            text=text,
            parse_mode="HTML",
        )


async def _on_subagent_spawned(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a subagent starts working."""
    user_id = self._user_for_task(data)
    role = data.get("role", "agent")
    # Skip channel-level task spawns (the main agent itself)
    if role == "channel":
        return
    if user_id:
        instruction = data.get("instruction", "")[:100]
        await self._notify_user(
            user_id,
            f"\U0001f916 <b>{role}</b> started working"
            f"{f': {instruction}...' if instruction else ''}",
        )


async def _on_subagent_completed(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a subagent finishes."""
    user_id = self._user_for_task(data)
    role = data.get("role", "agent")
    if role == "channel":
        return
    if user_id:
        duration = data.get("duration_ms", 0)
        duration_s = f"{duration / 1000:.1f}s" if duration else ""
        await self._notify_user(
            user_id,
            f"\u2705 <b>{role}</b> finished" f"{f' ({duration_s})' if duration_s else ''}",
        )


async def _on_subagent_failed(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a subagent fails."""
    user_id = self._user_for_task(data)
    role = data.get("role", "agent")
    if role == "channel":
        return
    if user_id:
        error = data.get("error", "unknown error")[:200]
        await self._notify_user(
            user_id,
            f"\u274c <b>{role}</b> failed: {error}",
        )


async def _on_project_started(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a project pipeline starts."""
    # Project events don't have task_id in _task_user_map, but
    # the project name is enough for context.
    project = data.get("project", "?")
    stages = data.get("stages", 0)
    # Try all registered users (projects are typically single-user)
    for uid in set(self._task_user_map.values()):
        await self._notify_user(
            uid,
            f"\U0001f680 Project <b>{project}</b> started " f"({stages} stages)",
        )


async def _on_project_stage_started(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a project stage begins."""
    stage = data.get("stage", "?")
    agents = data.get("agents", 0)
    for uid in set(self._task_user_map.values()):
        await self._notify_user(
            uid,
            f"\u25b6\ufe0f Stage <b>{stage}</b> started "
            f"({agents} agent{'s' if agents != 1 else ''})",
        )


async def _on_project_stage_completed(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a project stage finishes."""
    stage = data.get("stage", "?")
    duration = data.get("duration_ms", 0)
    duration_s = f"{duration / 1000:.1f}s" if duration else ""
    for uid in set(self._task_user_map.values()):
        await self._notify_user(
            uid,
            f"\u2705 Stage <b>{stage}</b> completed" f"{f' ({duration_s})' if duration_s else ''}",
        )


async def _on_project_completed(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a project pipeline finishes."""
    project = data.get("project", "?")
    duration = data.get("duration_ms", 0)
    duration_s = f"{duration / 1000:.1f}s" if duration else ""
    for uid in set(self._task_user_map.values()):
        await self._notify_user(
            uid,
            f"\U0001f389 Project <b>{project}</b> completed"
            f"{f' in {duration_s}' if duration_s else ''}!",
        )


async def _on_project_failed(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a project pipeline fails."""
    project = data.get("project", "?")
    error = data.get("error", "unknown")[:200]
    for uid in set(self._task_user_map.values()):
        await self._notify_user(
            uid,
            f"\u274c Project <b>{project}</b> failed: {error}",
        )


# ── Task completion notification (async project results) ──────


async def _on_task_completed_notify(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Send the full project result summary to the user."""
    user_id = data.get("user_id")
    if not user_id:
        users = set(self._task_user_map.values())
        user_id = next(iter(users), None) if users else None
    if not user_id:
        return

    summary = data.get("summary", data.get("result", ""))
    duration = data.get("duration_seconds", 0)

    # Truncate for Telegram (max 4096 chars)
    if len(summary) > 3800:
        summary = summary[:3800] + "\n\n...(truncated)"

    header = "\U0001f4e6 <b>Task completed</b>"
    if duration:
        header += f" ({duration}s)"

    await self._notify_user(
        user_id,
        f"{header}\n\n{summary}",
    )


# ── Controller events ──────────────────────────────────────────


async def _on_controller_task_started(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when the controller starts working on a task."""
    user_id = data.get("user_id")
    if not user_id:
        # Fall back to any known user
        users = set(self._task_user_map.values())
        user_id = next(iter(users), None) if users else None
    if user_id:
        instruction = data.get("instruction", "")[:100]
        await self._notify_user(
            user_id,
            f"\U0001f4cb Controller started: {instruction}",
        )


async def _on_controller_task_progress(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user of controller progress updates."""
    user_id = data.get("user_id")
    if not user_id:
        users = set(self._task_user_map.values())
        user_id = next(iter(users), None) if users else None
    if user_id:
        status = data.get("status", "working")
        await self._notify_user(
            user_id,
            f"\u2699\ufe0f Controller: {status}",
        )


async def _on_controller_task_completed(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when the controller completes a task."""
    user_id = data.get("user_id")
    if not user_id:
        users = set(self._task_user_map.values())
        user_id = next(iter(users), None) if users else None
    if user_id:
        summary = data.get("summary", "Done")[:300]
        await self._notify_user(
            user_id,
            f"\u2705 Controller completed: {summary}",
        )


async def _on_controller_task_failed(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when the controller fails a task."""
    user_id = data.get("user_id")
    if not user_id:
        users = set(self._task_user_map.values())
        user_id = next(iter(users), None) if users else None
    if user_id:
        error = data.get("error", "unknown")[:200]
        await self._notify_user(
            user_id,
            f"\u274c Controller failed: {error}",
        )


async def _on_controller_task_cancelled(self: TelegramChannel, data: dict[str, Any]) -> None:
    """Notify user when a controller task is cancelled."""
    user_id = data.get("user_id")
    if not user_id:
        users = set(self._task_user_map.values())
        user_id = next(iter(users), None) if users else None
    if user_id:
        order_id = data.get("order_id", "unknown")
        await self._notify_user(
            user_id,
            f"\u26d4 Controller task cancelled: {order_id}",
        )
