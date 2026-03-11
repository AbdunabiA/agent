"""Send file tool — let the LLM send files to the user via the active channel."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog

from agent.core.events import EventBus, Events
from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)

# Global state set by the channel before processing a message
_global_event_bus: EventBus | None = None
_global_context: dict[str, str | None] = {"channel": None, "user_id": None}


def set_file_send_bus(event_bus: EventBus) -> None:
    """Set the global EventBus instance for file sending.

    Args:
        event_bus: The shared EventBus.
    """
    global _global_event_bus
    _global_event_bus = event_bus


def set_file_send_context(
    channel: str | None = None, user_id: str | None = None
) -> None:
    """Update the current channel/user context for file delivery.

    Args:
        channel: Channel name (e.g. "telegram").
        user_id: User ID within the channel.
    """
    _global_context["channel"] = channel
    _global_context["user_id"] = user_id


def _resolve_path(path: str) -> Path:
    """Resolve a file path synchronously (for use in asyncio.to_thread).

    Args:
        path: Raw path string, supports ~ for home.

    Returns:
        Resolved absolute Path.
    """
    return Path(path).expanduser().resolve()


def _validate_file(resolved: Path) -> dict[str, Any] | None:
    """Validate file exists and is within size limits. Sync, run via to_thread.

    Returns:
        None if valid, or dict with 'error' key if invalid.
    """
    if not resolved.exists():
        return {"error": f"[ERROR] File not found: {resolved}"}

    if not resolved.is_file():
        return {"error": f"[ERROR] Not a file: {resolved}"}

    file_size = resolved.stat().st_size
    max_size = 50 * 1024 * 1024  # 50MB (Telegram bot limit)
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        return {
            "error": f"[ERROR] File too large ({size_mb:.1f} MB). Maximum is 50 MB."
        }

    if file_size == 0:
        return {"error": "[ERROR] File is empty (0 bytes)."}

    return None


@tool(
    name="send_file",
    description=(
        "Send a file, image, or video to the user in the current chat. "
        "Provide the absolute or relative path to the file. "
        "Images (jpg, png, gif, webp, bmp) are sent as inline photos. "
        "Videos (mp4, mov, avi, mkv, webm) are sent as inline videos. "
        "Other files are sent as document attachments. "
        "Use this when the user asks you to send, share, or deliver "
        "a file, image, photo, video, screenshot, or document."
    ),
    tier=ToolTier.MODERATE,
)
async def send_file(path: str, caption: str = "") -> str:
    """Send a file to the user via the active messaging channel.

    Args:
        path: Path to the file to send. Supports ~ for home directory.
        caption: Optional caption/description for the file.

    Returns:
        Confirmation message or error.
    """
    if _global_event_bus is None:
        return (
            "[ERROR] File sending not available — event bus not initialized."
        )

    channel = _global_context.get("channel")
    user_id = _global_context.get("user_id")

    if not channel or not user_id:
        return (
            "[ERROR] No active channel/user context. "
            "File sending only works via messaging channels."
        )

    # Resolve and validate on a thread to avoid blocking the event loop
    resolved = await asyncio.to_thread(_resolve_path, path)
    validation_err = await asyncio.to_thread(_validate_file, resolved)
    if validation_err is not None:
        return validation_err["error"]

    file_size = await asyncio.to_thread(lambda: resolved.stat().st_size)

    # Emit file send event — the channel picks this up
    await _global_event_bus.emit(
        Events.FILE_SEND,
        {
            "file_path": str(resolved),
            "file_name": resolved.name,
            "caption": caption,
            "channel": channel,
            "user_id": user_id,
        },
    )

    size_str = _format_size(file_size)
    logger.info(
        "file_send_requested",
        file_path=str(resolved),
        file_name=resolved.name,
        size=size_str,
        channel=channel,
        user_id=user_id,
    )

    return f"File sent: {resolved.name} ({size_str})"


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"
