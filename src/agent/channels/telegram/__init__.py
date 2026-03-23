"""Telegram channel package."""

import contextlib
import os  # noqa: F401 — re-exported for test patching compatibility
from pathlib import Path  # noqa: F401 — re-exported for test patching compatibility

from agent.channels.telegram._core import (  # noqa: F401
    _APPROVAL_TIMEOUT,
    _DEFAULT_SHORTCUTS,
    _STREAM_THRESHOLD,
    _TG_MAX_LENGTH,
    _TYPING_INTERVAL,
    _UPLOAD_DIR,
    AIOGRAM_AVAILABLE,
    TelegramChannel,
    _tool_explanation,
)

# Re-export aiogram types so existing patches against
# ``agent.channels.telegram.Bot`` etc. keep working.
with contextlib.suppress(ImportError):
    from aiogram import Bot, Dispatcher, Router  # noqa: F401

__all__ = ["TelegramChannel", "_tool_explanation", "AIOGRAM_AVAILABLE"]
