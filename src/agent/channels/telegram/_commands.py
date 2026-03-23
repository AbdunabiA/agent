"""Telegram command handlers — all /cmd_* methods."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.channels.telegram._core import TelegramChannel

logger = structlog.get_logger(__name__)

# Import constants from _core lazily to avoid circular imports at module level
_TG_MAX_LENGTH = 4096


async def _cmd_start(self: TelegramChannel, message: Any) -> None:
    """Handle /start command."""
    if not self._check_message(message):
        return
    await message.answer(
        "Hello! I'm your AI assistant. Send me a message and I'll help you out.\n"
        "Use /help to see available commands."
    )


async def _cmd_help(self: TelegramChannel, message: Any) -> None:
    """Handle /help command."""
    if not self._check_message(message):
        return
    await message.answer(
        "Available commands:\n"
        "/start — Welcome message\n"
        "/help — This help text\n"
        "/status — Agent status\n"
        "/tools — List available tools\n"
        "/new — Start a new conversation\n"
        "/session — Show current session info\n"
        "/run <shortcut> — Run a prompt shortcut (/run list for all)\n"
        "/soul — View/edit agent personality\n"
        "/backend — View/switch LLM backend\n"
        "/workdir — View/change working directory\n"
        "/remind <delay> <text> — Set a reminder\n"
        "/reminders — List pending reminders\n"
        "/tasks — Show running background tasks\n"
        "/stop — Cancel all running tasks\n"
        "/pause — Pause message processing\n"
        "/resume — Resume message processing\n"
        "/mute — Disable heartbeat\n"
        "/unmute — Enable heartbeat"
    )


async def _cmd_status(self: TelegramChannel, message: Any) -> None:
    """Handle /status command — show agent status."""
    if not self._check_message(message):
        return

    text = await self._build_status_text()

    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    if AIOGRAM_AVAILABLE:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="Costs", callback_data="nav:status:costs"),
                    InlineKeyboardButton(text="Health", callback_data="nav:status:health"),
                    InlineKeyboardButton(text="Audit", callback_data="nav:status:audit"),
                ],
            ]
        )
        await message.answer(text, reply_markup=keyboard)
    else:
        await message.answer(text)


async def _build_status_text(self: TelegramChannel) -> str:
    """Build the status text with cost and audit data."""
    from agent.tools.registry import registry

    heartbeat_status = "not configured"
    last_tick = "never"
    if self.heartbeat:
        heartbeat_status = "enabled" if self.heartbeat.is_enabled else "disabled"
        if self.heartbeat.last_tick:
            last_tick = self.heartbeat.last_tick.strftime("%Y-%m-%d %H:%M:%S")

    tools_count = len(registry.list_tools())
    sessions_count = self.session_store.active_count

    lines = [
        "Agent Status\n",
        f"Heartbeat: {heartbeat_status}",
        f"Last tick: {last_tick}\n",
        f"Tools: {tools_count} registered",
        f"Sessions: {sessions_count} active",
    ]

    # Cost summary
    try:
        if self.cost_tracker:
            stats = self.cost_tracker.get_stats("day")
            total_cost = stats.get("total_cost", 0)
            total_calls = stats.get("total_calls", 0)
            tokens = stats.get("total_tokens", {})
            total_tokens = tokens.get("input", 0) + tokens.get("output", 0)
            token_display = f"{total_tokens / 1000:.1f}K" if total_tokens else "0"
            lines.append(
                f"\nToday: {total_calls} calls | ${total_cost:.2f} | {token_display} tokens"
            )
    except Exception:
        pass

    # Audit summary
    try:
        if self.audit_log:
            audit_stats = await self.audit_log.get_stats()
            tool_calls = audit_stats.get("total_calls", 0)
            success_rate = audit_stats.get("success_rate", 0) * 100
            avg_ms = audit_stats.get("avg_duration_ms", 0)
            if tool_calls > 0:
                lines.append(
                    f"Tools: {tool_calls} calls | {success_rate:.1f}% success | avg {avg_ms}ms"
                )
    except Exception:
        pass

    return "\n".join(lines)


async def _cmd_tools(self: TelegramChannel, message: Any) -> None:
    """Handle /tools command — list registered tools."""
    if not self._check_message(message):
        return

    text, keyboard = await self._build_tools_text_and_keyboard()
    if text is None:
        await message.answer("No tools registered.")
        return

    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    if keyboard and AIOGRAM_AVAILABLE:
        await message.answer(text, reply_markup=keyboard)
    else:
        await message.answer(text)


async def _build_tools_text_and_keyboard(
    self: TelegramChannel,
) -> tuple[str | None, Any]:
    """Build tools list text with usage counts and toggle keyboard."""
    from agent.tools.registry import registry

    tools = registry.list_tools()
    if not tools:
        return None, None

    # Get per-tool usage counts from audit log
    tools_used: dict[str, int] = {}
    try:
        if self.audit_log:
            audit_stats = await self.audit_log.get_stats()
            tools_used = audit_stats.get("tools_used", {})
    except Exception:
        pass

    tier_emoji = {"safe": "\U0001f7e2", "moderate": "\U0001f7e1", "dangerous": "\U0001f534"}
    lines: list[str] = ["Available tools:\n"]
    for t in tools:
        emoji = tier_emoji.get(t.tier.value, "\u2753")
        status = "on" if t.enabled else "off"
        count = tools_used.get(t.name, 0)
        usage = f" ({count}x)" if count > 0 else ""
        lines.append(f"{emoji} {t.name} [{status}] — {t.description}{usage}")

    text = "\n".join(lines)

    # Build toggle buttons for first 8 tools
    keyboard = None
    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    if AIOGRAM_AVAILABLE:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        buttons: list[list[Any]] = []
        row: list[Any] = []
        for t in tools[:8]:
            label = f"{'Disable' if t.enabled else 'Enable'} {t.name}"
            row.append(
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"nav:tools:toggle:{t.name}",
                )
            )
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        if buttons:
            keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    return text, keyboard


async def _cmd_pause(self: TelegramChannel, message: Any) -> None:
    """Handle /pause command."""
    if not self._check_message(message):
        return
    self.pause()
    await message.answer("Message processing paused. Use /resume to continue.")


async def _cmd_resume(self: TelegramChannel, message: Any) -> None:
    """Handle /resume command."""
    if not self._check_message(message):
        return
    self.resume()
    await message.answer("Message processing resumed.")


async def _cmd_mute(self: TelegramChannel, message: Any) -> None:
    """Handle /mute command — disable heartbeat."""
    if not self._check_message(message):
        return
    if self.heartbeat:
        self.heartbeat.disable()
        await message.answer("Heartbeat muted.")
    else:
        await message.answer("Heartbeat is not configured.")


async def _cmd_unmute(self: TelegramChannel, message: Any) -> None:
    """Handle /unmute command — enable heartbeat."""
    if not self._check_message(message):
        return
    if self.heartbeat:
        self.heartbeat.enable()
        await message.answer("Heartbeat unmuted.")
    else:
        await message.answer("Heartbeat is not configured.")


async def _cmd_soul(self: TelegramChannel, message: Any) -> None:
    """Handle /soul command — view or edit soul.md.

    /soul — display current soul.md content
    /soul edit <text> — overwrite soul.md with new content
    """
    if not self._check_message(message):
        return

    text = message.text or ""
    parts = text.split(maxsplit=2)

    # Use package-level Path so tests can monkeypatch agent.channels.telegram.Path
    _path_cls = sys.modules["agent.channels.telegram"].Path
    soul_path = _path_cls("soul.md")

    if len(parts) >= 3 and parts[1].lower() == "edit":
        # /soul edit <new content>
        new_content = parts[2]
        try:
            await asyncio.to_thread(soul_path.write_text, new_content, encoding="utf-8")
            await message.answer("soul.md updated successfully.")
        except OSError as e:
            await message.answer(f"Failed to update soul.md: {e}")
        return

    # /soul — show current content
    exists = await asyncio.to_thread(soul_path.exists)
    if exists:
        try:
            content = await asyncio.to_thread(soul_path.read_text, encoding="utf-8")
            if len(content) > _TG_MAX_LENGTH - 50:
                content = content[: _TG_MAX_LENGTH - 50] + "\n...(truncated)"
            await message.answer(f"Current soul.md:\n\n{content}")
        except OSError as e:
            await message.answer(f"Failed to read soul.md: {e}")
    else:
        await message.answer("soul.md not found. Use /soul edit <text> to create it.")


async def _cmd_backend(self: TelegramChannel, message: Any) -> None:
    """Handle /backend command — view or switch LLM backend.

    /backend — show current backend
    /backend litellm — switch to litellm
    /backend claude-sdk — switch to claude-sdk
    """
    if not self._check_message(message):
        return

    from agent.config import get_config, update_config_section

    config = get_config()
    text = message.text or ""
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer(
            f"Current backend: {config.models.backend}\n\n" "Usage: /backend <litellm|claude-sdk>"
        )
        return

    new_backend = parts[1].strip().lower()
    valid = ("litellm", "claude-sdk")
    if new_backend not in valid:
        await message.answer(f"Invalid backend '{new_backend}'. Choose: {', '.join(valid)}")
        return

    try:
        await asyncio.to_thread(update_config_section, "models", {"backend": new_backend})
        await message.answer(f"Backend switched to: {new_backend}")
    except Exception as e:
        await message.answer(f"Failed to switch backend: {e}")


async def _cmd_workdir(self: TelegramChannel, message: Any) -> None:
    """Handle /workdir command — view or change working directory.

    /workdir — show current working directory
    /workdir <path> — change working directory
    """
    if not self._check_message(message):
        return

    from agent.config import get_config, update_config_section

    config = get_config()
    text = message.text or ""
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        wd = config.models.claude_sdk.working_dir
        await message.answer(f"Current working directory: {wd}\n\n" "Usage: /workdir <path>")
        return

    new_dir = parts[1].strip()
    resolved = await asyncio.to_thread(lambda: Path(new_dir).expanduser().resolve())
    if not await asyncio.to_thread(resolved.is_dir):
        await message.answer(f"Directory not found: {resolved}")
        return

    try:
        await asyncio.to_thread(
            update_config_section,
            "models",
            {"claude_sdk": {"working_dir": str(resolved)}},
        )
        await message.answer(f"Working directory changed to: {resolved}")
    except Exception as e:
        await message.answer(f"Failed to change working directory: {e}")


async def _cmd_new(self: TelegramChannel, message: Any) -> None:
    """Handle /new command — start a fresh conversation.

    Clears the current SDK session so the next message
    begins a new conversation without prior context.
    """
    if not self._check_message(message):
        return

    user_id = str(message.from_user.id)
    session_id = self._make_session_id(user_id)

    # Disconnect persistent SDK client and clear session
    if self.sdk_service is not None:
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
        await sdk.disconnect_client(session_id)

    session = await self.session_store.get(session_id)
    if session:
        session.metadata.pop("sdk_session_id", None)
        session.clear()
        logger.info("session_cleared", user_id=user_id)

    await message.answer(
        "New conversation started. " "Your next message will begin a fresh session."
    )


async def _cmd_session(self: TelegramChannel, message: Any) -> None:
    """Handle /session command — show current session info."""
    if not self._check_message(message):
        return

    user_id = str(message.from_user.id)
    session_id = self._make_session_id(user_id)

    session = await self.session_store.get(session_id)
    sdk_sid = session.metadata.get("sdk_session_id") if session else None

    backend = "claude-sdk" if self.sdk_service else "litellm"
    status = "active" if sdk_sid else "none"

    lines = [
        "Session Info\n",
        f"Backend: {backend}",
        f"Agent session: {session_id}",
        f"SDK session: {sdk_sid or 'none (next message starts new)'}",
        f"Status: {status}",
    ]

    if self.sdk_service:
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
        task_status = sdk.get_status(session_id)
        lines.append(f"Task status: {task_status}")

    # Add message/token info from session
    if session:
        msg_count = getattr(session, "message_count", 0)
        tokens_in = getattr(session, "tokens_in", 0)
        tokens_out = getattr(session, "tokens_out", 0)
        total_tokens = tokens_in + tokens_out
        if msg_count or total_tokens:
            lines.append("")
        if msg_count:
            lines.append(f"Messages: {msg_count}")
        if total_tokens:
            lines.append(f"Tokens: {total_tokens:,} (in: {tokens_in:,} + out: {tokens_out:,})")

    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    if AIOGRAM_AVAILABLE:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Clear Session",
                        callback_data="nav:session:clear",
                    ),
                    InlineKeyboardButton(
                        text="New Session",
                        callback_data="nav:session:new",
                    ),
                ],
            ]
        )
        await message.answer("\n".join(lines), reply_markup=keyboard)
    else:
        await message.answer("\n".join(lines))


async def _cmd_remind(self: TelegramChannel, message: Any) -> None:
    """Handle /remind command — set a reminder.

    Usage: /remind <delay> <description>
    Examples:
        /remind 5m Check the deployment
        /remind 1h Review the pull request
        /remind 30s Test notification
    """
    if not self._check_message(message):
        return

    if not self.scheduler:
        await message.answer("Scheduler is not configured.")
        return

    text = message.text or ""
    parts = text.split(maxsplit=2)

    if len(parts) < 3:
        await message.answer(
            "Usage: /remind <delay> <description>\n\n"
            "Examples:\n"
            "/remind 5m Check the deployment\n"
            "/remind 1h30m Review the pull request\n"
            "/remind 2h Call the client\n"
            "/remind 45s Quick test\n\n"
            "Any combination works: s=seconds, m=minutes, h=hours, d=days\n"
            "e.g. 90s, 15m, 2h, 1d, 1h45m, 3h30m"
        )
        return

    delay_str = parts[1]
    description = parts[2]
    user_id = str(message.from_user.id)

    from agent.tools.builtins.scheduler import _parse_delay

    delta = _parse_delay(delay_str)
    if delta is None:
        await message.answer(
            f"Could not parse delay '{delay_str}'. " "Use formats like 5m, 1h, 30s, 2h30m."
        )
        return

    if delta.total_seconds() < 10:
        await message.answer("Reminder delay must be at least 10 seconds.")
        return

    from datetime import datetime

    run_at = datetime.now() + delta
    task = await self.scheduler.add_reminder(
        description=description,
        run_at=run_at,
        channel="telegram",
        user_id=user_id,
    )

    await message.answer(
        f"Reminder set (id={task.id}).\n"
        f"I'll remind you about \"{description}\" at "
        f"{run_at.strftime('%H:%M:%S')} (in {delay_str})."
    )


async def _cmd_reminders(self: TelegramChannel, message: Any) -> None:
    """Handle /reminders command — list pending reminders."""
    if not self._check_message(message):
        return

    if not self.scheduler:
        await message.answer("Scheduler is not configured.")
        return

    tasks = self.scheduler.list_tasks()
    if not tasks:
        await message.answer("No scheduled reminders.")
        return

    lines: list[str] = ["Scheduled reminders:\n"]
    for t in tasks:
        status_icon = {
            "pending": "\u23f3",
            "running": "\U0001f504",
            "completed": "\u2705",
            "failed": "\u274c",
        }
        icon = status_icon.get(t.status, "\u2753")
        time_info = t.next_run.strftime("%H:%M:%S") if t.next_run else t.schedule
        lines.append(f"{icon} [{t.id}] {t.description} @ {time_info}")

    await message.answer("\n".join(lines))


def _get_shortcuts(self: TelegramChannel) -> list[dict[str, str]]:
    """Resolve prompt shortcuts — merge defaults with config overrides."""
    from agent.channels.telegram._core import _DEFAULT_SHORTCUTS
    from agent.config import get_config

    try:
        cfg = get_config()
        config_shortcuts = [
            {"alias": s.alias, "template": s.template, "description": s.description}
            for s in cfg.prompts.shortcuts
        ]
    except Exception:
        config_shortcuts = []

    # Config overrides defaults by alias
    merged: dict[str, dict[str, str]] = {}
    for s in _DEFAULT_SHORTCUTS:
        merged[s["alias"]] = s
    for s in config_shortcuts:
        merged[s["alias"]] = s
    return list(merged.values())


async def _cmd_run(self: TelegramChannel, message: Any) -> None:
    """Handle /run command — run a prompt shortcut.

    Usage:
        /run list           — show all shortcuts
        /run <alias> [args] — run a shortcut
    """
    if not self._check_message(message):
        return

    if self._paused:
        await message.answer("I'm currently paused. Use /resume to re-enable me.")
        return

    text = (message.text or "").strip()
    parts = text.split(maxsplit=2)

    shortcuts = self._get_shortcuts()

    # /run (no args) or /run list
    if len(parts) < 2 or parts[1].lower() == "list":
        lines = ["Prompt shortcuts:\n"]
        for s in shortcuts:
            desc = f" — {s['description']}" if s.get("description") else ""
            lines.append(f"  /run {s['alias']}{desc}")
        lines.append("\nUsage: /run <shortcut> [args]")
        await message.answer("\n".join(lines))
        return

    alias = parts[1].lower()
    args = parts[2] if len(parts) > 2 else ""

    # Find matching shortcut
    shortcut = next((s for s in shortcuts if s["alias"] == alias), None)
    if not shortcut:
        await message.answer(
            f"Unknown shortcut: {alias}\n" "Use /run list to see available shortcuts."
        )
        return

    # Interpolate {args}
    prompt = shortcut["template"]
    if "{args}" in prompt:
        if not args:
            await message.answer(
                f"Shortcut '{alias}' requires arguments.\n" f"Usage: /run {alias} <args>"
            )
            return
        prompt = prompt.replace("{args}", args)

    # Process through agent loop (same pattern as _handle_text)
    user_id = message.from_user.id

    agent_loop, session_store = self._resolve_components(str(user_id))
    session_id = self._make_session_id(str(user_id))
    session = await session_store.get_or_create(session_id=session_id, channel="telegram")

    status_message = None
    with contextlib.suppress(Exception):
        status_message = await message.answer("\u23f3 Processing\u2026")

    # Dispatch to background
    task = asyncio.create_task(
        self._run_text_task(
            user_text=prompt,
            session=session,
            agent_loop=agent_loop,
            status_message=status_message,
            user_id=str(user_id),
            message=message,
        ),
        name=f"run:{user_id}:{alias}",
    )
    self._register_background_task(str(user_id), task, f"/run {alias}")


async def _cmd_tasks(self: TelegramChannel, message: Any) -> None:
    """Handle /tasks — show running background tasks."""
    if not self._check_message(message):
        return

    active = self._get_active_tasks()
    if not active:
        await message.answer("No tasks running right now. I'm free to chat!")
        return

    lines = [f"\u2699\ufe0f <b>Running tasks ({len(active)}):</b>\n"]
    for _uid, desc in active:
        lines.append(f"\u2022 {desc}")
    lines.append("\nUse /stop to cancel all running tasks.")
    await message.answer("\n".join(lines), parse_mode="HTML")


async def _cmd_stop(self: TelegramChannel, message: Any) -> None:
    """Handle /stop — cancel all running background tasks for this user."""
    if not self._check_message(message):
        return

    user_id = str(message.from_user.id)
    cancelled = await self._cancel_user_tasks(user_id)

    if cancelled == 0:
        await message.answer("No tasks running to cancel.")
    else:
        await message.answer(f"\u26d4 Cancelled {cancelled} task(s).")


async def _cancel_user_tasks(self: TelegramChannel, user_id: str) -> int:
    """Cancel all running tasks for a user.

    When an orchestrator is available, cancels via the orchestrator
    (which handles SDK interruption, event emission, and cleanup).
    Otherwise falls back to raw asyncio task cancellation.

    Returns the number of tasks cancelled.
    """
    tasks = list(self._background_tasks.get(user_id, []))
    if not tasks:
        return 0

    cancelled = 0

    # Cancel via orchestrator when available (handles SDK interruption)
    if self.orchestrator is not None:
        session_id = self._make_session_id(user_id)
        session = await self.session_store.get(session_id)
        if session:
            with contextlib.suppress(Exception):
                await self.orchestrator.cancel(session.id)

    # Also cancel the asyncio wrapper tasks
    for task, _desc in tasks:
        if not task.done():
            task.cancel()
            cancelled += 1

    # Also interrupt the SDK session directly (in case no orchestrator)
    if self.orchestrator is None and self.sdk_service is not None:
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
        session_id = self._make_session_id(user_id)
        session = await self.session_store.get(session_id)
        if session:
            with contextlib.suppress(Exception):
                await sdk.cancel_task(session.id)

    # Wait briefly for tasks to handle cancellation
    for task, _desc in tasks:
        if not task.done():
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

    # Clean up
    self._background_tasks.pop(user_id, None)

    return cancelled
