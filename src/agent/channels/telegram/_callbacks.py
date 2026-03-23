"""Telegram callback handlers — inline button callbacks for approval and navigation."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.channels.telegram._core import TelegramChannel

logger = structlog.get_logger(__name__)


async def _handle_callback(self: TelegramChannel, callback: Any) -> None:
    """Route inline button callbacks to the correct handler."""
    data = callback.data or ""
    if data.startswith("nav:"):
        await self._handle_nav_callback(callback)
    else:
        await self._handle_approval_callback(callback)


async def _handle_approval_callback(self: TelegramChannel, callback: Any) -> None:
    """Handle inline button callbacks for tool approval.

    Callback data format: 'approve:{id}', 'deny:{id}', 'approve_session:{id}:{tool}'
    """
    data = callback.data or ""

    if data.startswith("approve_session:"):
        parts = data.split(":", 2)
        if len(parts) == 3:
            request_id = parts[1]
            fut = self._approval_futures.get(request_id)
            if fut and not fut.done():
                fut.set_result(True)
            await callback.message.edit_text(f"{callback.message.text}\n\nApproved (session)")
    elif data.startswith("approve:"):
        request_id = data.split(":", 1)[1]
        fut = self._approval_futures.get(request_id)
        if fut and not fut.done():
            fut.set_result(True)
        await callback.message.edit_text(f"{callback.message.text}\n\nApproved")
    elif data.startswith("deny:"):
        request_id = data.split(":", 1)[1]
        fut = self._approval_futures.get(request_id)
        if fut and not fut.done():
            fut.set_result(False)
        await callback.message.edit_text(f"{callback.message.text}\n\nDenied")

    await callback.answer()


async def _handle_nav_callback(self: TelegramChannel, callback: Any) -> None:
    """Handle navigation inline button callbacks.

    Routes by callback data prefix:
    - nav:status:costs / nav:status:health / nav:status:audit
    - nav:tools:toggle:<name>
    - nav:session:clear / nav:session:new
    """
    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    data = callback.data or ""

    try:
        if data == "nav:status:costs":
            await self._nav_status_costs(callback)
        elif data == "nav:status:health":
            await self._nav_status_health(callback)
        elif data == "nav:status:audit":
            await self._nav_status_audit(callback)
        elif data.startswith("nav:tools:toggle:"):
            tool_name = data.split(":", 3)[3]
            await self._nav_tools_toggle(callback, tool_name)
        elif data == "nav:session:clear":
            await self._nav_session_clear(callback)
        elif data == "nav:session:new":
            await self._nav_session_new(callback)
        elif data == "nav:back:status":
            if AIOGRAM_AVAILABLE:
                from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

                text = await self._build_status_text()
                keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Costs",
                                callback_data="nav:status:costs",
                            ),
                            InlineKeyboardButton(
                                text="Health",
                                callback_data="nav:status:health",
                            ),
                            InlineKeyboardButton(
                                text="Audit",
                                callback_data="nav:status:audit",
                            ),
                        ],
                    ]
                )
                await callback.message.edit_text(text, reply_markup=keyboard)
        elif data == "nav:back:tools":
            text, keyboard = await self._build_tools_text_and_keyboard()
            if text:
                await callback.message.edit_text(
                    text,
                    reply_markup=keyboard,
                )
        elif data == "nav:back:session":
            await callback.message.edit_text(
                "Use /session to view current session info.",
            )
    except Exception as e:
        logger.warning("nav_callback_error", error=str(e), data=data)

    await callback.answer()


async def _nav_status_costs(self: TelegramChannel, callback: Any) -> None:
    """Show cost breakdown detail view."""
    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    if not self.cost_tracker:
        await callback.message.edit_text("Cost tracker not available.")
        return

    stats = self.cost_tracker.get_stats("day")
    total_cost = stats.get("total_cost", 0)
    total_calls = stats.get("total_calls", 0)
    tokens = stats.get("total_tokens", {})

    lines = [
        "Cost Details (Today)\n",
        f"Total cost: ${total_cost:.4f}",
        f"Total calls: {total_calls}",
        f"Tokens in: {tokens.get('input', 0):,}",
        f"Tokens out: {tokens.get('output', 0):,}",
    ]

    by_model = stats.get("by_model", [])
    if by_model:
        lines.append("\nBy model:")
        for m in by_model:
            lines.append(f"  {m['model']}: ${m['cost']:.4f} ({m['percentage']:.0f}%)")

    if AIOGRAM_AVAILABLE:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Back", callback_data="nav:back:status")],
            ]
        )
        await callback.message.edit_text("\n".join(lines), reply_markup=keyboard)
    else:
        await callback.message.edit_text("\n".join(lines))


async def _nav_status_health(self: TelegramChannel, callback: Any) -> None:
    """Show health check summary."""
    from agent.channels.telegram._core import _TG_MAX_LENGTH, AIOGRAM_AVAILABLE
    from agent.config import get_config
    from agent.core.doctor import run_all_checks

    config = get_config()
    checks = await run_all_checks(config)

    status_icon = {"pass": "\u2705", "warn": "\u26a0\ufe0f", "fail": "\u274c"}
    lines = ["Health Checks\n"]
    for check in checks:
        icon = status_icon.get(check.status, "\u2753")
        lines.append(f"{icon} {check.name}: {check.message}")

    pass_count = sum(1 for c in checks if c.status == "pass")
    warn_count = sum(1 for c in checks if c.status == "warn")
    fail_count = sum(1 for c in checks if c.status == "fail")
    lines.append(f"\n{pass_count} pass | {warn_count} warn | {fail_count} fail")

    if AIOGRAM_AVAILABLE:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Back", callback_data="nav:back:status")],
            ]
        )
    else:
        keyboard = None

    # Truncate if too long for Telegram
    text = "\n".join(lines)
    if len(text) > _TG_MAX_LENGTH - 100:
        text = text[: _TG_MAX_LENGTH - 150] + "\n\n... (truncated)"
    await callback.message.edit_text(text, reply_markup=keyboard)


async def _nav_status_audit(self: TelegramChannel, callback: Any) -> None:
    """Show audit log summary."""
    from agent.channels.telegram._core import _TG_MAX_LENGTH, AIOGRAM_AVAILABLE

    if not self.audit_log:
        await callback.message.edit_text("Audit log not available.")
        return

    stats = await self.audit_log.get_stats()
    total = stats.get("total_calls", 0)
    success = stats.get("success_count", 0)
    errors = stats.get("error_count", 0)
    rate = stats.get("success_rate", 0) * 100
    avg_ms = stats.get("avg_duration_ms", 0)

    lines = [
        "Audit Summary\n",
        f"Total tool calls: {total}",
        f"Success: {success} | Errors: {errors}",
        f"Success rate: {rate:.1f}%",
        f"Avg duration: {avg_ms}ms",
    ]

    tools_used = stats.get("tools_used", {})
    if tools_used:
        lines.append("\nPer-tool breakdown:")
        for name, count in sorted(tools_used.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}x")

    if AIOGRAM_AVAILABLE:
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Back", callback_data="nav:back:status")],
            ]
        )
    else:
        keyboard = None

    text = "\n".join(lines)
    if len(text) > _TG_MAX_LENGTH - 100:
        text = text[: _TG_MAX_LENGTH - 150] + "\n\n... (truncated)"
    await callback.message.edit_text(text, reply_markup=keyboard)


async def _nav_tools_toggle(
    self: TelegramChannel,
    callback: Any,
    tool_name: str,
) -> None:
    """Toggle a tool on/off and refresh the tools view."""
    from agent.tools.registry import registry

    try:
        tool = next((t for t in registry.list_tools() if t.name == tool_name), None)
        if not tool:
            await callback.message.edit_text(f"Tool '{tool_name}' not found.")
            return

        if tool.enabled:
            registry.disable_tool(tool_name)
            action = "disabled"
        else:
            registry.enable_tool(tool_name)
            action = "enabled"

        logger.info("tool_toggled_via_telegram", tool=tool_name, action=action)
    except Exception as e:
        logger.warning("tool_toggle_error", tool=tool_name, error=str(e))
        await callback.message.edit_text(f"Error toggling {tool_name}: {e}")
        return

    # Refresh tools view
    text, keyboard = await self._build_tools_text_and_keyboard()
    if text:
        await callback.message.edit_text(text, reply_markup=keyboard)


async def _nav_session_clear(self: TelegramChannel, callback: Any) -> None:
    """Clear conversation history in the current session."""
    user_id = str(callback.from_user.id)
    session_id = self._make_session_id(user_id)

    session = await self.session_store.get(session_id)
    if session:
        session.clear()
        session.metadata.pop("sdk_session_id", None)

    # Disconnect any persistent SDK client for this user
    if self.orchestrator and self.orchestrator.sdk_service:
        with contextlib.suppress(Exception):
            await self.orchestrator.sdk_service.disconnect_client(session_id)

    await callback.message.edit_text("Session cleared. Your next message starts fresh.")


async def _nav_session_new(self: TelegramChannel, callback: Any) -> None:
    """Create a new session."""
    user_id = str(callback.from_user.id)
    session_id = self._make_session_id(user_id)

    session = await self.session_store.get(session_id)
    if session:
        session.metadata.pop("sdk_session_id", None)

    # Disconnect any persistent SDK client for this user
    if self.orchestrator and self.orchestrator.sdk_service:
        with contextlib.suppress(Exception):
            await self.orchestrator.sdk_service.disconnect_client(session_id)

    await callback.message.edit_text(
        "New conversation started. Your next message begins a fresh session."
    )
