"""Tests for Telegram inline keyboard approval flow."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.channels.telegram import TelegramChannel
from agent.core.events import EventBus
from agent.core.session import SessionStore


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def session_store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    loop = MagicMock()
    loop.system_prompt = "You are a helpful assistant."
    loop.llm = MagicMock()
    loop.process_message = AsyncMock()
    return loop


@pytest.fixture
def channel(
    event_bus: EventBus,
    session_store: SessionStore,
    mock_agent_loop: MagicMock,
) -> TelegramChannel:
    """Create TelegramChannel with mocked bot."""
    with patch("agent.channels.telegram.AIOGRAM_AVAILABLE", True), \
         patch("agent.channels.telegram.Bot"), \
         patch("agent.channels.telegram.Dispatcher"), \
         patch("agent.channels.telegram.Router"):
        cfg = MagicMock()
        cfg.token = "fake:token"
        cfg.allowed_users = []

        ch = TelegramChannel(
            config=cfg,
            event_bus=event_bus,
            session_store=session_store,
            agent_loop=mock_agent_loop,
        )
        ch._bot = MagicMock()
        ch._bot.send_message = AsyncMock()
        ch._bot.send_chat_action = AsyncMock()
        return ch


class TestApprovalRequest:
    """Test send_approval_request sends inline keyboard."""

    async def test_sends_inline_keyboard_message(
        self, channel: TelegramChannel
    ) -> None:
        """Approval request should send a message with inline keyboard."""
        # Simulate approval in background
        async def approve_later() -> None:
            await asyncio.sleep(0.05)
            req_id = list(channel._approval_futures.keys())[0]
            channel._approval_futures[req_id].set_result(True)

        task = asyncio.create_task(approve_later())

        result = await channel.send_approval_request(
            channel_user_id="42",
            tool_name="shell_exec",
            arguments={"cmd": "rm -rf /tmp/test"},
            request_id="req-001",
        )

        await task
        assert result is True
        channel._bot.send_message.assert_awaited_once()
        call_kwargs = channel._bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 42
        assert "reply_markup" in call_kwargs


class TestApproveCallback:
    """Test inline button callback handling."""

    async def test_approve_resolves_future_true(
        self, channel: TelegramChannel
    ) -> None:
        """'approve:{id}' callback should resolve future to True."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        channel._approval_futures["req-002"] = future

        callback = MagicMock()
        callback.data = "approve:req-002"
        callback.message = MagicMock()
        callback.message.text = "Tool Approval"
        callback.message.edit_text = AsyncMock()
        callback.answer = AsyncMock()

        await channel._handle_approval_callback(callback)

        assert future.done()
        assert future.result() is True
        callback.message.edit_text.assert_awaited_once()
        assert "Approved" in callback.message.edit_text.call_args[0][0]

    async def test_deny_resolves_future_false(
        self, channel: TelegramChannel
    ) -> None:
        """'deny:{id}' callback should resolve future to False."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        channel._approval_futures["req-003"] = future

        callback = MagicMock()
        callback.data = "deny:req-003"
        callback.message = MagicMock()
        callback.message.text = "Tool Approval"
        callback.message.edit_text = AsyncMock()
        callback.answer = AsyncMock()

        await channel._handle_approval_callback(callback)

        assert future.done()
        assert future.result() is False
        callback.message.edit_text.assert_awaited_once()
        assert "Denied" in callback.message.edit_text.call_args[0][0]

    async def test_approve_session_resolves_true(
        self, channel: TelegramChannel
    ) -> None:
        """'approve_session:{id}:{tool}' callback should resolve future to True."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        channel._approval_futures["req-004"] = future

        callback = MagicMock()
        callback.data = "approve_session:req-004:shell_exec"
        callback.message = MagicMock()
        callback.message.text = "Tool Approval"
        callback.message.edit_text = AsyncMock()
        callback.answer = AsyncMock()

        await channel._handle_approval_callback(callback)

        assert future.done()
        assert future.result() is True


class TestApprovalTimeout:
    """Test approval timeout behaviour."""

    async def test_timeout_returns_false(
        self, channel: TelegramChannel
    ) -> None:
        """Approval request should return False on timeout."""
        with patch("agent.channels.telegram._APPROVAL_TIMEOUT", 0.05):
            result = await channel.send_approval_request(
                channel_user_id="42",
                tool_name="dangerous_tool",
                arguments={"arg": "value"},
                request_id="req-timeout",
            )

        assert result is False
        # Future should be cleaned up
        assert "req-timeout" not in channel._approval_futures
