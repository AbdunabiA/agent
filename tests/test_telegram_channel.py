"""Tests for the Telegram channel adapter."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import TelegramConfig
from agent.core.events import EventBus
from agent.core.session import SessionStore

if TYPE_CHECKING:
    from agent.channels.telegram import TelegramChannel


def _make_tg_message(
    user_id: int = 111,
    text: str | None = "hello",
) -> MagicMock:
    """Create a mock aiogram Message."""
    msg = AsyncMock()
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.text = text
    msg.answer = AsyncMock()
    return msg


@pytest.fixture
def config() -> TelegramConfig:
    return TelegramConfig(enabled=True, token="fake-token", allowed_users=[])


@pytest.fixture
def config_restricted() -> TelegramConfig:
    return TelegramConfig(enabled=True, token="fake-token", allowed_users=[111, 222])


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def session_store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def mock_agent_loop() -> AsyncMock:
    loop = AsyncMock()
    response = AsyncMock()
    response.content = "Agent reply"
    loop.process_message.return_value = response
    return loop


@pytest.fixture
def mock_heartbeat() -> MagicMock:
    hb = MagicMock()
    hb.is_enabled = True
    hb.last_tick = None
    hb.enable = MagicMock()
    hb.disable = MagicMock()
    return hb


@pytest.fixture
def channel(
    config: TelegramConfig,
    event_bus: EventBus,
    session_store: SessionStore,
    mock_agent_loop: AsyncMock,
    mock_heartbeat: MagicMock,
) -> Any:
    from agent.channels.telegram import TelegramChannel

    with patch("agent.channels.telegram.Bot"), \
         patch("agent.channels.telegram.Dispatcher"), \
         patch("agent.channels.telegram.Router"):
        ch = TelegramChannel(
            config=config,
            event_bus=event_bus,
            session_store=session_store,
            agent_loop=mock_agent_loop,
            heartbeat=mock_heartbeat,
        )
        # Give it mock bot/dispatcher so send_message / send_typing / stop work
        ch._bot = AsyncMock()
        ch._dispatcher = AsyncMock()
    return ch


@pytest.fixture
def restricted_channel(
    config_restricted: TelegramConfig,
    event_bus: EventBus,
    session_store: SessionStore,
    mock_agent_loop: AsyncMock,
) -> Any:
    from agent.channels.telegram import TelegramChannel

    with patch("agent.channels.telegram.Bot"), \
         patch("agent.channels.telegram.Dispatcher"), \
         patch("agent.channels.telegram.Router"):
        ch = TelegramChannel(
            config=config_restricted,
            event_bus=event_bus,
            session_store=session_store,
            agent_loop=mock_agent_loop,
            heartbeat=None,
        )
        ch._bot = AsyncMock()
    return ch


# =====================================================================
# Allowlist
# =====================================================================

class TestAllowlist:
    """Allowlist security checks."""

    def test_empty_allowlist_allows_all(self, channel: TelegramChannel) -> None:
        assert channel._is_allowed(999) is True

    def test_restricted_allows_listed(self, restricted_channel: TelegramChannel) -> None:
        assert restricted_channel._is_allowed(111) is True
        assert restricted_channel._is_allowed(222) is True

    def test_restricted_blocks_unlisted(self, restricted_channel: TelegramChannel) -> None:
        assert restricted_channel._is_allowed(333) is False


# =====================================================================
# Handle text
# =====================================================================

class TestHandleText:
    """Text message processing."""

    async def test_processes_message(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        msg = _make_tg_message(text="What is 2+2?")
        await channel._handle_text(msg)

        mock_agent_loop.process_message.assert_called_once()
        call_args = mock_agent_loop.process_message.call_args
        assert call_args[0][0] == "What is 2+2?"

    async def test_sends_response(
        self,
        channel: TelegramChannel,
    ) -> None:
        msg = _make_tg_message(text="Hi")
        await channel._handle_text(msg)

        channel._bot.send_message.assert_called()

    async def test_emits_incoming_event(
        self,
        channel: TelegramChannel,
        event_bus: EventBus,
    ) -> None:
        events: list[dict[str, Any]] = []

        async def handler(data: dict[str, Any]) -> None:
            events.append(data)

        event_bus.on("message.incoming", handler)

        msg = _make_tg_message(text="test event")
        await channel._handle_text(msg)

        assert len(events) == 1
        assert events[0]["content"] == "test event"

    async def test_reuses_session_for_same_user(
        self,
        channel: TelegramChannel,
        session_store: SessionStore,
    ) -> None:
        msg1 = _make_tg_message(user_id=42, text="first")
        msg2 = _make_tg_message(user_id=42, text="second")

        await channel._handle_text(msg1)
        await channel._handle_text(msg2)

        # Should have one session for user 42
        session = await session_store.get("telegram:42")
        assert session is not None

    async def test_handles_agent_error(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        mock_agent_loop.process_message.side_effect = RuntimeError("LLM down")
        msg = _make_tg_message(text="boom")
        await channel._handle_text(msg)

        msg.answer.assert_called_with("Sorry, something went wrong processing your message.")

    async def test_blocks_when_paused(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        channel.pause()
        msg = _make_tg_message(text="ignored")
        await channel._handle_text(msg)

        mock_agent_loop.process_message.assert_not_called()
        msg.answer.assert_called_once()

    async def test_ignores_no_from_user(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        msg = _make_tg_message(text="anon")
        msg.from_user = None
        await channel._handle_text(msg)

        mock_agent_loop.process_message.assert_not_called()

    async def test_ignores_empty_text(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        msg = _make_tg_message(text=None)
        await channel._handle_text(msg)

        mock_agent_loop.process_message.assert_not_called()


# =====================================================================
# Commands
# =====================================================================

class TestCommands:
    """Bot command handlers."""

    async def test_cmd_start(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        await channel._cmd_start(msg)
        msg.answer.assert_called_once()
        assert "Hello" in msg.answer.call_args[0][0]

    async def test_cmd_help(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        await channel._cmd_help(msg)
        text = msg.answer.call_args[0][0]
        assert "/start" in text
        assert "/help" in text
        assert "/mute" in text

    async def test_cmd_status(
        self,
        channel: TelegramChannel,
        mock_heartbeat: MagicMock,
    ) -> None:
        msg = _make_tg_message()
        with patch("agent.tools.registry.registry") as mock_reg:
            mock_reg.list_tools.return_value = [MagicMock(), MagicMock()]
            await channel._cmd_status(msg)

        text = msg.answer.call_args[0][0]
        assert "enabled" in text
        assert "Tools: 2" in text

    async def test_cmd_tools_empty(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        with patch("agent.tools.registry.registry") as mock_reg:
            mock_reg.list_tools.return_value = []
            await channel._cmd_tools(msg)

        msg.answer.assert_called_with("No tools registered.")

    async def test_cmd_tools_with_tools(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()

        mock_tool = MagicMock()
        mock_tool.name = "shell_exec"
        mock_tool.description = "Run shell commands"
        mock_tool.tier.value = "moderate"
        mock_tool.enabled = True

        with patch("agent.tools.registry.registry") as mock_reg:
            mock_reg.list_tools.return_value = [mock_tool]
            await channel._cmd_tools(msg)

        text = msg.answer.call_args[0][0]
        assert "shell_exec" in text
        assert "[on]" in text

    async def test_cmd_pause(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        await channel._cmd_pause(msg)
        assert channel._paused is True
        assert "paused" in msg.answer.call_args[0][0].lower()

    async def test_cmd_resume(self, channel: TelegramChannel) -> None:
        channel.pause()
        msg = _make_tg_message()
        await channel._cmd_resume(msg)
        assert channel._paused is False
        assert "resumed" in msg.answer.call_args[0][0].lower()

    async def test_cmd_mute(
        self,
        channel: TelegramChannel,
        mock_heartbeat: MagicMock,
    ) -> None:
        msg = _make_tg_message()
        await channel._cmd_mute(msg)
        mock_heartbeat.disable.assert_called_once()
        assert "muted" in msg.answer.call_args[0][0].lower()

    async def test_cmd_unmute(
        self,
        channel: TelegramChannel,
        mock_heartbeat: MagicMock,
    ) -> None:
        msg = _make_tg_message()
        await channel._cmd_unmute(msg)
        mock_heartbeat.enable.assert_called_once()
        assert "unmuted" in msg.answer.call_args[0][0].lower()

    async def test_cmd_mute_no_heartbeat(
        self,
        config: TelegramConfig,
        event_bus: EventBus,
        session_store: SessionStore,
        mock_agent_loop: AsyncMock,
    ) -> None:
        from agent.channels.telegram import TelegramChannel

        with patch("agent.channels.telegram.Bot"), \
             patch("agent.channels.telegram.Dispatcher"), \
             patch("agent.channels.telegram.Router"):
            ch = TelegramChannel(
                config=config,
                event_bus=event_bus,
                session_store=session_store,
                agent_loop=mock_agent_loop,
                heartbeat=None,
            )
            ch._bot = AsyncMock()

        msg = _make_tg_message()
        await ch._cmd_mute(msg)
        assert "not configured" in msg.answer.call_args[0][0].lower()


# =====================================================================
# Split message
# =====================================================================

class TestSplitMessage:
    """Message splitting logic."""

    def test_short_message_unchanged(self) -> None:
        from agent.channels.telegram import TelegramChannel

        result = TelegramChannel._split_message("short text")
        assert result == ["short text"]

    def test_splits_at_newline(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "a" * 50 + "\n" + "b" * 50
        result = TelegramChannel._split_message(text, max_length=60)
        assert len(result) == 2
        assert result[0] == "a" * 50

    def test_splits_at_space(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "a" * 50 + " " + "b" * 50
        result = TelegramChannel._split_message(text, max_length=60)
        assert len(result) == 2
        assert result[0] == "a" * 50

    def test_hard_split(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "a" * 100
        result = TelegramChannel._split_message(text, max_length=60)
        assert len(result) == 2
        assert len(result[0]) == 60
        assert len(result[1]) == 40

    def test_multiple_chunks(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "word " * 100  # 500 chars
        result = TelegramChannel._split_message(text, max_length=50)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 50


# =====================================================================
# Lifecycle
# =====================================================================

class TestLifecycle:
    """Start/stop behavior."""

    def test_name(self, channel: TelegramChannel) -> None:
        assert channel.name == "telegram"

    async def test_stop_without_start(self, channel: TelegramChannel) -> None:
        """stop() should not raise if never started."""
        await channel.stop()

    async def test_start_no_token(
        self,
        event_bus: EventBus,
        session_store: SessionStore,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """start() should log warning and return if no token."""
        from agent.channels.telegram import TelegramChannel

        no_token_config = TelegramConfig(enabled=True, token=None)
        ch = TelegramChannel(
            config=no_token_config,
            event_bus=event_bus,
            session_store=session_store,
            agent_loop=mock_agent_loop,
        )
        await ch.start()
        assert ch.is_running is False


# =====================================================================
# Keep typing
# =====================================================================

class TestKeepTyping:
    """Typing indicator loop."""

    async def test_keep_typing_sends_and_cancels(
        self,
        channel: TelegramChannel,
    ) -> None:
        task = asyncio.create_task(channel._keep_typing("42"))

        # Let it run for a short while
        await asyncio.sleep(0.05)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should not raise — task cancelled cleanly
        assert task.done()
