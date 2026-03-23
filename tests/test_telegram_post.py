"""Tests for Telegram posting tools (post_to_channel, send_telegram_message, schedule_post)."""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.builtins import telegram_post

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module-level globals before each test."""
    telegram_post._global_event_bus = None
    telegram_post._global_scheduler = None
    telegram_post._channel_var.set(None)
    telegram_post._user_id_var.set(None)
    yield
    telegram_post._global_event_bus = None
    telegram_post._global_scheduler = None
    telegram_post._channel_var.set(None)
    telegram_post._user_id_var.set(None)


def _make_event_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.emit = AsyncMock()
    return bus


# -----------------------------------------------------------------------
# _parse_delay
# -----------------------------------------------------------------------


class TestParseDelay:
    def test_compact_minutes(self) -> None:
        assert telegram_post._parse_delay("5m") == timedelta(minutes=5)

    def test_compact_hours(self) -> None:
        assert telegram_post._parse_delay("2h") == timedelta(hours=2)

    def test_compact_combo(self) -> None:
        assert telegram_post._parse_delay("1h30m") == timedelta(hours=1, minutes=30)

    def test_natural_language(self) -> None:
        assert telegram_post._parse_delay("30 minutes") == timedelta(minutes=30)

    def test_plain_number(self) -> None:
        assert telegram_post._parse_delay("10") == timedelta(minutes=10)

    def test_invalid(self) -> None:
        assert telegram_post._parse_delay("asap") is None


# -----------------------------------------------------------------------
# post_to_channel
# -----------------------------------------------------------------------


class TestPostToChannel:
    async def test_emits_event(self) -> None:
        bus = _make_event_bus()
        telegram_post._global_event_bus = bus

        result = await telegram_post.post_to_channel(
            chat_id="@mychannel",
            text="Hello world",
        )

        assert "Posted to @mychannel" in result
        bus.emit.assert_called_once()
        call_args = bus.emit.call_args
        assert call_args[0][0] == "channel.post"
        data = call_args[0][1]
        assert data["chat_id"] == "@mychannel"
        assert data["text"] == "Hello world"
        assert data["channel"] == "telegram"

    async def test_no_event_bus(self) -> None:
        result = await telegram_post.post_to_channel(
            chat_id="@ch",
            text="hello",
        )
        assert "ERROR" in result

    async def test_empty_text(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        result = await telegram_post.post_to_channel(
            chat_id="@ch",
            text="",
        )
        assert "ERROR" in result

    async def test_empty_chat_id(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        result = await telegram_post.post_to_channel(
            chat_id="",
            text="hello",
        )
        assert "ERROR" in result

    async def test_with_photo_missing(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        result = await telegram_post.post_to_channel(
            chat_id="@ch",
            text="hello",
            photo_path="/nonexistent/photo.jpg",
        )
        assert "ERROR" in result
        assert "not found" in result.lower()

    async def test_with_photo_valid(self) -> None:
        bus = _make_event_bus()
        telegram_post._global_event_bus = bus

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            photo = f.name

        try:
            result = await telegram_post.post_to_channel(
                chat_id="@ch",
                text="check this out",
                photo_path=photo,
            )
            assert "Posted to @ch" in result
            assert "with photo" in result
            data = bus.emit.call_args[0][1]
            assert data["photo_path"]  # non-empty
        finally:
            os.unlink(photo)

    async def test_with_pin(self) -> None:
        bus = _make_event_bus()
        telegram_post._global_event_bus = bus

        result = await telegram_post.post_to_channel(
            chat_id="@ch",
            text="important",
            pin=True,
        )
        assert "pinned" in result
        data = bus.emit.call_args[0][1]
        assert data["pin"] is True


# -----------------------------------------------------------------------
# send_telegram_message
# -----------------------------------------------------------------------


class TestSendTelegramMessage:
    async def test_emits_event(self) -> None:
        bus = _make_event_bus()
        telegram_post._global_event_bus = bus

        result = await telegram_post.send_telegram_message(
            user_id="12345",
            text="Hello!",
        )

        assert "sent to user 12345" in result
        bus.emit.assert_called_once()
        data = bus.emit.call_args[0][1]
        assert data["user_id"] == "12345"
        assert data["text"] == "Hello!"
        assert data["channel"] == "telegram"

    async def test_no_event_bus(self) -> None:
        result = await telegram_post.send_telegram_message(
            user_id="123",
            text="hi",
        )
        assert "ERROR" in result

    async def test_empty_text(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        result = await telegram_post.send_telegram_message(
            user_id="123",
            text="  ",
        )
        assert "ERROR" in result

    async def test_empty_user_id(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        result = await telegram_post.send_telegram_message(
            user_id="",
            text="hello",
        )
        assert "ERROR" in result


# -----------------------------------------------------------------------
# schedule_post
# -----------------------------------------------------------------------


class TestSchedulePost:
    async def test_valid_schedule(self) -> None:
        bus = _make_event_bus()
        telegram_post._global_event_bus = bus

        mock_task = MagicMock()
        mock_task.id = "abc123"

        scheduler = AsyncMock()
        scheduler.add_reminder = AsyncMock(return_value=mock_task)
        telegram_post._global_scheduler = scheduler

        result = await telegram_post.schedule_post(
            chat_id="@ch",
            text="scheduled post",
            delay="30m",
        )

        assert "scheduled" in result.lower()
        assert "@ch" in result
        assert "abc123" in result
        scheduler.add_reminder.assert_called_once()
        # Description should contain the scheduled_post metadata prefix
        call_kwargs = scheduler.add_reminder.call_args[1]
        assert "[scheduled_post:" in call_kwargs["description"]
        assert "scheduled post" in call_kwargs["description"]

    async def test_invalid_delay(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        telegram_post._global_scheduler = MagicMock()

        result = await telegram_post.schedule_post(
            chat_id="@ch",
            text="hello",
            delay="asap",
        )
        assert "ERROR" in result
        assert "parse" in result.lower()

    async def test_too_short_delay(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        telegram_post._global_scheduler = MagicMock()

        result = await telegram_post.schedule_post(
            chat_id="@ch",
            text="hello",
            delay="5s",
        )
        assert "ERROR" in result
        assert "10 seconds" in result

    async def test_no_scheduler(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        # No scheduler set
        result = await telegram_post.schedule_post(
            chat_id="@ch",
            text="hello",
            delay="1h",
        )
        assert "ERROR" in result

    async def test_no_event_bus(self) -> None:
        result = await telegram_post.schedule_post(
            chat_id="@ch",
            text="hello",
            delay="1h",
        )
        assert "ERROR" in result

    async def test_empty_text(self) -> None:
        telegram_post._global_event_bus = _make_event_bus()
        result = await telegram_post.schedule_post(
            chat_id="@ch",
            text="",
            delay="1h",
        )
        assert "ERROR" in result


# -----------------------------------------------------------------------
# TelegramChannel event handlers
# -----------------------------------------------------------------------


class TestChannelPostHandler:
    """Tests for _on_channel_post and _on_send_message handlers."""

    def _make_channel(self) -> Any:
        """Create a minimal TelegramChannel mock for handler testing."""

        channel = MagicMock()
        channel._bot = AsyncMock()
        channel._bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
        channel._bot.send_photo = AsyncMock(return_value=MagicMock(message_id=2))
        channel._bot.pin_chat_message = AsyncMock()
        return channel

    async def test_post_text_only(self) -> None:
        from agent.channels.telegram import TelegramChannel

        ch = self._make_channel()

        await TelegramChannel._on_channel_post(
            ch,
            {
                "channel": "telegram",
                "chat_id": "@test",
                "text": "Hello",
                "parse_mode": "Markdown",
            },
        )

        ch._bot.send_message.assert_called_once_with(
            chat_id="@test",
            text="Hello",
            parse_mode="Markdown",
        )

    async def test_post_with_pin(self) -> None:
        from agent.channels.telegram import TelegramChannel

        ch = self._make_channel()

        await TelegramChannel._on_channel_post(
            ch,
            {
                "channel": "telegram",
                "chat_id": "@test",
                "text": "Pinned post",
                "pin": True,
            },
        )

        ch._bot.pin_chat_message.assert_called_once()

    async def test_post_ignores_non_telegram(self) -> None:
        from agent.channels.telegram import TelegramChannel

        ch = self._make_channel()

        await TelegramChannel._on_channel_post(
            ch,
            {
                "channel": "webchat",
                "chat_id": "@test",
                "text": "Hello",
            },
        )

        ch._bot.send_message.assert_not_called()

    async def test_post_missing_data(self) -> None:
        from agent.channels.telegram import TelegramChannel

        ch = self._make_channel()

        # Missing text
        await TelegramChannel._on_channel_post(
            ch,
            {
                "channel": "telegram",
                "chat_id": "@test",
            },
        )

        ch._bot.send_message.assert_not_called()

    async def test_send_message(self) -> None:
        from agent.channels.telegram import TelegramChannel

        ch = self._make_channel()

        await TelegramChannel._on_send_message(
            ch,
            {
                "channel": "telegram",
                "user_id": "12345",
                "text": "Hello user!",
                "parse_mode": "Markdown",
            },
        )

        ch._bot.send_message.assert_called_once_with(
            chat_id=12345,
            text="Hello user!",
            parse_mode="Markdown",
        )

    async def test_send_message_ignores_non_telegram(self) -> None:
        from agent.channels.telegram import TelegramChannel

        ch = self._make_channel()

        await TelegramChannel._on_send_message(
            ch,
            {
                "channel": "webchat",
                "user_id": "123",
                "text": "hi",
            },
        )

        ch._bot.send_message.assert_not_called()


# Allow mypy to accept the _make_channel mock return
