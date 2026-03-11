"""Tests for the abstract BaseChannel and message data classes."""

from __future__ import annotations

import pytest

from agent.channels.base import (
    Attachment,
    BaseChannel,
    IncomingMessage,
    OutgoingMessage,
)
from agent.core.events import EventBus
from agent.core.session import SessionStore


class ConcreteChannel(BaseChannel):
    """Minimal concrete channel for testing."""

    @property
    def name(self) -> str:
        return "test"

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send_message(self, message: OutgoingMessage) -> None:
        pass

    async def send_typing(self, channel_user_id: str) -> None:
        pass


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def session_store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def channel(event_bus: EventBus, session_store: SessionStore) -> ConcreteChannel:
    return ConcreteChannel(
        config={},
        event_bus=event_bus,
        session_store=session_store,
    )


class TestBaseChannelABC:
    """Test that BaseChannel cannot be instantiated directly."""

    def test_cannot_instantiate_abc(
        self, event_bus: EventBus, session_store: SessionStore
    ) -> None:
        with pytest.raises(TypeError):
            BaseChannel(config={}, event_bus=event_bus, session_store=session_store)  # type: ignore[abstract]

    def test_concrete_subclass_works(self, channel: ConcreteChannel) -> None:
        assert channel.name == "test"


class TestMakeSessionId:
    """Test deterministic session ID generation."""

    def test_returns_channel_colon_userid(self, channel: ConcreteChannel) -> None:
        assert channel._make_session_id("12345") == "test:12345"

    def test_different_users_get_different_ids(self, channel: ConcreteChannel) -> None:
        assert channel._make_session_id("1") != channel._make_session_id("2")


class TestRunningState:
    """Test is_running reflects _running."""

    def test_initially_false(self, channel: ConcreteChannel) -> None:
        assert channel.is_running is False

    async def test_reflects_running(self, channel: ConcreteChannel) -> None:
        await channel.start()
        assert channel.is_running is True
        await channel.stop()
        assert channel.is_running is False


class TestPauseResume:
    """Test pause/resume toggle."""

    def test_initially_not_paused(self, channel: ConcreteChannel) -> None:
        assert channel._paused is False

    def test_pause(self, channel: ConcreteChannel) -> None:
        channel.pause()
        assert channel._paused is True

    def test_resume(self, channel: ConcreteChannel) -> None:
        channel.pause()
        channel.resume()
        assert channel._paused is False


class TestApprovalRequest:
    """Test default approval behaviour."""

    async def test_default_auto_approves(self, channel: ConcreteChannel) -> None:
        result = await channel.send_approval_request(
            "user1", "shell_exec", {"cmd": "ls"}, request_id="test-req-1"
        )
        assert result is True


class TestDataClasses:
    """Test message and attachment data classes."""

    def test_incoming_message(self) -> None:
        msg = IncomingMessage(
            channel="telegram",
            channel_user_id="42",
            content="hello",
        )
        assert msg.channel == "telegram"
        assert msg.channel_user_id == "42"
        assert msg.content == "hello"
        assert msg.attachments == []
        assert msg.metadata == {}
        assert msg.message_id is None

    def test_outgoing_message(self) -> None:
        msg = OutgoingMessage(content="world", channel_user_id="42")
        assert msg.content == "world"
        assert msg.parse_mode is None
        assert msg.attachments == []

    def test_attachment(self) -> None:
        att = Attachment(type="photo", file_id="abc123", file_size=1024)
        assert att.type == "photo"
        assert att.file_id == "abc123"
        assert att.file_size == 1024
        assert att.data is None
