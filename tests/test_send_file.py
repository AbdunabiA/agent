"""Tests for the send_file tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.core.events import EventBus, Events
from agent.tools.builtins.send_file import (
    _global_context,
    send_file,
    set_file_send_bus,
    set_file_send_context,
)


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset global state between tests."""
    import agent.tools.builtins.send_file as mod

    original_bus = mod._global_event_bus
    original_ctx = dict(mod._global_context)
    yield
    mod._global_event_bus = original_bus
    mod._global_context.update(original_ctx)


@pytest.fixture()
def event_bus() -> EventBus:
    """Create an event bus with a mocked emit."""
    bus = EventBus()
    bus.emit = AsyncMock()  # type: ignore[method-assign]
    return bus


class TestSendFile:
    """Tests for the send_file tool function."""

    async def test_no_event_bus_returns_error(self):
        """Returns error when event bus is not initialized."""
        import agent.tools.builtins.send_file as mod

        mod._global_event_bus = None
        result = await send_file("/some/file.txt")
        assert "[ERROR]" in result
        assert "event bus" in result.lower()

    async def test_no_context_returns_error(self, event_bus: EventBus):
        """Returns error when channel/user context is not set."""
        set_file_send_bus(event_bus)
        set_file_send_context(channel=None, user_id=None)
        result = await send_file("/some/file.txt")
        assert "[ERROR]" in result
        assert "channel" in result.lower()

    async def test_file_not_found_returns_error(self, event_bus: EventBus):
        """Returns error for nonexistent file."""
        set_file_send_bus(event_bus)
        set_file_send_context(channel="telegram", user_id="123")
        result = await send_file("/nonexistent/path/file.txt")
        assert "[ERROR]" in result
        assert "not found" in result.lower()

    async def test_directory_returns_error(
        self, event_bus: EventBus, tmp_path: Path
    ):
        """Returns error when path is a directory."""
        set_file_send_bus(event_bus)
        set_file_send_context(channel="telegram", user_id="123")
        result = await send_file(str(tmp_path))
        assert "[ERROR]" in result
        assert "Not a file" in result

    async def test_empty_file_returns_error(
        self, event_bus: EventBus, tmp_path: Path
    ):
        """Returns error for empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        set_file_send_bus(event_bus)
        set_file_send_context(channel="telegram", user_id="123")
        result = await send_file(str(empty_file))
        assert "[ERROR]" in result
        assert "empty" in result.lower()

    async def test_successful_send_emits_event(
        self, event_bus: EventBus, tmp_path: Path
    ):
        """Emits FILE_SEND event and returns confirmation on success."""
        test_file = tmp_path / "hello.txt"
        test_file.write_text("Hello, world!")
        set_file_send_bus(event_bus)
        set_file_send_context(channel="telegram", user_id="42")

        result = await send_file(str(test_file), caption="Test file")

        assert "File sent" in result
        assert "hello.txt" in result
        event_bus.emit.assert_called_once()  # type: ignore[attr-defined]
        call_args = event_bus.emit.call_args  # type: ignore[attr-defined]
        assert call_args[0][0] == Events.FILE_SEND
        data = call_args[0][1]
        assert data["file_name"] == "hello.txt"
        assert data["caption"] == "Test file"
        assert data["channel"] == "telegram"
        assert data["user_id"] == "42"

    async def test_send_without_caption(
        self, event_bus: EventBus, tmp_path: Path
    ):
        """Works with empty caption."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")
        set_file_send_bus(event_bus)
        set_file_send_context(channel="telegram", user_id="99")

        result = await send_file(str(test_file))

        assert "File sent" in result
        data = event_bus.emit.call_args[0][1]  # type: ignore[attr-defined]
        assert data["caption"] == ""


class TestSetContext:
    """Tests for context management functions."""

    def test_set_context_updates_globals(self):
        """set_file_send_context updates the global context dict."""
        set_file_send_context(channel="webchat", user_id="abc")
        assert _global_context["channel"] == "webchat"
        assert _global_context["user_id"] == "abc"

    def test_set_context_clears(self):
        """Setting None clears the context."""
        set_file_send_context(channel="telegram", user_id="123")
        set_file_send_context(channel=None, user_id=None)
        assert _global_context["channel"] is None
        assert _global_context["user_id"] is None
