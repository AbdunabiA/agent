"""Tests for the async event bus."""

from __future__ import annotations

import pytest

from agent.core.events import EventBus


class TestEventBus:
    """Test EventBus functionality."""

    @pytest.fixture
    def bus(self) -> EventBus:
        return EventBus()

    async def test_register_and_emit(self, bus: EventBus) -> None:
        received: list[object] = []

        async def handler(data: object) -> None:
            received.append(data)

        bus.on("test.event", handler)
        await bus.emit("test.event", {"msg": "hello"})

        assert len(received) == 1
        assert received[0] == {"msg": "hello"}

    async def test_multiple_handlers(self, bus: EventBus) -> None:
        calls: list[str] = []

        async def handler1(data: object) -> None:
            calls.append("h1")

        async def handler2(data: object) -> None:
            calls.append("h2")

        bus.on("test.event", handler1)
        bus.on("test.event", handler2)
        await bus.emit("test.event")

        assert "h1" in calls
        assert "h2" in calls
        assert len(calls) == 2

    async def test_unregister_handler(self, bus: EventBus) -> None:
        calls: list[str] = []

        async def handler(data: object) -> None:
            calls.append("called")

        bus.on("test.event", handler)
        bus.off("test.event", handler)
        await bus.emit("test.event")

        assert len(calls) == 0

    async def test_handler_error_doesnt_break_others(self, bus: EventBus) -> None:
        calls: list[str] = []

        async def bad_handler(data: object) -> None:
            raise ValueError("oops")

        async def good_handler(data: object) -> None:
            calls.append("ok")

        bus.on("test.event", bad_handler)
        bus.on("test.event", good_handler)
        await bus.emit("test.event")

        assert calls == ["ok"]

    async def test_emit_no_handlers(self, bus: EventBus) -> None:
        # Should not raise
        await bus.emit("nonexistent.event", {"data": 123})

    async def test_clear_removes_all(self, bus: EventBus) -> None:
        calls: list[str] = []

        async def handler(data: object) -> None:
            calls.append("called")

        bus.on("event1", handler)
        bus.on("event2", handler)
        bus.clear()

        await bus.emit("event1")
        await bus.emit("event2")

        assert len(calls) == 0

    async def test_emit_with_none_data(self, bus: EventBus) -> None:
        received: list[object] = []

        async def handler(data: object) -> None:
            received.append(data)

        bus.on("test.event", handler)
        await bus.emit("test.event")

        assert received == [None]

    async def test_unregister_nonexistent_handler(self, bus: EventBus) -> None:
        async def handler(data: object) -> None:
            pass

        # Should not raise
        bus.off("test.event", handler)

    async def test_multiple_events_independent(self, bus: EventBus) -> None:
        event1_calls: list[str] = []
        event2_calls: list[str] = []

        async def handler1(data: object) -> None:
            event1_calls.append("e1")

        async def handler2(data: object) -> None:
            event2_calls.append("e2")

        bus.on("event1", handler1)
        bus.on("event2", handler2)

        await bus.emit("event1")
        assert len(event1_calls) == 1
        assert len(event2_calls) == 0

        await bus.emit("event2")
        assert len(event2_calls) == 1
