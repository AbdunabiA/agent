"""Async event bus for internal component communication.

Provides a lightweight pub/sub system where components can emit and
subscribe to events without tight coupling.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Type alias for async event handlers
EventHandler = Callable[..., Coroutine[Any, Any, None]]


class Events:
    """Event name constants used throughout the application."""

    MESSAGE_INCOMING = "message.incoming"
    MESSAGE_OUTGOING = "message.outgoing"
    TOOL_EXECUTE = "tool.execute"
    TOOL_RESULT = "tool.result"
    HEARTBEAT_TICK = "heartbeat.tick"
    HEARTBEAT_ACTION = "heartbeat.action"
    MEMORY_UPDATE = "memory.update"
    SKILL_LOADED = "skill.loaded"
    AGENT_ERROR = "agent.error"
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    VOICE_TRANSCRIBED = "voice.transcribed"
    VOICE_SYNTHESIZED = "voice.synthesized"
    FILE_SEND = "file.send"


class EventBus:
    """Async event bus for internal component communication.

    Usage:
        bus = EventBus()

        async def handler(data):
            print(f"Received: {data}")

        bus.on("message.incoming", handler)
        await bus.emit("message.incoming", {"text": "hello"})
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def on(self, event: str, handler: EventHandler) -> None:
        """Register a handler for an event.

        Args:
            event: Event name to subscribe to.
            handler: Async callable to invoke when event fires.
        """
        self._handlers[event].append(handler)

    def off(self, event: str, handler: EventHandler) -> None:
        """Unregister a handler for an event.

        Args:
            event: Event name to unsubscribe from.
            handler: The handler to remove.
        """
        if event in self._handlers:
            with contextlib.suppress(ValueError):
                self._handlers[event].remove(handler)

    async def emit(self, event: str, data: Any = None) -> None:
        """Emit an event to all registered handlers.

        Calls all handlers concurrently. Individual handler errors
        are logged but don't break other handlers.

        Args:
            event: Event name to emit.
            data: Optional data to pass to handlers.
        """
        handlers = self._handlers.get(event, [])
        if not handlers:
            return

        async def _safe_call(handler: EventHandler) -> None:
            try:
                await handler(data)
            except Exception as e:
                logger.error(
                    "event_handler_error",
                    event_name=event,
                    handler_name=handler.__name__,
                    error=str(e),
                )

        await asyncio.gather(*[_safe_call(h) for h in handlers])

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
