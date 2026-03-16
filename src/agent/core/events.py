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
    CHANNEL_POST = "channel.post"
    CHANNEL_SEND_MESSAGE = "channel.send_message"

    # Self-building skills
    SKILL_BUILD_REQUESTED = "skill.build.requested"
    SKILL_BUILD_COMPLETED = "skill.build.completed"

    # Sub-agent orchestration
    SUBAGENT_SPAWNED = "subagent.spawned"
    SUBAGENT_STARTED = "subagent.started"
    SUBAGENT_PROGRESS = "subagent.progress"
    SUBAGENT_COMPLETED = "subagent.completed"
    SUBAGENT_FAILED = "subagent.failed"
    SUBAGENT_CANCELLED = "subagent.cancelled"

    # Project pipeline
    PROJECT_STARTED = "project.started"
    PROJECT_STAGE_STARTED = "project.stage.started"
    PROJECT_STAGE_COMPLETED = "project.stage.completed"
    PROJECT_COMPLETED = "project.completed"
    PROJECT_FAILED = "project.failed"

    # Project feedback loops
    PROJECT_FEEDBACK_STARTED = "project.feedback.started"
    PROJECT_FEEDBACK_ITERATION = "project.feedback.iteration"
    PROJECT_FEEDBACK_PASSED = "project.feedback.passed"
    PROJECT_FEEDBACK_EXHAUSTED = "project.feedback.exhausted"

    # Inter-agent consultation
    AGENT_CONSULT_REQUESTED = "agent.consult.requested"
    AGENT_CONSULT_COMPLETED = "agent.consult.completed"
    AGENT_CONSULT_FAILED = "agent.consult.failed"

    # Inter-agent delegation
    AGENT_DELEGATION_REQUESTED = "agent.delegation.requested"
    AGENT_DELEGATION_COMPLETED = "agent.delegation.completed"
    AGENT_DELEGATION_FAILED = "agent.delegation.failed"

    # Discussion rounds
    DISCUSSION_STARTED = "discussion.started"
    DISCUSSION_ROUND_COMPLETED = "discussion.round.completed"
    DISCUSSION_CONSENSUS_REACHED = "discussion.consensus.reached"
    DISCUSSION_COMPLETED = "discussion.completed"

    # Controller agent
    CONTROLLER_TASK_STARTED = "controller.task.started"
    CONTROLLER_TASK_PROGRESS = "controller.task.progress"
    CONTROLLER_TASK_COMPLETED = "controller.task.completed"
    CONTROLLER_TASK_FAILED = "controller.task.failed"
    CONTROLLER_TASK_CANCELLED = "controller.task.cancelled"

    # Proactive autonomy
    PROACTIVE_MESSAGE = "proactive.message"
    SCHEDULED_TASK_DUE = "scheduled.task.due"
    MONITORING_ALERT = "monitoring.alert"


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
        handlers = list(self._handlers.get(event, []))
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
