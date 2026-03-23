"""Inter-agent message bus for real-time communication."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

if TYPE_CHECKING:
    from agent.core.events import EventBus
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)


@dataclass
class AgentMessage:
    """A message sent between agents."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    task_id: str = ""
    from_role: str = ""
    to_role: str | None = None  # None = broadcast
    to_team: str | None = None
    thread_id: str = ""
    content: str = ""
    msg_type: str = "question"  # question, answer, status, alert
    reply_to: str | None = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = self.id


class MessageBus:
    """Inter-agent message bus for real-time communication.

    Stores messages in memory (and optionally persists to SQLite).
    Supports point-to-point messages, broadcasts, threading, and
    read-status tracking.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        database: Database | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._db = database

        # task_id -> list of messages
        self._messages: dict[str, list[AgentMessage]] = {}
        # Set of (message_id, role) tuples marking reads
        self._read_status: set[tuple[str, str]] = set()
        # role -> list of async callbacks
        self._subscribers: dict[str, list[Callable[..., Any]]] = {}
        # Track which task_ids have been loaded from DB
        self._loaded_tasks: set[str] = set()

    async def send(self, msg: AgentMessage) -> str:
        """Send a message, store it, notify subscribers, and persist.

        Args:
            msg: The message to send.

        Returns:
            The message ID.
        """
        # Store in memory
        if msg.task_id not in self._messages:
            self._messages[msg.task_id] = []
        self._messages[msg.task_id].append(msg)

        logger.info(
            "message_sent",
            message_id=msg.id,
            from_role=msg.from_role,
            to_role=msg.to_role,
            task_id=msg.task_id,
            msg_type=msg.msg_type,
        )

        # Notify subscribers
        target_role = msg.to_role
        if target_role and target_role in self._subscribers:
            for callback in self._subscribers[target_role]:
                try:
                    await callback(msg)
                except Exception as e:
                    logger.error(
                        "subscriber_callback_error",
                        role=target_role,
                        error=str(e),
                    )

        # Emit event
        if self._event_bus is not None:
            from agent.core.events import Events

            await self._event_bus.emit(
                Events.MESSAGE_SENT,
                {
                    "message_id": msg.id,
                    "from_role": msg.from_role,
                    "to_role": msg.to_role,
                    "task_id": msg.task_id,
                    "msg_type": msg.msg_type,
                },
            )

        # Persist to database
        await self._persist(msg)

        return msg.id

    async def get_messages(
        self,
        task_id: str,
        role: str,
        unread_only: bool = False,
    ) -> list[AgentMessage]:
        """Get messages for a role within a task.

        Returns messages where to_role matches the given role, or
        to_role is None (broadcasts). Optionally filters to unread only.
        Marks returned messages as read.

        Args:
            task_id: The task context.
            role: The recipient role.
            unread_only: If True, only return unread messages.

        Returns:
            List of matching messages.
        """
        # Load from DB on first access for this task
        await self._load_messages(task_id)

        all_msgs = self._messages.get(task_id, [])

        # Filter: to_role matches or broadcast (to_role is None)
        matching = [m for m in all_msgs if m.to_role == role or m.to_role is None]

        if unread_only:
            matching = [m for m in matching if (m.id, role) not in self._read_status]

        # Mark as read
        for m in matching:
            self._read_status.add((m.id, role))

        # Emit read events
        if self._event_bus is not None and matching:
            from agent.core.events import Events

            for m in matching:
                await self._event_bus.emit(
                    Events.MESSAGE_READ,
                    {
                        "message_id": m.id,
                        "reader_role": role,
                        "task_id": task_id,
                    },
                )

        return matching

    async def get_thread(self, thread_id: str) -> list[AgentMessage]:
        """Get all messages in a thread.

        Args:
            thread_id: The thread ID to look up.

        Returns:
            List of messages in the thread, sorted by timestamp.
        """
        result: list[AgentMessage] = []
        for msgs in self._messages.values():
            for m in msgs:
                if m.thread_id == thread_id:
                    result.append(m)
        result.sort(key=lambda m: m.timestamp)
        return result

    async def broadcast(self, msg: AgentMessage) -> str:
        """Broadcast a message to all roles (to_role=None).

        Args:
            msg: The message to broadcast. to_role will be set to None.

        Returns:
            The message ID.
        """
        msg.to_role = None

        msg_id = await self.send(msg)

        # Emit broadcast event
        if self._event_bus is not None:
            from agent.core.events import Events

            await self._event_bus.emit(
                Events.MESSAGE_BROADCAST,
                {
                    "message_id": msg.id,
                    "from_role": msg.from_role,
                    "task_id": msg.task_id,
                    "msg_type": msg.msg_type,
                },
            )

        # Notify all subscribers (not just the target role)
        for role, callbacks in self._subscribers.items():
            if role != msg.from_role:
                for callback in callbacks:
                    try:
                        await callback(msg)
                    except Exception as e:
                        logger.error(
                            "broadcast_subscriber_error",
                            role=role,
                            error=str(e),
                        )

        return msg_id

    def subscribe(self, role: str, callback: Callable[..., Any]) -> None:
        """Register an async callback for messages to a role.

        Args:
            role: The role to subscribe for.
            callback: Async callable invoked with (AgentMessage).
        """
        if role not in self._subscribers:
            self._subscribers[role] = []
        self._subscribers[role].append(callback)
        logger.info("message_bus_subscribe", role=role)

    async def _persist(self, msg: AgentMessage) -> None:
        """Write a message to the agent_messages table.

        Args:
            msg: The message to persist.
        """
        if self._db is None:
            return

        try:
            await self._db.execute(
                """
                INSERT INTO agent_messages
                    (id, task_id, from_role, to_role, to_team, thread_id,
                     content, msg_type, reply_to, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.id,
                    msg.task_id,
                    msg.from_role,
                    msg.to_role,
                    msg.to_team,
                    msg.thread_id,
                    msg.content,
                    msg.msg_type,
                    msg.reply_to,
                    msg.timestamp,
                ),
            )
        except Exception as e:
            logger.error("message_persist_error", message_id=msg.id, error=str(e))

    async def _load_messages(self, task_id: str) -> None:
        """Load messages from DB on first access for a task_id.

        Args:
            task_id: The task to load messages for.
        """
        if task_id in self._loaded_tasks:
            return
        self._loaded_tasks.add(task_id)

        if self._db is None:
            return

        try:
            rows = await self._db.fetch_all(
                """
                SELECT id, task_id, from_role, to_role, to_team, thread_id,
                       content, msg_type, reply_to, timestamp
                FROM agent_messages
                WHERE task_id = ?
                ORDER BY timestamp ASC
                """,
                (task_id,),
            )

            if not rows:
                return

            existing_ids = {m.id for m in self._messages.get(task_id, [])}

            if task_id not in self._messages:
                self._messages[task_id] = []

            for row in rows:
                if row[0] not in existing_ids:
                    self._messages[task_id].append(
                        AgentMessage(
                            id=row[0],
                            task_id=row[1],
                            from_role=row[2],
                            to_role=row[3],
                            to_team=row[4],
                            thread_id=row[5],
                            content=row[6],
                            msg_type=row[7],
                            reply_to=row[8],
                            timestamp=row[9],
                        )
                    )
        except Exception as e:
            logger.error("message_load_error", task_id=task_id, error=str(e))
