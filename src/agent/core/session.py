"""Conversation session management.

Phase 1-3: In-memory only.
Phase 4: SQLite persistence with in-memory cache fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

if TYPE_CHECKING:
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass
class TokenUsage:
    """Token usage statistics for an LLM call."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    @property
    def estimated_cost(self) -> float:
        """Rough cost estimate based on model pricing.

        Uses approximate per-token pricing. Actual costs vary by model.
        """
        input_cost = self.input_tokens * 0.000003  # ~$3/M input tokens
        output_cost = self.output_tokens * 0.000015  # ~$15/M output tokens
        return input_cost + output_cost


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str | list[dict[str, Any]]
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    model: str | None = None
    usage: TokenUsage | None = None
    timestamp: datetime = field(default_factory=datetime.now)


def content_as_text(content: str | list[dict[str, Any]]) -> str:
    """Extract plain text from content (handles both str and multimodal list).

    Args:
        content: Message content — either a plain string or a list of content blocks.

    Returns:
        Concatenated text from all text blocks.
    """
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )


class Session:
    """Manages a single conversation session.

    In-memory message list with optional SQLite persistence.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.id = session_id or str(uuid4())
        self.messages: list[Message] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata: dict[str, object] = {}

    def add_message(self, message: Message) -> None:
        """Add a message to the session.

        Args:
            message: The message to add.
        """
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """Get message history formatted for LLM API calls.

        Returns list of dicts suitable for the LLM API, including
        tool_calls and tool_call_id where present.

        Args:
            max_messages: Maximum number of messages to return.

        Returns:
            List of message dicts suitable for LLM API.
        """
        recent = self.messages[-max_messages:]
        result: list[dict[str, Any]] = []

        for msg in recent:
            entry: dict[str, Any] = {"role": msg.role, "content": msg.content}

            # Include tool_calls for assistant messages
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else str(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Include tool_call_id for tool result messages
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id

            result.append(entry)

        return result

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.updated_at = datetime.now()

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this session."""
        total = 0
        for msg in self.messages:
            if msg.usage:
                total += msg.usage.total_tokens
        return total

    @property
    def message_count(self) -> int:
        """Number of messages in this session."""
        return len(self.messages)


_MAX_CACHED_SESSIONS = 200


class SessionStore:
    """Session store with optional SQLite persistence.

    Maintains an in-memory cache of sessions. When a Database is provided,
    sessions and messages are persisted to SQLite and can survive restarts.
    Falls back to pure in-memory when no database is available.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._sessions: dict[str, Session] = {}

    def _evict_oldest(self) -> None:
        """Evict oldest sessions from cache if over the limit."""
        if len(self._sessions) <= _MAX_CACHED_SESSIONS:
            return
        sorted_sessions = sorted(
            self._sessions.items(), key=lambda kv: kv[1].updated_at,
        )
        to_remove = len(self._sessions) - _MAX_CACHED_SESSIONS
        for sid, _ in sorted_sessions[:to_remove]:
            del self._sessions[sid]

    async def get_or_create(
        self,
        session_id: str | None = None,
        channel: str = "api",
    ) -> Session:
        """Get an existing session or create a new one.

        Checks in-memory cache first, then SQLite, then creates new.

        Args:
            session_id: Session ID to look up. Creates new if not found.
            channel: Channel identifier (e.g., 'api', 'telegram', 'webchat').

        Returns:
            The existing or newly created Session.
        """
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        # Check SQLite for existing session
        if session_id and self._db:
            session = await self._load_session_from_db(session_id)
            if session:
                self._sessions[session.id] = session
                return session

        session = Session(session_id=session_id)
        session.metadata["channel"] = channel
        self._sessions[session.id] = session
        self._evict_oldest()

        if self._db:
            await self._persist_session(session, channel)

        return session

    async def get(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            The Session if found, None otherwise.
        """
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Check SQLite
        if self._db:
            session = await self._load_session_from_db(session_id)
            if session:
                self._sessions[session.id] = session
                return session

        return None

    async def new_session(self, channel: str = "api") -> Session:
        """Create a new session.

        Args:
            channel: Channel identifier.

        Returns:
            The newly created Session.
        """
        session = Session()
        session.metadata["channel"] = channel
        self._sessions[session.id] = session
        self._evict_oldest()

        if self._db:
            await self._persist_session(session, channel)

        return session

    async def save_message(self, session_id: str, message: Message) -> None:
        """Persist a message to SQLite.

        Args:
            session_id: The session this message belongs to.
            message: The message to persist.
        """
        if not self._db:
            return

        msg_id = str(uuid4())
        tool_calls_json = None
        if message.tool_calls:
            tool_calls_json = json.dumps([
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in message.tool_calls
            ])

        token_usage_json = None
        if message.usage:
            token_usage_json = json.dumps({
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.total_tokens,
            })

        content_value = (
            json.dumps(message.content)
            if isinstance(message.content, list)
            else message.content
        )

        await self._db.db.execute(
            """INSERT INTO messages
               (id, conversation_id, role, content, tool_calls, tool_call_id,
                model, token_usage, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id,
                session_id,
                message.role,
                content_value,
                tool_calls_json,
                message.tool_call_id,
                message.model,
                token_usage_json,
                message.timestamp.isoformat(),
            ),
        )
        await self._db.db.execute(
            """UPDATE conversations
               SET message_count = message_count + 1, updated_at = ?
               WHERE id = ?""",
            (datetime.now().isoformat(), session_id),
        )
        await self._db.db.commit()

    async def load_history(self, session_id: str, limit: int = 50) -> list[Message]:
        """Load message history from SQLite.

        Args:
            session_id: The session to load messages for.
            limit: Maximum messages to load.

        Returns:
            List of Messages ordered by creation time.
        """
        if not self._db:
            return []

        async with self._db.db.execute(
            """SELECT * FROM messages
               WHERE conversation_id = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        return [self._row_to_message(row) for row in rows]

    async def list_sessions(
        self,
        channel: str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        """List sessions, optionally filtered by channel.

        Args:
            channel: Filter by channel name.
            limit: Maximum number of sessions to return.

        Returns:
            List of sessions sorted by last update time (newest first).
        """
        if self._db:
            return await self._list_sessions_from_db(channel, limit)

        sessions = list(self._sessions.values())
        if channel:
            sessions = [
                s for s in sessions if s.metadata.get("channel") == channel
            ]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]

    @property
    def active_count(self) -> int:
        """Number of active sessions in cache."""
        return len(self._sessions)

    async def remove(self, session_id: str) -> bool:
        """Remove a session by ID.

        Args:
            session_id: The session ID to remove.

        Returns:
            True if the session was removed, False if not found.
        """
        removed = session_id in self._sessions
        if removed:
            del self._sessions[session_id]

        if self._db:
            await self._db.db.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (session_id,)
            )
            cursor = await self._db.db.execute(
                "DELETE FROM conversations WHERE id = ?", (session_id,)
            )
            await self._db.db.commit()
            if not removed and cursor.rowcount > 0:
                removed = True

        return removed

    # --- Private helpers ---

    async def _persist_session(self, session: Session, channel: str) -> None:
        """Save a session to SQLite."""
        now = datetime.now().isoformat()
        await self._db.db.execute(
            """INSERT OR IGNORE INTO conversations
               (id, channel, channel_user_id, title, message_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, ?, ?)""",
            (session.id, channel, "", None, now, now),
        )
        await self._db.db.commit()

    async def _load_session_from_db(self, session_id: str) -> Session | None:
        """Load a session and its messages from SQLite."""
        async with self._db.db.execute(
            "SELECT * FROM conversations WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        session = Session(session_id=row["id"])
        session.created_at = datetime.fromisoformat(row["created_at"])
        session.updated_at = datetime.fromisoformat(row["updated_at"])
        session.metadata["channel"] = row["channel"]

        # Load messages
        messages = await self.load_history(session_id)
        session.messages = messages

        return session

    async def _list_sessions_from_db(
        self, channel: str | None, limit: int
    ) -> list[Session]:
        """List sessions from SQLite."""
        if channel:
            query = """SELECT * FROM conversations
                       WHERE channel = ?
                       ORDER BY updated_at DESC LIMIT ?"""
            params: tuple = (channel, limit)
        else:
            query = "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?"
            params = (limit,)

        async with self._db.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        sessions: list[Session] = []
        for row in rows:
            # Use cached session if available
            if row["id"] in self._sessions:
                sessions.append(self._sessions[row["id"]])
            else:
                session = Session(session_id=row["id"])
                session.created_at = datetime.fromisoformat(row["created_at"])
                session.updated_at = datetime.fromisoformat(row["updated_at"])
                session.metadata["channel"] = row["channel"]
                sessions.append(session)

        return sessions

    @staticmethod
    def _row_to_message(row: Any) -> Message:
        """Convert a database row to a Message."""
        tool_calls = None
        if row["tool_calls"]:
            raw = json.loads(row["tool_calls"])
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in raw
            ]

        usage = None
        if row["token_usage"]:
            raw_usage = json.loads(row["token_usage"])
            usage = TokenUsage(
                input_tokens=raw_usage["input_tokens"],
                output_tokens=raw_usage["output_tokens"],
                total_tokens=raw_usage["total_tokens"],
            )

        raw_content = row["content"]
        content: str | list[dict[str, Any]] = raw_content
        if raw_content and raw_content.startswith("["):
            try:
                parsed = json.loads(raw_content)
                if (
                    isinstance(parsed, list)
                    and parsed
                    and isinstance(parsed[0], dict)
                    and "type" in parsed[0]
                ):
                    content = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        return Message(
            role=row["role"],
            content=content,
            tool_calls=tool_calls,
            tool_call_id=row["tool_call_id"],
            model=row["model"],
            usage=usage,
            timestamp=datetime.fromisoformat(row["created_at"]),
        )
