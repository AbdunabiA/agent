"""Integration tests for session lifecycle and persistence.

Tests session creation, message persistence, eviction, and reload
using a real SQLite database (via tmp_path).
"""

from __future__ import annotations

from agent.core.session import (
    _MAX_CACHED_SESSIONS,
    Message,
    SessionStore,
    TokenUsage,
    ToolCall,
)
from agent.memory.database import Database


class TestSessionLifecycle:
    """End-to-end tests for session persistence."""

    async def test_create_session_and_add_messages(
        self,
        session_manager: SessionStore,
    ) -> None:
        """Create a session and add messages to it."""
        session = await session_manager.new_session(channel="api")

        msg1 = Message(role="user", content="Hello!")
        msg2 = Message(role="assistant", content="Hi there!")

        session.add_message(msg1)
        await session_manager.save_message(session.id, msg1)

        session.add_message(msg2)
        await session_manager.save_message(session.id, msg2)

        history = await session_manager.load_history(session.id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "Hello!"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there!"

    async def test_session_persists_across_reconnect(
        self,
        tmp_path: object,
    ) -> None:
        """Session and messages survive database close and reopen."""
        db_path = str(tmp_path / "lifecycle_test.db")

        # Phase 1: create session and add messages
        db1 = Database(db_path)
        await db1.connect()
        store1 = SessionStore(db=db1)
        session = await store1.new_session(channel="telegram")
        session_id = session.id

        for i in range(5):
            msg = Message(role="user", content=f"Message {i}")
            session.add_message(msg)
            await store1.save_message(session_id, msg)
        await db1.close()

        # Phase 2: reopen and verify
        db2 = Database(db_path)
        await db2.connect()
        store2 = SessionStore(db=db2)
        loaded = await store2.get(session_id)

        assert loaded is not None
        assert loaded.id == session_id
        assert loaded.metadata["channel"] == "telegram"
        assert len(loaded.messages) == 5
        for i in range(5):
            assert loaded.messages[i].content == f"Message {i}"
        await db2.close()

    async def test_eviction_triggers_persistence(
        self,
        test_database: Database,
    ) -> None:
        """When cache exceeds limit, evicted sessions are persisted to SQLite."""
        store = SessionStore(db=test_database)

        # Create sessions up to the limit
        session_ids = []
        for _i in range(_MAX_CACHED_SESSIONS + 5):
            session = await store.new_session(channel="api")
            session_ids.append(session.id)

        # Cache should be at or under the limit
        assert store.active_count <= _MAX_CACHED_SESSIONS

        # The earliest sessions should have been evicted from cache
        # but should still be retrievable from SQLite
        evicted_id = session_ids[0]
        assert evicted_id not in store._sessions  # not in cache

        reloaded = await store.get(evicted_id)
        assert reloaded is not None
        assert reloaded.id == evicted_id

    async def test_evicted_session_messages_intact(
        self,
        test_database: Database,
    ) -> None:
        """Messages added before eviction should survive after reload."""
        store = SessionStore(db=test_database)

        # Create a session and add messages
        target = await store.new_session(channel="api")
        target_id = target.id

        msg = Message(role="user", content="Important message before eviction")
        target.add_message(msg)
        await store.save_message(target_id, msg)

        # Force eviction by creating many more sessions
        for _ in range(_MAX_CACHED_SESSIONS + 10):
            await store.new_session(channel="api")

        # Target should have been evicted
        assert target_id not in store._sessions

        # Reload and verify messages
        reloaded = await store.get(target_id)
        assert reloaded is not None
        assert len(reloaded.messages) == 1
        assert reloaded.messages[0].content == "Important message before eviction"

    async def test_messages_with_tool_calls_persist(
        self,
        session_manager: SessionStore,
    ) -> None:
        """Messages with tool calls should round-trip through SQLite."""
        session = await session_manager.new_session(channel="api")

        msg = Message(
            role="assistant",
            content="Running command...",
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="shell_exec",
                    arguments={"command": "ls -la"},
                ),
            ],
        )
        session.add_message(msg)
        await session_manager.save_message(session.id, msg)

        history = await session_manager.load_history(session.id)
        assert len(history) == 1
        assert history[0].tool_calls is not None
        assert len(history[0].tool_calls) == 1
        assert history[0].tool_calls[0].name == "shell_exec"
        assert history[0].tool_calls[0].arguments == {"command": "ls -la"}

    async def test_messages_with_token_usage_persist(
        self,
        session_manager: SessionStore,
    ) -> None:
        """Messages with token usage metadata should round-trip through SQLite."""
        session = await session_manager.new_session(channel="api")

        msg = Message(
            role="assistant",
            content="Hello!",
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        )
        session.add_message(msg)
        await session_manager.save_message(session.id, msg)

        history = await session_manager.load_history(session.id)
        assert len(history) == 1
        assert history[0].usage is not None
        assert history[0].usage.input_tokens == 100
        assert history[0].usage.output_tokens == 50
        assert history[0].usage.total_tokens == 150

    async def test_multiple_sessions_different_channels(
        self,
        session_manager: SessionStore,
    ) -> None:
        """Sessions across different channels should be independently manageable."""
        api_session = await session_manager.new_session(channel="api")
        tg_session = await session_manager.new_session(channel="telegram")
        await session_manager.new_session(channel="webchat")

        api_sessions = await session_manager.list_sessions(channel="api")
        tg_sessions = await session_manager.list_sessions(channel="telegram")

        assert len(api_sessions) == 1
        assert api_sessions[0].id == api_session.id
        assert len(tg_sessions) == 1
        assert tg_sessions[0].id == tg_session.id

        all_sessions = await session_manager.list_sessions()
        assert len(all_sessions) == 3

    async def test_remove_session_deletes_messages(
        self,
        session_manager: SessionStore,
        test_database: Database,
    ) -> None:
        """Removing a session should also delete its messages from SQLite."""
        session = await session_manager.new_session(channel="api")
        for i in range(3):
            msg = Message(role="user", content=f"msg {i}")
            session.add_message(msg)
            await session_manager.save_message(session.id, msg)

        removed = await session_manager.remove(session.id)
        assert removed is True

        # Verify session is gone
        assert await session_manager.get(session.id) is None

        # Verify messages are gone from database
        async with test_database.db.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (session.id,),
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == 0

    async def test_get_or_create_returns_existing(
        self,
        session_manager: SessionStore,
    ) -> None:
        """get_or_create with a known ID should return the existing session."""
        original = await session_manager.new_session(channel="api")
        found = await session_manager.get_or_create(session_id=original.id)

        assert found.id == original.id

    async def test_session_message_count_tracking(
        self,
        session_manager: SessionStore,
        test_database: Database,
    ) -> None:
        """The conversations table should track message_count accurately."""
        session = await session_manager.new_session(channel="api")

        for i in range(4):
            msg = Message(role="user", content=f"msg {i}")
            await session_manager.save_message(session.id, msg)

        async with test_database.db.execute(
            "SELECT message_count FROM conversations WHERE id = ?",
            (session.id,),
        ) as cursor:
            row = await cursor.fetchone()
            assert row[0] == 4
