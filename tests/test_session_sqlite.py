"""Tests for SQLite-backed session persistence."""

from __future__ import annotations

import pytest

from agent.core.session import Message, SessionStore, TokenUsage, ToolCall
from agent.memory.database import Database


@pytest.fixture
async def db(tmp_path: object) -> Database:
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "session_test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def store(db: Database) -> SessionStore:
    """SessionStore with SQLite backend."""
    return SessionStore(db=db)


class TestSessionSQLite:
    """Tests for SQLite-backed SessionStore."""

    async def test_session_created_and_persisted(self, store: SessionStore) -> None:
        """Creating a session should persist it to SQLite."""
        session = await store.new_session(channel="telegram")

        assert session.id
        assert session.metadata["channel"] == "telegram"
        assert store.active_count == 1

    async def test_session_survives_reconnect(self, tmp_path: object) -> None:
        """Session should survive closing and reopening the database."""
        db_path = str(tmp_path / "persist_test.db")

        # Create session
        db1 = Database(db_path)
        await db1.connect()
        store1 = SessionStore(db=db1)
        session = await store1.new_session(channel="api")
        session_id = session.id
        await db1.close()

        # Reopen and verify
        db2 = Database(db_path)
        await db2.connect()
        store2 = SessionStore(db=db2)
        found = await store2.get(session_id)

        assert found is not None
        assert found.id == session_id
        assert found.metadata["channel"] == "api"
        await db2.close()

    async def test_messages_saved_to_sqlite(
        self, store: SessionStore, db: Database
    ) -> None:
        """Messages should be persisted to SQLite."""
        session = await store.new_session(channel="api")
        msg = Message(role="user", content="Hello!")

        session.add_message(msg)
        await store.save_message(session.id, msg)

        # Verify in database
        async with db.db.execute(
            "SELECT * FROM messages WHERE conversation_id = ?", (session.id,)
        ) as cursor:
            rows = await cursor.fetchall()

        assert len(rows) == 1
        assert rows[0]["role"] == "user"
        assert rows[0]["content"] == "Hello!"

    async def test_messages_with_tool_calls(self, store: SessionStore) -> None:
        """Messages with tool_calls should round-trip correctly."""
        session = await store.new_session(channel="api")
        msg = Message(
            role="assistant",
            content="Let me check that.",
            tool_calls=[
                ToolCall(id="tc_1", name="shell_exec", arguments={"command": "ls"})
            ],
        )
        session.add_message(msg)
        await store.save_message(session.id, msg)

        history = await store.load_history(session.id)
        assert len(history) == 1
        assert history[0].tool_calls is not None
        assert len(history[0].tool_calls) == 1
        assert history[0].tool_calls[0].name == "shell_exec"
        assert history[0].tool_calls[0].arguments == {"command": "ls"}

    async def test_messages_with_token_usage(self, store: SessionStore) -> None:
        """Messages with token usage should round-trip correctly."""
        session = await store.new_session(channel="api")
        msg = Message(
            role="assistant",
            content="Hello!",
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        session.add_message(msg)
        await store.save_message(session.id, msg)

        history = await store.load_history(session.id)
        assert len(history) == 1
        assert history[0].usage is not None
        assert history[0].usage.input_tokens == 10
        assert history[0].usage.output_tokens == 5
        assert history[0].usage.total_tokens == 15

    async def test_history_loaded_from_sqlite(self, tmp_path: object) -> None:
        """History should be loaded from SQLite on session retrieval."""
        db_path = str(tmp_path / "history_test.db")

        # Create session with messages
        db1 = Database(db_path)
        await db1.connect()
        store1 = SessionStore(db=db1)
        session = await store1.new_session(channel="api")
        session_id = session.id

        for i in range(3):
            msg = Message(role="user", content=f"Message {i}")
            session.add_message(msg)
            await store1.save_message(session_id, msg)
        await db1.close()

        # Reopen and load history
        db2 = Database(db_path)
        await db2.connect()
        store2 = SessionStore(db=db2)
        loaded = await store2.get(session_id)

        assert loaded is not None
        assert len(loaded.messages) == 3
        assert loaded.messages[0].content == "Message 0"
        assert loaded.messages[2].content == "Message 2"
        await db2.close()

    async def test_list_sessions_from_db(self, store: SessionStore) -> None:
        """list_sessions should return sessions from SQLite."""
        await store.new_session(channel="api")
        await store.new_session(channel="telegram")
        await store.new_session(channel="api")

        all_sessions = await store.list_sessions()
        assert len(all_sessions) == 3

        api_sessions = await store.list_sessions(channel="api")
        assert len(api_sessions) == 2

    async def test_list_sessions_limit(self, store: SessionStore) -> None:
        """list_sessions should respect limit."""
        for _ in range(10):
            await store.new_session()

        sessions = await store.list_sessions(limit=3)
        assert len(sessions) == 3

    async def test_get_or_create_existing(self, store: SessionStore) -> None:
        """get_or_create with known ID should return existing session."""
        original = await store.new_session(channel="api")
        found = await store.get_or_create(session_id=original.id)

        assert found.id == original.id

    async def test_get_or_create_new(self, store: SessionStore) -> None:
        """get_or_create with unknown ID should create new session."""
        session = await store.get_or_create(session_id="new-id", channel="webchat")

        assert session.id == "new-id"
        assert session.metadata["channel"] == "webchat"

    async def test_remove_session(self, store: SessionStore) -> None:
        """remove should delete session and its messages."""
        session = await store.new_session(channel="api")
        msg = Message(role="user", content="test")
        session.add_message(msg)
        await store.save_message(session.id, msg)

        removed = await store.remove(session.id)
        assert removed is True
        assert await store.get(session.id) is None

    async def test_multimodal_content_round_trip(self, store: SessionStore) -> None:
        """Multimodal (list) content should round-trip through SQLite."""
        session = await store.new_session(channel="api")
        multimodal_content = [
            {"type": "text", "text": "Screenshot: 1920x1080"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        msg = Message(role="tool", content=multimodal_content, tool_call_id="tc_mm")
        session.add_message(msg)
        await store.save_message(session.id, msg)

        history = await store.load_history(session.id)
        assert len(history) == 1
        loaded = history[0]
        assert isinstance(loaded.content, list)
        assert len(loaded.content) == 2
        assert loaded.content[0]["type"] == "text"
        assert loaded.content[0]["text"] == "Screenshot: 1920x1080"
        assert loaded.content[1]["type"] == "image_url"

    async def test_plain_string_content_still_works(self, store: SessionStore) -> None:
        """Plain string content should still load as string (not parsed as JSON)."""
        session = await store.new_session(channel="api")
        msg = Message(role="user", content="Hello world!")
        session.add_message(msg)
        await store.save_message(session.id, msg)

        history = await store.load_history(session.id)
        assert len(history) == 1
        assert isinstance(history[0].content, str)
        assert history[0].content == "Hello world!"

    async def test_json_like_string_not_parsed_as_multimodal(
        self, store: SessionStore,
    ) -> None:
        """A string that looks like JSON array should stay as string."""
        session = await store.new_session(channel="api")
        # User sends text that happens to be valid JSON array
        msg = Message(role="user", content='["hello", "world"]')
        session.add_message(msg)
        await store.save_message(session.id, msg)

        history = await store.load_history(session.id)
        assert len(history) == 1
        assert isinstance(history[0].content, str)
        assert history[0].content == '["hello", "world"]'

    async def test_message_count_updated(
        self, store: SessionStore, db: Database
    ) -> None:
        """message_count should be updated in conversations table."""
        session = await store.new_session(channel="api")

        for i in range(3):
            msg = Message(role="user", content=f"msg {i}")
            await store.save_message(session.id, msg)

        async with db.db.execute(
            "SELECT message_count FROM conversations WHERE id = ?",
            (session.id,),
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] == 3


class TestSessionStoreFallback:
    """Tests for in-memory fallback when no database."""

    async def test_fallback_works(self) -> None:
        """SessionStore should work without database."""
        store = SessionStore()
        session = await store.new_session(channel="api")

        assert session.id
        assert store.active_count == 1

    async def test_fallback_get_or_create(self) -> None:
        """get_or_create should work in memory-only mode."""
        store = SessionStore()
        s1 = await store.get_or_create(channel="api")
        s2 = await store.get_or_create(session_id=s1.id)

        assert s1.id == s2.id

    async def test_fallback_list_sessions(self) -> None:
        """list_sessions should work in memory-only mode."""
        store = SessionStore()
        await store.new_session(channel="api")
        await store.new_session(channel="telegram")

        sessions = await store.list_sessions()
        assert len(sessions) == 2

    async def test_fallback_remove(self) -> None:
        """remove should work in memory-only mode."""
        store = SessionStore()
        session = await store.new_session()

        removed = await store.remove(session.id)
        assert removed is True
        assert store.active_count == 0
