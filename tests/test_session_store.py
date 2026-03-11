"""Tests for SessionStore."""

from __future__ import annotations

from agent.core.session import SessionStore


class TestSessionStore:
    """Tests for the SessionStore class (in-memory fallback mode)."""

    async def test_new_session(self) -> None:
        """new_session should create a session and track it."""
        store = SessionStore()
        session = await store.new_session(channel="telegram")

        assert session.id
        assert session.metadata["channel"] == "telegram"
        assert store.active_count == 1

    async def test_get_existing(self) -> None:
        """get should return an existing session by ID."""
        store = SessionStore()
        session = await store.new_session()

        found = await store.get(session.id)
        assert found is session

    async def test_get_missing(self) -> None:
        """get should return None for unknown session ID."""
        store = SessionStore()
        assert await store.get("nonexistent-id") is None

    async def test_get_or_create_existing(self) -> None:
        """get_or_create with known ID should return existing session."""
        store = SessionStore()
        original = await store.new_session(channel="api")

        found = await store.get_or_create(session_id=original.id)
        assert found is original

    async def test_get_or_create_new(self) -> None:
        """get_or_create with unknown ID should create new session."""
        store = SessionStore()
        session = await store.get_or_create(session_id="new-id", channel="webchat")

        assert session.id == "new-id"
        assert session.metadata["channel"] == "webchat"
        assert store.active_count == 1

    async def test_get_or_create_no_id(self) -> None:
        """get_or_create with no ID should create new session."""
        store = SessionStore()
        session = await store.get_or_create()

        assert session.id
        assert store.active_count == 1

    async def test_list_sessions_all(self) -> None:
        """list_sessions should return all sessions sorted by updated_at."""
        store = SessionStore()
        s1 = await store.new_session(channel="api")
        s2 = await store.new_session(channel="telegram")
        s3 = await store.new_session(channel="api")

        sessions = await store.list_sessions()
        assert len(sessions) == 3
        # All sessions should be returned
        session_ids = {s.id for s in sessions}
        assert {s1.id, s2.id, s3.id} == session_ids

    async def test_list_sessions_filter_channel(self) -> None:
        """list_sessions with channel filter should only return matching."""
        store = SessionStore()
        await store.new_session(channel="api")
        await store.new_session(channel="telegram")
        await store.new_session(channel="api")

        api_sessions = await store.list_sessions(channel="api")
        assert len(api_sessions) == 2
        for s in api_sessions:
            assert s.metadata["channel"] == "api"

    async def test_list_sessions_limit(self) -> None:
        """list_sessions should respect the limit parameter."""
        store = SessionStore()
        for _ in range(10):
            await store.new_session()

        sessions = await store.list_sessions(limit=3)
        assert len(sessions) == 3

    async def test_remove_session(self) -> None:
        """remove should delete a session and return True."""
        store = SessionStore()
        session = await store.new_session()
        session_id = session.id

        assert await store.remove(session_id) is True
        assert await store.get(session_id) is None
        assert store.active_count == 0

    async def test_remove_nonexistent(self) -> None:
        """remove should return False for unknown session ID."""
        store = SessionStore()
        assert await store.remove("nonexistent") is False

    async def test_active_count(self) -> None:
        """active_count should reflect the number of tracked sessions."""
        store = SessionStore()
        assert store.active_count == 0

        await store.new_session()
        await store.new_session()
        assert store.active_count == 2
