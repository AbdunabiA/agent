"""Tests for the SQLite facts store."""

from __future__ import annotations

import pytest

from agent.memory.database import Database
from agent.memory.store import FactStore


@pytest.fixture
async def db(tmp_path: object) -> Database:
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "facts_test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def store(db: Database) -> FactStore:
    """Create a FactStore with test database."""
    return FactStore(db)


class TestFactStore:
    """Tests for FactStore."""

    async def test_set_creates_new_fact(self, store: FactStore) -> None:
        """set() should create a new fact."""
        fact = await store.set("user.name", "Abduvohid", category="user")

        assert fact.key == "user.name"
        assert fact.value == "Abduvohid"
        assert fact.category == "user"
        assert fact.confidence == 1.0
        assert fact.source == "user"
        assert fact.id  # Should have a UUID

    async def test_set_updates_existing_fact(self, store: FactStore) -> None:
        """set() with same key should update the existing fact."""
        await store.set("user.name", "Old Name")
        fact = await store.set("user.name", "New Name")

        assert fact.value == "New Name"
        assert await store.count() == 1

    async def test_set_updates_category_and_confidence(self, store: FactStore) -> None:
        """set() should update category and confidence on upsert."""
        await store.set("test.key", "val1", category="general", confidence=0.5)
        fact = await store.set("test.key", "val2", category="user", confidence=0.9)

        assert fact.category == "user"
        assert fact.confidence == 0.9

    async def test_get_returns_fact(self, store: FactStore) -> None:
        """get() should return a fact by exact key."""
        await store.set("user.name", "Abduvohid")
        fact = await store.get("user.name")

        assert fact is not None
        assert fact.key == "user.name"
        assert fact.value == "Abduvohid"

    async def test_get_returns_none_for_unknown_key(self, store: FactStore) -> None:
        """get() should return None for an unknown key."""
        fact = await store.get("nonexistent.key")
        assert fact is None

    async def test_get_updates_access_tracking(self, store: FactStore) -> None:
        """get() should update accessed_at and increment access_count."""
        await store.set("user.name", "Abduvohid")

        fact1 = await store.get("user.name")
        assert fact1.access_count == 1

        fact2 = await store.get("user.name")
        assert fact2.access_count == 2
        assert fact2.accessed_at >= fact1.accessed_at

    async def test_delete_removes_fact(self, store: FactStore) -> None:
        """delete() should remove a fact."""
        await store.set("user.name", "Abduvohid")
        deleted = await store.delete("user.name")

        assert deleted is True
        assert await store.get("user.name") is None
        assert await store.count() == 0

    async def test_delete_returns_false_for_unknown(self, store: FactStore) -> None:
        """delete() should return False for unknown key."""
        deleted = await store.delete("nonexistent")
        assert deleted is False

    async def test_search_by_prefix(self, store: FactStore) -> None:
        """search() should return all facts matching a key prefix."""
        await store.set("user.name", "Abduvohid")
        await store.set("user.email", "test@example.com")
        await store.set("preference.theme", "dark")

        results = await store.search("user")
        assert len(results) == 2
        keys = {f.key for f in results}
        assert keys == {"user.name", "user.email"}

    async def test_search_empty_results(self, store: FactStore) -> None:
        """search() should return empty list for no matches."""
        results = await store.search("nonexistent")
        assert results == []

    async def test_get_by_category(self, store: FactStore) -> None:
        """get_by_category() should return facts in a category."""
        await store.set("user.name", "Abduvohid", category="user")
        await store.set("user.email", "test@example.com", category="user")
        await store.set("pref.theme", "dark", category="preference")

        results = await store.get_by_category("user")
        assert len(results) == 2
        for fact in results:
            assert fact.category == "user"

    async def test_get_by_category_empty(self, store: FactStore) -> None:
        """get_by_category() should return empty list for unknown category."""
        results = await store.get_by_category("nonexistent")
        assert results == []

    async def test_get_relevant_returns_ranked(self, store: FactStore) -> None:
        """get_relevant() should return facts ranked by relevance."""
        await store.set("low.conf", "value", confidence=0.1)
        await store.set("high.conf", "value", confidence=1.0)

        # Access the high confidence one more to boost relevance
        await store.get("high.conf")
        await store.get("high.conf")

        results = await store.get_relevant(limit=10)
        assert len(results) == 2
        # High confidence + more accesses should rank first
        assert results[0].key == "high.conf"

    async def test_get_relevant_respects_limit(self, store: FactStore) -> None:
        """get_relevant() should respect the limit parameter."""
        for i in range(10):
            await store.set(f"fact.{i}", f"value_{i}")

        results = await store.get_relevant(limit=3)
        assert len(results) == 3

    async def test_get_all(self, store: FactStore) -> None:
        """get_all() should return all facts."""
        await store.set("a.key", "val_a")
        await store.set("b.key", "val_b")
        await store.set("c.key", "val_c")

        results = await store.get_all()
        assert len(results) == 3

    async def test_get_all_respects_limit(self, store: FactStore) -> None:
        """get_all() should respect the limit parameter."""
        for i in range(10):
            await store.set(f"fact.{i}", f"value_{i}")

        results = await store.get_all(limit=5)
        assert len(results) == 5

    async def test_count(self, store: FactStore) -> None:
        """count() should return the total number of facts."""
        assert await store.count() == 0

        await store.set("a", "1")
        assert await store.count() == 1

        await store.set("b", "2")
        assert await store.count() == 2

    async def test_different_sources(self, store: FactStore) -> None:
        """Facts should preserve source metadata."""
        f1 = await store.set("a", "1", source="user")
        f2 = await store.set("b", "2", source="extracted")
        f3 = await store.set("c", "3", source="inferred")

        assert f1.source == "user"
        assert f2.source == "extracted"
        assert f3.source == "inferred"

    async def test_get_by_priority(self, store: FactStore) -> None:
        """get_by_priority() should return facts with the given priority."""
        await store.set("task.urgent", "deploy now", priority="high", topic="deployment")
        await store.set("task.normal", "update docs", priority="normal")
        await store.set("task.also_high", "fix bug", priority="high", topic="bugs")

        high = await store.get_by_priority("high")
        assert len(high) == 2
        keys = {f.key for f in high}
        assert keys == {"task.urgent", "task.also_high"}

        normal = await store.get_by_priority("normal")
        assert len(normal) == 1
        assert normal[0].key == "task.normal"

    async def test_get_active_topics(self, store: FactStore) -> None:
        """get_active_topics() should return most accessed topic clusters."""
        await store.set("a", "1", topic="deployment")
        await store.set("b", "2", topic="deployment")
        await store.set("c", "3", topic="design")
        await store.set("d", "4", topic="")  # empty topic should be excluded

        # Access deployment facts to boost their counts
        await store.get("a")
        await store.get("b")

        topics = await store.get_active_topics(limit=5)
        assert isinstance(topics, list)
        assert len(topics) >= 1
        assert "deployment" in topics

    async def test_get_emotional_summary(self, store: FactStore) -> None:
        """get_emotional_summary() should return a summary string."""
        await store.set("proj.deploy", "Friday release", tone="urgent", emotion="concerned")
        await store.set("user.feedback", "loves the UI", tone="positive", emotion="excited")

        summary = await store.get_emotional_summary(limit=10)
        assert isinstance(summary, str)
        assert len(summary) > 0

    async def test_get_temporal_due_soon(self, store: FactStore) -> None:
        """get_temporal_due_soon() should return facts due within the time window."""
        from datetime import datetime, timedelta

        # Set a fact with a temporal reference in the near future
        future_time = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        await store.set(
            "task.soon",
            "review PR",
            temporal_reference=future_time,
            priority="high",
        )

        # Set a fact with a temporal reference far in the future (beyond 24h)
        far_future = (datetime.now() + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")
        await store.set(
            "task.later",
            "plan sprint",
            temporal_reference=far_future,
        )

        due_soon = await store.get_temporal_due_soon(hours=24)
        assert isinstance(due_soon, list)
        keys = {f.key for f in due_soon}
        assert "task.soon" in keys
        assert "task.later" not in keys
