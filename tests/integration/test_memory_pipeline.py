"""Integration tests for the memory fact storage and retrieval pipeline.

Tests the full flow: store -> retrieve -> relevance ranking -> update -> delete,
using a real SQLite database (via tmp_path).
"""

from __future__ import annotations

from agent.memory.store import FactStore


class TestMemoryPipeline:
    """End-to-end tests for fact storage and retrieval."""

    async def test_store_and_retrieve_fact(self, fact_store: FactStore) -> None:
        """Store a fact via set() and retrieve it via get()."""
        await fact_store.set("user.name", "Abduvohid", category="user")

        fact = await fact_store.get("user.name")

        assert fact is not None
        assert fact.key == "user.name"
        assert fact.value == "Abduvohid"
        assert fact.category == "user"

    async def test_get_relevant_returns_results(self, fact_store: FactStore) -> None:
        """get_relevant() should return stored facts."""
        await fact_store.set("user.name", "Abduvohid", category="user")
        await fact_store.set("user.lang", "Python", category="user")
        await fact_store.set("pref.theme", "dark", category="preference")

        results = await fact_store.get_relevant(limit=10)

        assert len(results) == 3

    async def test_relevance_ordering_by_confidence_and_access(
        self,
        fact_store: FactStore,
    ) -> None:
        """Facts with higher confidence and more accesses should rank higher."""
        await fact_store.set("low.priority", "value", confidence=0.1)
        await fact_store.set("high.priority", "value", confidence=1.0)

        # Boost high.priority by accessing it multiple times
        await fact_store.get("high.priority")
        await fact_store.get("high.priority")
        await fact_store.get("high.priority")

        results = await fact_store.get_relevant(limit=10)

        assert len(results) == 2
        assert results[0].key == "high.priority"
        assert results[1].key == "low.priority"

    async def test_fact_update_same_key_new_value(
        self,
        fact_store: FactStore,
    ) -> None:
        """Setting a fact with the same key should update its value."""
        await fact_store.set("project.name", "OldProject")
        updated = await fact_store.set("project.name", "NewProject")

        assert updated.value == "NewProject"
        assert await fact_store.count() == 1

        # Verify via direct retrieval
        fetched = await fact_store.get("project.name")
        assert fetched is not None
        assert fetched.value == "NewProject"

    async def test_fact_update_preserves_id_or_reuses_key(
        self,
        fact_store: FactStore,
    ) -> None:
        """Updating a fact should keep it as a single row (upsert on key)."""
        await fact_store.set("tool.favorite", "vim")
        await fact_store.set("tool.favorite", "neovim")
        await fact_store.set("tool.favorite", "helix")

        assert await fact_store.count() == 1
        fact = await fact_store.get("tool.favorite")
        assert fact is not None
        assert fact.value == "helix"

    async def test_delete_by_id(self, fact_store: FactStore) -> None:
        """delete_by_id should remove the fact with that specific ID."""
        fact = await fact_store.set("temp.key", "temp_value")
        fact_id = fact.id

        assert await fact_store.count() == 1

        deleted = await fact_store.delete_by_id(fact_id)
        assert deleted is True
        assert await fact_store.count() == 0

        # Verify it is truly gone
        assert await fact_store.get("temp.key") is None

    async def test_delete_by_id_returns_false_for_unknown(
        self,
        fact_store: FactStore,
    ) -> None:
        """delete_by_id should return False when the ID does not exist."""
        deleted = await fact_store.delete_by_id("nonexistent-uuid")
        assert deleted is False

    async def test_full_pipeline_store_search_update_delete(
        self,
        fact_store: FactStore,
    ) -> None:
        """Full pipeline: store multiple facts, search, update, delete, verify."""
        # Store
        await fact_store.set("user.name", "Abduvohid", category="user")
        await fact_store.set("user.email", "test@example.com", category="user")
        await fact_store.set("pref.editor", "vscode", category="preference")

        assert await fact_store.count() == 3

        # Search by prefix
        user_facts = await fact_store.search("user")
        assert len(user_facts) == 2

        # Update
        await fact_store.set("pref.editor", "cursor", category="preference")
        editor = await fact_store.get("pref.editor")
        assert editor is not None
        assert editor.value == "cursor"
        assert await fact_store.count() == 3  # still 3, not 4

        # Delete by key
        deleted = await fact_store.delete("user.email")
        assert deleted is True
        assert await fact_store.count() == 2

        # Verify remaining
        remaining = await fact_store.get_all()
        keys = {f.key for f in remaining}
        assert keys == {"user.name", "pref.editor"}

    async def test_category_filtering(self, fact_store: FactStore) -> None:
        """get_by_category should return only facts in the specified category."""
        await fact_store.set("user.name", "Test", category="user")
        await fact_store.set("user.age", "30", category="user")
        await fact_store.set("sys.os", "macOS", category="system")
        await fact_store.set("pref.lang", "en", category="preference")

        user_facts = await fact_store.get_by_category("user")
        assert len(user_facts) == 2
        for f in user_facts:
            assert f.category == "user"

        system_facts = await fact_store.get_by_category("system")
        assert len(system_facts) == 1
        assert system_facts[0].key == "sys.os"
