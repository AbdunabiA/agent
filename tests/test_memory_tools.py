"""Tests for memory tools (memory_set, memory_get, memory_search)."""

from __future__ import annotations

import pytest

from agent.memory.database import Database
from agent.memory.store import FactStore
from agent.tools.builtins.memory import (
    get_fact_store,
    memory_get,
    memory_search,
    memory_set,
    set_fact_store,
)
from agent.tools.registry import ToolTier, registry


@pytest.fixture
async def fact_store(tmp_path: object) -> FactStore:
    """Create a FactStore backed by a temporary database."""
    db = Database(str(tmp_path / "mem_tools_test.db"))
    await db.connect()
    store = FactStore(db)
    set_fact_store(store)
    yield store
    # Cleanup
    import agent.tools.builtins.memory as mem_mod

    mem_mod._global_fact_store = None
    await db.close()


class TestMemoryToolRegistration:
    """Tests for tool registration."""

    def test_memory_set_registered(self) -> None:
        """memory_set should be in the global registry."""
        tool_def = registry.get_tool("memory_set")
        assert tool_def is not None
        assert tool_def.tier == ToolTier.SAFE

    def test_memory_get_registered(self) -> None:
        """memory_get should be in the global registry."""
        tool_def = registry.get_tool("memory_get")
        assert tool_def is not None
        assert tool_def.tier == ToolTier.SAFE

    def test_memory_search_registered(self) -> None:
        """memory_search should be in the global registry."""
        tool_def = registry.get_tool("memory_search")
        assert tool_def is not None
        assert tool_def.tier == ToolTier.SAFE


class TestMemorySet:
    """Tests for memory_set tool."""

    async def test_stores_fact(self, fact_store: FactStore) -> None:
        """memory_set should store a fact."""
        result = await memory_set("user.name", "Abduvohid")
        assert "Stored" in result
        assert "user.name" in result

        fact = await fact_store.get("user.name")
        assert fact is not None
        assert fact.value == "Abduvohid"

    async def test_respects_category(self, fact_store: FactStore) -> None:
        """memory_set should use the provided category."""
        await memory_set("pref.theme", "dark", category="preference")

        fact = await fact_store.get("pref.theme")
        assert fact is not None
        assert fact.category == "preference"

    async def test_source_is_extracted(self, fact_store: FactStore) -> None:
        """memory_set should set source to 'extracted'."""
        await memory_set("test.key", "value")

        fact = await fact_store.get("test.key")
        assert fact is not None
        assert fact.source == "extracted"


class TestMemoryGet:
    """Tests for memory_get tool."""

    async def test_retrieves_existing(self, fact_store: FactStore) -> None:
        """memory_get should return the fact value for an existing key."""
        await fact_store.set("user.name", "Abduvohid", category="user")
        result = await memory_get("user.name")
        assert "Abduvohid" in result
        assert "user.name" in result

    async def test_returns_not_found(self, fact_store: FactStore) -> None:
        """memory_get should return 'No fact found' for unknown key."""
        result = await memory_get("nonexistent.key")
        assert "No fact found" in result


class TestMemorySearch:
    """Tests for memory_search tool."""

    async def test_finds_by_prefix(self, fact_store: FactStore) -> None:
        """memory_search should find facts by key prefix."""
        await fact_store.set("user.name", "Abduvohid")
        await fact_store.set("user.email", "test@example.com")
        await fact_store.set("project.name", "Agent")

        result = await memory_search("user")
        assert "2 fact(s)" in result
        assert "user.name" in result
        assert "user.email" in result

    async def test_finds_by_category(self, fact_store: FactStore) -> None:
        """memory_search with category should filter by category."""
        await fact_store.set("user.name", "Abduvohid", category="user")
        await fact_store.set("pref.theme", "dark", category="preference")

        result = await memory_search("pref", category="preference")
        assert "1 fact(s)" in result
        assert "pref.theme" in result

    async def test_returns_no_results_message(self, fact_store: FactStore) -> None:
        """memory_search should return a message when no facts match."""
        result = await memory_search("nonexistent")
        assert "No facts found" in result


class TestGetFactStoreGuard:
    """Tests for get_fact_store() without initialization."""

    def test_raises_when_not_initialized(self) -> None:
        """get_fact_store() should raise RuntimeError when not set."""
        import agent.tools.builtins.memory as mem_mod

        original = mem_mod._global_fact_store
        mem_mod._global_fact_store = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                get_fact_store()
        finally:
            mem_mod._global_fact_store = original
