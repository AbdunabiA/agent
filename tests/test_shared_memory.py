"""Tests for SharedMemoryLayer."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.workspaces.shared_memory import SharedMemoryLayer


@pytest.fixture
async def shared_memory(tmp_path: Path) -> SharedMemoryLayer:
    """Create and initialize a SharedMemoryLayer in a temp directory."""
    layer = SharedMemoryLayer(shared_dir=str(tmp_path / "_shared" / "data"))
    await layer.initialize()
    return layer


@pytest.mark.asyncio
class TestSharedMemoryLayer:
    async def test_set_and_get(self, shared_memory: SharedMemoryLayer) -> None:
        """set_shared_fact stores and get_shared_fact retrieves."""
        await shared_memory.set_shared_fact("user.name", "Abduvohid")
        value = await shared_memory.get_shared_fact("user.name")
        assert value == "Abduvohid"

    async def test_get_nonexistent(self, shared_memory: SharedMemoryLayer) -> None:
        """get_shared_fact returns None for missing key."""
        value = await shared_memory.get_shared_fact("nonexistent.key")
        assert value is None

    async def test_overwrite(self, shared_memory: SharedMemoryLayer) -> None:
        """Setting same key overwrites the value."""
        await shared_memory.set_shared_fact("user.name", "Alice")
        await shared_memory.set_shared_fact("user.name", "Bob")
        value = await shared_memory.get_shared_fact("user.name")
        assert value == "Bob"

    async def test_search(self, shared_memory: SharedMemoryLayer) -> None:
        """search_shared finds facts by key prefix."""
        await shared_memory.set_shared_fact("user.name", "Abduvohid")
        await shared_memory.set_shared_fact("user.email", "test@example.com")
        await shared_memory.set_shared_fact("project.name", "Agent")

        results = await shared_memory.search_shared("user")
        keys = [f.key for f in results]
        assert "user.name" in keys
        assert "user.email" in keys
        assert "project.name" not in keys

    async def test_search_empty(self, shared_memory: SharedMemoryLayer) -> None:
        """search_shared returns empty list when no matches."""
        results = await shared_memory.search_shared("nonexistent")
        assert results == []

    async def test_promote_to_shared(self, shared_memory: SharedMemoryLayer) -> None:
        """promote_to_shared stores fact with source annotation."""
        await shared_memory.promote_to_shared(
            "user.timezone", "UTC+5", source_workspace="work"
        )

        value = await shared_memory.get_shared_fact("user.timezone")
        assert value == "UTC+5"

        # Check source was set correctly
        fact = await shared_memory.fact_store.get("user.timezone")
        assert fact is not None
        assert "promoted:work" in fact.source

    async def test_initialize_creates_directory(self, tmp_path: Path) -> None:
        """initialize creates the shared data directory."""
        shared_dir = tmp_path / "new_shared" / "data"
        layer = SharedMemoryLayer(shared_dir=str(shared_dir))
        await layer.initialize()
        assert shared_dir.exists()

    async def test_not_initialized_raises(self, tmp_path: Path) -> None:
        """Operations before initialize() raise RuntimeError."""
        layer = SharedMemoryLayer(shared_dir=str(tmp_path / "uninit"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await layer.get_shared_fact("test")

    async def test_multiple_facts(self, shared_memory: SharedMemoryLayer) -> None:
        """Can store and retrieve many facts."""
        for i in range(10):
            await shared_memory.set_shared_fact(f"item.{i}", f"value_{i}")

        for i in range(10):
            val = await shared_memory.get_shared_fact(f"item.{i}")
            assert val == f"value_{i}"

    async def test_isolated_directory(self, tmp_path: Path) -> None:
        """Two SharedMemoryLayers in different dirs are isolated."""
        layer_a = SharedMemoryLayer(shared_dir=str(tmp_path / "a"))
        await layer_a.initialize()

        layer_b = SharedMemoryLayer(shared_dir=str(tmp_path / "b"))
        await layer_b.initialize()

        await layer_a.set_shared_fact("key", "from_a")
        await layer_b.set_shared_fact("key", "from_b")

        assert await layer_a.get_shared_fact("key") == "from_a"
        assert await layer_b.get_shared_fact("key") == "from_b"
