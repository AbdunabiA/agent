"""Tests for MemoryDecay."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent.memory.database import Database
from agent.memory.decay import MemoryDecay
from agent.memory.store import FactStore


@pytest.fixture
async def db(tmp_path):
    """Create a temporary database."""
    db = Database(str(tmp_path / "test.db"))
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
async def fact_store(db):
    """Create a FactStore backed by the temp db."""
    return FactStore(db)


@pytest.fixture
async def decay(fact_store):
    """Create a MemoryDecay instance."""
    return MemoryDecay(fact_store)


class TestMemoryDecay:
    """Tests for MemoryDecay class."""

    @pytest.mark.asyncio
    async def test_apply_decay_reduces_stale(
        self, fact_store: FactStore, decay: MemoryDecay, db: Database
    ) -> None:
        """Decay reduces confidence of stale facts."""
        await fact_store.set("old.fact", "value", confidence=0.9)

        # Make the fact stale by backdating accessed_at
        stale_time = (datetime.now() - timedelta(days=10)).isoformat()
        await db.db.execute(
            "UPDATE facts SET accessed_at = ? WHERE key = ?",
            (stale_time, "old.fact"),
        )
        await db.db.commit()

        affected = await decay.apply_decay(decay_rate=0.1, stale_days=7)
        assert affected == 1

        fact = await fact_store.get("old.fact")
        assert fact is not None
        assert fact.confidence == pytest.approx(0.8, abs=0.01)

    @pytest.mark.asyncio
    async def test_apply_decay_skips_recent(
        self, fact_store: FactStore, decay: MemoryDecay
    ) -> None:
        """Decay skips recently accessed facts."""
        await fact_store.set("recent.fact", "value", confidence=0.9)

        affected = await decay.apply_decay(stale_days=7)
        assert affected == 0

    @pytest.mark.asyncio
    async def test_apply_decay_respects_min_confidence(
        self, fact_store: FactStore, decay: MemoryDecay, db: Database
    ) -> None:
        """Confidence doesn't drop below min_confidence."""
        await fact_store.set("low.fact", "value", confidence=0.15)

        stale_time = (datetime.now() - timedelta(days=10)).isoformat()
        await db.db.execute(
            "UPDATE facts SET accessed_at = ? WHERE key = ?",
            (stale_time, "low.fact"),
        )
        await db.db.commit()

        await decay.apply_decay(decay_rate=0.1, min_confidence=0.1, stale_days=7)

        fact = await fact_store.get("low.fact")
        assert fact is not None
        assert fact.confidence >= 0.1

    @pytest.mark.asyncio
    async def test_cleanup_deletes_low_confidence(
        self, fact_store: FactStore, decay: MemoryDecay, db: Database
    ) -> None:
        """Cleanup deletes facts below min_confidence."""
        await fact_store.set("low.fact", "value", confidence=0.05)

        deleted = await decay.cleanup(min_confidence=0.1)
        assert deleted == 1

        fact = await fact_store.get("low.fact")
        assert fact is None

    @pytest.mark.asyncio
    async def test_cleanup_keeps_above_threshold(
        self, fact_store: FactStore, decay: MemoryDecay
    ) -> None:
        """Cleanup keeps facts above min_confidence."""
        await fact_store.set("healthy.fact", "value", confidence=0.5)

        deleted = await decay.cleanup(min_confidence=0.1)
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_apply_decay_returns_count(
        self, fact_store: FactStore, decay: MemoryDecay, db: Database
    ) -> None:
        """apply_decay returns correct affected count."""
        await fact_store.set("a", "1", confidence=0.9)
        await fact_store.set("b", "2", confidence=0.8)
        await fact_store.set("c", "3", confidence=0.7)

        stale_time = (datetime.now() - timedelta(days=10)).isoformat()
        await db.db.execute(
            "UPDATE facts SET accessed_at = ?", (stale_time,)
        )
        await db.db.commit()

        affected = await decay.apply_decay(decay_rate=0.05, stale_days=7)
        assert affected == 3
