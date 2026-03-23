"""SQLite-backed structured facts storage.

Provides CRUD operations and relevance-ranked retrieval
for the agent's structured memory (key-value facts).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from agent.memory.models import Fact

if TYPE_CHECKING:
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)


def _row_to_fact(row: object) -> Fact:
    """Convert a database row to a Fact."""
    return Fact(
        id=row["id"],
        key=row["key"],
        value=row["value"],
        category=row["category"],
        confidence=row["confidence"],
        source=row["source"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        accessed_at=datetime.fromisoformat(row["accessed_at"]),
        access_count=row["access_count"],
    )


class FactStore:
    """SQLite-backed structured facts storage.

    Usage:
        store = FactStore(db)
        await store.set("user.name", "Abduvohid", category="user")
        name = await store.get("user.name")
        facts = await store.search("user")
        top = await store.get_relevant(limit=10)
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    async def set(
        self,
        key: str,
        value: str,
        category: str = "general",
        source: str = "user",
        confidence: float = 1.0,
    ) -> Fact:
        """Set a fact. Updates if key exists, creates if not.

        Args:
            key: Dot-notation key (e.g. "user.name").
            value: The fact value.
            category: Grouping category.
            source: How the fact was learned.
            confidence: Confidence score (0.0 to 1.0).

        Returns:
            The created or updated Fact.
        """
        now = datetime.now().isoformat()
        fact_id = str(uuid4())

        # Use BEGIN IMMEDIATE to prevent race conditions between the
        # INSERT...ON CONFLICT and the subsequent SELECT.
        await self.db.db.execute("BEGIN IMMEDIATE")
        try:
            await self.db.db.execute(
                """INSERT INTO facts (id, key, value, category, confidence, source,
                                      created_at, updated_at, accessed_at, access_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(key) DO UPDATE SET
                       value = excluded.value,
                       category = excluded.category,
                       confidence = excluded.confidence,
                       source = excluded.source,
                       updated_at = excluded.updated_at
                """,
                (fact_id, key, value, category, confidence, source, now, now, now),
            )

            async with self.db.db.execute("SELECT * FROM facts WHERE key = ?", (key,)) as cursor:
                row = await cursor.fetchone()

            await self.db.db.execute("COMMIT")
        except Exception:
            await self.db.db.execute("ROLLBACK")
            raise

        fact = _row_to_fact(row)
        logger.debug("fact_set", key=key, category=category, source=source)
        return fact

    async def get(self, key: str) -> Fact | None:
        """Get a fact by exact key. Updates accessed_at and access_count.

        Args:
            key: The exact key to look up.

        Returns:
            The Fact if found, None otherwise.
        """
        # Atomic update + fetch in one transaction
        now = datetime.now().isoformat()
        await self.db.db.execute(
            "UPDATE facts SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
            (now, key),
        )
        await self.db.db.commit()

        async with self.db.db.execute("SELECT * FROM facts WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return _row_to_fact(row)

    async def delete(self, key: str) -> bool:
        """Delete a fact by key.

        Args:
            key: The key to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = await self.db.db.execute("DELETE FROM facts WHERE key = ?", (key,))
        await self.db.db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("fact_deleted", key=key)
        return deleted

    async def delete_by_id(self, fact_id: str) -> bool:
        """Delete a fact by its ID. Returns True if deleted."""
        cursor = await self.db.db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        await self.db.db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("fact_deleted_by_id", fact_id=fact_id)
        return deleted

    async def search(self, prefix: str, limit: int = 50, offset: int = 0) -> list[Fact]:
        """Search facts by key prefix.

        'user' matches 'user.name', 'user.email', etc.

        Args:
            prefix: Key prefix to search for.
            limit: Maximum results.
            offset: Number of results to skip.

        Returns:
            List of matching Facts.
        """
        async with self.db.db.execute(
            "SELECT * FROM facts WHERE key LIKE ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (f"{prefix}%", limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        return [_row_to_fact(row) for row in rows]

    async def get_by_category(
        self,
        category: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Fact]:
        """Get all facts in a category.

        Args:
            category: Category to filter by.
            limit: Maximum results.
            offset: Number of results to skip.

        Returns:
            List of matching Facts.
        """
        async with self.db.db.execute(
            "SELECT * FROM facts WHERE category = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (category, limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        return [_row_to_fact(row) for row in rows]

    async def get_relevant(self, limit: int = 20) -> list[Fact]:
        """Get most relevant facts for context injection.

        Ranked by: confidence * recency_score * frequency_score
        - recency: higher if accessed recently (exponential decay over days)
        - frequency: log-scaled access count

        Args:
            limit: Maximum number of facts to return.

        Returns:
            List of Facts ranked by relevance.
        """
        # SQLite relevance scoring:
        # confidence * (1.0 / (1.0 + days_since_access)) * (1.0 + log(1 + access_count))
        async with self.db.db.execute(
            """SELECT *,
                confidence *
                (1.0 / (1.0 + MIN(
                    julianday('now') - julianday(COALESCE(accessed_at, created_at)),
                    365))) *
                (1.0 + ln(1.0 + COALESCE(access_count, 0))) AS relevance
               FROM facts
               ORDER BY relevance DESC
               LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [_row_to_fact(row) for row in rows]

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[Fact]:
        """Get all facts sorted by updated_at desc.

        Args:
            limit: Maximum results.
            offset: Number of results to skip.

        Returns:
            List of all Facts.
        """
        async with self.db.db.execute(
            "SELECT * FROM facts ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        return [_row_to_fact(row) for row in rows]

    async def count(self) -> int:
        """Count total facts.

        Returns:
            Number of facts in the store.
        """
        async with self.db.db.execute("SELECT COUNT(*) FROM facts") as cursor:
            row = await cursor.fetchone()
            return row[0]
