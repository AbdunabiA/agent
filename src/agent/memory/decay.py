"""Memory confidence decay.

Reduces confidence of stale facts over time and cleans up
facts that fall below minimum confidence thresholds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.memory.store import FactStore

logger = structlog.get_logger(__name__)


class MemoryDecay:
    """Applies time-based decay to fact confidence scores.

    Facts not accessed within ``stale_days`` have their confidence
    reduced by ``decay_rate``. Facts that drop below ``min_confidence``
    can be cleaned up.

    Usage::

        decay = MemoryDecay(fact_store)
        affected = await decay.apply_decay()
        deleted = await decay.cleanup()
    """

    def __init__(self, fact_store: FactStore) -> None:
        self.fact_store = fact_store

    async def apply_decay(
        self,
        decay_rate: float = 0.05,
        min_confidence: float = 0.1,
        stale_days: int = 7,
    ) -> int:
        """Reduce confidence of facts not accessed in stale_days.

        Applies a single-step decay: ``confidence = confidence - decay_rate``,
        clamped to ``min_confidence``.

        Args:
            decay_rate: Amount to subtract from confidence per decay pass.
            min_confidence: Floor value — confidence won't drop below this.
            stale_days: Days since last access before a fact is considered stale.

        Returns:
            Number of facts affected.
        """
        db = self.fact_store.db
        cursor = await db.db.execute(
            """UPDATE facts
               SET confidence = MAX(?, confidence - ?),
                   updated_at = datetime('now')
               WHERE julianday('now') - julianday(accessed_at) > ?
                 AND confidence > ?
            """,
            (min_confidence, decay_rate, stale_days, min_confidence),
        )
        await db.db.commit()
        affected = cursor.rowcount
        if affected > 0:
            logger.info(
                "memory_decay_applied",
                affected=affected,
                decay_rate=decay_rate,
                stale_days=stale_days,
            )
        return affected

    async def cleanup(self, min_confidence: float = 0.1) -> int:
        """Delete facts whose confidence has dropped below the threshold.

        Args:
            min_confidence: Facts with confidence below this are deleted.

        Returns:
            Number of facts deleted.
        """
        db = self.fact_store.db
        cursor = await db.db.execute(
            "DELETE FROM facts WHERE confidence < ?",
            (min_confidence,),
        )
        await db.db.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("memory_cleanup", deleted=deleted, threshold=min_confidence)
        return deleted
