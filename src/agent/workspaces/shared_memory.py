"""Shared memory layer for cross-workspace access.

Some facts should be accessible across all workspaces:
- User identity (name, preferences)
- Global settings
- Cross-project knowledge

Shared memory is stored in a separate "_shared" directory
alongside workspace-specific memory.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog

from agent.memory.store import FactStore
from agent.tools.registry import ToolTier, registry

logger = structlog.get_logger(__name__)


class SharedMemoryLayer:
    """Manages shared facts accessible from all workspaces.

    Stored in: workspaces/_shared/data/

    When context is assembled for an LLM call:
    1. Workspace-specific facts (highest priority)
    2. Shared facts (lower priority, included if relevant)
    """

    def __init__(self, shared_dir: str = "workspaces/_shared/data") -> None:
        self.shared_dir = Path(shared_dir)
        self.fact_store: FactStore | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize shared memory stores."""
        await asyncio.to_thread(os.makedirs, self.shared_dir, exist_ok=True)

        from agent.memory.database import Database

        db = Database(str(self.shared_dir / "shared.db"))
        await db.connect()

        self.fact_store = FactStore(db)
        self._initialized = True
        logger.info("shared_memory_initialized", path=str(self.shared_dir))

    def _ensure_initialized(self) -> None:
        """Guard: raise if initialize() hasn't been called."""
        if not self._initialized or not self.fact_store:
            raise RuntimeError(
                "SharedMemoryLayer not initialized. Call initialize() first."
            )

    async def set_shared_fact(
        self,
        key: str,
        value: str,
        source: str = "shared",
    ) -> None:
        """Store a fact in shared memory.

        Args:
            key: Dot-notation key (e.g. "user.name").
            value: The fact value.
            source: How the fact was learned.
        """
        self._ensure_initialized()
        await self.fact_store.set(key, value, source=source, confidence=1.0)
        logger.debug("shared_fact_set", key=key)

    async def get_shared_fact(self, key: str) -> str | None:
        """Get a fact from shared memory.

        Args:
            key: The exact key to look up.

        Returns:
            The fact value, or None if not found.
        """
        self._ensure_initialized()
        fact = await self.fact_store.get(key)
        return fact.value if fact else None

    async def search_shared(self, query: str, limit: int = 5) -> list:
        """Search shared memory facts.

        Args:
            query: Key prefix to search for.
            limit: Maximum results.

        Returns:
            List of matching Facts.
        """
        self._ensure_initialized()
        return await self.fact_store.search(query, limit=limit)

    async def promote_to_shared(
        self,
        key: str,
        value: str,
        source_workspace: str,
    ) -> None:
        """Promote a workspace-local fact to shared memory.

        Called when a fact is deemed useful across workspaces
        (e.g., user.name should be shared).

        Args:
            key: Fact key.
            value: Fact value.
            source_workspace: Which workspace the fact came from.
        """
        await self.set_shared_fact(
            key, value, source=f"promoted:{source_workspace}"
        )
        logger.info(
            "fact_promoted_to_shared",
            key=key,
            source_workspace=source_workspace,
        )


def register_shared_memory_tools(shared_memory: SharedMemoryLayer) -> None:
    """Register shared memory tools in the global registry.

    Args:
        shared_memory: The SharedMemoryLayer instance.
    """

    @registry.tool(
        name="shared_memory_set",
        description=(
            "Store a fact in shared memory (accessible from all workspaces). "
            "Use for user-wide facts like name, preferences, or global settings."
        ),
        tier=ToolTier.MODERATE,
    )
    async def shared_memory_set(key: str, value: str) -> str:
        """Store a shared fact.

        Args:
            key: Fact key (e.g. "user.name").
            value: Fact value.
        """
        await shared_memory.set_shared_fact(key, value)
        return f"Shared fact set: {key} = {value}"

    @registry.tool(
        name="shared_memory_get",
        description="Retrieve a fact from shared memory.",
        tier=ToolTier.SAFE,
    )
    async def shared_memory_get(key: str) -> str:
        """Get a shared fact.

        Args:
            key: Fact key to look up.
        """
        value = await shared_memory.get_shared_fact(key)
        if value is None:
            return f"No shared fact found for key: {key}"
        return f"{key} = {value}"

    @registry.tool(
        name="shared_memory_search",
        description="Search shared memory across all workspaces.",
        tier=ToolTier.SAFE,
    )
    async def shared_memory_search(query: str) -> str:
        """Search shared facts.

        Args:
            query: Key prefix to search for.
        """
        facts = await shared_memory.search_shared(query)
        if not facts:
            return "No shared facts found."
        lines = ["Shared facts:"]
        for f in facts:
            lines.append(f"  - {f.key}: {f.value}")
        return "\n".join(lines)
