"""Memory tools — let the LLM read/write structured facts at runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.memory.store import FactStore

_global_fact_store: FactStore | None = None


def set_fact_store(store: FactStore) -> None:
    """Set the global FactStore instance (called during agent startup).

    Args:
        store: The initialized FactStore.
    """
    global _global_fact_store
    _global_fact_store = store


def get_fact_store() -> FactStore:
    """Get the global FactStore instance.

    Returns:
        The shared FactStore.

    Raises:
        RuntimeError: If set_fact_store() hasn't been called yet.
    """
    if _global_fact_store is None:
        raise RuntimeError(
            "FactStore not initialized. Call set_fact_store() during startup."
        )
    return _global_fact_store


@tool(
    name="memory_set",
    description=(
        "Store a fact in the agent's long-term memory. "
        "Use dot-notation keys like 'user.name', 'preference.language', 'project.stack'. "
        "If the key already exists, the value is updated."
    ),
    tier=ToolTier.SAFE,
)
async def memory_set(
    key: str,
    value: str,
    category: str = "general",
) -> str:
    """Store a fact in memory.

    Args:
        key: Dot-notation key (e.g. "user.name").
        value: The fact value.
        category: Grouping category (user, preference, project, system, general).

    Returns:
        Confirmation message.
    """
    store = get_fact_store()
    await store.set(key, value, category=category, source="extracted")
    return f"Stored: {key} = {value}"


@tool(
    name="memory_get",
    description=(
        "Retrieve a specific fact from the agent's long-term memory by its exact key."
    ),
    tier=ToolTier.SAFE,
)
async def memory_get(key: str) -> str:
    """Retrieve a fact by key.

    Args:
        key: The exact key to look up.

    Returns:
        The fact value, or a message if not found.
    """
    store = get_fact_store()
    fact = await store.get(key)
    if fact is None:
        return f"No fact found for key: {key}"
    return f"{fact.key} = {fact.value} (category={fact.category}, confidence={fact.confidence})"


@tool(
    name="memory_search",
    description=(
        "Search the agent's long-term memory by key prefix and/or category. "
        "For example, searching 'user' finds 'user.name', 'user.email', etc."
    ),
    tier=ToolTier.SAFE,
)
async def memory_search(
    query: str,
    category: str | None = None,
) -> str:
    """Search facts by prefix and optional category.

    Args:
        query: Key prefix to search for.
        category: Optional category filter.

    Returns:
        Formatted list of matching facts.
    """
    store = get_fact_store()

    if category:
        facts = await store.get_by_category(category)
        # Further filter by prefix
        facts = [f for f in facts if f.key.startswith(query)]
    else:
        facts = await store.search(query)

    if not facts:
        return f"No facts found matching '{query}'"

    lines = [f"- {f.key} = {f.value}" for f in facts]
    return f"Found {len(facts)} fact(s):\n" + "\n".join(lines)
