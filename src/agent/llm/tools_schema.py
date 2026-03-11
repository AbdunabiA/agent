"""Generate tool schemas from the registry for LLM function calling."""

from __future__ import annotations

from typing import Any

from agent.tools.registry import registry


def get_available_tools() -> list[dict[str, Any]]:
    """Return tool definitions formatted for LLM function calling.

    Returns:
        List of tool schema dicts from the global registry.
    """
    return registry.get_tool_schemas(enabled_only=True)
