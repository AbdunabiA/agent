"""Scoped tool registry for sub-agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class ScopedToolRegistry:
    """A filtered view of a parent ToolRegistry.

    Implements the same interface as ToolRegistry but only exposes
    tools matching the allow/deny configuration.
    """

    def __init__(
        self,
        parent: ToolRegistry,
        allowed_tools: list[str] | None = None,
        denied_tools: set[str] | None = None,
        exclude_dangerous: bool = True,
    ) -> None:
        self._parent = parent
        self._allowed = set(allowed_tools) if allowed_tools else None
        self._denied = denied_tools or set()
        self._exclude_dangerous = exclude_dangerous

    def _is_tool_allowed(self, name: str) -> bool:
        """Check if a tool passes the scope filters."""
        from agent.tools.registry import ToolTier

        if name in self._denied:
            return False

        if self._allowed is not None and name not in self._allowed:
            return False

        if self._exclude_dangerous:
            tool_def = self._parent.get_tool(name)
            if tool_def and tool_def.tier == ToolTier.DANGEROUS:
                return False

        return True

    def get_tool(self, name: str) -> Any:
        """Look up a tool by name, respecting scope."""
        if not self._is_tool_allowed(name):
            return None
        return self._parent.get_tool(name)

    def get_tool_schemas(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Get filtered tool schemas."""
        all_schemas = self._parent.get_tool_schemas(enabled_only=enabled_only)
        return [s for s in all_schemas if self._is_tool_allowed(s["function"]["name"])]

    def list_tools(self) -> list[Any]:
        """List all allowed tools."""
        return [t for t in self._parent.list_tools() if self._is_tool_allowed(t.name)]

    def enable_tool(self, name: str) -> None:
        """Scoped registries cannot enable tools in the parent."""
        logger.debug("scoped_registry_enable_noop", tool=name)

    def disable_tool(self, name: str) -> None:
        """Scoped registries cannot disable tools in the parent."""
        logger.debug("scoped_registry_disable_noop", tool=name)

    def unregister_tool(self, name: str) -> None:
        """Scoped registries cannot unregister tools from the parent."""
        logger.debug("scoped_registry_unregister_noop", tool=name)
