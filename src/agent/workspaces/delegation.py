"""Inter-workspace delegation.

One workspace's agent can delegate a task to another workspace's agent.
Use cases:
- "work" agent asks "research" agent to look something up
- "personal" agent asks "work" agent about a project status
- Any agent can query another's memory without full context

Delegation is implemented as tools: delegate_task, search_workspace_memory,
list_workspaces.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import structlog

from agent.tools.registry import ToolTier, registry

if TYPE_CHECKING:
    from agent.workspaces.manager import WorkspaceManager

logger = structlog.get_logger(__name__)


class WorkspaceDelegator:
    """Handles delegation between workspaces.

    When a workspace delegates to another:
    1. A new session is created in the target workspace
    2. The delegated message is processed by the target's agent loop
    3. The response is returned to the source workspace
    4. The target workspace's memory and tools are used
    """

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        application_factory: Callable[[str], Awaitable[Any]],
    ) -> None:
        """
        Args:
            workspace_manager: For resolving workspaces.
            application_factory: Async callable(workspace_name) -> Application.
        """
        self.workspace_manager = workspace_manager
        self.application_factory = application_factory
        self._active_apps: dict[str, Any] = {}

    async def delegate(
        self,
        target_workspace: str,
        message: str,
        source_workspace: str = "",
        context: str = "",
    ) -> str:
        """Send a message to another workspace and get a response.

        Args:
            target_workspace: Name of workspace to delegate to.
            message: The message/question/task to send.
            source_workspace: Name of the requesting workspace.
            context: Optional context to include.

        Returns:
            The response from the target workspace's agent.
        """
        self.workspace_manager.resolve(target_workspace)  # validate exists
        app = await self._get_app(target_workspace)

        session = await app.session_store.get_or_create(
            session_id=f"delegation:{source_workspace}",
            channel="delegation",
        )

        delegation_msg = message
        if context:
            delegation_msg = (
                f"[Delegated from '{source_workspace}' workspace]\n"
                f"Context: {context}\n\n"
                f"Request: {message}"
            )

        response = await app.agent_loop.process_message(
            delegation_msg, session, trigger="delegation"
        )
        return response.content

    async def query_memory(
        self,
        target_workspace: str,
        query: str,
    ) -> str:
        """Search another workspace's memory without full agent processing.

        Faster than delegation -- just searches facts and vectors.

        Args:
            target_workspace: Name of workspace to query.
            query: What to search for.

        Returns:
            Formatted search results or a "not found" message.
        """
        self.workspace_manager.resolve(target_workspace)  # validate exists
        app = await self._get_app(target_workspace)

        results: list[str] = []

        fact_store = getattr(app, "fact_store", None)
        if fact_store:
            facts = await fact_store.search(query, limit=5)
            if facts:
                lines = [f"  - {f.key}: {f.value}" for f in facts]
                results.append("Facts:\n" + "\n".join(lines))

        vector_store = getattr(app, "vector_store", None)
        if vector_store:
            vectors = await vector_store.search(query, limit=3)
            if vectors:
                lines = [f"  - {v.text[:200]}" for v in vectors]
                results.append("Related context:\n" + "\n".join(lines))

        if results:
            return "\n\n".join(results)
        return f"No relevant information found in '{target_workspace}' workspace."

    async def _get_app(self, workspace_name: str) -> Any:
        """Get or create an Application instance for a workspace."""
        if workspace_name not in self._active_apps:
            app = await self.application_factory(workspace_name)
            self._active_apps[workspace_name] = app
        return self._active_apps[workspace_name]


def register_delegation_tools(
    delegator: WorkspaceDelegator,
    workspace_manager: WorkspaceManager,
) -> None:
    """Register delegation tools in the global registry.

    Args:
        delegator: The WorkspaceDelegator instance.
        workspace_manager: For listing available workspaces.
    """

    @registry.tool(
        name="delegate_task",
        description=(
            "Delegate a task or question to another workspace's agent. "
            "Each workspace has its own memory, personality, and tools. "
            "Use this when a task would be better handled by a different workspace "
            "or when you need information from another workspace's memory."
        ),
        tier=ToolTier.MODERATE,
    )
    async def delegate_task(
        workspace: str,
        message: str,
        context: str = "",
    ) -> str:
        """Delegate to another workspace.

        Args:
            workspace: Target workspace name (e.g., "research", "work").
            message: The task or question to send.
            context: Optional context to provide.
        """
        available = workspace_manager.discover()
        if workspace not in available:
            return (
                f"Workspace '{workspace}' not found. "
                f"Available: {', '.join(available)}"
            )
        return await delegator.delegate(workspace, message, context=context)

    @registry.tool(
        name="search_workspace_memory",
        description=(
            "Search another workspace's memory for relevant information "
            "without running a full agent conversation. Fast memory-only lookup."
        ),
        tier=ToolTier.SAFE,
    )
    async def search_workspace_memory(workspace: str, query: str) -> str:
        """Search another workspace's memory.

        Args:
            workspace: Target workspace name.
            query: What to search for.
        """
        available = workspace_manager.discover()
        if workspace not in available:
            return (
                f"Workspace '{workspace}' not found. "
                f"Available: {', '.join(available)}"
            )
        return await delegator.query_memory(workspace, query)

    @registry.tool(
        name="list_workspaces",
        description="List all available workspaces with their descriptions.",
        tier=ToolTier.SAFE,
    )
    async def list_workspaces() -> str:
        """List workspaces."""
        workspaces = workspace_manager.discover()
        if not workspaces:
            return "No workspaces found."

        lines = ["Available workspaces:"]
        for name in workspaces:
            try:
                ws = workspace_manager.resolve(name)
                desc = ws.description or "(no description)"
                lines.append(f"  - {name}: {desc}")
            except Exception:
                lines.append(f"  - {name}: (error loading)")
        return "\n".join(lines)
