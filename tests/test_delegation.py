"""Tests for WorkspaceDelegator and delegation tools."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config import AgentConfig, WorkspacesSection
from agent.workspaces.delegation import WorkspaceDelegator
from agent.workspaces.manager import WorkspaceManager


@pytest.fixture
def ws_config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        workspaces=WorkspacesSection(directory=str(tmp_path / "workspaces")),
    )


@pytest.fixture
def manager(ws_config: AgentConfig) -> WorkspaceManager:
    mgr = WorkspaceManager(ws_config)
    mgr.create("default", display_name="Default", description="Default workspace")
    mgr.create("work", display_name="Work", description="Work workspace")
    mgr.create("research", display_name="Research", description="Research workspace")
    return mgr


def _make_mock_app(response_content: str = "mock response") -> SimpleNamespace:
    """Create a mock application with session_store and agent_loop."""
    session = MagicMock()
    session.id = "test-session"

    session_store = AsyncMock()
    session_store.get_or_create = AsyncMock(return_value=session)

    response = SimpleNamespace(content=response_content)
    agent_loop = AsyncMock()
    agent_loop.process_message = AsyncMock(return_value=response)

    fact = SimpleNamespace(key="project.name", value="Agent")
    fact_store = AsyncMock()
    fact_store.search = AsyncMock(return_value=[fact])

    vector = SimpleNamespace(text="We discussed the deployment strategy for the agent project.")
    vector_store = AsyncMock()
    vector_store.search = AsyncMock(return_value=[vector])

    return SimpleNamespace(
        session_store=session_store,
        agent_loop=agent_loop,
        fact_store=fact_store,
        vector_store=vector_store,
    )


@pytest.mark.asyncio
class TestDelegator:
    async def test_delegate_sends_message(self, manager: WorkspaceManager) -> None:
        """delegate sends message to target workspace and returns response."""
        mock_app = _make_mock_app("I found the answer")
        factory = AsyncMock(return_value=mock_app)

        delegator = WorkspaceDelegator(manager, factory)
        result = await delegator.delegate("work", "What is the project status?")

        assert result == "I found the answer"
        mock_app.agent_loop.process_message.assert_awaited_once()
        call_args = mock_app.agent_loop.process_message.call_args
        assert "What is the project status?" in call_args[0][0]

    async def test_delegate_with_context(self, manager: WorkspaceManager) -> None:
        """delegate includes context in the message when provided."""
        mock_app = _make_mock_app("response with context")
        factory = AsyncMock(return_value=mock_app)

        delegator = WorkspaceDelegator(manager, factory)
        result = await delegator.delegate(
            "work",
            "Check status",
            source_workspace="personal",
            context="User asked about deployment",
        )

        assert result == "response with context"
        call_args = mock_app.agent_loop.process_message.call_args
        msg = call_args[0][0]
        assert "personal" in msg
        assert "User asked about deployment" in msg
        assert "Check status" in msg

    async def test_delegate_without_context(self, manager: WorkspaceManager) -> None:
        """delegate without context sends plain message."""
        mock_app = _make_mock_app("plain response")
        factory = AsyncMock(return_value=mock_app)

        delegator = WorkspaceDelegator(manager, factory)
        result = await delegator.delegate("work", "Hello")

        assert result == "plain response"
        call_args = mock_app.agent_loop.process_message.call_args
        assert call_args[0][0] == "Hello"

    async def test_delegate_nonexistent_workspace(
        self, manager: WorkspaceManager,
    ) -> None:
        """delegate raises for nonexistent workspace."""
        factory = AsyncMock()
        delegator = WorkspaceDelegator(manager, factory)

        with pytest.raises(Exception, match="not found"):
            await delegator.delegate("nonexistent", "Hello")

    async def test_query_memory(self, manager: WorkspaceManager) -> None:
        """query_memory searches target workspace facts and vectors."""
        mock_app = _make_mock_app()
        factory = AsyncMock(return_value=mock_app)

        delegator = WorkspaceDelegator(manager, factory)
        result = await delegator.query_memory("work", "project")

        assert "project.name" in result
        assert "Agent" in result
        assert "deployment strategy" in result

    async def test_query_memory_empty(self, manager: WorkspaceManager) -> None:
        """query_memory returns not found message when no results."""
        mock_app = _make_mock_app()
        mock_app.fact_store.search = AsyncMock(return_value=[])
        mock_app.vector_store.search = AsyncMock(return_value=[])
        factory = AsyncMock(return_value=mock_app)

        delegator = WorkspaceDelegator(manager, factory)
        result = await delegator.query_memory("work", "nonexistent topic")

        assert "No relevant information" in result

    async def test_query_memory_nonexistent_workspace(
        self, manager: WorkspaceManager,
    ) -> None:
        """query_memory raises for nonexistent workspace."""
        factory = AsyncMock()
        delegator = WorkspaceDelegator(manager, factory)

        with pytest.raises(Exception, match="not found"):
            await delegator.query_memory("nonexistent", "test")

    async def test_app_cached(self, manager: WorkspaceManager) -> None:
        """Application instances are cached per workspace."""
        mock_app = _make_mock_app()
        factory = AsyncMock(return_value=mock_app)

        delegator = WorkspaceDelegator(manager, factory)
        await delegator.delegate("work", "first")
        await delegator.delegate("work", "second")

        # Factory called only once for the same workspace
        factory.assert_awaited_once_with("work")

    async def test_different_workspaces_different_apps(
        self, manager: WorkspaceManager,
    ) -> None:
        """Different workspaces get different app instances."""
        mock_app1 = _make_mock_app("from work")
        mock_app2 = _make_mock_app("from research")

        call_count = 0

        async def factory(name: str) -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            return mock_app1 if name == "work" else mock_app2

        delegator = WorkspaceDelegator(manager, factory)
        r1 = await delegator.delegate("work", "hello")
        r2 = await delegator.delegate("research", "hello")

        assert r1 == "from work"
        assert r2 == "from research"
        assert call_count == 2
