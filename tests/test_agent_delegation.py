"""Tests for controlled inter-agent delegation (Feature 2)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import OrchestrationConfig
from agent.core.events import EventBus, Events
from agent.core.orchestrator import ScopedToolRegistry, SubAgentOrchestrator
from agent.core.subagent import (
    AgentTeam,
    DelegationMode,
    DelegationRequest,
    DelegationResult,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
    SubAgentTask,
)
from agent.tools.registry import ToolDefinition, ToolTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_def(
    name: str, tier: ToolTier = ToolTier.SAFE, enabled: bool = True,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        tier=tier,
        parameters={"type": "object", "properties": {}},
        function=AsyncMock(),
        category="builtin",
        enabled=enabled,
    )


def _make_parent_registry(tools: list[ToolDefinition] | None = None) -> MagicMock:
    tools = tools or []
    tool_map = {t.name: t for t in tools}
    registry = MagicMock()
    registry.get_tool.side_effect = lambda name: tool_map.get(name)
    registry.get_tool_schemas.return_value = [
        {"function": {"name": t.name}} for t in tools if t.enabled
    ]
    registry.list_tools.return_value = tools
    return registry


def _make_orchestrator(
    teams: list[AgentTeam] | None = None,
    tools: list[ToolDefinition] | None = None,
) -> SubAgentOrchestrator:
    agent_loop = MagicMock()
    agent_loop.llm = MagicMock()
    agent_loop.tool_executor = MagicMock()
    config = OrchestrationConfig(
        enabled=True, max_concurrent_agents=5, subagent_timeout=30,
    )
    event_bus = EventBus()
    registry = _make_parent_registry(tools)
    return SubAgentOrchestrator(
        agent_loop=agent_loop,
        config=config,
        event_bus=event_bus,
        tool_registry=registry,
        teams=teams,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDelegation:
    @pytest.mark.asyncio
    async def test_delegation_sync_returns_result(self) -> None:
        """Sync delegation waits and returns specialist output."""
        role = SubAgentRole(name="writer", persona="You write docs.")
        team = AgentTeam(name="docs", description="Documentation", roles=[role])
        orch = _make_orchestrator(teams=[team])

        expected = SubAgentResult(
            task_id="t1",
            role_name="writer",
            status=SubAgentStatus.COMPLETED,
            output="Here is the documentation.",
            token_usage=200,
            duration_ms=1000,
        )

        with patch.object(orch, "spawn_subagent", return_value=expected):
            request = DelegationRequest(
                delegating_agent_id="a1",
                delegating_role="developer",
                target_team="docs",
                target_role="writer",
                instruction="Write API docs for the users endpoint.",
                mode=DelegationMode.SYNC,
            )
            result = await orch.handle_delegation(request, nesting_depth=0)

        assert result.status == SubAgentStatus.COMPLETED
        assert result.output == "Here is the documentation."
        assert result.token_usage == 200

    @pytest.mark.asyncio
    async def test_delegation_async_returns_task_id(self) -> None:
        """Async delegation returns pending status with task_id immediately."""
        role = SubAgentRole(name="tester", persona="You test.")
        team = AgentTeam(name="qa", description="QA", roles=[role])
        orch = _make_orchestrator(teams=[team])

        async def slow_spawn(task: SubAgentTask) -> SubAgentResult:
            await asyncio.sleep(10)
            return SubAgentResult(
                task_id=task.task_id,
                role_name="tester",
                status=SubAgentStatus.COMPLETED,
                output="Tests pass.",
            )

        with patch.object(orch, "spawn_subagent", side_effect=slow_spawn):
            request = DelegationRequest(
                delegating_agent_id="a1",
                delegating_role="developer",
                target_team="qa",
                target_role="tester",
                instruction="Run integration tests.",
                mode=DelegationMode.ASYNC,
            )
            result = await orch.handle_delegation(request, nesting_depth=0)

        assert result.status == SubAgentStatus.PENDING
        assert result.task_id != ""

        # Clean up the async task
        for task in orch._running_tasks.values():
            task.cancel()

    @pytest.mark.asyncio
    async def test_delegation_async_status_check(self) -> None:
        """Async delegation result retrievable via get_status after completion."""
        role = SubAgentRole(name="tester", persona="You test.")
        team = AgentTeam(name="qa", description="QA", roles=[role])
        orch = _make_orchestrator(teams=[team])

        async def instant_spawn(task: SubAgentTask) -> SubAgentResult:
            result = SubAgentResult(
                task_id=task.task_id,
                role_name="tester",
                status=SubAgentStatus.COMPLETED,
                output="All green.",
            )
            orch._results[task.task_id] = result
            return result

        with patch.object(orch, "spawn_subagent", side_effect=instant_spawn):
            request = DelegationRequest(
                delegating_agent_id="a1",
                delegating_role="developer",
                target_team="qa",
                target_role="tester",
                instruction="Run tests.",
                mode=DelegationMode.ASYNC,
            )
            result = await orch.handle_delegation(request, nesting_depth=0)
            task_id = result.task_id

            # Allow the fire-and-forget task to complete
            await asyncio.sleep(0.1)

        status = orch.get_status(task_id)
        assert status is not None
        assert status.status == SubAgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_delegation_prevents_recursion(self) -> None:
        """Delegation refused at nesting depth >= 1."""
        role = SubAgentRole(name="writer", persona="Writer.")
        team = AgentTeam(name="docs", description="Docs", roles=[role])
        orch = _make_orchestrator(teams=[team])

        request = DelegationRequest(
            delegating_agent_id="a1",
            delegating_role="sub_developer",
            target_team="docs",
            target_role="writer",
            instruction="Write something.",
        )
        result = await orch.handle_delegation(request, nesting_depth=1)

        assert result.status == SubAgentStatus.FAILED
        assert "recursion" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delegation_unknown_team(self) -> None:
        """Error on unknown team."""
        orch = _make_orchestrator(teams=[])

        request = DelegationRequest(
            delegating_agent_id="a1",
            delegating_role="developer",
            target_team="nonexistent",
            target_role="writer",
            instruction="Write docs.",
        )
        result = await orch.handle_delegation(request, nesting_depth=0)

        assert result.status == SubAgentStatus.FAILED
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_delegation_unknown_role(self) -> None:
        """Error on unknown role."""
        role = SubAgentRole(name="writer", persona="Writer.")
        team = AgentTeam(name="docs", description="Docs", roles=[role])
        orch = _make_orchestrator(teams=[team])

        request = DelegationRequest(
            delegating_agent_id="a1",
            delegating_role="developer",
            target_team="docs",
            target_role="nonexistent",
            instruction="Write docs.",
        )
        result = await orch.handle_delegation(request, nesting_depth=0)

        assert result.status == SubAgentStatus.FAILED
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_delegation_specialist_failure(self) -> None:
        """Specialist failure propagated correctly."""
        role = SubAgentRole(name="buggy", persona="Buggy.")
        team = AgentTeam(name="qa", description="QA", roles=[role])
        orch = _make_orchestrator(teams=[team])

        failed_result = SubAgentResult(
            task_id="t1",
            role_name="buggy",
            status=SubAgentStatus.FAILED,
            error="Segfault",
        )

        with patch.object(orch, "spawn_subagent", return_value=failed_result):
            request = DelegationRequest(
                delegating_agent_id="a1",
                delegating_role="developer",
                target_team="qa",
                target_role="buggy",
                instruction="Do something.",
                mode=DelegationMode.SYNC,
            )
            result = await orch.handle_delegation(request, nesting_depth=0)

        assert result.status == SubAgentStatus.FAILED
        assert result.error == "Segfault"

    @pytest.mark.asyncio
    async def test_delegation_emits_events(self) -> None:
        """Correct events emitted during delegation."""
        role = SubAgentRole(name="writer", persona="Writer.")
        team = AgentTeam(name="docs", description="Docs", roles=[role])
        orch = _make_orchestrator(teams=[team])

        events_received: list[str] = []

        async def capture(name: str):
            events_received.append(name)

        orch.event_bus.on(
            Events.AGENT_DELEGATION_REQUESTED,
            lambda d: capture("requested"),
        )
        orch.event_bus.on(
            Events.AGENT_DELEGATION_COMPLETED,
            lambda d: capture("completed"),
        )

        success_result = SubAgentResult(
            task_id="t1",
            role_name="writer",
            status=SubAgentStatus.COMPLETED,
            output="Done.",
        )

        with patch.object(orch, "spawn_subagent", return_value=success_result):
            request = DelegationRequest(
                delegating_agent_id="a1",
                delegating_role="developer",
                target_team="docs",
                target_role="writer",
                instruction="Write it.",
                mode=DelegationMode.SYNC,
            )
            await orch.handle_delegation(request, nesting_depth=0)

        assert "requested" in events_received
        assert "completed" in events_received

    @pytest.mark.asyncio
    async def test_delegate_tool_excluded_at_depth_1(self) -> None:
        """Scoped registry at depth 1 hides delegate_to_specialist."""
        delegate_tool = _make_tool_def("delegate_to_specialist", ToolTier.MODERATE)
        read_tool = _make_tool_def("read_file", ToolTier.SAFE)
        orch = _make_orchestrator(tools=[delegate_tool, read_tool])

        role = SubAgentRole(name="worker", persona="Worker.")
        scoped = orch._create_scoped_registry(role, nesting_depth=1)

        assert scoped.get_tool("delegate_to_specialist") is None
        assert scoped.get_tool("read_file") is not None
