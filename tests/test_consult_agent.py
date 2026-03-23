"""Tests for inter-agent consultation (Feature 1)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import OrchestrationConfig
from agent.core.events import EventBus, Events
from agent.core.orchestrator import SubAgentOrchestrator
from agent.core.subagent import (
    AgentTeam,
    ConsultRequest,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
)
from agent.tools.registry import ToolDefinition, ToolTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_def(
    name: str,
    tier: ToolTier = ToolTier.SAFE,
    enabled: bool = True,
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
        enabled=True,
        max_concurrent_agents=5,
        subagent_timeout=30,
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


class TestConsultAgent:
    @pytest.mark.asyncio
    async def test_consult_resolves_team_and_role(self) -> None:
        """Happy path — consulted agent answers the question."""
        role = SubAgentRole(name="expert", persona="You are an expert.")
        team = AgentTeam(name="research", description="R&D", roles=[role])
        orch = _make_orchestrator(teams=[team])

        expected = SubAgentResult(
            task_id="t1",
            role_name="expert",
            status=SubAgentStatus.COMPLETED,
            output="The answer is 42.",
            token_usage=100,
            duration_ms=500,
        )

        with patch.object(orch, "spawn_subagent", return_value=expected):
            request = ConsultRequest(
                requesting_agent_id="a1",
                requesting_role="developer",
                target_team="research",
                target_role="expert",
                question="What is the meaning of life?",
                context="We are building software.",
            )
            response = await orch.handle_consult(request, nesting_depth=0)

        assert response.status == SubAgentStatus.COMPLETED
        assert response.answer == "The answer is 42."
        assert response.token_usage == 100

    @pytest.mark.asyncio
    async def test_consult_unknown_team_returns_error(self) -> None:
        """Unknown team returns error."""
        orch = _make_orchestrator(teams=[])

        request = ConsultRequest(
            requesting_agent_id="a1",
            requesting_role="developer",
            target_team="nonexistent",
            target_role="expert",
            question="Hello?",
        )
        response = await orch.handle_consult(request, nesting_depth=0)

        assert response.status == SubAgentStatus.FAILED
        assert "not found" in response.error

    @pytest.mark.asyncio
    async def test_consult_unknown_role_returns_error(self) -> None:
        """Known team but unknown role returns error."""
        role = SubAgentRole(name="analyst", persona="You analyze.")
        team = AgentTeam(name="research", description="R&D", roles=[role])
        orch = _make_orchestrator(teams=[team])

        request = ConsultRequest(
            requesting_agent_id="a1",
            requesting_role="developer",
            target_team="research",
            target_role="nonexistent",
            question="Hello?",
        )
        response = await orch.handle_consult(request, nesting_depth=0)

        assert response.status == SubAgentStatus.FAILED
        assert "not found" in response.error

    @pytest.mark.asyncio
    async def test_consult_prevents_recursion(self) -> None:
        """Consultation refused at nesting depth >= 1."""
        role = SubAgentRole(name="expert", persona="Expert.")
        team = AgentTeam(name="research", description="R&D", roles=[role])
        orch = _make_orchestrator(teams=[team])

        request = ConsultRequest(
            requesting_agent_id="a1",
            requesting_role="developer",
            target_team="research",
            target_role="expert",
            question="Can you consult someone?",
        )
        response = await orch.handle_consult(request, nesting_depth=1)

        assert response.status == SubAgentStatus.FAILED
        assert "recursion" in response.error.lower()

    @pytest.mark.asyncio
    async def test_consult_timeout(self) -> None:
        """Consulted agent times out."""
        role = SubAgentRole(name="slow_expert", persona="Slow.")
        team = AgentTeam(name="research", description="R&D", roles=[role])
        orch = _make_orchestrator(teams=[team])

        timeout_result = SubAgentResult(
            task_id="t1",
            role_name="slow_expert",
            status=SubAgentStatus.FAILED,
            error="Timed out after 30s",
        )

        with patch.object(orch, "spawn_subagent", return_value=timeout_result):
            request = ConsultRequest(
                requesting_agent_id="a1",
                requesting_role="developer",
                target_team="research",
                target_role="slow_expert",
                question="This will time out",
            )
            response = await orch.handle_consult(request, nesting_depth=0)

        assert response.status == SubAgentStatus.FAILED
        assert "Timed out" in response.error

    @pytest.mark.asyncio
    async def test_consult_emits_events(self) -> None:
        """Correct events emitted during consultation."""
        role = SubAgentRole(name="expert", persona="Expert.")
        team = AgentTeam(name="research", description="R&D", roles=[role])
        orch = _make_orchestrator(teams=[team])

        events_received: list[str] = []

        async def capture_event(data: dict) -> None:
            pass

        orch.event_bus.on(
            Events.AGENT_CONSULT_REQUESTED,
            lambda d: _capture(events_received, "requested"),
        )
        orch.event_bus.on(
            Events.AGENT_CONSULT_COMPLETED,
            lambda d: _capture(events_received, "completed"),
        )

        success_result = SubAgentResult(
            task_id="t1",
            role_name="expert",
            status=SubAgentStatus.COMPLETED,
            output="Answer.",
        )

        with patch.object(orch, "spawn_subagent", return_value=success_result):
            request = ConsultRequest(
                requesting_agent_id="a1",
                requesting_role="developer",
                target_team="research",
                target_role="expert",
                question="Q?",
            )
            await orch.handle_consult(request, nesting_depth=0)

        assert "requested" in events_received
        assert "completed" in events_received

    @pytest.mark.asyncio
    async def test_consult_tool_excluded_at_depth_1(self) -> None:
        """Scoped registry at depth 1 hides consult_agent."""
        consult_tool = _make_tool_def("consult_agent", ToolTier.MODERATE)
        # Use "file_read" — a tool in the essential_mcp_tools allowlist
        read_tool = _make_tool_def("file_read", ToolTier.SAFE)
        orch = _make_orchestrator(tools=[consult_tool, read_tool])

        role = SubAgentRole(name="worker", persona="Worker.")
        scoped = orch._create_scoped_registry(role, nesting_depth=1)

        assert scoped.get_tool("consult_agent") is None
        assert scoped.get_tool("file_read") is not None

    @pytest.mark.asyncio
    async def test_consult_tool_available_at_depth_0(self) -> None:
        """Scoped registry at depth 0 exposes consult_agent."""
        consult_tool = _make_tool_def("consult_agent", ToolTier.MODERATE)
        # Use "file_read" — a tool in the essential_mcp_tools allowlist
        read_tool = _make_tool_def("file_read", ToolTier.SAFE)
        orch = _make_orchestrator(tools=[consult_tool, read_tool])

        role = SubAgentRole(name="worker", persona="Worker.")
        scoped = orch._create_scoped_registry(role, nesting_depth=0)

        assert scoped.get_tool("consult_agent") is not None
        assert scoped.get_tool("file_read") is not None


# Helpers for event capture


async def _capture(events_list: list[str], name: str) -> None:
    events_list.append(name)
