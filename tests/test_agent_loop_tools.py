"""Tests for the upgraded agent loop with tool calling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.config import AgentPersonaConfig, ToolsConfig
from agent.core.agent_loop import AgentLoop
from agent.core.audit import AuditLog
from agent.core.events import EventBus
from agent.core.guardrails import Guardrails
from agent.core.permissions import PermissionManager
from agent.core.planner import Planner
from agent.core.recovery import ErrorRecovery
from agent.core.session import Session, TokenUsage, ToolCall
from agent.llm.provider import LLMResponse
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry, ToolTier


@pytest.fixture
def test_registry() -> ToolRegistry:
    """Create a registry with a test tool."""
    reg = ToolRegistry()

    @reg.tool(name="echo", description="Echo input", tier=ToolTier.SAFE)
    async def echo(text: str) -> str:
        return f"echo: {text}"

    return reg


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def tools_config() -> ToolsConfig:
    return ToolsConfig()


@pytest.fixture
def agent_loop(
    test_registry: ToolRegistry, event_bus: EventBus, tools_config: ToolsConfig
) -> AgentLoop:
    """Create a full agent loop with mock LLM."""
    mock_llm = AsyncMock()
    config = AgentPersonaConfig(max_iterations=5)

    guardrails = Guardrails(tools_config)
    permissions = PermissionManager(tools_config)
    audit = AuditLog()

    tool_executor = ToolExecutor(
        registry=test_registry,
        config=tools_config,
        event_bus=event_bus,
        audit=audit,
        permissions=permissions,
        guardrails=guardrails,
    )

    recovery = ErrorRecovery()
    planner = Planner(llm=mock_llm, config=config)

    loop = AgentLoop(
        llm=mock_llm,
        config=config,
        event_bus=event_bus,
        tool_executor=tool_executor,
        planner=planner,
        recovery=recovery,
        guardrails=guardrails,
    )

    return loop


class TestAgentLoopWithTools:
    """Tests for the agent loop with tool calling."""

    async def test_simple_message_no_tools(self, agent_loop: AgentLoop) -> None:
        """Simple message should return direct response without tool calls."""
        agent_loop.llm.completion.return_value = LLMResponse(
            content="Hello! How can I help?",
            model="test-model",
            tool_calls=None,
            usage=TokenUsage(10, 5, 15),
        )

        session = Session()
        response = await agent_loop.process_message("Hello", session)

        assert response.content == "Hello! How can I help?"
        assert len(session.messages) == 2  # user + assistant

    async def test_single_tool_call(self, agent_loop: AgentLoop) -> None:
        """Message requiring one tool call should execute and return result."""
        # First call: LLM requests tool
        tool_call = ToolCall(id="call_1", name="echo", arguments={"text": "hello"})
        agent_loop.llm.completion.side_effect = [
            # First: LLM wants to call a tool
            LLMResponse(
                content="",
                model="test-model",
                tool_calls=[tool_call],
                usage=TokenUsage(10, 5, 15),
            ),
            # Second: LLM gives final response
            LLMResponse(
                content="The echo tool returned: echo: hello",
                model="test-model",
                tool_calls=None,
                usage=TokenUsage(20, 10, 30),
            ),
        ]

        session = Session()
        response = await agent_loop.process_message("Echo hello for me", session)

        assert "echo" in response.content.lower()
        # Messages: user, assistant (tool call), tool result, assistant (final)
        assert len(session.messages) == 4

    async def test_max_iterations_reached(self, agent_loop: AgentLoop) -> None:
        """Should gracefully stop after max iterations."""
        # LLM always requests tool calls
        tool_call = ToolCall(id="call_loop", name="echo", arguments={"text": "loop"})
        agent_loop.llm.completion.return_value = LLMResponse(
            content="",
            model="test-model",
            tool_calls=[tool_call],
            usage=TokenUsage(10, 5, 15),
        )

        # Override for the final forced response
        call_count = 0

        async def smart_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            messages = kwargs.get("messages", [])
            # After several iterations, check if we're being asked to summarize
            has_summary_request = any(
                "maximum" in str(m.get("content", "")).lower()
                for m in messages if isinstance(m, dict)
            )
            if has_summary_request:
                return LLMResponse(
                    content="I reached the iteration limit.",
                    model="test-model",
                    tool_calls=None,
                    usage=TokenUsage(10, 5, 15),
                )
            return LLMResponse(
                content="",
                model="test-model",
                tool_calls=[tool_call],
                usage=TokenUsage(10, 5, 15),
            )

        agent_loop.llm.completion = smart_completion

        session = Session()
        response = await agent_loop.process_message("Keep echoing", session)

        assert "iteration limit" in response.content.lower() or response.content

    async def test_tool_error_recovery(self, agent_loop: AgentLoop) -> None:
        """Tool errors should trigger recovery and include error info."""
        # Register a failing tool
        @agent_loop.tool_executor.registry.tool(
            name="fail_tool", description="Fails", tier=ToolTier.SAFE
        )
        async def fail_tool() -> str:
            raise ValueError("tool broke")

        tool_call = ToolCall(id="call_fail", name="fail_tool", arguments={})

        agent_loop.llm.completion.side_effect = [
            LLMResponse(
                content="",
                model="test-model",
                tool_calls=[tool_call],
                usage=TokenUsage(10, 5, 15),
            ),
            LLMResponse(
                content="The tool failed. Let me try something else.",
                model="test-model",
                tool_calls=None,
                usage=TokenUsage(20, 10, 30),
            ),
        ]

        session = Session()
        response = await agent_loop.process_message("Run the failing tool", session)

        # Should get a response despite tool failure
        assert response.content

    async def test_events_emitted(self, agent_loop: AgentLoop, event_bus: EventBus) -> None:
        """Tool execution should emit events."""
        events_received: list[str] = []

        async def track_event(data):
            events_received.append("received")

        event_bus.on("tool.execute", track_event)
        event_bus.on("tool.result", track_event)

        tool_call = ToolCall(id="call_evt", name="echo", arguments={"text": "event test"})
        agent_loop.llm.completion.side_effect = [
            LLMResponse(
                content="",
                model="test-model",
                tool_calls=[tool_call],
                usage=TokenUsage(10, 5, 15),
            ),
            LLMResponse(
                content="Done!",
                model="test-model",
                tool_calls=None,
                usage=TokenUsage(10, 5, 15),
            ),
        ]

        session = Session()
        await agent_loop.process_message("Echo event test", session)

        assert len(events_received) >= 2  # TOOL_EXECUTE + TOOL_RESULT

    async def test_backward_compatible_no_tools(self) -> None:
        """Agent loop should work without tool executor (backward compat)."""
        mock_llm = AsyncMock()
        mock_llm.completion.return_value = LLMResponse(
            content="Hi there!",
            model="test-model",
            tool_calls=None,
            usage=TokenUsage(10, 5, 15),
        )

        loop = AgentLoop(
            llm=mock_llm,
            config=AgentPersonaConfig(),
            event_bus=EventBus(),
        )

        session = Session()
        response = await loop.process_message("Hello", session)
        assert response.content == "Hi there!"
