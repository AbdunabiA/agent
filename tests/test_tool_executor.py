"""Tests for the tool executor."""

from __future__ import annotations

import asyncio

import pytest

from agent.config import ToolsConfig
from agent.core.audit import AuditLog
from agent.core.events import EventBus
from agent.core.guardrails import Guardrails
from agent.core.permissions import PermissionManager
from agent.core.session import ToolCall
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry, ToolTier


@pytest.fixture
def test_registry() -> ToolRegistry:
    """Create a registry with test tools."""
    reg = ToolRegistry()

    @reg.tool(name="echo", description="Echo input", tier=ToolTier.SAFE)
    async def echo(text: str) -> str:
        return f"echo: {text}"

    @reg.tool(name="slow_tool", description="Slow tool", tier=ToolTier.SAFE)
    async def slow_tool(seconds: int = 5) -> str:
        await asyncio.sleep(seconds)
        return "done"

    @reg.tool(name="failing_tool", description="Always fails", tier=ToolTier.SAFE)
    async def failing_tool() -> str:
        raise ValueError("intentional failure")

    @reg.tool(name="big_output", description="Produces big output", tier=ToolTier.SAFE)
    async def big_output() -> str:
        return "x" * 100_000  # 100KB

    return reg


@pytest.fixture
def tools_config() -> ToolsConfig:
    return ToolsConfig()


@pytest.fixture
def executor(test_registry: ToolRegistry, tools_config: ToolsConfig) -> ToolExecutor:
    event_bus = EventBus()
    audit = AuditLog()
    permissions = PermissionManager(tools_config)
    guardrails = Guardrails(tools_config)
    return ToolExecutor(
        registry=test_registry,
        config=tools_config,
        event_bus=event_bus,
        audit=audit,
        permissions=permissions,
        guardrails=guardrails,
    )


class TestToolExecutor:
    """Tests for ToolExecutor."""

    async def test_execute_safe_tool(self, executor: ToolExecutor) -> None:
        """A safe tool should auto-approve and return result."""
        tc = ToolCall(id="call_1", name="echo", arguments={"text": "hello"})
        result = await executor.execute(tc, session_id="test-session")

        assert result.success is True
        assert result.output == "echo: hello"
        assert result.tool_name == "echo"

    async def test_execute_unknown_tool(self, executor: ToolExecutor) -> None:
        """Unknown tool should return error result."""
        tc = ToolCall(id="call_2", name="nonexistent", arguments={})
        result = await executor.execute(tc, session_id="test-session")

        assert result.success is False
        assert "not found" in result.output.lower() or "not registered" in result.error.lower()

    async def test_execute_disabled_tool(self, executor: ToolExecutor) -> None:
        """Disabled tool should return error result."""
        executor.registry.disable_tool("echo")
        tc = ToolCall(id="call_3", name="echo", arguments={"text": "hello"})
        result = await executor.execute(tc, session_id="test-session")

        assert result.success is False
        assert "disabled" in result.output.lower()

    async def test_execute_with_timeout(self, executor: ToolExecutor) -> None:
        """Tool that exceeds timeout should be killed."""
        tc = ToolCall(id="call_4", name="slow_tool", arguments={"seconds": 60, "timeout": 1})
        result = await executor.execute(tc, session_id="test-session")

        assert result.success is False
        assert "timed out" in result.output.lower() or "timed out" in (result.error or "").lower()

    async def test_output_truncation(self, executor: ToolExecutor) -> None:
        """Large output should be truncated."""
        tc = ToolCall(id="call_5", name="big_output", arguments={})
        result = await executor.execute(tc, session_id="test-session")

        assert result.success is True
        # Output should be truncated to ~50KB
        assert len(result.output) <= 60_000  # Some overhead for truncation message

    async def test_audit_log_created(self, executor: ToolExecutor) -> None:
        """Every execution should create an audit log entry."""
        tc = ToolCall(id="call_6", name="echo", arguments={"text": "audit test"})
        await executor.execute(tc, session_id="test-session")

        entries = await executor.audit.get_entries()
        assert len(entries) >= 1
        assert entries[0].tool_name == "echo"
        assert entries[0].status == "success"

    async def test_failing_tool_returns_error(self, executor: ToolExecutor) -> None:
        """A tool that raises should return error result."""
        tc = ToolCall(id="call_7", name="failing_tool", arguments={})
        result = await executor.execute(tc, session_id="test-session")

        assert result.success is False
        assert (
            "intentional failure" in result.output
            or "intentional failure" in (result.error or "")
        )

    async def test_parallel_execution(self, executor: ToolExecutor) -> None:
        """Parallel execution should run tools concurrently."""
        calls = [
            ToolCall(id="p1", name="echo", arguments={"text": "a"}),
            ToolCall(id="p2", name="echo", arguments={"text": "b"}),
            ToolCall(id="p3", name="failing_tool", arguments={}),
        ]
        results = await executor.execute_parallel(calls, session_id="test-session")

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is False

    async def test_duration_tracked(self, executor: ToolExecutor) -> None:
        """Execution duration should be tracked."""
        tc = ToolCall(id="call_8", name="echo", arguments={"text": "timing"})
        result = await executor.execute(tc, session_id="test-session")

        assert result.duration_ms >= 0
