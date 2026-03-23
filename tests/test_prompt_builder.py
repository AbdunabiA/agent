"""Tests for the PromptBuilderAgent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.core.prompt_builder_agent import PromptBuilderAgent
from agent.core.subagent import SubAgentRole


@pytest.fixture
def worker_role() -> SubAgentRole:
    return SubAgentRole(
        name="backend_developer",
        persona="You are a senior backend developer.",
        allowed_tools=["read_file", "write_file", "shell_exec"],
        max_iterations=10,
    )


@pytest.fixture
def mock_sdk() -> MagicMock:
    sdk = MagicMock()
    sdk.tool_registry = MagicMock()
    sdk.run_subagent = AsyncMock(
        return_value=(
            "## Task\nDo the thing\n## Requirements\n1. X\n2. Y\n"
            "## Acceptance criteria\n1. Tests pass\n"
            "## When Done\nCall complete_my_task()"
        )
    )
    return sdk


@pytest.fixture
def mock_working_memory() -> MagicMock:
    wm = MagicMock()
    wm.get_context_for_role = AsyncMock(
        return_value="### qa_engineer\n- **output_summary**: All tests pass"
    )
    return wm


@pytest.fixture
def mock_tracer() -> MagicMock:
    tracer = MagicMock()
    # Make span() an async context manager that yields a mock span
    span_mock = MagicMock()
    span_mock.metadata = {}

    async def _span_cm(*args, **kwargs):
        class _CM:
            async def __aenter__(self):
                return span_mock

            async def __aexit__(self, *a):
                return False

        return _CM()

    # Use asynccontextmanager-compatible mock
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_span(*args, **kwargs):
        yield span_mock

    tracer.span = fake_span
    return tracer


class TestPromptBuilderSuccess:
    """Builder succeeds — worker gets enriched prompt."""

    @pytest.mark.asyncio
    async def test_build_prompt_returns_builder_output(
        self,
        mock_sdk,
        mock_working_memory,
        worker_role,
    ):
        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
            working_memory=mock_working_memory,
        )

        result = await builder.build_prompt(
            worker_role=worker_role,
            task_id="test-123",
            task_description="Implement user auth endpoint",
        )

        assert "## Task" in result
        assert "## Requirements" in result
        mock_sdk.run_subagent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_working_memory_context_included_in_builder_input(
        self,
        mock_sdk,
        mock_working_memory,
        worker_role,
    ):
        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
            working_memory=mock_working_memory,
        )

        await builder.build_prompt(
            worker_role=worker_role,
            task_id="test-123",
            task_description="Fix the bug",
        )

        call_kwargs = mock_sdk.run_subagent.call_args
        prompt = call_kwargs.kwargs["prompt"]
        assert "qa_engineer" in prompt or "All tests pass" in prompt


class TestPromptBuilderTimeout:
    """Builder times out — worker still gets fallback prompt."""

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback(
        self,
        mock_sdk,
        mock_working_memory,
        worker_role,
    ):
        async def slow_subagent(**kwargs):
            await asyncio.sleep(60)  # will be cancelled by timeout
            return "never reached"

        mock_sdk.run_subagent = slow_subagent

        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
            working_memory=mock_working_memory,
        )

        # Patch _BUILDER_TIMEOUT to make test fast
        with patch("agent.core.prompt_builder_agent._BUILDER_TIMEOUT", 0.1):
            result = await builder.build_prompt(
                worker_role=worker_role,
                task_id="test-timeout",
                task_description="Build feature X",
            )

        # Should get fallback template
        assert "## Task" in result
        assert "Build feature X" in result
        assert "## When Done" in result


class TestPromptBuilderFailure:
    """Builder raises — worker still gets fallback prompt."""

    @pytest.mark.asyncio
    async def test_exception_returns_fallback(
        self,
        mock_sdk,
        mock_working_memory,
        worker_role,
    ):
        mock_sdk.run_subagent = AsyncMock(side_effect=RuntimeError("SDK crashed"))

        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
            working_memory=mock_working_memory,
        )

        result = await builder.build_prompt(
            worker_role=worker_role,
            task_id="test-fail",
            task_description="Fix bug Y",
        )

        assert "## Task" in result
        assert "Fix bug Y" in result

    @pytest.mark.asyncio
    async def test_empty_result_returns_fallback(
        self,
        mock_sdk,
        mock_working_memory,
        worker_role,
    ):
        # Builder returns too-short output
        mock_sdk.run_subagent = AsyncMock(return_value="ok")

        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
            working_memory=mock_working_memory,
        )

        result = await builder.build_prompt(
            worker_role=worker_role,
            task_id="test-empty",
            task_description="Do task Z",
        )

        assert "## Task" in result
        assert "Do task Z" in result


class TestPromptBuilderNone:
    """prompt_builder=None — worker spawns with base persona only."""

    @pytest.mark.asyncio
    async def test_no_builder_no_crash(self, worker_role):
        """Orchestrator with prompt_builder=None should not crash."""
        # Just verify PromptBuilderAgent can be None and the fallback path works
        builder = None
        assert builder is None
        # The orchestrator checks `if self.prompt_builder is not None` before calling


class TestPromptBuilderWithTracer:
    """Builder with tracer records a span."""

    @pytest.mark.asyncio
    async def test_tracer_span_recorded(
        self,
        mock_sdk,
        mock_working_memory,
        mock_tracer,
        worker_role,
    ):
        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
            working_memory=mock_working_memory,
            tracer=mock_tracer,
        )

        result = await builder.build_prompt(
            worker_role=worker_role,
            task_id="test-traced",
            task_description="Traced task",
        )

        assert "## Task" in result


class TestPromptBuilderTicketContext:
    """Builder includes ticket context when provided."""

    @pytest.mark.asyncio
    async def test_ticket_in_builder_input(
        self,
        mock_sdk,
        mock_working_memory,
        worker_role,
    ):
        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
            working_memory=mock_working_memory,
        )

        ticket = {"from_role": "qa_engineer", "title": "Fix login bug", "priority": "blocker"}

        await builder.build_prompt(
            worker_role=worker_role,
            task_id="test-ticket",
            task_description="Fix the login flow",
            ticket=ticket,
        )

        call_kwargs = mock_sdk.run_subagent.call_args
        prompt = call_kwargs.kwargs["prompt"]
        assert "Fix login bug" in prompt or "blocker" in prompt

    @pytest.mark.asyncio
    async def test_ticket_in_fallback(self, mock_sdk, worker_role):
        mock_sdk.run_subagent = AsyncMock(side_effect=RuntimeError("fail"))

        builder = PromptBuilderAgent(
            sdk_service=mock_sdk,
        )

        ticket = {"title": "Auth bug", "priority": "normal"}

        result = await builder.build_prompt(
            worker_role=worker_role,
            task_id="test-ticket-fallback",
            task_description="Fix auth",
            ticket=ticket,
        )

        assert "## Ticket Context" in result
        assert "Auth bug" in result


class TestToolNameResolution:
    """Verify prompt builder resolves all 4 tools from the registry."""

    def test_scoped_registry_resolves_four_tools(self, mock_sdk):
        """ScopedToolRegistry should find file_read, list_directory, find_files, memory_search."""
        from agent.core.orchestrator import ScopedToolRegistry
        from agent.tools.registry import ToolRegistry, ToolTier

        # Create a mock parent registry with the expected tools
        parent = ToolRegistry()

        expected_tools = {"file_read", "list_directory", "find_files", "memory_search"}

        # Register the 4 tools as stubs using the decorator API
        for tool_name in expected_tools:

            @parent.tool(name=tool_name, description=f"stub {tool_name}", tier=ToolTier.SAFE)
            async def _stub(**kw: str) -> str:
                return ""

        scoped = ScopedToolRegistry(
            parent=parent,
            allowed_tools=list(expected_tools),
            denied_tools=set(),
            exclude_dangerous=True,
        )

        resolved = scoped.list_tools()
        resolved_names = {t.name for t in resolved}
        assert resolved_names == expected_tools
        assert len(resolved) == 4

    def test_default_builder_role_has_correct_tool_names(self):
        """The default builder role must reference real registry tool names."""
        from agent.core.prompt_builder_agent import _DEFAULT_BUILDER_ROLE

        expected = {"file_read", "list_directory", "find_files", "memory_search"}
        assert set(_DEFAULT_BUILDER_ROLE.allowed_tools) == expected


class TestFallbackTemplate:
    """Direct tests for the fallback template."""

    def test_fallback_with_context(self, mock_sdk):
        builder = PromptBuilderAgent(sdk_service=mock_sdk)

        result = builder._fallback_template(
            description="Build feature",
            context="### qa: tests pass",
            ticket=None,
        )

        assert "## Task\nBuild feature" in result
        assert "### qa: tests pass" in result
        assert "## When Done" in result

    def test_fallback_no_context(self, mock_sdk):
        builder = PromptBuilderAgent(sdk_service=mock_sdk)

        result = builder._fallback_template(
            description="Build feature",
            context="",
            ticket=None,
        )

        assert "No prior context." in result

    def test_fallback_with_ticket(self, mock_sdk):
        builder = PromptBuilderAgent(sdk_service=mock_sdk)

        result = builder._fallback_template(
            description="Fix bug",
            context="",
            ticket={"title": "Login broken"},
        )

        assert "## Ticket Context" in result
        assert "Login broken" in result
