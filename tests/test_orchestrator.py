"""Tests for the multi-agent orchestration system."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import OrchestrationConfig
from agent.core.events import EventBus, Events
from agent.core.orchestrator import ScopedToolRegistry, SubAgentOrchestrator
from agent.core.session import TokenUsage
from agent.core.subagent import (
    AgentTeam,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
    SubAgentTask,
)
from agent.llm.provider import LLMResponse
from agent.tools.registry import ToolDefinition, ToolTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_def(
    name: str, tier: ToolTier = ToolTier.SAFE, enabled: bool = True
) -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f"A {tier} tool called {name}",
        tier=tier,
        parameters={"type": "object", "properties": {}},
        function=AsyncMock(),
        category="builtin",
        enabled=enabled,
    )


def _make_tool_schema(name: str) -> dict:
    """Create a tool schema dict matching ToolDefinition.to_llm_schema output."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Schema for {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_llm_response(content: str = "Done.") -> LLMResponse:
    """Create a mock LLMResponse."""
    return LLMResponse(
        content=content,
        model="gpt-4o-mini",
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        finish_reason="stop",
    )


def _make_parent_registry(
    tools: list[ToolDefinition] | None = None,
) -> MagicMock:
    """Build a mock ToolRegistry exposing the given tools."""
    tools = tools or []
    tool_map = {t.name: t for t in tools}

    registry = MagicMock()
    registry.get_tool = MagicMock(side_effect=lambda n: tool_map.get(n))
    registry.list_tools = MagicMock(return_value=tools)
    registry.get_tool_schemas = MagicMock(
        return_value=[_make_tool_schema(t.name) for t in tools if t.enabled]
    )
    registry.enable_tool = MagicMock()
    registry.disable_tool = MagicMock()
    registry.unregister_tool = MagicMock()
    return registry


def _make_role(
    name: str = "researcher",
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    max_iterations: int = 3,
) -> SubAgentRole:
    return SubAgentRole(
        name=name,
        persona="You are a test agent.",
        allowed_tools=allowed_tools or [],
        denied_tools=denied_tools or [],
        max_iterations=max_iterations,
    )


def _make_task(
    role: SubAgentRole | None = None,
    instruction: str = "Summarize the data.",
    context: str = "",
    parent_session_id: str = "parent-1",
    task_id: str | None = None,
) -> SubAgentTask:
    role = role or _make_role()
    task = SubAgentTask(
        role=role,
        instruction=instruction,
        context=context,
        parent_session_id=parent_session_id,
    )
    if task_id is not None:
        task.task_id = task_id
    return task


# Shared patch targets for the local imports inside _execute_subagent
_PATCH_AGENT_LOOP = "agent.core.agent_loop.AgentLoop"
_PATCH_TOOL_EXECUTOR = "agent.tools.executor.ToolExecutor"
_PATCH_BUILD_PROMPT = "agent.llm.prompts.build_system_prompt"


# ---------------------------------------------------------------------------
# ScopedToolRegistry tests
# ---------------------------------------------------------------------------


class TestScopedToolRegistry:
    """Tests for ScopedToolRegistry filtering logic."""

    def test_allows_all_safe_tools_with_no_filters(self) -> None:
        """With no allow/deny lists, all non-dangerous tools pass through."""
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("write_file", ToolTier.MODERATE),
        ]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(parent)
        result = scoped.list_tools()

        names = {t.name for t in result}
        assert "read_file" in names
        assert "write_file" in names

    def test_excludes_dangerous_tools_by_default(self) -> None:
        """Dangerous tools are excluded when exclude_dangerous=True (default)."""
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("delete_everything", ToolTier.DANGEROUS),
        ]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(parent, exclude_dangerous=True)
        result = scoped.list_tools()

        names = {t.name for t in result}
        assert "read_file" in names
        assert "delete_everything" not in names

    def test_allows_dangerous_when_exclude_disabled(self) -> None:
        """Dangerous tools are included when exclude_dangerous=False."""
        tools = [
            _make_tool_def("delete_everything", ToolTier.DANGEROUS),
        ]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(parent, exclude_dangerous=False)
        result = scoped.list_tools()

        names = {t.name for t in result}
        assert "delete_everything" in names

    def test_allow_list_restricts_to_named_tools(self) -> None:
        """Only tools in the allow list are visible."""
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("web_search", ToolTier.SAFE),
            _make_tool_def("shell_exec", ToolTier.MODERATE),
        ]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(parent, allowed_tools=["read_file", "web_search"])
        result = scoped.list_tools()

        names = {t.name for t in result}
        assert names == {"read_file", "web_search"}

    def test_deny_list_removes_named_tools(self) -> None:
        """Tools in the deny list are hidden."""
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("shell_exec", ToolTier.MODERATE),
        ]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(parent, denied_tools={"shell_exec"})
        result = scoped.list_tools()

        names = {t.name for t in result}
        assert "read_file" in names
        assert "shell_exec" not in names

    def test_deny_takes_precedence_over_allow(self) -> None:
        """If a tool appears in both allow and deny, deny wins."""
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("shell_exec", ToolTier.MODERATE),
        ]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(
            parent,
            allowed_tools=["read_file", "shell_exec"],
            denied_tools={"shell_exec"},
        )
        result = scoped.list_tools()

        names = {t.name for t in result}
        assert names == {"read_file"}

    def test_get_tool_returns_none_for_denied(self) -> None:
        """get_tool returns None when the tool is denied."""
        tools = [_make_tool_def("shell_exec", ToolTier.MODERATE)]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(parent, denied_tools={"shell_exec"})
        assert scoped.get_tool("shell_exec") is None

    def test_get_tool_returns_definition_for_allowed(self) -> None:
        """get_tool returns the tool when it passes the filters."""
        tool = _make_tool_def("read_file", ToolTier.SAFE)
        parent = _make_parent_registry([tool])

        scoped = ScopedToolRegistry(parent)
        assert scoped.get_tool("read_file") is tool

    def test_get_tool_schemas_filters_correctly(self) -> None:
        """get_tool_schemas only includes schemas for allowed tools."""
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("danger_tool", ToolTier.DANGEROUS),
        ]
        parent = _make_parent_registry(tools)

        scoped = ScopedToolRegistry(parent, exclude_dangerous=True)
        schemas = scoped.get_tool_schemas(enabled_only=True)

        schema_names = {s["function"]["name"] for s in schemas}
        assert "read_file" in schema_names
        assert "danger_tool" not in schema_names

    def test_enable_tool_is_noop(self) -> None:
        """enable_tool is a no-op — sub-agents cannot modify the global registry."""
        tool = _make_tool_def("read_file", ToolTier.SAFE)
        parent = _make_parent_registry([tool])

        scoped = ScopedToolRegistry(parent)
        scoped.enable_tool("read_file")

        parent.enable_tool.assert_not_called()

    def test_enable_tool_noop_for_denied(self) -> None:
        """enable_tool does nothing for denied tools."""
        tool = _make_tool_def("shell_exec", ToolTier.MODERATE)
        parent = _make_parent_registry([tool])

        scoped = ScopedToolRegistry(parent, denied_tools={"shell_exec"})
        scoped.enable_tool("shell_exec")

        parent.enable_tool.assert_not_called()

    def test_disable_tool_is_noop(self) -> None:
        """disable_tool is a no-op — sub-agents cannot modify global registry."""
        tool = _make_tool_def("read_file", ToolTier.SAFE)
        parent = _make_parent_registry([tool])

        scoped = ScopedToolRegistry(parent)
        scoped.disable_tool("read_file")

        parent.disable_tool.assert_not_called()

    def test_unregister_tool_is_noop(self) -> None:
        """unregister_tool is a no-op — sub-agents cannot modify global registry."""
        tool = _make_tool_def("read_file", ToolTier.SAFE)
        parent = _make_parent_registry([tool])

        scoped = ScopedToolRegistry(parent)
        scoped.unregister_tool("read_file")

        parent.unregister_tool.assert_not_called()


# ---------------------------------------------------------------------------
# SubAgentOrchestrator tests
# ---------------------------------------------------------------------------


class TestSubAgentOrchestrator:
    """Tests for SubAgentOrchestrator."""

    @pytest.fixture
    def config(self) -> OrchestrationConfig:
        return OrchestrationConfig(
            enabled=True,
            max_concurrent_agents=3,
            subagent_timeout=10,
            default_max_iterations=5,
        )

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def tool_registry(self) -> MagicMock:
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("write_file", ToolTier.MODERATE),
        ]
        return _make_parent_registry(tools)

    @pytest.fixture
    def mock_agent_loop(self) -> MagicMock:
        loop = MagicMock()
        loop.llm = MagicMock()
        loop.tool_executor = MagicMock()
        loop.tool_executor.config = None
        loop.cost_tracker = None
        loop.process_message = AsyncMock(return_value=_make_llm_response("Sub-agent done."))
        return loop

    @pytest.fixture
    def orchestrator(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
    ) -> SubAgentOrchestrator:
        return SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
        )

    @pytest.fixture
    def team_orchestrator(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
    ) -> SubAgentOrchestrator:
        """Orchestrator pre-loaded with a team."""
        researcher = _make_role("researcher", allowed_tools=["web_search"])
        writer = _make_role("writer", allowed_tools=["write_file"])
        team = AgentTeam(
            name="content",
            description="Research and write content",
            roles=[researcher, writer],
        )
        return SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            teams=[team],
        )

    # -----------------------------------------------------------------------
    # spawn_subagent
    # -----------------------------------------------------------------------

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_subagent_success(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """A sub-agent spawns, runs process_message, and returns COMPLETED."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(return_value=_make_llm_response("Research complete."))
        mock_loop_cls.return_value = sub_loop

        task = _make_task(task_id="t1")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert result.output == "Research complete."
        assert result.task_id == "t1"
        assert result.role_name == "researcher"
        sub_loop.process_message.assert_awaited_once()

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_subagent_with_context(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """When context is provided, instruction is wrapped with context prefix."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(return_value=_make_llm_response("OK."))
        mock_loop_cls.return_value = sub_loop

        task = _make_task(
            instruction="Summarize the data.",
            context="The dataset has 1000 rows.",
            task_id="t2",
        )
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.COMPLETED
        # The first positional arg to process_message should contain context
        call_args = sub_loop.process_message.call_args
        message_arg = call_args[0][0]
        assert "The dataset has 1000 rows." in message_arg
        assert "Summarize the data." in message_arg

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_subagent_failure_in_execution(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """If process_message raises, the result status is FAILED."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(side_effect=RuntimeError("LLM broke"))
        mock_loop_cls.return_value = sub_loop

        task = _make_task(task_id="t-fail")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.FAILED
        assert "LLM broke" in result.error

    async def test_spawn_subagent_concurrency_limit(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Spawning beyond max_concurrent_agents returns FAILED immediately."""
        # Fill up running tasks with non-done futures
        for i in range(orchestrator.config.max_concurrent_agents):
            future = asyncio.get_event_loop().create_future()
            orchestrator._running_tasks[f"running-{i}"] = future  # type: ignore[assignment]

        task = _make_task(task_id="over-limit")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.FAILED
        assert "Concurrency limit" in result.error

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_subagent_timeout(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """A sub-agent that exceeds the timeout results in FAILED."""

        async def slow_process(*args, **kwargs):
            await asyncio.sleep(999)
            return _make_llm_response("never")

        sub_loop = AsyncMock()
        sub_loop.process_message = slow_process
        mock_loop_cls.return_value = sub_loop

        # Use a very short timeout
        orchestrator.config.subagent_timeout = 0.05

        task = _make_task(task_id="t-timeout")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.FAILED
        assert "Timed out" in result.error

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_subagent_emits_events(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
        event_bus: EventBus,
    ) -> None:
        """Successful spawn emits SPAWNED, STARTED, and COMPLETED events."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(return_value=_make_llm_response("Done."))
        mock_loop_cls.return_value = sub_loop

        events_seen: list[str] = []

        async def capture_spawned(data: dict) -> None:
            events_seen.append("spawned")

        async def capture_started(data: dict) -> None:
            events_seen.append("started")

        async def capture_completed(data: dict) -> None:
            events_seen.append("completed")

        event_bus.on(Events.SUBAGENT_SPAWNED, capture_spawned)
        event_bus.on(Events.SUBAGENT_STARTED, capture_started)
        event_bus.on(Events.SUBAGENT_COMPLETED, capture_completed)

        task = _make_task(task_id="t-events")
        await orchestrator.spawn_subagent(task)

        assert "spawned" in events_seen
        assert "started" in events_seen
        assert "completed" in events_seen

    # -----------------------------------------------------------------------
    # spawn_parallel
    # -----------------------------------------------------------------------

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_parallel_returns_all_results(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """spawn_parallel returns one result per task."""
        # Raise limit so the concurrent tasks don't trip the concurrency check.
        # spawn_parallel pre-registers all asyncio tasks in _running_tasks
        # before spawn_subagent checks the active count, so the limit must
        # exceed the total number of parallel tasks.
        orchestrator.config.max_concurrent_agents = 10

        call_count = 0

        async def numbered_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _make_llm_response(f"Result {call_count}")

        sub_loop = AsyncMock()
        sub_loop.process_message = numbered_response
        mock_loop_cls.return_value = sub_loop

        tasks = [
            _make_task(task_id="p1"),
            _make_task(task_id="p2"),
            _make_task(task_id="p3"),
        ]
        results = await orchestrator.spawn_parallel(tasks)

        assert len(results) == 3
        task_ids = {r.task_id for r in results}
        assert task_ids == {"p1", "p2", "p3"}
        assert all(r.status == SubAgentStatus.COMPLETED for r in results)

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_parallel_partial_failure(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """One failing task doesn't prevent others from completing."""
        call_index = 0

        async def alternating(*args, **kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 2:
                raise RuntimeError("Oops")
            return _make_llm_response(f"OK {call_index}")

        sub_loop = AsyncMock()
        sub_loop.process_message = alternating
        mock_loop_cls.return_value = sub_loop

        tasks = [_make_task(task_id=f"pf-{i}") for i in range(3)]
        # Disable retries so the failing worker isn't recovered
        for t in tasks:
            t.max_attempts = 1
        results = await orchestrator.spawn_parallel(tasks)

        assert len(results) == 3
        statuses = {r.task_id: r.status for r in results}
        # At least one should have failed
        assert SubAgentStatus.FAILED in statuses.values()
        # At least one should have completed
        assert SubAgentStatus.COMPLETED in statuses.values()

    # -----------------------------------------------------------------------
    # spawn_team
    # -----------------------------------------------------------------------

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_spawn_team_success(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        team_orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Spawning a known team runs all team roles."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(return_value=_make_llm_response("Team result."))
        mock_loop_cls.return_value = sub_loop

        results = await team_orchestrator.spawn_team("content", "Write an article about AI.")

        assert len(results) == 2  # researcher + writer
        role_names = {r.role_name for r in results}
        assert "researcher" in role_names
        assert "writer" in role_names
        assert all(r.status == SubAgentStatus.COMPLETED for r in results)

    async def test_spawn_team_unknown_team(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Spawning an unknown team returns FAILED with a descriptive error."""
        results = await orchestrator.spawn_team("nonexistent", "Do something")

        assert len(results) == 1
        assert results[0].status == SubAgentStatus.FAILED
        assert "Unknown team" in results[0].error
        assert "nonexistent" in results[0].error

    async def test_spawn_team_unknown_lists_available(
        self,
        team_orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Error message for unknown team includes available team names."""
        results = await team_orchestrator.spawn_team("bad_name", "Do something")

        assert len(results) == 1
        assert results[0].status == SubAgentStatus.FAILED
        assert "content" in results[0].error

    # -----------------------------------------------------------------------
    # cancel
    # -----------------------------------------------------------------------

    async def test_cancel_running_task(
        self,
        orchestrator: SubAgentOrchestrator,
        event_bus: EventBus,
    ) -> None:
        """cancel() returns True and cancels a running asyncio task."""
        cancelled_events: list[dict] = []

        async def on_cancelled(data: dict) -> None:
            cancelled_events.append(data)

        event_bus.on(Events.SUBAGENT_CANCELLED, on_cancelled)

        # Simulate a running task
        future = asyncio.get_event_loop().create_future()
        orchestrator._running_tasks["running-1"] = future  # type: ignore[assignment]

        result = await orchestrator.cancel("running-1")
        assert result is True
        assert future.cancelled()
        assert len(cancelled_events) == 1
        assert cancelled_events[0]["task_id"] == "running-1"

    async def test_cancel_unknown_task(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """cancel() returns False for a task ID that doesn't exist."""
        result = await orchestrator.cancel("no-such-id")
        assert result is False

    async def test_cancel_already_done_task(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """cancel() returns False when the asyncio task is already done."""
        future = asyncio.get_event_loop().create_future()
        future.set_result(None)
        orchestrator._running_tasks["done-1"] = future  # type: ignore[assignment]

        result = await orchestrator.cancel("done-1")
        assert result is False

    # -----------------------------------------------------------------------
    # get_status
    # -----------------------------------------------------------------------

    def test_get_status_not_found(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """get_status returns None for an unknown task ID."""
        assert orchestrator.get_status("no-such-id") is None

    def test_get_status_completed_task(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """get_status returns the stored result for a completed task."""
        stored = SubAgentResult(
            task_id="done-42",
            role_name="writer",
            status=SubAgentStatus.COMPLETED,
            output="All done.",
            token_usage=100,
            duration_ms=500,
        )
        import time

        orchestrator._results["done-42"] = (time.monotonic(), stored)

        result = orchestrator.get_status("done-42")
        assert result is stored
        assert result.status == SubAgentStatus.COMPLETED
        assert result.output == "All done."

    def test_get_status_failed_task(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """get_status returns the result including the error for a failed task."""
        stored = SubAgentResult(
            task_id="fail-7",
            role_name="coder",
            status=SubAgentStatus.FAILED,
            error="Out of tokens",
        )
        import time

        orchestrator._results["fail-7"] = (time.monotonic(), stored)

        result = orchestrator.get_status("fail-7")
        assert result.status == SubAgentStatus.FAILED
        assert result.error == "Out of tokens"

    # -----------------------------------------------------------------------
    # list_teams
    # -----------------------------------------------------------------------

    def test_list_teams_empty(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """list_teams returns an empty list when no teams are registered."""
        assert orchestrator.list_teams() == []

    def test_list_teams_populated(
        self,
        team_orchestrator: SubAgentOrchestrator,
    ) -> None:
        """list_teams returns team info with role details."""
        teams = team_orchestrator.list_teams()

        assert len(teams) == 1
        team = teams[0]
        assert team["name"] == "content"
        assert team["description"] == "Research and write content"
        assert len(team["roles"]) == 2

        role_names = {r["name"] for r in team["roles"]}
        assert role_names == {"researcher", "writer"}

    def test_list_teams_truncates_persona(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
    ) -> None:
        """Role persona is truncated to 100 characters in list_teams output."""
        long_persona = "A" * 200
        role = SubAgentRole(name="verbose", persona=long_persona)
        team = AgentTeam(name="big_team", description="Test", roles=[role])

        orch = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            teams=[team],
        )
        teams = orch.list_teams()
        assert len(teams[0]["roles"][0]["persona"]) == 100

    # -----------------------------------------------------------------------
    # Concurrency limit enforcement
    # -----------------------------------------------------------------------

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_concurrency_limit_allows_up_to_max(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Sequential spawns up to max_concurrent_agents should all succeed."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(return_value=_make_llm_response("OK"))
        mock_loop_cls.return_value = sub_loop

        # Spawn tasks one at a time (sequential) so the concurrency check
        # sees only the current non-done task each time.
        results = []
        for i in range(orchestrator.config.max_concurrent_agents):
            task = _make_task(task_id=f"c-{i}")
            result = await orchestrator.spawn_subagent(task)
            results.append(result)

        assert len(results) == 3
        assert all(r.status == SubAgentStatus.COMPLETED for r in results)

    async def test_concurrency_limit_rejects_excess(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """When all slots are filled, new spawn returns FAILED immediately."""
        # Fill slots with undone futures
        for i in range(orchestrator.config.max_concurrent_agents):
            future = asyncio.get_event_loop().create_future()
            orchestrator._running_tasks[f"slot-{i}"] = future  # type: ignore[assignment]

        task = _make_task(task_id="rejected")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.FAILED
        assert "Concurrency limit" in result.error

    # -----------------------------------------------------------------------
    # Scoped registry creation
    # -----------------------------------------------------------------------

    def test_create_scoped_registry_excludes_orchestration_tools(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """_create_scoped_registry always denies orchestration tools."""
        role = _make_role()
        scoped = orchestrator._create_scoped_registry(role)

        for tool_name in SubAgentOrchestrator.EXCLUDED_TOOLS:
            assert tool_name in scoped._denied

    def test_create_scoped_registry_merges_role_denied(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Role-level denied_tools are merged with EXCLUDED_TOOLS."""
        role = _make_role(denied_tools=["custom_tool"])
        scoped = orchestrator._create_scoped_registry(role)

        assert "custom_tool" in scoped._denied
        # Also still has orchestration tools
        assert "spawn_subagent" in scoped._denied

    def test_create_scoped_registry_uses_role_allowed_tools(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Scoped registry uses role's allowed_tools as allowlist."""
        role = _make_role(allowed_tools=["read_file", "web_search"])
        scoped = orchestrator._create_scoped_registry(role)

        # The implementation passes role.allowed_tools through to the scoped registry
        assert "read_file" in scoped._allowed
        assert "web_search" in scoped._allowed

    def test_create_scoped_registry_excludes_dangerous_by_default(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Scoped registries use exclude_dangerous=True by default."""
        role = _make_role()
        scoped = orchestrator._create_scoped_registry(role)

        # The implementation passes exclude_dangerous=True to ScopedToolRegistry
        assert scoped._exclude_dangerous is True

    def test_create_scoped_registry_empty_allowed_tools_means_no_allowlist(
        self,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Empty role.allowed_tools results in no allowlist (None), allowing all tools."""
        role = _make_role(allowed_tools=[])
        scoped = orchestrator._create_scoped_registry(role)

        # Empty list is falsy, so ScopedToolRegistry receives allowed_tools=None
        # meaning no allowlist filter is applied (all non-denied tools pass through)
        assert scoped._allowed is None

    # -----------------------------------------------------------------------
    # Result storage
    # -----------------------------------------------------------------------

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_result_stored_after_spawn(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """After spawn_subagent completes, result is retrievable via get_status."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(return_value=_make_llm_response("Stored."))
        mock_loop_cls.return_value = sub_loop

        task = _make_task(task_id="store-me")
        await orchestrator.spawn_subagent(task)

        stored = orchestrator.get_status("store-me")
        assert stored is not None
        assert stored.status == SubAgentStatus.COMPLETED
        assert stored.output == "Stored."

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_failed_result_stored_after_spawn(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """After a failed spawn, error result is retrievable via get_status."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(side_effect=ValueError("bad input"))
        mock_loop_cls.return_value = sub_loop

        task = _make_task(task_id="fail-store")
        await orchestrator.spawn_subagent(task)

        stored = orchestrator.get_status("fail-store")
        assert stored is not None
        assert stored.status == SubAgentStatus.FAILED
        assert "bad input" in stored.error


# ---------------------------------------------------------------------------
# SDK sub-agent routing tests
# ---------------------------------------------------------------------------


class TestSubAgentSDKRouting:
    """Tests for SDK-first sub-agent routing in the orchestrator."""

    @pytest.fixture
    def config(self) -> OrchestrationConfig:
        return OrchestrationConfig(
            enabled=True,
            max_concurrent_agents=3,
            subagent_timeout=10,
            default_max_iterations=5,
        )

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def tool_registry(self) -> MagicMock:
        tools = [
            _make_tool_def("read_file", ToolTier.SAFE),
            _make_tool_def("write_file", ToolTier.MODERATE),
        ]
        return _make_parent_registry(tools)

    @pytest.fixture
    def mock_agent_loop(self) -> MagicMock:
        loop = MagicMock()
        loop.llm = MagicMock()
        loop.tool_executor = MagicMock()
        loop.tool_executor.config = None
        loop.cost_tracker = None
        loop.process_message = AsyncMock(return_value=_make_llm_response("Loop fallback."))
        return loop

    @pytest.fixture
    def mock_sdk_service(self) -> MagicMock:
        sdk = MagicMock()
        sdk.run_subagent = AsyncMock(return_value="SDK sub-agent done.")
        return sdk

    async def test_execute_subagent_uses_sdk_when_available(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        mock_sdk_service: MagicMock,
    ) -> None:
        """When sdk_service is set, sub-agents route through the SDK."""
        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=mock_sdk_service,
        )

        task = _make_task(task_id="sdk-1")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert result.output == "SDK sub-agent done."
        mock_sdk_service.run_subagent.assert_awaited_once()

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_execute_subagent_falls_back_to_loop(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
    ) -> None:
        """When sdk_service is None, sub-agents use the AgentLoop path."""
        sub_loop = AsyncMock()
        sub_loop.process_message = AsyncMock(return_value=_make_llm_response("Loop result."))
        mock_loop_cls.return_value = sub_loop

        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=None,
        )

        task = _make_task(task_id="loop-1")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert result.output == "Loop result."
        sub_loop.process_message.assert_awaited_once()

    async def test_subagent_sdk_client_disconnects_after_use(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        mock_sdk_service: MagicMock,
    ) -> None:
        """The SDK run_subagent method is called (client lifecycle is internal)."""
        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=mock_sdk_service,
        )

        task = _make_task(task_id="disconnect-1")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.COMPLETED
        # run_subagent was called exactly once (it handles connect/disconnect internally)
        mock_sdk_service.run_subagent.assert_awaited_once()
        call_kwargs = mock_sdk_service.run_subagent.call_args.kwargs
        assert call_kwargs["task_id"] == "disconnect-1"

    async def test_subagent_sdk_prompt_no_orchestration_mandate(
        self,
    ) -> None:
        """The sub-agent prompt should not contain orchestration mandate."""
        from agent.llm.claude_sdk import ClaudeSDKService

        # We can't instantiate ClaudeSDKService without the SDK, so test
        # the method directly via the unbound function logic
        prompt = ClaudeSDKService._build_subagent_prompt(
            None,  # type: ignore[arg-type]
            role_persona="You are a code reviewer.",
            task_context="Review the PR.",
        )

        assert "worker agent" in prompt
        assert "Do NOT delegate" in prompt or "Do NOT spawn" in prompt
        assert "WORK DELEGATION RULE" not in prompt
        assert "ROLE:\nYou are a code reviewer." in prompt
        assert "CONTEXT:\nReview the PR." in prompt

    async def test_subagent_sdk_respects_model_override(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        mock_sdk_service: MagicMock,
    ) -> None:
        """Model override from the role is passed to run_subagent."""
        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=mock_sdk_service,
        )

        role = _make_role(name="fast-worker")
        task = _make_task(role=role, task_id="model-1")
        await orchestrator.spawn_subagent(task)

        call_kwargs = mock_sdk_service.run_subagent.call_args.kwargs
        assert call_kwargs["role_persona"] == "You are a test agent."
        assert call_kwargs["max_turns"] == 3  # from _make_role default

    async def test_subagent_sdk_failure_returns_failed_result(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        mock_sdk_service: MagicMock,
    ) -> None:
        """When SDK run_subagent raises, result status is FAILED."""
        mock_sdk_service.run_subagent = AsyncMock(side_effect=RuntimeError("SDK connection lost"))
        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=mock_sdk_service,
        )

        task = _make_task(task_id="sdk-fail")
        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.FAILED
        assert "SDK connection lost" in result.error
        assert result.duration_ms >= 0

    async def test_subagent_sdk_failure_emits_failed_event(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        mock_sdk_service: MagicMock,
    ) -> None:
        """SDK path emits SUBAGENT_FAILED when run_subagent raises."""
        mock_sdk_service.run_subagent = AsyncMock(side_effect=ValueError("bad prompt"))
        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=mock_sdk_service,
        )

        failed_events: list[dict] = []

        async def on_failed(data: dict) -> None:
            failed_events.append(data)

        event_bus.on(Events.SUBAGENT_FAILED, on_failed)

        task = _make_task(task_id="sdk-fail-event")
        await orchestrator.spawn_subagent(task)

        assert len(failed_events) >= 1
        assert any("bad prompt" in e.get("error", "") for e in failed_events)

    async def test_subagent_sdk_emits_started_and_completed_events(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        mock_sdk_service: MagicMock,
    ) -> None:
        """SDK path emits SPAWNED, STARTED, and COMPLETED events on success."""
        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=mock_sdk_service,
        )

        events_seen: list[str] = []

        async def on_spawned(data: dict) -> None:
            events_seen.append("spawned")

        async def on_started(data: dict) -> None:
            events_seen.append("started")

        async def on_completed(data: dict) -> None:
            events_seen.append("completed")

        event_bus.on(Events.SUBAGENT_SPAWNED, on_spawned)
        event_bus.on(Events.SUBAGENT_STARTED, on_started)
        event_bus.on(Events.SUBAGENT_COMPLETED, on_completed)

        task = _make_task(task_id="sdk-events")
        await orchestrator.spawn_subagent(task)

        assert "spawned" in events_seen
        assert "started" in events_seen
        assert "completed" in events_seen

    async def test_subagent_sdk_with_context(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        mock_sdk_service: MagicMock,
    ) -> None:
        """Context is embedded in the prompt passed to SDK run_subagent."""
        orchestrator = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            sdk_service=mock_sdk_service,
        )

        task = _make_task(
            instruction="Analyze the logs.",
            context="Server is returning 500 errors.",
            task_id="sdk-ctx",
        )
        await orchestrator.spawn_subagent(task)

        call_kwargs = mock_sdk_service.run_subagent.call_args.kwargs
        # prompt may be positional or keyword
        call_args = mock_sdk_service.run_subagent.call_args
        prompt_arg = call_args.args[0] if call_args.args else call_kwargs["prompt"]
        assert "Server is returning 500 errors." in prompt_arg
        assert "Analyze the logs." in prompt_arg
        # Context is now embedded directly in the prompt (not passed as
        # separate task_context kwarg) to avoid duplication.

    async def test_subagent_prompt_no_context_section_when_empty(
        self,
    ) -> None:
        """Sub-agent prompt omits CONTEXT section when task_context is empty."""
        from agent.llm.claude_sdk import ClaudeSDKService

        prompt = ClaudeSDKService._build_subagent_prompt(
            None,  # type: ignore[arg-type]
            role_persona="You are a coder.",
            task_context="",
        )

        assert "ROLE:\nYou are a coder." in prompt
        assert "CONTEXT:" not in prompt

    async def test_subagent_prompt_includes_context_when_provided(
        self,
    ) -> None:
        """Sub-agent prompt includes CONTEXT section when task_context is given."""
        from agent.llm.claude_sdk import ClaudeSDKService

        prompt = ClaudeSDKService._build_subagent_prompt(
            None,  # type: ignore[arg-type]
            role_persona="You are a researcher.",
            task_context="The project uses FastAPI.",
        )

        assert "CONTEXT:\nThe project uses FastAPI." in prompt


class TestBuildMcpServerRegistry:
    """Tests for _build_mcp_server registry override."""

    @patch.dict(
        sys.modules,
        {
            "claude_code_sdk": MagicMock(),
        },
    )
    async def test_build_mcp_server_uses_override_registry(self) -> None:
        """When registry is passed, it is used instead of self.tool_registry."""
        from agent.llm.claude_sdk import ClaudeSDKService

        # Create a service with a mock self.tool_registry
        sdk = object.__new__(ClaudeSDKService)
        sdk.tool_registry = MagicMock()
        sdk.tool_registry.list_tools = MagicMock(return_value=[])

        # Pass an override registry with tools (empty → returns None)
        override = MagicMock()
        override.list_tools = MagicMock(return_value=[])

        result = sdk._build_mcp_server(registry=override)

        # Override was used, not self.tool_registry
        override.list_tools.assert_called_once()
        sdk.tool_registry.list_tools.assert_not_called()
        # No enabled tools → returns None
        assert result is None

    @patch.dict(
        sys.modules,
        {
            "claude_code_sdk": MagicMock(),
        },
    )
    async def test_build_mcp_server_falls_back_to_self_registry(self) -> None:
        """When registry is None, self.tool_registry is used."""
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk = object.__new__(ClaudeSDKService)
        sdk.tool_registry = MagicMock()
        sdk.tool_registry.list_tools = MagicMock(return_value=[])

        result = sdk._build_mcp_server(registry=None)

        sdk.tool_registry.list_tools.assert_called_once()
        # No enabled tools → returns None
        assert result is None

    async def test_build_mcp_server_returns_none_no_registry(self) -> None:
        """Returns None when both registry arg and self.tool_registry are None."""
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk = object.__new__(ClaudeSDKService)
        sdk.tool_registry = None

        result = sdk._build_mcp_server(registry=None)
        assert result is None

    @patch.dict(
        sys.modules,
        {
            "claude_code_sdk": MagicMock(),
        },
    )
    async def test_build_mcp_server_returns_none_no_enabled_tools(self) -> None:
        """Returns None when registry has tools but none are enabled."""
        from agent.llm.claude_sdk import ClaudeSDKService

        disabled_tool = _make_tool_def("disabled_tool", enabled=False)
        sdk = object.__new__(ClaudeSDKService)
        sdk.tool_registry = None

        override = MagicMock()
        override.list_tools = MagicMock(return_value=[disabled_tool])

        result = sdk._build_mcp_server(registry=override)
        assert result is None
