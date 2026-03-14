"""Sub-agent orchestrator — spawns and manages concurrent sub-agents.

Implements the orchestrator pattern: main agent spawns sub-agents with
scoped tools, configurable personas, and independent sessions.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import Events
from agent.core.session import Session
from agent.core.subagent import (
    AgentTeam,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
    SubAgentTask,
)

if TYPE_CHECKING:
    from agent.config import OrchestrationConfig
    from agent.core.agent_loop import AgentLoop
    from agent.core.events import EventBus
    from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class SubAgentOrchestrator:
    """Spawns and manages concurrent sub-agents.

    Sub-agents get scoped tool registries, independent sessions,
    and configurable personas. Orchestration tools are always excluded
    from sub-agents to prevent recursive spawning.
    """

    # Tools that sub-agents are never allowed to use
    EXCLUDED_TOOLS = {
        "spawn_subagent",
        "spawn_parallel_agents",
        "spawn_team",
        "list_agent_teams",
        "get_subagent_status",
        "cancel_subagent",
    }

    def __init__(
        self,
        agent_loop: AgentLoop,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        teams: list[AgentTeam] | None = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.config = config
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self.teams = {t.name: t for t in (teams or [])}

        self._running_tasks: dict[str, asyncio.Task[SubAgentResult]] = {}
        self._results: dict[str, SubAgentResult] = {}

    async def spawn_subagent(self, task: SubAgentTask) -> SubAgentResult:
        """Spawn a single sub-agent and wait for its result.

        Creates a scoped session and tool registry, builds a sub-agent
        prompt, and runs the agent loop with constraints.

        Args:
            task: The sub-agent task to execute.

        Returns:
            SubAgentResult with output or error.
        """
        # Check concurrency limit
        active = sum(1 for t in self._running_tasks.values() if not t.done())
        if active >= self.config.max_concurrent_agents:
            return SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.FAILED,
                error=f"Concurrency limit reached ({self.config.max_concurrent_agents})",
            )

        await self.event_bus.emit(Events.SUBAGENT_SPAWNED, {
            "task_id": task.task_id,
            "role": task.role.name,
            "instruction": task.instruction[:200],
        })

        # Track the running task for concurrency enforcement
        exec_task = asyncio.ensure_future(self._execute_subagent(task))
        self._running_tasks[task.task_id] = exec_task

        # Run with timeout
        result: SubAgentResult
        try:
            result = await asyncio.wait_for(
                exec_task,
                timeout=self.config.subagent_timeout,
            )
        except TimeoutError:
            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.FAILED,
                error=f"Timed out after {self.config.subagent_timeout}s",
            )
            await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                "task_id": task.task_id,
                "error": result.error,
            })
        except asyncio.CancelledError:
            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.CANCELLED,
            )
            await self.event_bus.emit(Events.SUBAGENT_CANCELLED, {
                "task_id": task.task_id,
            })
        except Exception as e:
            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.FAILED,
                error=f"Unexpected error: {e}",
            )
            await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                "task_id": task.task_id,
                "error": result.error,
            })
        finally:
            self._running_tasks.pop(task.task_id, None)

        self._results[task.task_id] = result
        return result

    async def spawn_parallel(self, tasks: list[SubAgentTask]) -> list[SubAgentResult]:
        """Spawn multiple sub-agents concurrently.

        Args:
            tasks: List of sub-agent tasks to execute in parallel.

        Returns:
            List of SubAgentResults in the same order as tasks.
        """
        # Wrap each in an asyncio Task for concurrent execution
        # Note: don't pre-add to _running_tasks to avoid double-counting
        # in spawn_subagent's concurrency check
        async_tasks: list[asyncio.Task[SubAgentResult]] = []
        for task in tasks:
            atask = asyncio.create_task(self.spawn_subagent(task))
            async_tasks.append(atask)

        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        final: list[SubAgentResult] = []
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                final.append(SubAgentResult(
                    task_id=tasks[i].task_id,
                    role_name=tasks[i].role.name,
                    status=SubAgentStatus.FAILED,
                    error=str(r),
                ))
            else:
                final.append(r)

        return final

    async def spawn_team(
        self,
        team_name: str,
        instruction: str,
        context: str = "",
        parent_session_id: str = "",
    ) -> list[SubAgentResult]:
        """Spawn a pre-defined team of sub-agents.

        Args:
            team_name: Name of the registered team.
            instruction: Task instruction for all team members.
            context: Shared context for the team.
            parent_session_id: Parent session ID.

        Returns:
            List of results from all team members.
        """
        team = self.teams.get(team_name)
        if not team:
            available = ", ".join(self.teams.keys()) or "none"
            return [SubAgentResult(
                task_id="",
                role_name="",
                status=SubAgentStatus.FAILED,
                error=f"Unknown team: '{team_name}'. Available: {available}",
            )]

        tasks = [
            SubAgentTask(
                role=role,
                instruction=instruction,
                context=context,
                parent_session_id=parent_session_id,
            )
            for role in team.roles
        ]

        return await self.spawn_parallel(tasks)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running sub-agent.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if cancelled, False if not found.
        """
        atask = self._running_tasks.get(task_id)
        if atask and not atask.done():
            atask.cancel()
            await self.event_bus.emit(Events.SUBAGENT_CANCELLED, {
                "task_id": task_id,
            })
            return True
        return False

    def get_status(self, task_id: str) -> SubAgentResult | None:
        """Get the status/result of a sub-agent task.

        Args:
            task_id: The task ID to query.

        Returns:
            SubAgentResult if found, None otherwise.
        """
        return self._results.get(task_id)

    def list_teams(self) -> list[dict[str, Any]]:
        """List all registered teams.

        Returns:
            List of team info dicts.
        """
        return [
            {
                "name": team.name,
                "description": team.description,
                "roles": [
                    {
                        "name": r.name,
                        "persona": r.persona[:100],
                        "allowed_tools": r.allowed_tools,
                        "max_iterations": r.max_iterations,
                    }
                    for r in team.roles
                ],
            }
            for team in self.teams.values()
        ]

    async def _execute_subagent(self, task: SubAgentTask) -> SubAgentResult:
        """Run a sub-agent task to completion.

        Creates a lightweight AgentLoop with a scoped tool registry
        and processes the instruction.
        """
        from agent.core.agent_loop import AgentLoop
        from agent.tools.executor import ToolExecutor

        start = asyncio.get_event_loop().time()

        await self.event_bus.emit(Events.SUBAGENT_STARTED, {
            "task_id": task.task_id,
            "role": task.role.name,
        })

        # Create scoped registry
        scoped_registry = self._create_scoped_registry(task.role)

        # Create a minimal config for the sub-agent
        from agent.config import AgentPersonaConfig

        sub_config = AgentPersonaConfig(
            name=task.role.name,
            persona=task.role.persona,
            max_iterations=task.role.max_iterations,
        )

        # Create tool executor with scoped registry, reusing parent's safety components
        parent_executor = self.agent_loop.tool_executor
        assert parent_executor is not None, "Parent executor must be set"
        sub_executor = ToolExecutor(
            registry=scoped_registry,  # type: ignore[arg-type]
            config=parent_executor.config,
            event_bus=self.event_bus,
            audit=parent_executor.audit,
            permissions=parent_executor.permissions,
            guardrails=parent_executor.guardrails,
        )

        # Create sub-agent loop (lightweight, no memory/planning)
        sub_loop = AgentLoop(
            llm=self.agent_loop.llm,
            config=sub_config,
            event_bus=self.event_bus,
            tool_executor=sub_executor,
            cost_tracker=self.agent_loop.cost_tracker,
        )

        # Build instruction with context
        full_instruction = task.instruction
        if task.context:
            full_instruction = f"Context:\n{task.context}\n\nTask:\n{task.instruction}"

        # Create session
        session = Session(
            session_id=f"subagent:{task.parent_session_id}:{task.task_id}"
        )

        try:
            response = await sub_loop.process_message(
                full_instruction, session, trigger="subagent"
            )

            duration_ms = int((asyncio.get_event_loop().time() - start) * 1000)

            # Count tool calls and tokens
            tool_calls_made = sum(
                1 for m in session.messages
                if m.tool_calls
                for _ in m.tool_calls
            )
            total_tokens = sum(
                (m.usage.total_tokens if m.usage else 0)
                for m in session.messages
            )
            iterations = sum(
                1 for m in session.messages if m.role == "assistant"
            )

            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.COMPLETED,
                output=response.content,
                token_usage=total_tokens,
                duration_ms=duration_ms,
                tool_calls_made=tool_calls_made,
                iterations=iterations,
            )

            await self.event_bus.emit(Events.SUBAGENT_COMPLETED, {
                "task_id": task.task_id,
                "role": task.role.name,
                "tokens": total_tokens,
                "duration_ms": duration_ms,
            })

            return result

        except Exception as e:
            duration_ms = int((asyncio.get_event_loop().time() - start) * 1000)
            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )

            await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                "task_id": task.task_id,
                "error": str(e),
            })

            return result

    def _create_scoped_registry(self, role: SubAgentRole) -> ScopedToolRegistry:
        """Create a filtered copy of the tool registry for a sub-agent.

        Applies allow/deny lists and always excludes orchestration tools
        and dangerous-tier tools.
        """
        return ScopedToolRegistry(
            parent=self.tool_registry,
            allowed_tools=role.allowed_tools or None,
            denied_tools=set(role.denied_tools) | self.EXCLUDED_TOOLS,
            exclude_dangerous=True,
        )


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
        return [
            s for s in all_schemas
            if self._is_tool_allowed(s["function"]["name"])
        ]

    def list_tools(self) -> list[Any]:
        """List all allowed tools."""
        return [
            t for t in self._parent.list_tools()
            if self._is_tool_allowed(t.name)
        ]

    def enable_tool(self, name: str) -> None:
        """Enable a tool (delegates to parent)."""
        if self._is_tool_allowed(name):
            self._parent.enable_tool(name)

    def disable_tool(self, name: str) -> None:
        """Disable a tool (delegates to parent)."""
        self._parent.disable_tool(name)

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool (delegates to parent)."""
        self._parent.unregister_tool(name)
