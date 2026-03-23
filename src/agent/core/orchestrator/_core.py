"""Sub-agent orchestrator — spawns and manages concurrent sub-agents.

Implements the orchestrator pattern: main agent spawns sub-agents with
scoped tools, configurable personas, and independent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import Events
from agent.core.orchestrator._scoped_registry import ScopedToolRegistry
from agent.core.session import Session
from agent.core.subagent import (
    AgentTeam,
    Project,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
    SubAgentTask,
)

if TYPE_CHECKING:
    from agent.config import OrchestrationConfig
    from agent.core.agent_loop import AgentLoop
    from agent.core.events import EventBus
    from agent.llm.claude_sdk import ClaudeSDKService
    from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

# Hard limit on sub-agent nesting to prevent runaway recursion.
MAX_NESTING_DEPTH = 3


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
        "run_project",
        "list_projects",
        "assign_work",
        "check_work_status",
        "direct_controller",
    }

    # Tools blocked at nesting depth >= 1 (prevents recursive consult/delegation)
    NESTED_EXCLUDED_TOOLS = {
        "consult_agent",
        "delegate_to_specialist",
    }

    def __init__(
        self,
        agent_loop: AgentLoop,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        teams: list[AgentTeam] | None = None,
        sdk_service: ClaudeSDKService | None = None,
        task_board: Any | None = None,
    ) -> None:
        self.agent_loop = agent_loop
        self.config = config
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self.sdk_service = sdk_service
        self.task_board = task_board
        self.teams = {t.name: t for t in (teams or [])}
        self.projects: dict[str, Project] = {}

        # Inter-agent message bus
        from agent.core.message_bus import MessageBus

        self.message_bus = MessageBus(event_bus=self.event_bus)

        self._running_tasks: dict[str, asyncio.Task[SubAgentResult]] = {}
        self._results: dict[str, tuple[float, SubAgentResult]] = {}
        self._async_futures: dict[str, asyncio.Future[SubAgentResult]] = {}
        self._task_nesting_depths: dict[str, int] = {}
        self._spawn_lock = asyncio.Lock()
        self._max_results = 500  # prune _results after this many entries

    async def spawn_subagent(self, task: SubAgentTask) -> SubAgentResult:
        """Spawn a single sub-agent and wait for its result.

        Delegates to _single_spawn_attempt for the actual execution.
        Use _run_worker_with_retry for automatic retry with backoff.

        Args:
            task: The sub-agent task to execute.

        Returns:
            SubAgentResult with output or error.
        """
        return await self._single_spawn_attempt(task)

    async def _single_spawn_attempt(
        self,
        task: SubAgentTask,
    ) -> SubAgentResult:
        """Execute a single attempt to run a sub-agent task.

        This is the atomic operation that _run_worker_with_retry calls
        repeatedly. Handles concurrency limits, timeout, execution,
        and result storage — but no retry logic.

        Args:
            task: The sub-agent task to execute.

        Returns:
            SubAgentResult with output or error.
        """
        # Hard nesting limit to prevent runaway recursion.
        if task.nesting_depth >= MAX_NESTING_DEPTH:
            logger.warning(
                "spawn_subagent_nesting_limit",
                task_id=task.task_id,
                nesting_depth=task.nesting_depth,
                max_nesting_depth=MAX_NESTING_DEPTH,
            )
            return SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.FAILED,
                error=(
                    f"Maximum nesting depth exceeded ({task.nesting_depth} >= "
                    f"{MAX_NESTING_DEPTH}). Sub-agents cannot spawn beyond this limit."
                ),
            )

        # Serialize concurrency check + task registration to prevent
        # parallel spawns from all passing the check simultaneously.
        async with self._spawn_lock:
            active = sum(1 for t in self._running_tasks.values() if not t.done())
            if active >= self.config.max_concurrent_agents:
                return SubAgentResult(
                    task_id=task.task_id,
                    role_name=task.role.name,
                    status=SubAgentStatus.FAILED,
                    error=f"Concurrency limit reached ({self.config.max_concurrent_agents})",
                )

            # Track the running task for concurrency enforcement.
            # Each sub-agent runs in a copied context so parallel tasks
            # get independent ContextVar values (e.g. nesting_depth).
            import contextvars

            ctx = contextvars.copy_context()
            exec_task = asyncio.get_running_loop().create_task(
                self._execute_subagent(task),
                context=ctx,
            )
            self._running_tasks[task.task_id] = exec_task

        await self.event_bus.emit(
            Events.SUBAGENT_SPAWNED,
            {
                "task_id": task.task_id,
                "role": task.role.name,
                "instruction": task.instruction[:200],
                "parent_session_id": task.parent_session_id or "",
            },
        )

        # Run with timeout
        result: SubAgentResult
        try:
            result = await asyncio.wait_for(
                asyncio.shield(exec_task),
                timeout=self.config.subagent_timeout,
            )
        except TimeoutError:
            exec_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(exec_task, timeout=5.0)
            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.FAILED,
                error=f"Timed out after {self.config.subagent_timeout}s",
            )
            await self.event_bus.emit(
                Events.SUBAGENT_FAILED,
                {
                    "task_id": task.task_id,
                    "error": result.error,
                },
            )
        except asyncio.CancelledError:
            exec_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(exec_task, timeout=5.0)
            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.CANCELLED,
            )
            with contextlib.suppress(Exception):
                await self.event_bus.emit(
                    Events.SUBAGENT_CANCELLED,
                    {
                        "task_id": task.task_id,
                    },
                )
        except Exception as e:
            result = SubAgentResult(
                task_id=task.task_id,
                role_name=task.role.name,
                status=SubAgentStatus.FAILED,
                error=f"Unexpected error: {e}",
            )
            await self.event_bus.emit(
                Events.SUBAGENT_FAILED,
                {
                    "task_id": task.task_id,
                    "error": result.error,
                },
            )
        finally:
            self._running_tasks.pop(task.task_id, None)

        self._results[task.task_id] = (time.monotonic(), result)

        # Prune old results to prevent unbounded memory growth
        if len(self._results) > self._max_results:
            excess = len(self._results) - self._max_results
            oldest_keys = sorted(
                self._results,
                key=lambda k: self._results[k][0],
            )[:excess]
            for old_key in oldest_keys:
                self._results.pop(old_key, None)

        return result

    async def _run_worker_with_retry(
        self,
        task: SubAgentTask,
    ) -> SubAgentResult:
        """Run a sub-agent task with retry and exponential backoff.

        Attempts up to ``task.max_attempts`` times.  Retries on FAILED
        status; stops immediately on COMPLETED or CANCELLED.

        Emits:
            WORKER_RETRYING:  between failed attempts (with attempt# and delay)
            WORKER_SUCCEEDED: on successful completion
            WORKER_FAILED:    when all attempts exhausted
        """
        last_result: SubAgentResult | None = None

        for attempt in range(1, task.max_attempts + 1):
            result = await self._single_spawn_attempt(task)
            last_result = result

            if result.status == SubAgentStatus.COMPLETED:
                await self.event_bus.emit(
                    Events.WORKER_SUCCEEDED,
                    {
                        "task_id": task.task_id,
                        "role": task.role.name,
                        "attempt": attempt,
                    },
                )
                return result

            if result.status == SubAgentStatus.CANCELLED:
                return result

            # FAILED — retry if attempts remain
            if attempt < task.max_attempts:
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, …
                await self.event_bus.emit(
                    Events.WORKER_RETRYING,
                    {
                        "task_id": task.task_id,
                        "role": task.role.name,
                        "attempt": attempt,
                        "retry_in_seconds": delay,
                        "error": result.error,
                    },
                )
                logger.info(
                    "worker_retrying",
                    task_id=task.task_id,
                    role=task.role.name,
                    attempt=attempt,
                    retry_in=delay,
                )
                await asyncio.sleep(delay)

        # All attempts exhausted
        assert last_result is not None  # max_attempts >= 1 guarantees this
        await self.event_bus.emit(
            Events.WORKER_FAILED,
            {
                "task_id": task.task_id,
                "role": task.role.name,
                "error": last_result.error,
                "attempts": task.max_attempts,
                "critical": task.critical,
            },
        )
        return last_result

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
                final.append(
                    SubAgentResult(
                        task_id=tasks[i].task_id,
                        role_name=tasks[i].role.name,
                        status=SubAgentStatus.FAILED,
                        error=str(r),
                    )
                )
            else:
                final.append(r)

        return final

    async def spawn_team(
        self,
        team_name: str,
        instruction: str,
        context: str = "",
        parent_session_id: str = "",
        max_rounds: int = 0,
    ) -> list[SubAgentResult]:
        """Spawn a pre-defined team of sub-agents.

        Args:
            team_name: Name of the registered team.
            instruction: Task instruction for all team members.
            context: Shared context for the team.
            parent_session_id: Parent session ID.
            max_rounds: If > 0, delegates to run_iterative_team for
                multi-round collaboration.

        Returns:
            List of results from all team members.
        """
        team = self.teams.get(team_name)
        if not team:
            available = ", ".join(self.teams.keys()) or "none"
            return [
                SubAgentResult(
                    task_id="",
                    role_name="",
                    status=SubAgentStatus.FAILED,
                    error=f"Unknown team: '{team_name}'. Available: {available}",
                )
            ]

        tasks = [
            SubAgentTask(
                role=role,
                instruction=instruction,
                context=context,
                parent_session_id=parent_session_id,
            )
            for role in team.roles
        ]

        if max_rounds > 0:
            from uuid import uuid4

            task_id = f"team-{uuid4().hex[:8]}"
            result = await self.run_iterative_team(
                task_id=task_id,
                team=tasks,
                max_rounds=max_rounds,
            )
            return result["results"]

        return await self.spawn_parallel(tasks)

    async def run_iterative_team(
        self,
        task_id: str,
        team: list[SubAgentTask],
        max_rounds: int,
    ) -> dict[str, Any]:
        """Run a team iteratively over multiple rounds.

        Each round spawns tasks in parallel. After each round, checks the
        task_board for pending work. If no pending tasks remain, the team
        finishes successfully. If max_rounds is reached or a cycle is
        detected (3 consecutive rounds with the same set of roles running),
        the team stops.

        Args:
            task_id: Parent orchestration task ID.
            team: List of SubAgentTasks representing the team members.
            max_rounds: Maximum number of rounds to execute.

        Returns:
            Dict with keys: success, rounds_completed, results.
        """
        all_results: list[SubAgentResult] = []
        previous_role_sets: list[frozenset[str]] = []
        rounds_completed = 0

        # Set all team members' task_id to the parent task_id so board
        # operations (post_task, get_my_tasks, has_pending) share scope.
        for t in team:
            t.task_id = task_id

        # Build a lookup of role_name -> task for the team
        role_task_map: dict[str, SubAgentTask] = {}
        for t in team:
            role_task_map[t.role.name] = t

        all_role_names = frozenset(role_task_map.keys())

        for round_num in range(max_rounds):
            await self.event_bus.emit(
                Events.ROUND_STARTED,
                {
                    "task_id": task_id,
                    "round": round_num,
                },
            )

            # Determine which roles to run this round
            if round_num == 0:
                # First round: run all roles
                roles_to_run = all_role_names
            elif self.task_board is not None:
                # Subsequent rounds: run roles that have pending tasks
                roles_with_pending = await self.task_board.get_roles_with_pending(task_id)
                # Only run roles that are in our team
                roles_to_run = frozenset(roles_with_pending & all_role_names)
                if not roles_to_run:
                    # No pending work for any team member — done
                    rounds_completed = round_num
                    await self.event_bus.emit(
                        Events.ROUND_COMPLETED,
                        {
                            "task_id": task_id,
                            "round": round_num,
                        },
                    )
                    break
            else:
                # No task board — no way to determine pending work, stop
                rounds_completed = round_num
                await self.event_bus.emit(
                    Events.ROUND_COMPLETED,
                    {
                        "task_id": task_id,
                        "round": round_num,
                    },
                )
                break

            # Cycle detection: if 3 consecutive rounds have same roles
            previous_role_sets.append(roles_to_run)
            if len(previous_role_sets) >= 3:
                last_three = previous_role_sets[-3:]
                if last_three[0] == last_three[1] == last_three[2]:
                    rounds_completed = round_num + 1
                    # Run this round before breaking (cycle detected after)
                    round_tasks = [role_task_map[r] for r in roles_to_run]
                    round_results = await self._run_round(round_tasks)
                    all_results.extend(round_results)
                    await self.event_bus.emit(
                        Events.ROUND_COMPLETED,
                        {
                            "task_id": task_id,
                            "round": round_num,
                        },
                    )
                    # Emit team finished with cycle
                    has_pending = False
                    if self.task_board is not None:
                        has_pending = await self.task_board.has_pending(task_id)
                    await self.event_bus.emit(
                        Events.TEAM_FINISHED,
                        {
                            "task_id": task_id,
                            "rounds_completed": rounds_completed,
                            "success": not has_pending,
                            "reason": "cycle_detected",
                        },
                    )
                    return {
                        "success": not has_pending,
                        "rounds_completed": rounds_completed,
                        "results": all_results,
                    }

            # Run this round's tasks
            round_tasks = [role_task_map[r] for r in roles_to_run]
            round_results = await self._run_round(round_tasks)
            all_results.extend(round_results)
            rounds_completed = round_num + 1

            await self.event_bus.emit(
                Events.ROUND_COMPLETED,
                {
                    "task_id": task_id,
                    "round": round_num,
                },
            )

            # Check if there are still pending tasks
            if self.task_board is not None:
                has_pending = await self.task_board.has_pending(task_id)
                if not has_pending:
                    break
            else:
                # No board means we can't check — one round is enough
                break

        # Determine final status
        has_pending = False
        if self.task_board is not None:
            has_pending = await self.task_board.has_pending(task_id)

        success = not has_pending

        await self.event_bus.emit(
            Events.TEAM_FINISHED,
            {
                "task_id": task_id,
                "rounds_completed": rounds_completed,
                "success": success,
            },
        )

        return {
            "success": success,
            "rounds_completed": rounds_completed,
            "results": all_results,
        }

    async def _run_round(
        self,
        tasks: list[SubAgentTask],
    ) -> list[SubAgentResult]:
        """Run a single round of tasks, catching exceptions per-task.

        Unlike spawn_parallel (which uses spawn_subagent with concurrency
        checks), this directly calls _run_worker_with_retry and handles
        exceptions so one failure doesn't cancel siblings.
        """

        async def _safe_run(task: SubAgentTask) -> SubAgentResult:
            try:
                return await self._run_worker_with_retry(task)
            except Exception as e:
                return SubAgentResult(
                    task_id=task.task_id,
                    role_name=task.role.name,
                    status=SubAgentStatus.FAILED,
                    error=str(e),
                )

        coros = [_safe_run(t) for t in tasks]
        results = await asyncio.gather(*coros)
        return list(results)

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
            await self.event_bus.emit(
                Events.SUBAGENT_CANCELLED,
                {
                    "task_id": task_id,
                },
            )
            return True
        # Also check async futures from fire-and-forget delegations
        afut = self._async_futures.get(task_id)
        if afut and not afut.done():
            afut.cancel()
            await self.event_bus.emit(
                Events.SUBAGENT_CANCELLED,
                {
                    "task_id": task_id,
                },
            )
            return True
        return False

    async def shutdown(self) -> None:
        """Cancel all running tasks and async futures, clean up resources."""
        # Cancel running tasks
        for task_id, atask in list(self._running_tasks.items()):
            if not atask.done():
                atask.cancel()
                logger.info("shutdown_cancel_task", task_id=task_id)
        # Cancel async futures
        for task_id, fut in list(self._async_futures.items()):
            if not fut.done():
                fut.cancel()
                logger.info("shutdown_cancel_future", task_id=task_id)
        # Wait briefly for everything to finish
        all_pending = [t for t in list(self._running_tasks.values()) if not t.done()] + [
            f for f in list(self._async_futures.values()) if not f.done()
        ]
        if all_pending:
            await asyncio.gather(*all_pending, return_exceptions=True)
        self._running_tasks.clear()
        self._async_futures.clear()
        logger.info("orchestrator_shutdown_complete")

    def get_status(self, task_id: str) -> SubAgentResult | None:
        """Get the status/result of a sub-agent task.

        Args:
            task_id: The task ID to query.

        Returns:
            SubAgentResult if found, None otherwise.
        """
        entry = self._results.get(task_id)
        if entry is None:
            return None
        return entry[1]

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

    def list_projects(self) -> list[dict[str, Any]]:
        """List all registered projects.

        Returns:
            List of project info dicts.
        """
        return [
            {
                "name": proj.name,
                "description": proj.description,
                "stages": [
                    {
                        "name": s.name,
                        "agents": [{"team": a.team, "role": a.role} for a in s.agents],
                    }
                    for s in proj.stages
                ],
            }
            for proj in self.projects.values()
        ]

    async def run_channel_task(
        self,
        prompt: str,
        task_id: str,
        session: Session,
        *,
        on_progress: Any | None = None,
        on_permission: Any | None = None,
    ) -> SubAgentResult:
        """Run a user message as an orchestrated task.

        This is the entry point for channels (Telegram, webchat) to dispatch
        work through the orchestrator.  If an SDK service is available, it
        routes through the SDK; otherwise it falls back to the agent loop.

        The task is tracked, has timeouts, and can be cancelled via
        ``cancel(task_id)``.

        Args:
            prompt: The user message to process.
            task_id: Unique ID for this task (e.g. session ID).
            session: The conversation session.
            on_progress: Optional async callback ``(event) -> None`` for
                streaming progress (tool use, text chunks).
            on_permission: Optional async callback for tool approval.

        Returns:
            SubAgentResult with the response text or error.
        """
        # Serialize concurrency check + task registration under lock
        async with self._spawn_lock:
            active = sum(1 for t in self._running_tasks.values() if not t.done())
            if active >= self.config.max_concurrent_agents:
                return SubAgentResult(
                    task_id=task_id,
                    role_name="channel",
                    status=SubAgentStatus.FAILED,
                    error=f"Too many tasks running ({active}). " f"Use /stop to cancel some.",
                )

            exec_coro = self._execute_channel_task(
                prompt,
                task_id,
                session,
                on_progress=on_progress,
                on_permission=on_permission,
            )
            import contextvars

            ctx = contextvars.copy_context()
            exec_task = asyncio.get_running_loop().create_task(
                exec_coro,
                context=ctx,
            )
            self._running_tasks[task_id] = exec_task

        await self.event_bus.emit(
            Events.SUBAGENT_SPAWNED,
            {
                "task_id": task_id,
                "role": "channel",
                "instruction": prompt[:200],
            },
        )

        result: SubAgentResult
        try:
            result = await asyncio.wait_for(
                asyncio.shield(exec_task),
                timeout=self.config.subagent_timeout,
            )
        except TimeoutError:
            exec_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(exec_task, timeout=5.0)
            result = SubAgentResult(
                task_id=task_id,
                role_name="channel",
                status=SubAgentStatus.FAILED,
                error=f"Task timed out after {self.config.subagent_timeout}s.",
            )
            with contextlib.suppress(Exception):
                await self.event_bus.emit(
                    Events.SUBAGENT_FAILED,
                    {
                        "task_id": task_id,
                        "error": result.error,
                    },
                )
        except asyncio.CancelledError:
            exec_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await exec_task
            result = SubAgentResult(
                task_id=task_id,
                role_name="channel",
                status=SubAgentStatus.CANCELLED,
            )
            with contextlib.suppress(Exception):
                await self.event_bus.emit(
                    Events.SUBAGENT_CANCELLED,
                    {
                        "task_id": task_id,
                    },
                )
        except Exception as e:
            result = SubAgentResult(
                task_id=task_id,
                role_name="channel",
                status=SubAgentStatus.FAILED,
                error=f"Unexpected error: {e}",
            )
            with contextlib.suppress(Exception):
                await self.event_bus.emit(
                    Events.SUBAGENT_FAILED,
                    {
                        "task_id": task_id,
                        "error": result.error,
                    },
                )
        finally:
            self._running_tasks.pop(task_id, None)
            self._task_nesting_depths.pop(task_id, None)

        self._results[task_id] = (time.monotonic(), result)

        # Prune old results to prevent unbounded memory growth
        if len(self._results) > self._max_results:
            excess = len(self._results) - self._max_results
            oldest_keys = sorted(
                self._results,
                key=lambda k: self._results[k][0],
            )[:excess]
            for old_key in oldest_keys:
                self._results.pop(old_key, None)

        return result

    async def _execute_channel_task(
        self,
        prompt: str,
        task_id: str,
        session: Session,
        *,
        on_progress: Any | None = None,
        on_permission: Any | None = None,
    ) -> SubAgentResult:
        """Execute a channel task via SDK or agent loop."""
        import time as _time

        start = _time.monotonic()

        await self.event_bus.emit(
            Events.SUBAGENT_STARTED,
            {
                "task_id": task_id,
                "role": "channel",
            },
        )

        try:
            # Prefer SDK path when available
            if self.sdk_service is not None:
                response_text = await self._execute_via_sdk(
                    prompt,
                    task_id,
                    session,
                    on_progress=on_progress,
                    on_permission=on_permission,
                )
            else:
                response = await self.agent_loop.process_message(
                    prompt, session, trigger="user_message"
                )
                response_text = response.content

            duration_ms = int((_time.monotonic() - start) * 1000)

            result = SubAgentResult(
                task_id=task_id,
                role_name="channel",
                status=SubAgentStatus.COMPLETED,
                output=response_text,
                duration_ms=duration_ms,
            )

            await self.event_bus.emit(
                Events.SUBAGENT_COMPLETED,
                {
                    "task_id": task_id,
                    "role": "channel",
                    "duration_ms": duration_ms,
                },
            )

            return result

        except asyncio.CancelledError:
            raise  # let outer handler deal with it
        except Exception as e:
            duration_ms = int((_time.monotonic() - start) * 1000)
            result = SubAgentResult(
                task_id=task_id,
                role_name="channel",
                status=SubAgentStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms,
            )
            await self.event_bus.emit(
                Events.SUBAGENT_FAILED,
                {
                    "task_id": task_id,
                    "error": str(e),
                },
            )
            return result

    async def _execute_via_sdk(
        self,
        prompt: str,
        task_id: str,
        session: Session,
        *,
        on_progress: Any | None = None,
        on_permission: Any | None = None,
    ) -> str:
        """Execute a task through the Claude SDK, streaming events."""

        sdk: Any = self.sdk_service
        accumulated = ""
        sdk_session_id = session.metadata.get("sdk_session_id")

        channel = str(session.metadata.get("channel", "cli"))

        async for event in sdk.run_task_stream(
            prompt=prompt,
            task_id=task_id,
            session_id=sdk_session_id,
            on_permission=on_permission,
            channel=channel,
        ):
            if event.type == "text":
                if not (event.data and event.data.get("subagent")):
                    accumulated += event.content
            elif event.type == "result":
                sdk_sid = event.data.get("session_id")
                if sdk_sid:
                    session.metadata["sdk_session_id"] = sdk_sid
                if event.content and len(event.content) > len(accumulated):
                    accumulated = event.content
            elif event.type == "error":
                raise RuntimeError(event.content)

            # Forward progress events to the caller
            if on_progress is not None:
                try:
                    await on_progress(event)
                except Exception as _prog_err:
                    logger.debug("progress_callback_error", error=str(_prog_err))

        return accumulated or "[No response]"

    def _create_scoped_registry(
        self,
        role: SubAgentRole,
        nesting_depth: int = 0,
    ) -> ScopedToolRegistry:
        """Create a filtered copy of the tool registry for a sub-agent.

        Applies allow/deny lists and always excludes orchestration tools
        and dangerous-tier tools. At nesting depth >= 1, also excludes
        consult/delegation tools to prevent recursive chains.
        """
        denied = set(role.denied_tools) | self.EXCLUDED_TOOLS
        if nesting_depth >= 1:
            denied |= self.NESTED_EXCLUDED_TOOLS
        return ScopedToolRegistry(
            parent=self.tool_registry,
            allowed_tools=role.allowed_tools or None,
            denied_tools=denied,
            exclude_dangerous=True,
        )


# --- Method binding from split modules ---
from agent.core.orchestrator._projects import (  # noqa: E402
    _evaluate_review_output,
    _evaluate_with_quality_gate,
    _find_stage_by_name,
    _run_stage,
    run_project,
)

SubAgentOrchestrator.run_project = run_project  # type: ignore[assignment]
SubAgentOrchestrator._find_stage_by_name = _find_stage_by_name  # type: ignore[assignment]
SubAgentOrchestrator._run_stage = _run_stage  # type: ignore[assignment]
SubAgentOrchestrator._evaluate_review_output = _evaluate_review_output  # type: ignore[assignment]
SubAgentOrchestrator._evaluate_with_quality_gate = _evaluate_with_quality_gate  # type: ignore[assignment]

from agent.core.orchestrator._discussion import (  # noqa: E402
    _run_discussion_stage,
)

SubAgentOrchestrator._run_discussion_stage = _run_discussion_stage  # type: ignore[assignment]

from agent.core.orchestrator._consultation import (  # noqa: E402
    handle_consult,
    handle_delegation,
)

SubAgentOrchestrator.handle_consult = handle_consult  # type: ignore[assignment]
SubAgentOrchestrator.handle_delegation = handle_delegation  # type: ignore[assignment]

from agent.core.orchestrator._execution import (  # noqa: E402
    _execute_subagent,
    _execute_subagent_via_loop,
    _execute_subagent_via_sdk,
)

SubAgentOrchestrator._execute_subagent = _execute_subagent  # type: ignore[assignment]
SubAgentOrchestrator._execute_subagent_via_loop = _execute_subagent_via_loop  # type: ignore[assignment]
SubAgentOrchestrator._execute_subagent_via_sdk = _execute_subagent_via_sdk  # type: ignore[assignment]
