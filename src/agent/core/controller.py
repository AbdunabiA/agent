"""Controller agent — project manager layer between main agent and workers.

The controller receives high-level work orders from the main agent,
plans and decomposes them, spawns worker sub-agents, monitors progress,
and reports results back. This keeps the main agent free for conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import Events
from agent.core.subagent import (
    ControllerDirective,
    ControllerTaskState,
    ControllerWorkOrder,
    SubAgentRole,
)

if TYPE_CHECKING:
    from agent.config import OrchestrationConfig
    from agent.core.events import EventBus
    from agent.core.orchestrator import SubAgentOrchestrator
    from agent.core.role_registry import RoleRegistry
    from agent.llm.claude_sdk import ClaudeSDKService

logger = structlog.get_logger(__name__)

# Terminal statuses — tasks in these states are eligible for cleanup.
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

# Maximum finished tasks to retain before pruning.
_MAX_FINISHED_TASKS = 200


CONTROLLER_SYSTEM_PROMPT = """\
You are a Project Manager agent. You receive work orders and execute them
by delegating to specialized worker agents.

## Your Process
1. ANALYZE the work order — understand what needs to be done
2. PLAN — decide whether to use a single agent, parallel agents, a team, or a pipeline
3. EXECUTE — spawn the appropriate agents using your tools
4. MONITOR — check on progress, handle failures
5. REPORT — when done, provide a clear summary of what was accomplished

## Decision Guide
- Quick question or tiny task → spawn_subagent with one worker
- Code review, project analysis, feature work → spawn_subagent
  with a thorough worker (give detailed instructions including
  which files to read)
- Multiple independent tasks → spawn_parallel_agents
  (but use sparingly to avoid rate limits)
- Sequential pipeline (plan→build→review) → run_project
- Multi-component projects (SaaS, platform, app with multiple
  frontends) → plan_and_build

## Rules
- You are the coordinator, NOT the worker. Never do the work directly.
- Always delegate to specialized agents with clear, specific instructions.
- Give workers DETAILED instructions: specify file paths,
  what to analyze, what format the report should be in.
- Workers have Read, Bash, Glob, Grep, Edit, Write tools —
  tell them to USE these tools to read files.
- If a worker fails, analyze why and retry with adjusted
  instructions.
- Always use mode='sync' when calling spawn_subagent. This
  ensures you receive results and can report them back.
- After receiving worker results, compile them into a clear, complete summary.
"""


class ControllerAgent:
    """Project manager agent that handles work coordination.

    Receives work orders and directives via an async queue,
    plans and decomposes tasks, spawns workers, and reports results.
    """

    def __init__(
        self,
        orchestrator: SubAgentOrchestrator,
        sdk_service: ClaudeSDKService | None,
        event_bus: EventBus,
        config: OrchestrationConfig,
        role_registry: RoleRegistry | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.sdk_service = sdk_service
        self.event_bus = event_bus
        self.config = config
        self.role_registry: RoleRegistry | None = role_registry

        self._task_queue: asyncio.Queue[ControllerWorkOrder | ControllerDirective] = asyncio.Queue()
        self._active_tasks: dict[str, ControllerTaskState] = {}
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        """Start the controller's processing loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

        # Listen for worker failure events to handle critical vs non-critical
        self.event_bus.on(Events.WORKER_FAILED, self._on_worker_failed)
        self.event_bus.on(Events.WORKER_RETRYING, self._on_worker_retrying)

        logger.info("controller_started")

    async def stop(self) -> None:
        """Gracefully shut down the controller."""
        self._running = False

        # FIX #5: Stop the loop first so no new workers are spawned,
        # then cancel remaining workers.
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task

        # Unregister event handlers to prevent memory leaks
        self.event_bus.off(Events.WORKER_FAILED, self._on_worker_failed)
        self.event_bus.off(Events.WORKER_RETRYING, self._on_worker_retrying)

        for _task_id, task in list(self._worker_tasks.items()):
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        logger.info("controller_stopped")

    async def submit_order(self, order: ControllerWorkOrder) -> None:
        """Submit a work order to the controller. Returns immediately."""
        await self._task_queue.put(order)
        logger.info(
            "controller_order_submitted",
            order_id=order.order_id,
            instruction=order.instruction[:100],
        )

    async def submit_directive(self, directive: ControllerDirective) -> None:
        """Submit a directive to the controller. Returns immediately."""
        await self._task_queue.put(directive)
        logger.info(
            "controller_directive_submitted",
            order_id=directive.order_id,
            command=directive.command,
        )

    def get_task_summary(self, order_id: str) -> str:
        """Return human-readable status for one task."""
        state = self._active_tasks.get(order_id)
        if not state:
            return f"No task found with order_id: {order_id}"

        lines = [
            f"Order: {state.order_id}",
            f"Status: {state.status}",
        ]
        if state.worker_task_ids:
            lines.append(f"Workers: {', '.join(state.worker_task_ids)}")
        if state.summary:
            lines.append(f"Summary: {state.summary}")
        if state.error:
            lines.append(f"Error: {state.error}")
        return "\n".join(lines)

    def get_all_tasks_summary(self) -> str:
        """Return summary of all active tasks."""
        if not self._active_tasks:
            return "No active tasks."

        lines = [f"Active Tasks ({len(self._active_tasks)}):"]
        for order_id, state in self._active_tasks.items():
            status_icon = {
                "pending": "\u23f3",
                "planning": "\U0001f9e0",
                "executing": "\u2699\ufe0f",
                "completed": "\u2705",
                "failed": "\u274c",
                "cancelled": "\u26d4",
            }.get(state.status, "\u2753")
            lines.append(f"  {status_icon} {order_id}: {state.status}")
            if state.summary:
                lines.append(f"     {state.summary[:100]}")
        return "\n".join(lines)

    async def _on_worker_retrying(self, data: Any) -> None:
        """Handle a worker retrying event — log for observability."""
        if data is None:
            return
        logger.warning(
            "worker_retrying",
            task_id=data.get("task_id"),
            role=data.get("role"),
            attempt=data.get("attempt"),
            max_attempts=data.get("max_attempts"),
            error=data.get("error"),
            retry_in_seconds=data.get("retry_in_seconds"),
        )

    async def _on_worker_failed(self, data: Any) -> None:
        """Handle a worker failure after all retries are exhausted.

        For tasks managed by the controller, check if the failed worker
        is critical.  Critical failures mark the whole order as failed;
        non-critical failures are logged as warnings.
        """
        if data is None:
            return

        task_id = data.get("task_id", "")
        role = data.get("role", "unknown")
        error = data.get("error", "unknown error")
        attempts = data.get("attempts", 0)

        # Find the order that owns this worker (if any)
        for order_id, state in self._active_tasks.items():
            if task_id in state.worker_task_ids:
                # Check if the SubAgentTask was marked critical
                # (we flag via metadata on the task_id)
                is_critical = task_id.startswith("critical-")

                if is_critical and state.status not in _TERMINAL_STATUSES:
                    state.status = "failed"
                    state.error = (
                        f"Critical worker '{role}' failed after " f"{attempts} attempts: {error}"
                    )
                    logger.error(
                        "critical_worker_failed",
                        order_id=order_id,
                        role=role,
                        error=error,
                    )
                    await self.event_bus.emit(
                        Events.CONTROLLER_TASK_FAILED,
                        {
                            "order_id": order_id,
                            "error": state.error,
                            "user_id": state.user_id,
                        },
                    )
                    await self.event_bus.emit(
                        Events.TASK_FAILED_NOTIFY,
                        {
                            "task_id": order_id,
                            "user_id": state.user_id,
                            "error": state.error,
                            "attempts": attempts,
                        },
                    )
                else:
                    logger.warning(
                        "non_critical_worker_failed",
                        order_id=order_id,
                        role=role,
                        error=error,
                        attempts=attempts,
                    )
                break
        else:
            # Worker not owned by any controller order — just log
            logger.warning(
                "worker_failed_no_order",
                task_id=task_id,
                role=role,
                error=error,
                attempts=attempts,
            )

    async def _run_loop(self) -> None:
        """Main processing loop — reads from queue and dispatches."""
        while self._running:
            try:
                item = await asyncio.wait_for(
                    self._task_queue.get(),
                    timeout=1.0,
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # FIX #4: Wrap dispatch in try/except so a bad directive
            # doesn't kill the entire processing loop.
            try:
                if isinstance(item, ControllerWorkOrder):
                    task = asyncio.create_task(self._handle_order(item))
                    self._worker_tasks[item.order_id] = task
                    task.add_done_callback(
                        lambda t, oid=item.order_id: self._worker_tasks.pop(oid, None)
                    )
                elif isinstance(item, ControllerDirective):
                    await self._handle_directive(item)
            except Exception as e:
                logger.error(
                    "controller_dispatch_error",
                    item_type=type(item).__name__,
                    error=str(e),
                )

    async def _handle_directive(self, directive: ControllerDirective) -> None:
        """Handle a control directive (stop, pause, redirect)."""
        state = self._active_tasks.get(directive.order_id)
        if not state:
            logger.warning(
                "controller_directive_no_task",
                order_id=directive.order_id,
                command=directive.command,
            )
            return

        if directive.command == "stop":
            # Cancel the worker task for this order
            worker = self._worker_tasks.get(directive.order_id)
            if worker and not worker.done():
                worker.cancel()

            # FIX #3: Only update state and emit if not already terminal.
            # The worker's CancelledError handler also sets status, so
            # whichever runs first "wins" and the other becomes a no-op.
            if state.status not in _TERMINAL_STATUSES:
                state.status = "cancelled"
                await self.event_bus.emit(
                    Events.CONTROLLER_TASK_CANCELLED,
                    {
                        "order_id": directive.order_id,
                        "user_id": state.user_id,
                    },
                )
                logger.info(
                    "controller_task_cancelled",
                    order_id=directive.order_id,
                )
        else:
            logger.warning(
                "controller_unknown_directive",
                order_id=directive.order_id,
                command=directive.command,
            )

    async def _handle_order(self, order: ControllerWorkOrder) -> None:
        """Handle a work order by running the controller's LLM session."""
        start_time = time.monotonic()
        state = ControllerTaskState(
            order_id=order.order_id,
            status="planning",
            user_id=order.user_id,
        )
        self._active_tasks[order.order_id] = state

        await self.event_bus.emit(
            Events.CONTROLLER_TASK_STARTED,
            {
                "order_id": order.order_id,
                "instruction": order.instruction,
                "user_id": order.user_id,
            },
        )

        try:
            result_text = await self._execute_order(order, state)

            # Guard against race with directive "stop" setting status
            # to "cancelled" while _execute_order was awaited.
            if state.status not in _TERMINAL_STATUSES:
                # Detect SDK/auth errors in the result text that would
                # otherwise be reported as "completed" to the user.
                _error_indicators = (
                    "[SDK_ERROR]",
                    "Failed to authenticate",
                    "authentication_error",
                    "OAuth token has expired",
                )
                is_error_result = result_text and any(
                    ind in result_text for ind in _error_indicators
                )

                if is_error_result:
                    state.status = "failed"
                    state.error = result_text[:500] if result_text else "Unknown error"
                    logger.warning(
                        "controller_order_error_in_result",
                        order_id=order.order_id,
                        error=state.error[:200],
                    )
                    await self.event_bus.emit(
                        Events.CONTROLLER_TASK_FAILED,
                        {
                            "order_id": order.order_id,
                            "error": state.error,
                            "user_id": order.user_id,
                        },
                    )
                    await self.event_bus.emit(
                        Events.TASK_FAILED_NOTIFY,
                        {
                            "task_id": order.order_id,
                            "user_id": order.user_id,
                            "error": state.error,
                            "attempts": 1,
                        },
                    )
                else:
                    state.status = "completed"
                    # Keep full result for notifications — Telegram handler
                    # handles splitting long messages automatically.
                    state.summary = result_text or "Completed"

                    await self.event_bus.emit(
                        Events.CONTROLLER_TASK_COMPLETED,
                        {
                            "order_id": order.order_id,
                            "summary": state.summary,
                            "user_id": order.user_id,
                        },
                    )
                    await self.event_bus.emit(
                        Events.TASK_COMPLETED_NOTIFY,
                        {
                            "task_id": order.order_id,
                            "user_id": order.user_id,
                            "result": state.summary,
                            "duration_seconds": int(time.monotonic() - start_time),
                        },
                    )

        except asyncio.CancelledError:
            # FIX #3: Only emit if directive handler hasn't already
            # marked this as cancelled (prevents double-emission).
            if state.status not in _TERMINAL_STATUSES:
                state.status = "cancelled"
                # FIX #2: Protect the emit itself from CancelledError
                # during shutdown, when the event loop is tearing down.
                # CancelledError is a BaseException, not Exception.
                with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                    await self.event_bus.emit(
                        Events.CONTROLLER_TASK_CANCELLED,
                        {
                            "order_id": order.order_id,
                            "user_id": order.user_id,
                        },
                    )
        except Exception as e:
            # Only update state if not already terminal (e.g. directive
            # may have already cancelled this task).
            if state.status not in _TERMINAL_STATUSES:
                state.status = "failed"
                state.error = str(e)
                logger.error(
                    "controller_order_failed",
                    order_id=order.order_id,
                    error=str(e),
                )
                with contextlib.suppress(asyncio.CancelledError):
                    await self.event_bus.emit(
                        Events.CONTROLLER_TASK_FAILED,
                        {
                            "order_id": order.order_id,
                            "error": str(e),
                            "user_id": order.user_id,
                        },
                    )
                with contextlib.suppress(asyncio.CancelledError):
                    await self.event_bus.emit(
                        Events.TASK_FAILED_NOTIFY,
                        {
                            "task_id": order.order_id,
                            "user_id": order.user_id,
                            "error": str(e),
                            "attempts": 1,
                        },
                    )
        finally:
            # FIX #1: Prune finished tasks to prevent unbounded growth.
            self._prune_finished_tasks()

    def _prune_finished_tasks(self) -> None:
        """Remove oldest terminal-state tasks when the dict grows too large."""
        finished = [
            oid for oid, s in list(self._active_tasks.items()) if s.status in _TERMINAL_STATUSES
        ]
        if len(finished) > _MAX_FINISHED_TASKS:
            # Remove the oldest entries (dict preserves insertion order).
            to_remove = finished[: len(finished) - _MAX_FINISHED_TASKS]
            for oid in to_remove:
                self._active_tasks.pop(oid, None)

    async def _execute_order(
        self,
        order: ControllerWorkOrder,
        state: ControllerTaskState,
    ) -> str:
        """Execute a work order using the SDK or fallback to direct orchestration."""
        prompt = f"## Work Order\n" f"**Instruction:** {order.instruction}\n"
        if order.context:
            prompt += f"**Context:** {order.context}\n"

        task_id = f"controller-{order.order_id}"

        if self.sdk_service is not None:
            return await self._execute_via_sdk(prompt, task_id, order, state)

        return await self._execute_fallback(order, state)

    async def _execute_via_sdk(
        self,
        prompt: str,
        task_id: str,
        order: ControllerWorkOrder,
        state: ControllerTaskState,
    ) -> str:
        """Run the controller via Claude SDK with orchestration tools."""
        from agent.core.orchestrator import ScopedToolRegistry

        # Build a scoped registry with orchestration tools but NOT controller tools
        controller_excluded = {"assign_work", "check_work_status", "direct_controller"}
        scoped_registry = ScopedToolRegistry(
            parent=self.orchestrator.tool_registry,
            denied_tools=controller_excluded,
        )

        state.status = "executing"
        # FIX #6: Include user_id in progress events.
        await self.event_bus.emit(
            Events.CONTROLLER_TASK_PROGRESS,
            {
                "order_id": state.order_id,
                "status": "executing",
                "user_id": order.user_id,
            },
        )

        # Inject available roles into the system prompt
        system_prompt = CONTROLLER_SYSTEM_PROMPT
        if self.role_registry is not None:
            roster = self.role_registry.get_roster_description()
            system_prompt += (
                "\n## Available Specialist Roles\n"
                f"{roster}\n\n"
                "Pick only the roles needed for the task. "
                "Prefer fewer roles over more.\n"
            )

        result = await self.sdk_service.run_subagent(
            prompt=prompt,
            task_id=task_id,
            role_persona=system_prompt,
            scoped_registry=scoped_registry,
            max_turns=self.config.controller_max_turns,
            model=self.config.controller_model,
            denied_builtins={
                "Read",
                "Glob",
                "Grep",
                "Edit",
                "Write",
                "Bash",
                "WebFetch",
                "WebSearch",
                "LS",
                "Agent",
                "Explore",
                "NotebookEdit",
            },
        )

        # If the controller SDK session returned an error or empty marker,
        # fall back to direct orchestration so the work still gets done.
        if not result or result == "[No response from sub-agent]":
            logger.warning(
                "controller_sdk_empty_response",
                task_id=task_id,
                order_id=order.order_id,
            )
            return await self._execute_fallback(order, state)

        if result.strip().startswith("[SDK_ERROR]"):
            logger.warning(
                "controller_sdk_error_response",
                task_id=task_id,
                order_id=order.order_id,
                error=result[:200],
            )
            return await self._execute_fallback(order, state)

        return result

    async def _select_roles_for_task(
        self,
        instruction: str,
        context: str = "",
    ) -> tuple[list[SubAgentRole], int, str]:
        """Use the LLM to pick which roles are needed for a task.

        When no SDK/LLM is available, falls back to a heuristic selection.

        Args:
            instruction: The work order instruction text.
            context: Optional additional context.

        Returns:
            Tuple of (selected_roles, max_rounds, execution_order).
            execution_order is ``"parallel"`` or ``"sequential"``.
        """
        from agent.core.subagent import SubAgentRole

        if self.role_registry is None:
            # No registry — fall back to a generic worker
            return (
                [
                    SubAgentRole(
                        name="controller-worker",
                        persona="You are a capable assistant. Complete the given task.",
                        max_iterations=self.config.default_max_iterations,
                    )
                ],
                1,
                "parallel",
            )

        roster = self.role_registry.get_roster_description()

        # Try LLM-based selection via the SDK
        if self.sdk_service is not None:
            try:
                return await self._select_roles_via_llm(
                    instruction,
                    context,
                    roster,
                )
            except Exception as e:
                logger.warning("role_selection_llm_failed", error=str(e))

        # Heuristic fallback when no LLM is available
        return self._select_roles_heuristic(instruction)

    async def _select_roles_via_llm(
        self,
        instruction: str,
        context: str,
        roster: str,
    ) -> tuple[list[SubAgentRole], int, str]:
        """Ask the LLM which roles to use."""
        import json

        prompt = (
            "You are selecting which AI agent roles to assign for a task.\n\n"
            f"Available roles:\n{roster}\n\n"
            f"Task: {instruction}\n"
        )
        if context:
            prompt += f"Context: {context}\n"
        prompt += (
            "\nSelect ONLY the roles actually needed. Prefer fewer roles.\n"
            "Rules:\n"
            "- Include architect if task involves new features or design\n"
            "- Include qa_engineer if code will be written\n"
            "- Only include security_reviewer for auth, payments, or sensitive data\n"
            "- Only include devops_engineer if deployment or infra changes needed\n"
            "- max_rounds: 2 for simple tasks, 3-4 for QA cycles, 5 for complex\n\n"
            "Respond with JSON only, no markdown:\n"
            '{"roles": ["role_name_1"], "max_rounds": 3, '
            '"execution_order": "parallel", "reasoning": "one sentence"}\n'
        )

        assert self.sdk_service is not None
        from agent.core.orchestrator import ScopedToolRegistry

        # Role selection is a pure-LLM call — no tools needed.
        empty_registry = ScopedToolRegistry(
            parent=self.orchestrator.tool_registry,
            denied_tools=set(t.name for t in self.orchestrator.tool_registry.list_tools()),
        )
        raw = await self.sdk_service.run_subagent(
            prompt=prompt,
            task_id=f"role-select-{id(instruction) % 100000:05d}",
            role_persona="You are a task planning assistant. Return only JSON.",
            scoped_registry=empty_registry,
            max_turns=1,
        )

        # Extract JSON from response (may contain markdown fences)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        parsed = json.loads(text)
        role_names: list[str] = parsed.get("roles", [])
        max_rounds: int = min(int(parsed.get("max_rounds", 3)), 5)
        execution_order: str = parsed.get("execution_order", "parallel")
        reasoning: str = parsed.get("reasoning", "")

        assert self.role_registry is not None
        # LLM may return "team/role" format from roster — strip team prefix
        cleaned_names: list[str] = []
        for r in role_names:
            clean = r.split("/")[-1] if "/" in r else r
            cleaned_names.append(clean)

        roles = [
            self.role_registry.get_role(r)
            for r in cleaned_names
            if self.role_registry.get_role(r) is not None
        ]

        if not roles:
            raise ValueError("LLM returned no valid roles")

        logger.info(
            "roles_selected",
            roles=[r.name for r in roles],
            max_rounds=max_rounds,
            execution_order=execution_order,
            reasoning=reasoning,
        )

        return roles, max_rounds, execution_order

    def _select_roles_heuristic(
        self,
        instruction: str,
    ) -> tuple[list[SubAgentRole], int, str]:
        """Keyword-based heuristic for role selection when no LLM is available.

        Args:
            instruction: The task instruction text.

        Returns:
            Tuple of (selected_roles, max_rounds, execution_order).
        """
        from agent.core.subagent import SubAgentRole

        assert self.role_registry is not None
        text = instruction.lower()
        selected: list[SubAgentRole] = []

        # Map keywords to roles
        keyword_map: list[tuple[list[str], str]] = [
            (["architect", "design", "api", "structure", "refactor"], "architect"),
            (
                ["backend", "python", "api", "database", "implement", "fix", "bug", "code"],
                "backend_developer",
            ),
            (
                ["frontend", "react", "ui", "component", "css", "tailwind"],
                "frontend_developer",
            ),
            (["test", "qa", "coverage", "pytest", "verify"], "qa_engineer"),
            (
                ["security", "auth", "password", "token", "vulnerability", "cve"],
                "security_reviewer",
            ),
            (
                ["deploy", "docker", "ci", "cd", "pipeline", "infrastructure"],
                "devops_engineer",
            ),
            (["docs", "documentation", "readme", "guide"], "technical_writer"),
            (
                ["performance", "optimize", "benchmark", "profile", "bottleneck"],
                "performance_engineer",
            ),
        ]

        seen: set[str] = set()
        for keywords, role_name in keyword_map:
            if any(kw in text for kw in keywords):
                role = self.role_registry.get_role(role_name)
                if role and role_name not in seen:
                    selected.append(role)
                    seen.add(role_name)

        # Default: backend_developer if nothing matched
        if not selected:
            role = self.role_registry.get_role("backend_developer")
            if role:
                selected.append(role)
            else:
                # Absolute fallback
                selected.append(
                    SubAgentRole(
                        name="controller-worker",
                        persona="You are a capable assistant. Complete the given task.",
                        max_iterations=self.config.default_max_iterations,
                    )
                )

        # Determine max_rounds: add QA cycle rounds if QA is selected
        has_qa = any(r.name == "qa_engineer" for r in selected)
        max_rounds = 3 if has_qa else 2

        logger.info(
            "roles_selected_heuristic",
            roles=[r.name for r in selected],
            max_rounds=max_rounds,
        )

        return selected, max_rounds, "parallel"

    async def _execute_fallback(
        self,
        order: ControllerWorkOrder,
        state: ControllerTaskState,
    ) -> str:
        """Fallback: select roles dynamically and run via the orchestrator."""
        from agent.core.subagent import SubAgentTask

        # Only emit progress if not already executing (avoids duplicate
        # events when falling back from _execute_via_sdk).
        if state.status != "executing":
            state.status = "executing"
            await self.event_bus.emit(
                Events.CONTROLLER_TASK_PROGRESS,
                {
                    "order_id": state.order_id,
                    "status": "executing",
                    "user_id": order.user_id,
                },
            )

        # Dynamic role selection
        roles, max_rounds, _exec_order = await self._select_roles_for_task(
            instruction=order.instruction,
            context=order.context,
        )

        worker_timeout = self.config.subagent_timeout

        if len(roles) == 1 and max_rounds <= 1:
            # Single role, single round — use simple spawn
            task = SubAgentTask(
                role=roles[0],
                instruction=order.instruction,
                context=order.context,
                timeout_seconds=worker_timeout,
            )
            result = await self.orchestrator.spawn_subagent(task)
            state.worker_task_ids.append(result.task_id)

            if result.status == "completed":
                return result.output
            raise RuntimeError(f"Worker failed: {result.error or 'unknown error'}")

        # Multiple roles — use iterative team
        tasks = [
            SubAgentTask(
                role=role,
                instruction=order.instruction,
                context=order.context,
                timeout_seconds=worker_timeout,
            )
            for role in roles
        ]

        task_id = f"controller-team-{order.order_id}"

        # Timeout: per-agent timeout × agents × rounds + headroom
        team_timeout = max(
            self.config.subagent_timeout * len(roles) * max_rounds,
            600,  # minimum 10 minutes
        )
        try:
            team_result = await asyncio.wait_for(
                self.orchestrator.run_iterative_team(
                    task_id=task_id,
                    team=tasks,
                    max_rounds=max_rounds,
                ),
                timeout=team_timeout,
            )
        except TimeoutError:
            raise RuntimeError(
                f"Team execution timed out after {team_timeout}s "
                f"(roles: {[r.name for r in roles]})"
            ) from None

        for r in team_result.get("results", []):
            state.worker_task_ids.append(r.task_id)

        # Build summary
        lines: list[str] = []
        rounds = team_result.get("rounds_completed", 0)
        success = team_result.get("success", False)
        lines.append(f"Team completed in {rounds} round(s). Success: {success}")

        board_summary = team_result.get("board_summary", "")
        if board_summary:
            lines.append(board_summary)

        for r in team_result.get("results", []):
            if r.output:
                lines.append(f"\n--- {r.role_name} ---")
                lines.append(r.output)

        return "\n".join(lines)
