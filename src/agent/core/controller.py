"""Controller agent — project manager layer between main agent and workers.

The controller receives high-level work orders from the main agent,
plans and decomposes them, spawns worker sub-agents, monitors progress,
and reports results back. This keeps the main agent free for conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import Events
from agent.core.subagent import (
    ControllerDirective,
    ControllerTaskState,
    ControllerWorkOrder,
)

if TYPE_CHECKING:
    from agent.config import OrchestrationConfig
    from agent.core.events import EventBus
    from agent.core.orchestrator import SubAgentOrchestrator
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
- Simple task (one skill needed) → spawn_subagent
- Multiple independent tasks → spawn_parallel_agents
- Multi-role collaboration → spawn_team
- Sequential pipeline (plan→build→review) → run_project

## Rules
- You are the coordinator, NOT the worker. Never do the work directly.
- Always delegate to specialized agents with clear, specific instructions.
- If a worker fails, analyze why and retry with adjusted instructions.
- Keep your summaries concise — the user gets notified automatically.
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
    ) -> None:
        self.orchestrator = orchestrator
        self.sdk_service = sdk_service
        self.event_bus = event_bus
        self.config = config

        self._task_queue: asyncio.Queue[ControllerWorkOrder | ControllerDirective] = (
            asyncio.Queue()
        )
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

        for task_id, task in list(self._worker_tasks.items()):
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

    async def _run_loop(self) -> None:
        """Main processing loop — reads from queue and dispatches."""
        while self._running:
            try:
                item = await asyncio.wait_for(
                    self._task_queue.get(), timeout=1.0,
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
                await self.event_bus.emit(Events.CONTROLLER_TASK_CANCELLED, {
                    "order_id": directive.order_id,
                    "user_id": state.user_id,
                })
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
        state = ControllerTaskState(
            order_id=order.order_id,
            status="planning",
            user_id=order.user_id,
        )
        self._active_tasks[order.order_id] = state

        await self.event_bus.emit(Events.CONTROLLER_TASK_STARTED, {
            "order_id": order.order_id,
            "instruction": order.instruction,
            "user_id": order.user_id,
        })

        try:
            result_text = await self._execute_order(order, state)

            # Guard against race with directive "stop" setting status
            # to "cancelled" while _execute_order was awaited.
            if state.status not in _TERMINAL_STATUSES:
                state.status = "completed"
                state.summary = result_text[:500] if result_text else "Completed"

                await self.event_bus.emit(Events.CONTROLLER_TASK_COMPLETED, {
                    "order_id": order.order_id,
                    "summary": state.summary,
                    "user_id": order.user_id,
                })

        except asyncio.CancelledError:
            # FIX #3: Only emit if directive handler hasn't already
            # marked this as cancelled (prevents double-emission).
            if state.status not in _TERMINAL_STATUSES:
                state.status = "cancelled"
                # FIX #2: Protect the emit itself from CancelledError
                # during shutdown, when the event loop is tearing down.
                # CancelledError is a BaseException, not Exception.
                with contextlib.suppress(asyncio.CancelledError):
                    await self.event_bus.emit(Events.CONTROLLER_TASK_CANCELLED, {
                        "order_id": order.order_id,
                        "user_id": order.user_id,
                    })
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
                    await self.event_bus.emit(Events.CONTROLLER_TASK_FAILED, {
                        "order_id": order.order_id,
                        "error": str(e),
                        "user_id": order.user_id,
                    })
        finally:
            # FIX #1: Prune finished tasks to prevent unbounded growth.
            self._prune_finished_tasks()

    def _prune_finished_tasks(self) -> None:
        """Remove oldest terminal-state tasks when the dict grows too large."""
        finished = [
            oid for oid, s in self._active_tasks.items()
            if s.status in _TERMINAL_STATUSES
        ]
        if len(finished) > _MAX_FINISHED_TASKS:
            # Remove the oldest entries (dict preserves insertion order).
            to_remove = finished[: len(finished) - _MAX_FINISHED_TASKS]
            for oid in to_remove:
                del self._active_tasks[oid]

    async def _execute_order(
        self, order: ControllerWorkOrder, state: ControllerTaskState,
    ) -> str:
        """Execute a work order using the SDK or fallback to direct orchestration."""
        prompt = (
            f"## Work Order\n"
            f"**Instruction:** {order.instruction}\n"
        )
        if order.context:
            prompt += f"**Context:** {order.context}\n"

        task_id = f"controller-{order.order_id}"

        if self.sdk_service is not None:
            return await self._execute_via_sdk(prompt, task_id, order, state)

        return await self._execute_fallback(order, state)

    async def _execute_via_sdk(
        self, prompt: str, task_id: str,
        order: ControllerWorkOrder, state: ControllerTaskState,
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
        await self.event_bus.emit(Events.CONTROLLER_TASK_PROGRESS, {
            "order_id": state.order_id,
            "status": "executing",
            "user_id": order.user_id,
        })

        result = await self.sdk_service.run_subagent(
            prompt=prompt,
            task_id=task_id,
            role_persona=CONTROLLER_SYSTEM_PROMPT,
            scoped_registry=scoped_registry,
            max_turns=self.config.controller_max_turns,
            model=self.config.controller_model,
        )
        return result

    async def _execute_fallback(
        self, order: ControllerWorkOrder, state: ControllerTaskState,
    ) -> str:
        """Fallback: directly spawn a single subagent via the orchestrator."""
        from agent.core.subagent import SubAgentRole, SubAgentTask

        state.status = "executing"
        # FIX #6: Include user_id in progress events.
        await self.event_bus.emit(Events.CONTROLLER_TASK_PROGRESS, {
            "order_id": state.order_id,
            "status": "executing",
            "user_id": order.user_id,
        })

        role = SubAgentRole(
            name="controller-worker",
            persona="You are a capable assistant. Complete the given task.",
            max_iterations=self.config.default_max_iterations,
        )
        task = SubAgentTask(
            role=role,
            instruction=order.instruction,
            context=order.context,
        )

        result = await self.orchestrator.spawn_subagent(task)
        state.worker_task_ids.append(result.task_id)

        # FIX #7: StrEnum compares directly — no need for .value
        if result.status == "completed":
            return result.output
        raise RuntimeError(
            f"Worker failed: {result.error or 'unknown error'}"
        )
