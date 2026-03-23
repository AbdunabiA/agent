"""Consultation and delegation methods for SubAgentOrchestrator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import Events
from agent.core.subagent import (
    ConsultRequest,
    ConsultResponse,
    DelegationMode,
    DelegationRequest,
    DelegationResult,
    SubAgentStatus,
    SubAgentTask,
)

if TYPE_CHECKING:
    from agent.core.orchestrator._core import SubAgentOrchestrator

logger = structlog.get_logger(__name__)


async def handle_consult(
    self: SubAgentOrchestrator,
    request: ConsultRequest,
    nesting_depth: int = 0,
) -> ConsultResponse:
    """Handle a consultation request from one agent to another.

    Spawns a nested sub-agent to answer the question. Recursion is
    capped: consulted agents at depth >= 1 cannot consult others.

    Args:
        request: The consultation request.
        nesting_depth: Current nesting depth of the requesting agent.

    Returns:
        ConsultResponse with the answer or error.
    """
    import time as _time

    start = _time.monotonic()

    if nesting_depth >= 1:
        return ConsultResponse(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.FAILED,
            error="Cannot consult at nesting depth >= 1 (recursion limit)",
        )

    await self.event_bus.emit(
        Events.AGENT_CONSULT_REQUESTED,
        {
            "request_id": request.request_id,
            "requesting_role": request.requesting_role,
            "target_team": request.target_team,
            "target_role": request.target_role,
        },
    )

    # Resolve team and role
    team = self.teams.get(request.target_team)
    if not team:
        available = ", ".join(self.teams.keys()) or "none"
        error = f"Team '{request.target_team}' not found. Available: {available}"
        await self.event_bus.emit(
            Events.AGENT_CONSULT_FAILED,
            {
                "request_id": request.request_id,
                "error": error,
            },
        )
        return ConsultResponse(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.FAILED,
            error=error,
        )

    role = next(
        (r for r in team.roles if r.name == request.target_role),
        None,
    )
    if not role:
        available_roles = ", ".join(r.name for r in team.roles)
        error = (
            f"Role '{request.target_role}' not found in team "
            f"'{request.target_team}'. Available: {available_roles}"
        )
        await self.event_bus.emit(
            Events.AGENT_CONSULT_FAILED,
            {
                "request_id": request.request_id,
                "error": error,
            },
        )
        return ConsultResponse(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.FAILED,
            error=error,
        )

    # Build consultation instruction
    instruction = (
        f"You have been consulted by {request.requesting_role}.\n\n" f"Question: {request.question}"
    )
    if request.context:
        instruction += f"\n\nContext: {request.context}"
    instruction += "\n\nProvide a focused, expert answer."

    task = SubAgentTask(
        role=role,
        instruction=instruction,
        nesting_depth=nesting_depth + 1,
    )

    result = await self.spawn_subagent(task)

    duration_ms = int((_time.monotonic() - start) * 1000)

    if result.status == SubAgentStatus.COMPLETED:
        await self.event_bus.emit(
            Events.AGENT_CONSULT_COMPLETED,
            {
                "request_id": request.request_id,
                "target_role": request.target_role,
                "duration_ms": duration_ms,
            },
        )
        return ConsultResponse(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.COMPLETED,
            answer=result.output,
            token_usage=result.token_usage,
            duration_ms=duration_ms,
        )
    else:
        await self.event_bus.emit(
            Events.AGENT_CONSULT_FAILED,
            {
                "request_id": request.request_id,
                "error": result.error,
            },
        )
        return ConsultResponse(
            request_id=request.request_id,
            target_role=request.target_role,
            status=result.status,
            error=result.error,
            duration_ms=duration_ms,
        )


async def handle_delegation(
    self: SubAgentOrchestrator,
    request: DelegationRequest,
    nesting_depth: int = 0,
) -> DelegationResult:
    """Handle a delegation request from one agent to a specialist.

    Supports sync (wait for result) and async (fire-and-forget) modes.

    Args:
        request: The delegation request.
        nesting_depth: Current nesting depth of the delegating agent.

    Returns:
        DelegationResult with output, error, or task_id for async.
    """
    import time as _time

    start = _time.monotonic()

    if nesting_depth >= 1:
        return DelegationResult(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.FAILED,
            error="Cannot delegate at nesting depth >= 1 (recursion limit)",
        )

    await self.event_bus.emit(
        Events.AGENT_DELEGATION_REQUESTED,
        {
            "request_id": request.request_id,
            "delegating_role": request.delegating_role,
            "target_team": request.target_team,
            "target_role": request.target_role,
            "mode": request.mode,
        },
    )

    # Resolve team and role
    team = self.teams.get(request.target_team)
    if not team:
        available = ", ".join(self.teams.keys()) or "none"
        error = f"Team '{request.target_team}' not found. Available: {available}"
        await self.event_bus.emit(
            Events.AGENT_DELEGATION_FAILED,
            {
                "request_id": request.request_id,
                "error": error,
            },
        )
        return DelegationResult(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.FAILED,
            error=error,
        )

    role = next(
        (r for r in team.roles if r.name == request.target_role),
        None,
    )
    if not role:
        available_roles = ", ".join(r.name for r in team.roles)
        error = (
            f"Role '{request.target_role}' not found in team "
            f"'{request.target_team}'. Available: {available_roles}"
        )
        await self.event_bus.emit(
            Events.AGENT_DELEGATION_FAILED,
            {
                "request_id": request.request_id,
                "error": error,
            },
        )
        return DelegationResult(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.FAILED,
            error=error,
        )

    # Build delegation instruction
    instruction = request.instruction
    if request.context:
        instruction = f"Context:\n{request.context}\n\nTask:\n{instruction}"

    task = SubAgentTask(
        role=role,
        instruction=instruction,
        nesting_depth=nesting_depth + 1,
    )

    if request.mode == DelegationMode.ASYNC:
        # Fire-and-forget: schedule the task and return task_id.
        # spawn_subagent manages _running_tasks internally — don't
        # register the outer wrapper here to avoid double-registration
        # and broken cancellation tracking.
        # Store the Future in _async_futures so exceptions aren't lost.
        fut = asyncio.ensure_future(self.spawn_subagent(task))
        self._async_futures[task.task_id] = fut

        def _on_done(f: asyncio.Future[Any], tid: str = task.task_id) -> None:
            self._async_futures.pop(tid, None)
            if not f.cancelled():
                exc = f.exception()
                if exc is not None:
                    logger.error(
                        "async_delegation_failed",
                        task_id=tid,
                        error=str(exc),
                    )

        fut.add_done_callback(_on_done)
        return DelegationResult(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.PENDING,
            task_id=task.task_id,
        )

    # Sync mode: wait for result
    result = await self.spawn_subagent(task)

    duration_ms = int((_time.monotonic() - start) * 1000)

    if result.status == SubAgentStatus.COMPLETED:
        await self.event_bus.emit(
            Events.AGENT_DELEGATION_COMPLETED,
            {
                "request_id": request.request_id,
                "target_role": request.target_role,
                "duration_ms": duration_ms,
            },
        )
        return DelegationResult(
            request_id=request.request_id,
            target_role=request.target_role,
            status=SubAgentStatus.COMPLETED,
            output=result.output,
            task_id=task.task_id,
            token_usage=result.token_usage,
            duration_ms=duration_ms,
        )
    else:
        await self.event_bus.emit(
            Events.AGENT_DELEGATION_FAILED,
            {
                "request_id": request.request_id,
                "error": result.error,
            },
        )
        return DelegationResult(
            request_id=request.request_id,
            target_role=request.target_role,
            status=result.status,
            error=result.error,
            task_id=task.task_id,
            duration_ms=duration_ms,
        )
