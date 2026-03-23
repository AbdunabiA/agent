"""Sub-agent execution methods for SubAgentOrchestrator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from agent.core.events import Events
from agent.core.orchestrator._scoped_registry import ScopedToolRegistry
from agent.core.session import Session
from agent.core.subagent import (
    SubAgentResult,
    SubAgentStatus,
    SubAgentTask,
)

if TYPE_CHECKING:
    from agent.core.orchestrator._core import SubAgentOrchestrator

logger = structlog.get_logger(__name__)


async def _execute_subagent(
    self: SubAgentOrchestrator,
    task: SubAgentTask,
) -> SubAgentResult:
    """Run a sub-agent task to completion.

    Routes through the Claude SDK when available, falling back to the
    AgentLoop path otherwise. Sets task status to RUNNING before execution.
    """
    task.status = SubAgentStatus.RUNNING
    self._task_nesting_depths[task.task_id] = task.nesting_depth
    scoped_registry = self._create_scoped_registry(
        task.role,
        nesting_depth=task.nesting_depth,
    )

    # Set context var so orchestration tools know the caller's nesting depth.
    # NOTE: Each sub-agent task is wrapped in copy_context() at spawn time
    # (see spawn_subagent) so parallel sub-agents don't overwrite each
    # other's nesting depth.
    from agent.tools.builtins.orchestration import set_nesting_depth

    set_nesting_depth(task.nesting_depth)

    # Inject message bus so collaboration tools can send/receive messages
    from agent.tools.builtins.collaboration import set_message_bus

    set_message_bus(self.message_bus)

    try:
        if self.sdk_service is not None:
            return await _execute_subagent_via_sdk(self, task, scoped_registry)
        return await _execute_subagent_via_loop(self, task, scoped_registry)
    finally:
        self._task_nesting_depths.pop(task.task_id, None)


async def _execute_subagent_via_loop(
    self: SubAgentOrchestrator,
    task: SubAgentTask,
    scoped_registry: ScopedToolRegistry,
) -> SubAgentResult:
    """Run a sub-agent task via AgentLoop (LiteLLM path)."""
    from agent.core.agent_loop import AgentLoop
    from agent.tools.executor import ToolExecutor

    # Guard: LLM must be available for the LiteLLM path
    if self.agent_loop.llm is None:
        return SubAgentResult(
            task_id=task.task_id,
            role_name=task.role.name,
            status=SubAgentStatus.FAILED,
            error="No LLM provider configured for sub-agent execution",
        )

    start = asyncio.get_event_loop().time()

    await self.event_bus.emit(
        Events.SUBAGENT_STARTED,
        {
            "task_id": task.task_id,
            "role": task.role.name,
        },
    )

    # Create a minimal config for the sub-agent
    from agent.config import AgentPersonaConfig

    sub_config = AgentPersonaConfig(
        name=task.role.name,
        persona=task.role.persona,
        max_iterations=task.role.max_iterations,
    )

    # Create tool executor with scoped registry, reusing parent's safety components
    parent_executor = self.agent_loop.tool_executor
    if parent_executor is None:
        logger.error(
            "subagent_missing_parent_executor",
            task_id=task.task_id,
            role=task.role.name,
        )
        return SubAgentResult(
            task_id=task.task_id,
            role_name=task.role.name,
            status=SubAgentStatus.FAILED,
            error="Parent tool executor must be set before spawning sub-agents",
        )
    sub_executor = ToolExecutor(
        registry=scoped_registry,  # type: ignore[arg-type]
        config=parent_executor.config,
        event_bus=self.event_bus,
        audit=parent_executor.audit,
        permissions=parent_executor.permissions,
        guardrails=parent_executor.guardrails,
    )

    # Build sub-agent-specific system prompt (no orchestration to prevent recursion)
    sub_prompt = (
        f"ROLE: {task.role.name}\n{task.role.persona}\n\n"
        "INSTRUCTIONS:\nYou are a focused worker agent. Complete the task "
        "directly using the tools available to you. Do NOT delegate work "
        "to sub-agents."
    )

    # Create sub-agent loop (lightweight, no memory/planning, no orchestration)
    sub_loop = AgentLoop(
        llm=self.agent_loop.llm,
        config=sub_config,
        event_bus=self.event_bus,
        tool_executor=sub_executor,
        cost_tracker=self.agent_loop.cost_tracker,
        orchestration_enabled=False,  # Prevent recursive delegation
    )
    # Override the system prompt to prevent orchestration instructions
    sub_loop.system_prompt = sub_prompt

    # Build instruction with context
    full_instruction = task.instruction
    if task.context:
        full_instruction = f"Context:\n{task.context}\n\nTask:\n{task.instruction}"

    # Create session
    session = Session(session_id=f"subagent:{task.parent_session_id}:{task.task_id}")

    try:
        response = await sub_loop.process_message(
            full_instruction,
            session,
            trigger="subagent",
            tool_registry_override=scoped_registry,
        )

        duration_ms = int((asyncio.get_event_loop().time() - start) * 1000)

        # Count tool calls and tokens
        tool_calls_made = sum(1 for m in session.messages if m.tool_calls for _ in m.tool_calls)
        total_tokens = sum((m.usage.total_tokens if m.usage else 0) for m in session.messages)
        iterations = sum(1 for m in session.messages if m.role == "assistant")

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

        await self.event_bus.emit(
            Events.SUBAGENT_COMPLETED,
            {
                "task_id": task.task_id,
                "role": task.role.name,
                "tokens": total_tokens,
                "duration_ms": duration_ms,
            },
        )

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

        await self.event_bus.emit(
            Events.SUBAGENT_FAILED,
            {
                "task_id": task.task_id,
                "error": str(e),
            },
        )

        return result


async def _execute_subagent_via_sdk(
    self: SubAgentOrchestrator,
    task: SubAgentTask,
    scoped_registry: ScopedToolRegistry,
) -> SubAgentResult:
    """Run a sub-agent task via Claude SDK (no API key needed)."""
    import time as _time

    start = _time.monotonic()

    await self.event_bus.emit(
        Events.SUBAGENT_STARTED,
        {
            "task_id": task.task_id,
            "role": task.role.name,
        },
    )

    # Build instruction with context
    full_instruction = task.instruction
    if task.context:
        full_instruction = f"Context:\n{task.context}\n\nTask:\n{task.instruction}"

    # Build a sub-agent tool executor for safety routing
    sub_executor = None
    parent_executor = self.agent_loop.tool_executor
    if parent_executor is not None:
        from agent.tools.executor import ToolExecutor

        sub_executor = ToolExecutor(
            registry=scoped_registry,  # type: ignore[arg-type]
            config=parent_executor.config,
            event_bus=self.event_bus,
            audit=parent_executor.audit,
            permissions=parent_executor.permissions,
            guardrails=parent_executor.guardrails,
        )

    try:
        response_text = await self.sdk_service.run_subagent(  # type: ignore[union-attr]
            prompt=full_instruction,
            task_id=task.task_id,
            role_persona=task.role.persona,
            scoped_registry=scoped_registry,
            model=getattr(task.role, "model", None),
            max_turns=task.role.max_iterations,
            task_context=task.context,
            tool_executor=sub_executor,
            nesting_depth=task.nesting_depth,
        )

        duration_ms = int((_time.monotonic() - start) * 1000)

        # Read metrics collected during SDK sub-agent execution
        sdk_metrics = {}
        if self.sdk_service and hasattr(self.sdk_service, "_subagent_metrics"):
            sdk_metrics = self.sdk_service._subagent_metrics.pop(
                task.task_id,
                {},
            )

        result = SubAgentResult(
            task_id=task.task_id,
            role_name=task.role.name,
            status=SubAgentStatus.COMPLETED,
            output=response_text,
            duration_ms=duration_ms,
            tool_calls_made=int(sdk_metrics.get("tool_calls", 0) or 0),
            iterations=int(sdk_metrics.get("iterations", 0) or 0),
        )

        await self.event_bus.emit(
            Events.SUBAGENT_COMPLETED,
            {
                "task_id": task.task_id,
                "role": task.role.name,
                "duration_ms": duration_ms,
            },
        )

        return result

    except Exception as e:
        duration_ms = int((_time.monotonic() - start) * 1000)
        result = SubAgentResult(
            task_id=task.task_id,
            role_name=task.role.name,
            status=SubAgentStatus.FAILED,
            error=str(e),
            duration_ms=duration_ms,
        )

        await self.event_bus.emit(
            Events.SUBAGENT_FAILED,
            {
                "task_id": task.task_id,
                "error": str(e),
            },
        )

        return result
