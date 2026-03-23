"""Orchestration tools — let the LLM spawn sub-agents and teams."""

from __future__ import annotations

import asyncio
import contextvars
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog

from agent.core.events import Events
from agent.core.subagent import SubAgentStatus
from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from agent.core.orchestrator import SubAgentOrchestrator

_global_orchestrator: SubAgentOrchestrator | None = None

# Callback to register task→user mapping for status updates.
# Set by the Telegram channel (or any channel that wants notifications).
_task_user_callback: Callable[[str, str], None] | None = None


def set_task_user_callback(
    callback: Callable[[str, str], None] | None,
) -> None:
    """Set a callback(task_id, user_id) for tracking task ownership."""
    global _task_user_callback
    _task_user_callback = callback


def _register_task_user(task_id: str) -> None:
    """Register the current user_id for a task via context var."""
    if _task_user_callback is None:
        return
    try:
        from agent.tools.builtins.scheduler import _user_id_var

        user_id = _user_id_var.get("")
        if user_id:
            _task_user_callback(task_id, user_id)
    except Exception:
        pass


# Context var tracks the nesting depth of the currently executing task.
# Set by the orchestrator before running each subagent so that tools
# can determine the correct depth without scanning all running tasks.
_nesting_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "orchestration_nesting_depth",
    default=0,
)


def set_nesting_depth(depth: int) -> None:
    """Set the nesting depth for the current task context."""
    _nesting_depth_var.set(depth)


def get_nesting_depth() -> int:
    """Get the nesting depth for the current task context."""
    return _nesting_depth_var.get()


def set_orchestrator(orchestrator: SubAgentOrchestrator) -> None:
    """Set the global SubAgentOrchestrator instance (called during startup).

    Args:
        orchestrator: The initialized SubAgentOrchestrator.
    """
    global _global_orchestrator
    _global_orchestrator = orchestrator


def get_orchestrator() -> SubAgentOrchestrator:
    """Get the global SubAgentOrchestrator instance.

    Returns:
        The shared SubAgentOrchestrator.

    Raises:
        RuntimeError: If set_orchestrator() hasn't been called yet.
    """
    if _global_orchestrator is None:
        raise RuntimeError(
            "Orchestrator not initialized. " "Enable orchestration in config and restart."
        )
    return _global_orchestrator


@tool(
    name="spawn_subagent",
    description=(
        "Spawn a sub-agent with a specific role and instruction. "
        "The sub-agent runs independently with its own session and "
        "scoped tools. Use this for delegating focused tasks like "
        "research, code review, or data analysis."
    ),
    tier=ToolTier.MODERATE,
)
async def spawn_subagent_tool(
    role_name: str,
    instruction: str,
    persona: str = "You are a helpful assistant.",
    context: str = "",
    model: str = "",
    allowed_tools: str = "",
    max_iterations: int = 5,
    mode: str = "sync",
) -> str:
    """Spawn a single sub-agent.

    Args:
        role_name: Name for this sub-agent role.
        instruction: What the sub-agent should do.
        persona: Sub-agent personality/system prompt.
        context: Additional context to provide.
        model: Optional model override.
        allowed_tools: Comma-separated list of tool names (empty = all safe+moderate).
        max_iterations: Max tool-calling iterations.
        mode: "sync" to wait for results, "async" to fire and forget.

    Returns:
        Sub-agent result.
    """
    from agent.core.subagent import SubAgentRole, SubAgentTask

    orchestrator = get_orchestrator()

    # Look up the role in the registry — use the detailed persona
    # from teams/*.yaml instead of the generic one the controller passes.
    registry_role: SubAgentRole | None = None
    try:
        if hasattr(orchestrator, "_role_registry"):
            registry_role = orchestrator._role_registry.get_role(role_name)  # type: ignore[union-attr]
    except Exception:
        pass

    # Also try via controller's registry
    if registry_role is None:
        try:
            from agent.tools.builtins.controller import get_controller

            controller = get_controller()
            if controller.role_registry is not None:
                clean_name = role_name.split("/")[-1] if "/" in role_name else role_name
                registry_role = controller.role_registry.get_role(clean_name)
        except Exception:
            pass

    if registry_role is not None:
        role = SubAgentRole(
            name=registry_role.name,
            persona=registry_role.persona,
            model=model or registry_role.model,
            allowed_tools=registry_role.allowed_tools,
            denied_tools=registry_role.denied_tools,
            max_iterations=(
                min(max_iterations, 100) if max_iterations != 5 else registry_role.max_iterations
            ),
        )
        logger.info(
            "spawn_subagent_role_from_registry",
            role=role.name,
            persona_len=len(role.persona),
        )
    else:
        tools_list = [t.strip() for t in allowed_tools.split(",") if t.strip()] or []
        role = SubAgentRole(
            name=role_name,
            persona=persona,
            model=model or None,
            allowed_tools=tools_list,
            max_iterations=min(max_iterations, 100),
        )

    task = SubAgentTask(
        role=role,
        instruction=instruction,
        context=context,
        timeout_seconds=orchestrator.config.subagent_timeout,
    )

    import uuid as _uuid

    task_id = task.task_id or f"sa-{_uuid.uuid4().hex[:8]}"
    task.task_id = task_id
    _register_task_user(task_id)

    if mode == "sync":
        result = await orchestrator.spawn_subagent(task)
        if result.status == SubAgentStatus.COMPLETED:
            return f"Sub-agent '{role_name}' completed:\n\n{result.output}"
        return f"Sub-agent '{role_name}' failed: {result.error or 'unknown error'}"

    # Fire and forget — the subagent runs in the background.
    # Status updates are pushed to the user via event bus.
    future = asyncio.ensure_future(orchestrator.spawn_subagent(task))
    orchestrator._async_futures[task_id] = future

    return (
        f"Sub-agent '{role_name}' spawned (task_id: {task_id}).\n"
        f"It will work in the background. Status updates are sent "
        f"automatically. Use get_subagent_status to check results.\n"
        f"You are now free to continue chatting with the user."
    )


@tool(
    name="spawn_parallel_agents",
    description=(
        "Spawn multiple sub-agents in parallel. Each agent runs "
        "independently and results are collected when all finish. "
        "Provide a JSON array of agent configs."
    ),
    tier=ToolTier.MODERATE,
)
async def spawn_parallel_tool(agents: str, mode: str = "sync") -> str:
    """Spawn multiple sub-agents concurrently.

    Args:
        agents: JSON array of objects with keys: role_name, instruction,
                persona (optional), context (optional), allowed_tools (optional).
        mode: "sync" to wait for all results, "async" to fire and forget.

    Returns:
        Combined results from all sub-agents.
    """
    from agent.core.subagent import SubAgentRole, SubAgentTask

    orchestrator = get_orchestrator()

    try:
        agent_specs = json.loads(agents)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    if not isinstance(agent_specs, list):
        return "Expected a JSON array of agent specifications"

    tasks: list[SubAgentTask] = []
    for spec in agent_specs:
        if not isinstance(spec, dict):
            continue
        raw_tools = spec.get("allowed_tools", [])
        if isinstance(raw_tools, str):
            allowed = [t.strip() for t in raw_tools.split(",") if t.strip()]
        else:
            allowed = list(raw_tools)
        role = SubAgentRole(
            name=spec.get("role_name", "agent"),
            persona=spec.get("persona", "You are a helpful assistant."),
            model=spec.get("model") or None,
            allowed_tools=allowed,
            max_iterations=min(spec.get("max_iterations", 60), 100),
        )
        tasks.append(
            SubAgentTask(
                role=role,
                instruction=spec.get("instruction", ""),
                context=spec.get("context", ""),
                timeout_seconds=orchestrator.config.subagent_timeout,
            )
        )

    if not tasks:
        return "No valid agent specs provided"

    import uuid as _uuid

    group_id = f"par-{_uuid.uuid4().hex[:8]}"
    _register_task_user(group_id)

    if mode == "sync":
        results = await orchestrator.spawn_parallel(tasks)
        lines = [f"All {len(results)} agents completed:\n"]
        for r in results:
            status_icon = "\u2705" if r.status == SubAgentStatus.COMPLETED else "\u274c"
            lines.append(f"{status_icon} {r.role_name}:")
            if r.output:
                lines.append(r.output)
            if r.error:
                lines.append(f"Error: {r.error}")
            lines.append("")
        return "\n".join(lines)

    # Fire and forget — agents run in the background.
    future = asyncio.ensure_future(orchestrator.spawn_parallel(tasks))
    orchestrator._async_futures[group_id] = future

    agent_names = [t.role.name for t in tasks]
    return (
        f"Spawned {len(tasks)} agents in parallel (group: {group_id}):\n"
        f"  {', '.join(agent_names)}\n"
        f"They will work in the background. Status updates are sent "
        f"automatically. You are free to continue chatting."
    )


@tool(
    name="spawn_team",
    description=(
        "Spawn a pre-defined team of sub-agents to work on a task together. "
        "Each team member has a specialized role and tool set. "
        "Use list_agent_teams to see available teams."
    ),
    tier=ToolTier.MODERATE,
)
async def spawn_team_tool(
    team_name: str,
    instruction: str,
    context: str = "",
    mode: str = "sync",
) -> str:
    """Spawn a team of sub-agents.

    Args:
        team_name: Name of the pre-defined team.
        instruction: Task for the team to accomplish.
        context: Additional context for all team members.
        mode: "sync" to wait for results, "async" to fire and forget.

    Returns:
        Combined team results.
    """
    orchestrator = get_orchestrator()

    import uuid as _uuid

    group_id = f"team-{_uuid.uuid4().hex[:8]}"
    _register_task_user(group_id)

    if mode == "sync":
        results = await orchestrator.spawn_team(
            team_name=team_name,
            instruction=instruction,
            context=context,
        )
        lines = [f"Team '{team_name}' completed ({len(results)} agents):\n"]
        for r in results:
            status_icon = "\u2705" if r.status == SubAgentStatus.COMPLETED else "\u274c"
            lines.append(f"{status_icon} {r.role_name}:")
            if r.output:
                lines.append(r.output)
            if r.error:
                lines.append(f"Error: {r.error}")
            lines.append("")
        return "\n".join(lines)

    # Fire and forget — team runs in the background.
    future = asyncio.ensure_future(
        orchestrator.spawn_team(
            team_name=team_name,
            instruction=instruction,
            context=context,
        )
    )
    orchestrator._async_futures[group_id] = future

    team = orchestrator.teams.get(team_name)
    role_names = [r.name for r in team.roles] if team else ["(unknown)"]
    return (
        f"Team '{team_name}' spawned (group: {group_id}):\n"
        f"  Roles: {', '.join(role_names)}\n"
        f"They will work in the background. Status updates are sent "
        f"automatically. You are free to continue chatting."
    )


@tool(
    name="list_agent_teams",
    description="List all pre-defined sub-agent teams and their roles.",
    tier=ToolTier.SAFE,
)
async def list_teams_tool() -> str:
    """List available teams.

    Returns:
        Formatted list of teams.
    """
    orchestrator = get_orchestrator()
    teams = orchestrator.list_teams()

    if not teams:
        return "No teams configured. Add teams in agent.yaml under orchestration.teams."

    lines = [f"Available Teams ({len(teams)}):"]
    for team in teams:
        lines.append(f"\n  {team['name']}: {team['description']}")
        for role in team["roles"]:
            tools_str = ", ".join(role["allowed_tools"]) if role["allowed_tools"] else "all"
            lines.append(f"    - {role['name']}: {role['persona']} (tools: {tools_str})")

    return "\n".join(lines)


@tool(
    name="get_subagent_status",
    description="Get the status or result of a previously spawned sub-agent.",
    tier=ToolTier.SAFE,
)
async def get_status_tool(task_id: str) -> str:
    """Get sub-agent status.

    Args:
        task_id: The task ID returned when the sub-agent was spawned.

    Returns:
        Status information.
    """
    orchestrator = get_orchestrator()
    result = orchestrator.get_status(task_id)

    if not result:
        return f"No sub-agent found with task ID: {task_id}"

    lines = [
        f"Sub-agent '{result.role_name}' (task {result.task_id})",
        f"Status: {result.status}",
    ]
    if result.output:
        lines.append(f"Output: {result.output}")
    if result.error:
        lines.append(f"Error: {result.error}")

    return "\n".join(lines)


@tool(
    name="cancel_subagent",
    description="Cancel a running sub-agent by its task ID.",
    tier=ToolTier.MODERATE,
)
async def cancel_subagent_tool(task_id: str) -> str:
    """Cancel a running sub-agent.

    Args:
        task_id: The task ID to cancel.

    Returns:
        Status message.
    """
    orchestrator = get_orchestrator()
    cancelled = await orchestrator.cancel(task_id)

    if cancelled:
        return f"Sub-agent {task_id} cancelled."
    return f"Sub-agent {task_id} not found or already completed."


@tool(
    name="consult_agent",
    description=(
        "Consult another agent for their expert opinion mid-task. "
        "Specify a team and role to consult. The consulted agent "
        "runs synchronously and returns their response. Use this "
        "when you need a specialist's input before continuing."
    ),
    tier=ToolTier.MODERATE,
)
async def consult_agent_tool(
    team_name: str,
    role_name: str,
    question: str,
    context: str = "",
) -> str:
    """Consult another agent for expert input.

    Args:
        team_name: Name of the team the target agent belongs to.
        role_name: Role name of the agent to consult.
        question: The question to ask the consulted agent.
        context: Optional additional context for the consultation.

    Returns:
        The consulted agent's response.
    """
    from agent.core.subagent import ConsultRequest

    orchestrator = get_orchestrator()

    # Use context var for accurate caller-specific nesting depth
    nesting_depth = get_nesting_depth()

    import uuid as _uuid

    request = ConsultRequest(
        requesting_agent_id=f"tool-{_uuid.uuid4().hex[:8]}",
        requesting_role="current_agent",
        target_team=team_name,
        target_role=role_name,
        question=question,
        context=context,
    )

    response = await orchestrator.handle_consult(request, nesting_depth)

    if response.status == SubAgentStatus.COMPLETED:
        lines = [
            f"Consultation with {response.target_role} [{response.status}]:",
            "",
            response.answer,
            "",
            f"({response.duration_ms}ms, {response.token_usage} tokens)",
        ]
    else:
        lines = [
            f"Consultation with {response.target_role} [{response.status}]:",
            f"Error: {response.error}",
        ]

    return "\n".join(lines)


@tool(
    name="delegate_to_specialist",
    description=(
        "Delegate a focused subtask to a specialist agent from a "
        "pre-defined team. The specialist runs with its own tools "
        "and returns a deliverable. Use mode='sync' to wait for "
        "results, or mode='async' to continue working and check "
        "results later with get_subagent_status."
    ),
    tier=ToolTier.MODERATE,
)
async def delegate_to_specialist_tool(
    team_name: str,
    role_name: str,
    instruction: str,
    context: str = "",
    mode: str = "sync",
) -> str:
    """Delegate a subtask to a specialist agent.

    Args:
        team_name: Name of the team the specialist belongs to.
        role_name: Role name of the specialist.
        instruction: What the specialist should do.
        context: Optional additional context.
        mode: "sync" to wait for results, "async" to fire and forget.

    Returns:
        Specialist's output or task_id for async polling.
    """
    from agent.core.subagent import DelegationMode, DelegationRequest

    orchestrator = get_orchestrator()

    # Use context var for accurate caller-specific nesting depth
    nesting_depth = get_nesting_depth()

    delegation_mode = DelegationMode.ASYNC if mode == "async" else DelegationMode.SYNC

    import uuid as _uuid

    request = DelegationRequest(
        delegating_agent_id=f"tool-{_uuid.uuid4().hex[:8]}",
        delegating_role="current_agent",
        target_team=team_name,
        target_role=role_name,
        instruction=instruction,
        context=context,
        mode=delegation_mode,
    )

    result = await orchestrator.handle_delegation(request, nesting_depth)

    if result.status == SubAgentStatus.PENDING:
        return (
            f"Delegation to {result.target_role} started (async).\n"
            f"Task ID: {result.task_id}\n"
            f"Use get_subagent_status to check results."
        )
    elif result.status == SubAgentStatus.COMPLETED:
        lines = [
            f"Delegation to {result.target_role} [{result.status}]:",
            "",
            result.output,
            "",
            f"({result.duration_ms}ms, {result.token_usage} tokens)",
        ]
        return "\n".join(lines)
    else:
        return f"Delegation to {result.target_role} [{result.status}]: " f"{result.error}"


@tool(
    name="run_project",
    description=(
        "Run a cross-team project pipeline. Projects execute stages "
        "sequentially — within each stage, agents run in parallel. "
        "Each stage's output feeds into the next stage as context. "
        "Use list_projects to see available projects."
    ),
    tier=ToolTier.MODERATE,
)
async def run_project_tool(
    project_name: str,
    instruction: str,
    context: str = "",
    mode: str = "auto",
) -> str:
    """Run a project pipeline.

    Args:
        project_name: Name of the project to run.
        instruction: Task instruction for the project.
        context: Initial context for the first stage.
        mode: "sync" to wait for results, "async" to fire and forget.

    Returns:
        Combined project results from all stages.
    """
    orchestrator = get_orchestrator()

    import uuid as _uuid

    task_id = f"proj-{_uuid.uuid4().hex[:8]}"

    # Validate project exists before registering user
    proj = orchestrator.projects.get(project_name)
    if not proj:
        available = ", ".join(orchestrator.projects.keys()) or "none"
        return f"Unknown project: '{project_name}'. Available: {available}"

    _register_task_user(task_id)

    # Auto mode: use async for multi-stage projects to avoid blocking
    if mode == "auto":
        stage_count = len([s for s in proj.stages if not s.feedback_target])
        mode = "async" if stage_count > 1 else "sync"

    if mode == "sync":
        project_result = await orchestrator.run_project(
            project_name=project_name,
            instruction=instruction,
            context=context,
        )
        status_icon = "\u2705" if project_result.status == SubAgentStatus.COMPLETED else "\u274c"
        lines = [
            f"{status_icon} Project '{project_name}' " f"({len(project_result.stages)} stages):\n"
        ]
        if project_result.final_output:
            lines.append(project_result.final_output[:2000])
        else:
            for stage in project_result.stages:
                lines.append(f"\n--- {stage.stage_name} ---")
                if stage.combined_output:
                    lines.append(stage.combined_output)
                for r in stage.results:
                    if r.error:
                        lines.append(f"Error ({r.role_name}): {r.error}")
        if project_result.error:
            lines.append(f"\nProject error: {project_result.error}")
        return "\n".join(lines)

    # Fire and forget — project pipeline runs in the background.
    # Status updates for each stage/agent are pushed via event bus.
    future = asyncio.ensure_future(
        orchestrator.run_project(
            project_name=project_name,
            instruction=instruction,
            context=context,
        )
    )
    orchestrator._async_futures[task_id] = future

    # Capture user_id NOW (before the callback fires) via the context var
    # that was set by the Telegram channel's _register_task_user callback.
    _captured_user_id = ""
    try:
        from agent.tools.builtins.scheduler import _user_id_var

        _captured_user_id = _user_id_var.get("")
    except Exception:
        pass

    # Done-callback: emit TASK_COMPLETED_NOTIFY with full project summary
    def _on_project_done(
        fut: asyncio.Future,  # type: ignore[type-arg]
        _user_id: str = _captured_user_id,
    ) -> None:
        # CancelledError is a BaseException — must be caught explicitly
        # so it doesn't produce an unhandled exception traceback.
        if fut.cancelled():
            logger.info(
                "project_cancelled",
                task_id=task_id,
            )
            return

        exc = fut.exception()
        if exc is not None:
            logger.error(
                "project_done_error",
                task_id=task_id,
                error=str(exc),
            )
            return

        try:
            project_result = fut.result()
            # Build summary from stage results
            summary_parts: list[str] = []
            for sr in project_result.stages:
                if sr.combined_output:
                    summary_parts.append(f"[{sr.stage_name}]:\n{sr.combined_output[:300]}")
            summary = "\n\n".join(summary_parts)
            if project_result.final_output:
                summary = project_result.final_output

            asyncio.get_event_loop().create_task(
                orchestrator.event_bus.emit(
                    Events.TASK_COMPLETED_NOTIFY,
                    {
                        "task_id": task_id,
                        "user_id": _user_id,
                        "result": summary or "Project completed.",
                        "duration_seconds": round(
                            project_result.duration_ms / 1000,
                            1,
                        ),
                        "summary": summary,
                    },
                )
            )
        except Exception:
            pass

    future.add_done_callback(_on_project_done)

    stage_names = [s.name for s in proj.stages] if proj else ["(unknown)"]
    return (
        f"Project '{project_name}' started (task: {task_id}).\n"
        f"  Stages: {' -> '.join(stage_names)}\n"
        f"Status updates for each stage and agent will be sent "
        f"automatically. You are free to continue chatting with the user.\n"
        f"Use get_subagent_status with task_id '{task_id}' to check progress."
    )


@tool(
    name="list_projects",
    description="List all available cross-team project pipelines.",
    tier=ToolTier.SAFE,
)
async def list_projects_tool() -> str:
    """List available projects.

    Returns:
        Formatted list of projects and their stages.
    """
    orchestrator = get_orchestrator()
    projects = orchestrator.list_projects()

    if not projects:
        return "No projects configured. " "Add YAML files to teams/projects/ to define pipelines."

    lines = [f"Available Projects ({len(projects)}):"]
    for proj in projects:
        lines.append(f"\n  {proj['name']}: {proj['description']}")
        for stage in proj["stages"]:
            agents_str = ", ".join(f"{a['team']}/{a['role']}" for a in stage["agents"])
            lines.append(f"    Stage '{stage['name']}': {agents_str}")

    return "\n".join(lines)
