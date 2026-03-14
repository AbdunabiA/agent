"""Orchestration tools — let the LLM spawn sub-agents and teams."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.core.orchestrator import SubAgentOrchestrator

_global_orchestrator: SubAgentOrchestrator | None = None


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
            "Orchestrator not initialized. "
            "Enable orchestration in config and restart."
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

    Returns:
        Sub-agent result.
    """
    from agent.core.subagent import SubAgentRole, SubAgentTask

    orchestrator = get_orchestrator()

    tools_list = [t.strip() for t in allowed_tools.split(",") if t.strip()] or []

    role = SubAgentRole(
        name=role_name,
        persona=persona,
        model=model or None,
        allowed_tools=tools_list,
        max_iterations=min(max_iterations, 10),
    )

    task = SubAgentTask(
        role=role,
        instruction=instruction,
        context=context,
    )

    result = await orchestrator.spawn_subagent(task)

    lines = [
        f"Sub-agent '{result.role_name}' [{result.status}]",
    ]
    if result.output:
        lines.append(f"\nOutput:\n{result.output}")
    if result.error:
        lines.append(f"\nError: {result.error}")
    lines.append(
        f"\nStats: {result.iterations} iterations, "
        f"{result.tool_calls_made} tool calls, "
        f"{result.token_usage} tokens, "
        f"{result.duration_ms}ms"
    )

    return "\n".join(lines)


@tool(
    name="spawn_parallel_agents",
    description=(
        "Spawn multiple sub-agents in parallel. Each agent runs "
        "independently and results are collected when all finish. "
        "Provide a JSON array of agent configs."
    ),
    tier=ToolTier.MODERATE,
)
async def spawn_parallel_tool(agents: str) -> str:
    """Spawn multiple sub-agents concurrently.

    Args:
        agents: JSON array of objects with keys: role_name, instruction,
                persona (optional), context (optional), allowed_tools (optional).

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
            allowed_tools=allowed,
            max_iterations=min(spec.get("max_iterations", 5), 10),
        )
        tasks.append(SubAgentTask(
            role=role,
            instruction=spec.get("instruction", ""),
            context=spec.get("context", ""),
        ))

    if not tasks:
        return "No valid agent specs provided"

    results = await orchestrator.spawn_parallel(tasks)

    lines = [f"Parallel Results ({len(results)} agents):"]
    for r in results:
        lines.append(f"\n--- {r.role_name} [{r.status}] ---")
        if r.output:
            lines.append(r.output[:1000])
        if r.error:
            lines.append(f"Error: {r.error}")
        lines.append(
            f"({r.iterations} iterations, {r.tool_calls_made} tool calls, "
            f"{r.token_usage} tokens, {r.duration_ms}ms)"
        )

    return "\n".join(lines)


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
) -> str:
    """Spawn a team of sub-agents.

    Args:
        team_name: Name of the pre-defined team.
        instruction: Task for the team to accomplish.
        context: Additional context for all team members.

    Returns:
        Combined team results.
    """
    orchestrator = get_orchestrator()

    results = await orchestrator.spawn_team(
        team_name=team_name,
        instruction=instruction,
        context=context,
    )

    lines = [f"Team '{team_name}' Results ({len(results)} agents):"]
    for r in results:
        lines.append(f"\n--- {r.role_name} [{r.status}] ---")
        if r.output:
            lines.append(r.output[:1000])
        if r.error:
            lines.append(f"Error: {r.error}")

    return "\n".join(lines)


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
        lines.append(f"Output: {result.output[:500]}")
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
