"""Planner tools — entry point for the intelligent project planner."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.core.project_planner import ProjectPlannerService

logger = structlog.get_logger(__name__)

_global_planner_service: ProjectPlannerService | None = None


def set_planner_service(service: ProjectPlannerService) -> None:
    """Set the global ProjectPlannerService (called during startup)."""
    global _global_planner_service
    _global_planner_service = service


def get_planner_service() -> ProjectPlannerService:
    """Get the global ProjectPlannerService.

    Raises:
        RuntimeError: If set_planner_service() hasn't been called.
    """
    if _global_planner_service is None:
        raise RuntimeError(
            "ProjectPlannerService not initialized. "
            "Enable orchestration with claude-sdk backend and restart."
        )
    return _global_planner_service


@tool(
    name="plan_and_build",
    description=(
        "Intelligently plan and build a complex multi-component project. "
        "Gathers requirements interactively, decomposes into micro-tasks "
        "with dependency graph, then executes tasks one-by-one with "
        "quality gates. Use for SaaS, platforms, apps with multiple "
        "frontends, or any project requiring 10+ coordinated tasks."
    ),
    tier=ToolTier.MODERATE,
)
async def plan_and_build_tool(
    instruction: str,
    skip_requirements: bool = False,
) -> str:
    """Plan and build a complex project.

    Args:
        instruction: High-level project description (e.g. "Build a task
            management SaaS with web and mobile apps").
        skip_requirements: If True, skip the interactive requirements
            gathering phase and use the instruction directly.

    Returns:
        Summary of the project execution result.
    """
    service = get_planner_service()

    try:
        result = await service.plan_and_build(
            instruction=instruction,
            skip_requirements=skip_requirements,
        )

        if result.success:
            lines = [
                "Project completed successfully!",
                f"Tasks: {result.tasks_completed}/{result.total_tasks} passed",
            ]
            if result.spec:
                lines.insert(0, f"Project: {result.spec.title}")
            if result.summary:
                lines.append(f"\n{result.summary}")
            return "\n".join(lines)
        else:
            lines = [
                "Project partially completed.",
                f"Tasks: {result.tasks_completed}/{result.total_tasks} passed, "
                f"{result.tasks_failed} failed",
            ]
            if result.summary:
                lines.append(f"\n{result.summary}")
            return "\n".join(lines)

    except Exception as e:
        logger.error("plan_and_build_failed", error=str(e))
        return f"Project planning failed: {e}"
