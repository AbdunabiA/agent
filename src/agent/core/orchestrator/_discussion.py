"""Discussion stage methods for SubAgentOrchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent.core.events import Events
from agent.core.subagent import (
    ProjectStage,
    ProjectStageResult,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
    SubAgentTask,
)

if TYPE_CHECKING:
    from agent.core.orchestrator._core import SubAgentOrchestrator

logger = structlog.get_logger(__name__)


async def _run_discussion_stage(
    self: SubAgentOrchestrator,
    project_name: str,
    stage: ProjectStage,
    instruction: str,
    context: str,
    parent_session_id: str,
) -> tuple[ProjectStageResult, bool]:
    """Execute a discussion-mode project stage.

    Agents take turns responding over multiple rounds. An optional
    moderator summarizes each round and checks for consensus.

    Args:
        project_name: Name of the parent project.
        stage: The discussion stage to execute.
        instruction: Task instruction for agents.
        context: Accumulated context from prior stages.
        parent_session_id: Parent session ID.

    Returns:
        Tuple of (stage_result, has_failure).
    """
    import time as _time

    if stage.discussion is None:
        raise ValueError(f"Stage '{stage.name}' missing discussion config")

    stage_start = _time.monotonic()
    discussion = stage.discussion

    await self.event_bus.emit(
        Events.PROJECT_STAGE_STARTED,
        {
            "project": project_name,
            "stage": stage.name,
            "agents": len(stage.agents),
            "mode": "discussion",
        },
    )
    await self.event_bus.emit(
        Events.DISCUSSION_STARTED,
        {
            "project": project_name,
            "stage": stage.name,
            "rounds": discussion.rounds,
            "agents": len(stage.agents),
        },
    )

    # Resolve all agent refs to roles
    agent_roles: list[tuple[str, SubAgentRole]] = []
    for agent_ref in stage.agents:
        team = self.teams.get(agent_ref.team)
        if not team:
            error = f"RESOLUTION_ERROR: Stage '{stage.name}': " f"team '{agent_ref.team}' not found"
            return ProjectStageResult(
                stage_name=stage.name,
                results=[
                    SubAgentResult(
                        task_id="",
                        role_name="",
                        status=SubAgentStatus.FAILED,
                        error=error,
                    )
                ],
            ), True

        role = next(
            (r for r in team.roles if r.name == agent_ref.role),
            None,
        )
        if not role:
            error = (
                f"RESOLUTION_ERROR: Stage '{stage.name}': "
                f"role '{agent_ref.role}' not found "
                f"in team '{agent_ref.team}'"
            )
            return ProjectStageResult(
                stage_name=stage.name,
                results=[
                    SubAgentResult(
                        task_id="",
                        role_name="",
                        status=SubAgentStatus.FAILED,
                        error=error,
                    )
                ],
            ), True

        agent_roles.append((f"{agent_ref.team}/{agent_ref.role}", role))

    # Resolve moderator if configured
    moderator_role: SubAgentRole | None = None
    moderator_label: str = ""
    if discussion.moderator:
        mod_team = self.teams.get(discussion.moderator.team)
        if mod_team:
            moderator_role = next(
                (r for r in mod_team.roles if r.name == discussion.moderator.role),
                None,
            )
            moderator_label = f"{discussion.moderator.team}/{discussion.moderator.role}"

    # Warn if consensus_required but no moderator to evaluate it
    if discussion.consensus_required and not moderator_role:
        logger.warning(
            "consensus_requires_moderator",
            project=project_name,
            stage=stage.name,
            msg="consensus_required=True but no moderator configured; "
            "consensus can never be reached, all rounds will execute",
        )

    transcript = ""
    all_results: list[SubAgentResult] = []
    has_failure = False

    consensus_reached = False
    for round_num in range(1, discussion.rounds + 1):
        if round_num == 1 and len(agent_roles) > 1:
            # Round 1: all agents start from the same context — run in parallel
            tasks = []
            for _label, role in agent_roles:
                agent_context = context + ("\n\nRound 1: Share your initial thoughts.")
                tasks.append(
                    SubAgentTask(
                        role=role,
                        instruction=instruction,
                        context=agent_context,
                        parent_session_id=parent_session_id,
                    )
                )

            round_results = await self.spawn_parallel(tasks)

            for (label, _role), result in zip(agent_roles, round_results, strict=False):
                all_results.append(result)
                if result.status == SubAgentStatus.COMPLETED and result.output:
                    transcript += f"\n\n[Round {round_num}] [{label}]:\n" f"{result.output}"
                elif result.status == SubAgentStatus.FAILED:
                    has_failure = True
                    transcript += f"\n\n[Round {round_num}] [{label}] FAILED: " f"{result.error}"
        else:
            # Subsequent rounds (or single-agent round 1): agents respond
            # sequentially to build on each other's contributions.
            for label, role in agent_roles:
                agent_context = context
                if transcript:
                    agent_context += f"\n\n--- Discussion so far ---\n{transcript}"
                if round_num == 1 and not transcript:
                    agent_context += f"\n\nRound {round_num}: Share your initial thoughts."
                else:
                    agent_context += f"\n\nRound {round_num}: " f"It's your turn to contribute."

                task = SubAgentTask(
                    role=role,
                    instruction=instruction,
                    context=agent_context,
                    parent_session_id=parent_session_id,
                )
                result = await self.spawn_subagent(task)
                all_results.append(result)

                if result.status == SubAgentStatus.COMPLETED and result.output:
                    transcript += f"\n\n[Round {round_num}] [{label}]:\n" f"{result.output}"
                elif result.status == SubAgentStatus.FAILED:
                    has_failure = True
                    transcript += f"\n\n[Round {round_num}] [{label}] FAILED: " f"{result.error}"

        # Moderator summary
        if moderator_role:
            mod_context = context
            if transcript:
                mod_context += f"\n\n--- Discussion so far ---\n{transcript}"
            mod_context += (
                "\n\nSummarize the discussion so far. "
                "State CONSENSUS if all participants agree, "
                "or CONTINUE if more discussion needed."
            )

            mod_task = SubAgentTask(
                role=moderator_role,
                instruction=instruction,
                context=mod_context,
                parent_session_id=parent_session_id,
            )
            mod_result = await self.spawn_subagent(mod_task)
            all_results.append(mod_result)

            if mod_result.status == SubAgentStatus.COMPLETED:
                transcript += (
                    f"\n\n[Round {round_num}] "
                    f"[Moderator - {moderator_label}]:\n"
                    f"{mod_result.output}"
                )

                consensus_reached = (
                    discussion.consensus_required
                    and "CONSENSUS" in (mod_result.output or "").upper()
                )

                if consensus_reached:
                    await self.event_bus.emit(
                        Events.DISCUSSION_CONSENSUS_REACHED,
                        {
                            "project": project_name,
                            "stage": stage.name,
                            "round": round_num,
                        },
                    )

        await self.event_bus.emit(
            Events.DISCUSSION_ROUND_COMPLETED,
            {
                "project": project_name,
                "stage": stage.name,
                "round": round_num,
            },
        )

        if consensus_reached:
            break

    stage_duration = int((_time.monotonic() - stage_start) * 1000)

    await self.event_bus.emit(
        Events.DISCUSSION_COMPLETED,
        {
            "project": project_name,
            "stage": stage.name,
            "duration_ms": stage_duration,
        },
    )

    stage_result = ProjectStageResult(
        stage_name=stage.name,
        results=all_results,
        combined_output=transcript.strip(),
        duration_ms=stage_duration,
    )

    await self.event_bus.emit(
        Events.PROJECT_STAGE_COMPLETED,
        {
            "project": project_name,
            "stage": stage.name,
            "agents_completed": sum(1 for r in all_results if r.status == SubAgentStatus.COMPLETED),
            "agents_failed": sum(1 for r in all_results if r.status == SubAgentStatus.FAILED),
            "duration_ms": stage_duration,
        },
    )

    return stage_result, has_failure
