"""Sub-agent orchestrator — spawns and manages concurrent sub-agents.

Implements the orchestrator pattern: main agent spawns sub-agents with
scoped tools, configurable personas, and independent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from uuid import uuid4 as _uuid4

import structlog

from agent.core.events import Events
from agent.core.session import Session
from agent.core.subagent import (
    AgentTeam,
    ConsultRequest,
    ConsultResponse,
    DelegationMode,
    DelegationRequest,
    DelegationResult,
    Project,
    ProjectResult,
    ProjectStage,
    ProjectStageResult,
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
    ) -> None:
        self.agent_loop = agent_loop
        self.config = config
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self.sdk_service = sdk_service
        self.teams = {t.name: t for t in (teams or [])}
        self.projects: dict[str, Project] = {}

        self._running_tasks: dict[str, asyncio.Task[SubAgentResult]] = {}
        self._results: dict[str, SubAgentResult] = {}
        self._async_futures: dict[str, asyncio.Future[SubAgentResult]] = {}
        self._task_nesting_depths: dict[str, int] = {}
        self._spawn_lock = asyncio.Lock()
        self._max_results = 500  # prune _results after this many entries

    async def spawn_subagent(self, task: SubAgentTask) -> SubAgentResult:
        """Spawn a single sub-agent and wait for its result.

        Creates a scoped session and tool registry, builds a sub-agent
        prompt, and runs the agent loop with constraints.

        Args:
            task: The sub-agent task to execute.

        Returns:
            SubAgentResult with output or error.
        """
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
                self._execute_subagent(task), context=ctx,
            )
            self._running_tasks[task.task_id] = exec_task

        await self.event_bus.emit(Events.SUBAGENT_SPAWNED, {
            "task_id": task.task_id,
            "role": task.role.name,
            "instruction": task.instruction[:200],
            "parent_session_id": task.parent_session_id or "",
        })

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
            await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                "task_id": task.task_id,
                "error": result.error,
            })
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

        # Prune old results to prevent unbounded memory growth
        if len(self._results) > self._max_results:
            excess = len(self._results) - self._max_results
            for old_key in list(self._results)[:excess]:
                del self._results[old_key]

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
        # Also check async futures from fire-and-forget delegations
        afut = self._async_futures.get(task_id)
        if afut and not afut.done():
            afut.cancel()
            await self.event_bus.emit(Events.SUBAGENT_CANCELLED, {
                "task_id": task_id,
            })
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
        all_pending = [
            t for t in list(self._running_tasks.values()) if not t.done()
        ] + [
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
                        "agents": [
                            {"team": a.team, "role": a.role}
                            for a in s.agents
                        ],
                    }
                    for s in proj.stages
                ],
            }
            for proj in self.projects.values()
        ]

    def _find_stage_by_name(
        self, project: Project, name: str,
    ) -> ProjectStage | None:
        """Find a stage in a project by name."""
        return next((s for s in project.stages if s.name == name), None)

    async def _run_stage(
        self,
        project_name: str,
        stage: ProjectStage,
        instruction: str,
        context: str,
        parent_session_id: str,
        feedback_iteration: int = 0,
    ) -> tuple[ProjectStageResult, bool]:
        """Execute a single project stage.

        Args:
            project_name: Name of the parent project.
            stage: The stage to execute.
            instruction: Task instruction for agents.
            context: Accumulated context from prior stages.
            parent_session_id: Parent session ID.
            feedback_iteration: Which feedback iteration (0 = first run).

        Returns:
            Tuple of (stage_result, has_failure).
        """
        import time as _time

        stage_start = _time.monotonic()

        await self.event_bus.emit(Events.PROJECT_STAGE_STARTED, {
            "project": project_name,
            "stage": stage.name,
            "agents": len(stage.agents),
        })

        # Resolve agent refs to SubAgentTasks
        tasks: list[SubAgentTask] = []
        resolution_error: str | None = None

        for agent_ref in stage.agents:
            team = self.teams.get(agent_ref.team)
            if not team:
                resolution_error = (
                    f"RESOLUTION_ERROR: Stage '{stage.name}': "
                    f"team '{agent_ref.team}' not found"
                )
                break

            role = next(
                (r for r in team.roles if r.name == agent_ref.role), None,
            )
            if not role:
                available_roles = ", ".join(r.name for r in team.roles)
                resolution_error = (
                    f"RESOLUTION_ERROR: Stage '{stage.name}': "
                    f"role '{agent_ref.role}' not found "
                    f"in team '{agent_ref.team}'. Available: {available_roles}"
                )
                break

            stage_context = context if context else ""

            tasks.append(SubAgentTask(
                role=role,
                instruction=instruction,
                context=stage_context,
                parent_session_id=parent_session_id,
            ))

        if resolution_error:
            stage_result = ProjectStageResult(
                stage_name=stage.name,
                results=[SubAgentResult(
                    task_id="",
                    role_name="",
                    status=SubAgentStatus.FAILED,
                    error=resolution_error,
                )],
                feedback_iteration=feedback_iteration,
            )
            return stage_result, True

        # Run agents — parallel or sequential based on stage config
        if stage.parallel:
            results = await self.spawn_parallel(tasks)
        else:
            results = []
            for t in tasks:
                results.append(await self.spawn_subagent(t))

        stage_duration = int((_time.monotonic() - stage_start) * 1000)

        # Combine outputs for next stage's context
        stage_outputs: list[str] = []
        has_failure = False
        for r in results:
            if r.status == SubAgentStatus.COMPLETED and r.output:
                stage_outputs.append(
                    f"[{r.role_name}]:\n{r.output}"
                )
            elif r.status == SubAgentStatus.FAILED:
                has_failure = True
                stage_outputs.append(
                    f"[{r.role_name}] FAILED: {r.error}"
                )

        combined = "\n\n---\n\n".join(stage_outputs)

        stage_result = ProjectStageResult(
            stage_name=stage.name,
            results=results,
            combined_output=combined,
            duration_ms=stage_duration,
            feedback_iteration=feedback_iteration,
        )

        await self.event_bus.emit(Events.PROJECT_STAGE_COMPLETED, {
            "project": project_name,
            "stage": stage.name,
            "agents_completed": sum(
                1 for r in results if r.status == SubAgentStatus.COMPLETED
            ),
            "agents_failed": sum(
                1 for r in results if r.status == SubAgentStatus.FAILED
            ),
            "duration_ms": stage_duration,
        })

        return stage_result, has_failure

    async def _evaluate_review_output(self, review_output: str) -> tuple[bool, str]:
        """Evaluate review output for pass/fail using a single LLM call.

        Returns (passed: bool, summary: str).
        """
        eval_prompt = (
            "Evaluate the following review/test output. "
            "Does it indicate all checks PASSED, or are there FAILURES/issues?\n\n"
            "Respond with exactly one line:\n"
            "PASS: <one-sentence summary>\n"
            "or\n"
            "FAIL: <one-sentence summary of issues>\n\n"
            f"Review output:\n{review_output[:3000]}"
        )

        try:
            if self.sdk_service is not None:
                # Use an empty scoped registry so the evaluator has no
                # tool access — it only needs to return a PASS/FAIL line.
                empty_registry = ScopedToolRegistry(
                    parent=self.tool_registry,
                    allowed_tools=[],
                )
                response_text = await self.sdk_service.run_subagent(
                    prompt=eval_prompt,
                    task_id=f"eval-{_uuid4().hex[:8]}",
                    role_persona="You are a concise evaluator.",
                    scoped_registry=empty_registry,
                    max_turns=1,
                )
            elif self.agent_loop.llm is not None:
                from agent.llm.provider import LLMResponse
                resp: LLMResponse = await self.agent_loop.llm.completion(
                    messages=[{"role": "user", "content": eval_prompt}],
                    max_tokens=200,
                )
                response_text = resp.content
            else:
                return False, "No LLM available for evaluation"

            response_text = response_text.strip()
            if response_text.upper().startswith("PASS"):
                summary = response_text.split(":", 1)[1].strip() if ":" in response_text else ""
                return True, summary
            elif response_text.upper().startswith("FAIL"):
                summary = response_text.split(":", 1)[1].strip() if ":" in response_text else ""
                return False, summary
            else:
                # Can't parse — default to FAIL (safe)
                return False, response_text[:200]
        except Exception as e:
            logger.warning("feedback_evaluation_error", error=str(e))
            return False, f"Evaluation error: {e}"

    async def run_project(
        self,
        project_name: str,
        instruction: str,
        context: str = "",
        parent_session_id: str = "",
    ) -> ProjectResult:
        """Run a cross-team project pipeline.

        Executes stages sequentially. Within each stage, all agents run
        in parallel. Each stage's combined output is passed as context
        to the next stage. Stages with feedback config trigger iterative
        fix→verify loops on failure.

        Args:
            project_name: Name of the registered project.
            instruction: The task instruction for the project.
            context: Initial context (added to first stage).
            parent_session_id: Parent session ID.

        Returns:
            ProjectResult with all stage results and final output.
        """
        import time as _time

        project = self.projects.get(project_name)
        if not project:
            available = ", ".join(self.projects.keys()) or "none"
            return ProjectResult(
                project_name=project_name,
                status=SubAgentStatus.FAILED,
                error=f"Unknown project: '{project_name}'. Available: {available}",
            )

        start = _time.monotonic()

        await self.event_bus.emit(Events.PROJECT_STARTED, {
            "project": project_name,
            "stages": len(project.stages),
            "instruction": instruction[:200],
        })

        stage_results: list[ProjectStageResult] = []
        accumulated_context = context
        total_feedback_iterations = 0

        for stage in project.stages:
            # Skip feedback-target stages in normal flow
            if stage.feedback_target:
                continue

            if stage.mode == "discussion" and stage.discussion:
                stage_result, has_failure = await self._run_discussion_stage(
                    project_name, stage, instruction, accumulated_context,
                    parent_session_id,
                )
            else:
                stage_result, has_failure = await self._run_stage(
                    project_name, stage, instruction, accumulated_context,
                    parent_session_id,
                )

            stage_results.append(stage_result)

            # Handle resolution errors (stage_result has a failed result with error)
            if has_failure and stage_result.results and stage_result.results[0].error:
                first_error = stage_result.results[0].error
                if first_error.startswith("RESOLUTION_ERROR:"):
                    duration_ms = int((_time.monotonic() - start) * 1000)
                    await self.event_bus.emit(Events.PROJECT_FAILED, {
                        "project": project_name,
                        "stage": stage.name,
                        "error": first_error,
                    })
                    return ProjectResult(
                        project_name=project_name,
                        status=SubAgentStatus.FAILED,
                        stages=stage_results,
                        error=first_error,
                        duration_ms=duration_ms,
                    )
            combined = stage_result.combined_output

            # Stop pipeline on stage failure (GAP 11) — unless feedback loop handles it
            if has_failure and not stage.feedback:
                total_duration = int((_time.monotonic() - start) * 1000)
                failure_msg = f"Stage '{stage.name}' had agent failures"

                await self.event_bus.emit(Events.PROJECT_FAILED, {
                    "project": project_name,
                    "stage": stage.name,
                    "error": failure_msg,
                })

                return ProjectResult(
                    project_name=project_name,
                    status=SubAgentStatus.FAILED,
                    stages=stage_results,
                    final_output=combined,
                    duration_ms=total_duration,
                    error=failure_msg,
                )

            # --- Feedback loop ---
            if stage.feedback:
                fix_stage = self._find_stage_by_name(
                    project, stage.feedback.fix_stage,
                )
                if fix_stage is None:
                    logger.warning(
                        "feedback_fix_stage_not_found",
                        fix_stage=stage.feedback.fix_stage,
                    )
                else:
                    passed, eval_summary = await self._evaluate_review_output(
                        combined,
                    )

                    if not passed:
                        await self.event_bus.emit(
                            Events.PROJECT_FEEDBACK_STARTED, {
                                "project": project_name,
                                "review_stage": stage.name,
                                "fix_stage": stage.feedback.fix_stage,
                                "max_retries": stage.feedback.max_retries,
                                "issues": eval_summary,
                            },
                        )

                        feedback_passed = False
                        for iteration in range(1, stage.feedback.max_retries + 1):
                            total_feedback_iterations += 1

                            await self.event_bus.emit(
                                Events.PROJECT_FEEDBACK_ITERATION, {
                                    "project": project_name,
                                    "iteration": iteration,
                                    "max_retries": stage.feedback.max_retries,
                                    "issues": eval_summary,
                                },
                            )

                            # Build fix context with review feedback
                            fix_context = accumulated_context
                            if fix_context:
                                fix_context += (
                                    f"\n\n--- Review feedback (iteration "
                                    f"{iteration}) ---\n\n{combined}"
                                    f"\n\nIssues to fix: {eval_summary}"
                                )
                            else:
                                fix_context = (
                                    f"--- Review feedback (iteration "
                                    f"{iteration}) ---\n\n{combined}"
                                    f"\n\nIssues to fix: {eval_summary}"
                                )

                            # Run fix stage
                            fix_result, fix_failed = await self._run_stage(
                                project_name, fix_stage, instruction,
                                fix_context, parent_session_id,
                                feedback_iteration=iteration,
                            )
                            stage_results.append(fix_result)

                            if fix_failed:
                                # Fix stage itself failed — update summary
                                # so downstream telemetry reflects the real
                                # failure, not the prior review's issues.
                                eval_summary = (
                                    f"Fix stage failed on iteration {iteration}"
                                )
                                break

                            # Build re-review context
                            review_context = accumulated_context
                            if review_context:
                                review_context += (
                                    f"\n\n--- Fix output (iteration "
                                    f"{iteration}) ---\n\n"
                                    f"{fix_result.combined_output}"
                                    f"\n\nPlease re-verify the fixes."
                                )
                            else:
                                review_context = (
                                    f"--- Fix output (iteration "
                                    f"{iteration}) ---\n\n"
                                    f"{fix_result.combined_output}"
                                    f"\n\nPlease re-verify the fixes."
                                )

                            # Re-run review stage
                            review_result, review_failed = await self._run_stage(
                                project_name, stage, instruction,
                                review_context, parent_session_id,
                                feedback_iteration=iteration,
                            )
                            stage_results.append(review_result)
                            combined = review_result.combined_output

                            if review_failed:
                                break

                            # Re-evaluate
                            passed, eval_summary = (
                                await self._evaluate_review_output(combined)
                            )
                            if passed:
                                await self.event_bus.emit(
                                    Events.PROJECT_FEEDBACK_PASSED, {
                                        "project": project_name,
                                        "iteration": iteration,
                                        "summary": eval_summary,
                                    },
                                )
                                feedback_passed = True
                                break

                        if not feedback_passed:
                            await self.event_bus.emit(
                                Events.PROJECT_FEEDBACK_EXHAUSTED, {
                                    "project": project_name,
                                    "review_stage": stage.name,
                                    "max_retries": stage.feedback.max_retries,
                                    "last_issues": eval_summary,
                                },
                            )
                            total_duration = int(
                                (_time.monotonic() - start) * 1000
                            )
                            return ProjectResult(
                                project_name=project_name,
                                status=SubAgentStatus.FAILED,
                                stages=stage_results,
                                final_output=combined,
                                duration_ms=total_duration,
                                error=(
                                    f"Feedback loop exhausted after "
                                    f"{stage.feedback.max_retries} retries "
                                    f"on stage '{stage.name}'"
                                ),
                                feedback_iterations=total_feedback_iterations,
                            )

            # Feed this stage's output as context to the next stage
            if accumulated_context:
                accumulated_context += (
                    f"\n\n--- Previous stage: {stage.name} ---\n\n"
                )
                accumulated_context += combined
            else:
                accumulated_context = (
                    f"--- Previous stage: {stage.name} ---\n\n{combined}"
                )

            # Cap accumulated context to prevent unbounded growth
            _max_context = 50_000
            if len(accumulated_context) > _max_context:
                accumulated_context = (
                    "... (earlier context truncated) ...\n\n"
                    + accumulated_context[-_max_context:]
                )

        # Final result
        total_duration = int((_time.monotonic() - start) * 1000)

        # Use last stage's combined output as final output
        if stage_results:
            final_output = stage_results[-1].combined_output
            final_status = SubAgentStatus.COMPLETED
        elif project.stages:
            # All stages were skipped (e.g. all are feedback_target)
            logger.warning(
                "project_all_stages_skipped",
                project=project_name,
                total_stages=len(project.stages),
            )
            final_output = ""
            final_status = SubAgentStatus.FAILED
        else:
            # Empty project (no stages defined) — vacuously complete
            logger.warning("project_no_stages_defined", project=project_name)
            final_output = ""
            final_status = SubAgentStatus.COMPLETED

        project_result = ProjectResult(
            project_name=project_name,
            status=final_status,
            stages=stage_results,
            final_output=final_output,
            duration_ms=total_duration,
            feedback_iterations=total_feedback_iterations,
        )

        await self.event_bus.emit(Events.PROJECT_COMPLETED, {
            "project": project_name,
            "stages_completed": len(stage_results),
            "duration_ms": total_duration,
        })

        return project_result

    async def handle_consult(
        self, request: ConsultRequest, nesting_depth: int = 0,
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

        await self.event_bus.emit(Events.AGENT_CONSULT_REQUESTED, {
            "request_id": request.request_id,
            "requesting_role": request.requesting_role,
            "target_team": request.target_team,
            "target_role": request.target_role,
        })

        # Resolve team and role
        team = self.teams.get(request.target_team)
        if not team:
            available = ", ".join(self.teams.keys()) or "none"
            error = f"Team '{request.target_team}' not found. Available: {available}"
            await self.event_bus.emit(Events.AGENT_CONSULT_FAILED, {
                "request_id": request.request_id,
                "error": error,
            })
            return ConsultResponse(
                request_id=request.request_id,
                target_role=request.target_role,
                status=SubAgentStatus.FAILED,
                error=error,
            )

        role = next(
            (r for r in team.roles if r.name == request.target_role), None,
        )
        if not role:
            available_roles = ", ".join(r.name for r in team.roles)
            error = (
                f"Role '{request.target_role}' not found in team "
                f"'{request.target_team}'. Available: {available_roles}"
            )
            await self.event_bus.emit(Events.AGENT_CONSULT_FAILED, {
                "request_id": request.request_id,
                "error": error,
            })
            return ConsultResponse(
                request_id=request.request_id,
                target_role=request.target_role,
                status=SubAgentStatus.FAILED,
                error=error,
            )

        # Build consultation instruction
        instruction = (
            f"You have been consulted by {request.requesting_role}.\n\n"
            f"Question: {request.question}"
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
            await self.event_bus.emit(Events.AGENT_CONSULT_COMPLETED, {
                "request_id": request.request_id,
                "target_role": request.target_role,
                "duration_ms": duration_ms,
            })
            return ConsultResponse(
                request_id=request.request_id,
                target_role=request.target_role,
                status=SubAgentStatus.COMPLETED,
                answer=result.output,
                token_usage=result.token_usage,
                duration_ms=duration_ms,
            )
        else:
            await self.event_bus.emit(Events.AGENT_CONSULT_FAILED, {
                "request_id": request.request_id,
                "error": result.error,
            })
            return ConsultResponse(
                request_id=request.request_id,
                target_role=request.target_role,
                status=result.status,
                error=result.error,
                duration_ms=duration_ms,
            )

    async def handle_delegation(
        self, request: DelegationRequest, nesting_depth: int = 0,
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

        await self.event_bus.emit(Events.AGENT_DELEGATION_REQUESTED, {
            "request_id": request.request_id,
            "delegating_role": request.delegating_role,
            "target_team": request.target_team,
            "target_role": request.target_role,
            "mode": request.mode,
        })

        # Resolve team and role
        team = self.teams.get(request.target_team)
        if not team:
            available = ", ".join(self.teams.keys()) or "none"
            error = f"Team '{request.target_team}' not found. Available: {available}"
            await self.event_bus.emit(Events.AGENT_DELEGATION_FAILED, {
                "request_id": request.request_id,
                "error": error,
            })
            return DelegationResult(
                request_id=request.request_id,
                target_role=request.target_role,
                status=SubAgentStatus.FAILED,
                error=error,
            )

        role = next(
            (r for r in team.roles if r.name == request.target_role), None,
        )
        if not role:
            available_roles = ", ".join(r.name for r in team.roles)
            error = (
                f"Role '{request.target_role}' not found in team "
                f"'{request.target_team}'. Available: {available_roles}"
            )
            await self.event_bus.emit(Events.AGENT_DELEGATION_FAILED, {
                "request_id": request.request_id,
                "error": error,
            })
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
            await self.event_bus.emit(Events.AGENT_DELEGATION_COMPLETED, {
                "request_id": request.request_id,
                "target_role": request.target_role,
                "duration_ms": duration_ms,
            })
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
            await self.event_bus.emit(Events.AGENT_DELEGATION_FAILED, {
                "request_id": request.request_id,
                "error": result.error,
            })
            return DelegationResult(
                request_id=request.request_id,
                target_role=request.target_role,
                status=result.status,
                error=result.error,
                task_id=task.task_id,
                duration_ms=duration_ms,
            )

    async def _run_discussion_stage(
        self,
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

        await self.event_bus.emit(Events.PROJECT_STAGE_STARTED, {
            "project": project_name,
            "stage": stage.name,
            "agents": len(stage.agents),
            "mode": "discussion",
        })
        await self.event_bus.emit(Events.DISCUSSION_STARTED, {
            "project": project_name,
            "stage": stage.name,
            "rounds": discussion.rounds,
            "agents": len(stage.agents),
        })

        # Resolve all agent refs to roles
        agent_roles: list[tuple[str, SubAgentRole]] = []
        for agent_ref in stage.agents:
            team = self.teams.get(agent_ref.team)
            if not team:
                error = (
                    f"RESOLUTION_ERROR: Stage '{stage.name}': "
                    f"team '{agent_ref.team}' not found"
                )
                return ProjectStageResult(
                    stage_name=stage.name,
                    results=[SubAgentResult(
                        task_id="",
                        role_name="",
                        status=SubAgentStatus.FAILED,
                        error=error,
                    )],
                ), True

            role = next(
                (r for r in team.roles if r.name == agent_ref.role), None,
            )
            if not role:
                error = (
                    f"RESOLUTION_ERROR: Stage '{stage.name}': "
                    f"role '{agent_ref.role}' not found "
                    f"in team '{agent_ref.team}'"
                )
                return ProjectStageResult(
                    stage_name=stage.name,
                    results=[SubAgentResult(
                        task_id="",
                        role_name="",
                        status=SubAgentStatus.FAILED,
                        error=error,
                    )],
                ), True

            agent_roles.append((f"{agent_ref.team}/{agent_ref.role}", role))

        # Resolve moderator if configured
        moderator_role: SubAgentRole | None = None
        moderator_label: str = ""
        if discussion.moderator:
            mod_team = self.teams.get(discussion.moderator.team)
            if mod_team:
                moderator_role = next(
                    (r for r in mod_team.roles
                     if r.name == discussion.moderator.role),
                    None,
                )
                moderator_label = (
                    f"{discussion.moderator.team}/{discussion.moderator.role}"
                )

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
                    agent_context = context + (
                        "\n\nRound 1: Share your initial thoughts."
                    )
                    tasks.append(SubAgentTask(
                        role=role,
                        instruction=instruction,
                        context=agent_context,
                        parent_session_id=parent_session_id,
                    ))

                round_results = await self.spawn_parallel(tasks)

                for (label, _role), result in zip(agent_roles, round_results, strict=False):
                    all_results.append(result)
                    if result.status == SubAgentStatus.COMPLETED and result.output:
                        transcript += (
                            f"\n\n[Round {round_num}] [{label}]:\n"
                            f"{result.output}"
                        )
                    elif result.status == SubAgentStatus.FAILED:
                        has_failure = True
                        transcript += (
                            f"\n\n[Round {round_num}] [{label}] FAILED: "
                            f"{result.error}"
                        )
            else:
                # Subsequent rounds (or single-agent round 1): agents respond
                # sequentially to build on each other's contributions.
                for label, role in agent_roles:
                    agent_context = context
                    if transcript:
                        agent_context += (
                            f"\n\n--- Discussion so far ---\n{transcript}"
                        )
                    if round_num == 1 and not transcript:
                        agent_context += (
                            f"\n\nRound {round_num}: Share your initial thoughts."
                        )
                    else:
                        agent_context += (
                            f"\n\nRound {round_num}: "
                            f"It's your turn to contribute."
                        )

                    task = SubAgentTask(
                        role=role,
                        instruction=instruction,
                        context=agent_context,
                        parent_session_id=parent_session_id,
                    )
                    result = await self.spawn_subagent(task)
                    all_results.append(result)

                    if result.status == SubAgentStatus.COMPLETED and result.output:
                        transcript += (
                            f"\n\n[Round {round_num}] [{label}]:\n"
                            f"{result.output}"
                        )
                    elif result.status == SubAgentStatus.FAILED:
                        has_failure = True
                        transcript += (
                            f"\n\n[Round {round_num}] [{label}] FAILED: "
                            f"{result.error}"
                        )

            # Moderator summary
            if moderator_role:
                mod_context = context
                if transcript:
                    mod_context += (
                        f"\n\n--- Discussion so far ---\n{transcript}"
                    )
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
                            Events.DISCUSSION_CONSENSUS_REACHED, {
                                "project": project_name,
                                "stage": stage.name,
                                "round": round_num,
                            },
                        )

            await self.event_bus.emit(Events.DISCUSSION_ROUND_COMPLETED, {
                "project": project_name,
                "stage": stage.name,
                "round": round_num,
            })

            if consensus_reached:
                break

        stage_duration = int((_time.monotonic() - stage_start) * 1000)

        await self.event_bus.emit(Events.DISCUSSION_COMPLETED, {
            "project": project_name,
            "stage": stage.name,
            "duration_ms": stage_duration,
        })

        stage_result = ProjectStageResult(
            stage_name=stage.name,
            results=all_results,
            combined_output=transcript.strip(),
            duration_ms=stage_duration,
        )

        await self.event_bus.emit(Events.PROJECT_STAGE_COMPLETED, {
            "project": project_name,
            "stage": stage.name,
            "agents_completed": sum(
                1 for r in all_results
                if r.status == SubAgentStatus.COMPLETED
            ),
            "agents_failed": sum(
                1 for r in all_results
                if r.status == SubAgentStatus.FAILED
            ),
            "duration_ms": stage_duration,
        })

        return stage_result, has_failure

    async def _execute_subagent(self, task: SubAgentTask) -> SubAgentResult:
        """Run a sub-agent task to completion.

        Routes through the Claude SDK when available, falling back to the
        AgentLoop path otherwise. Sets task status to RUNNING before execution.
        """
        task.status = SubAgentStatus.RUNNING
        self._task_nesting_depths[task.task_id] = task.nesting_depth
        scoped_registry = self._create_scoped_registry(
            task.role, nesting_depth=task.nesting_depth,
        )

        # Set context var so orchestration tools know the caller's nesting depth.
        # NOTE: Each sub-agent task is wrapped in copy_context() at spawn time
        # (see spawn_subagent) so parallel sub-agents don't overwrite each
        # other's nesting depth.
        from agent.tools.builtins.orchestration import set_nesting_depth
        set_nesting_depth(task.nesting_depth)

        try:
            if self.sdk_service is not None:
                return await self._execute_subagent_via_sdk(task, scoped_registry)
            return await self._execute_subagent_via_loop(task, scoped_registry)
        finally:
            self._task_nesting_depths.pop(task.task_id, None)

    async def _execute_subagent_via_loop(
        self, task: SubAgentTask, scoped_registry: ScopedToolRegistry,
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

        await self.event_bus.emit(Events.SUBAGENT_STARTED, {
            "task_id": task.task_id,
            "role": task.role.name,
        })

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
            raise RuntimeError("Parent tool executor must be set before spawning sub-agents")
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
        session = Session(
            session_id=f"subagent:{task.parent_session_id}:{task.task_id}"
        )

        try:
            response = await sub_loop.process_message(
                full_instruction, session, trigger="subagent",
                tool_registry_override=scoped_registry,
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

    async def _execute_subagent_via_sdk(
        self, task: SubAgentTask, scoped_registry: ScopedToolRegistry,
    ) -> SubAgentResult:
        """Run a sub-agent task via Claude SDK (no API key needed)."""
        import time as _time

        start = _time.monotonic()

        await self.event_bus.emit(Events.SUBAGENT_STARTED, {
            "task_id": task.task_id,
            "role": task.role.name,
        })

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
                    task.task_id, {},
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

            await self.event_bus.emit(Events.SUBAGENT_COMPLETED, {
                "task_id": task.task_id,
                "role": task.role.name,
                "duration_ms": duration_ms,
            })

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

            await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                "task_id": task.task_id,
                "error": str(e),
            })

            return result

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
                    error=f"Too many tasks running ({active}). "
                          f"Use /stop to cancel some.",
                )

            exec_coro = self._execute_channel_task(
                prompt, task_id, session,
                on_progress=on_progress,
                on_permission=on_permission,
            )
            import contextvars
            ctx = contextvars.copy_context()
            exec_task = asyncio.get_running_loop().create_task(
                exec_coro, context=ctx,
            )
            self._running_tasks[task_id] = exec_task

        await self.event_bus.emit(Events.SUBAGENT_SPAWNED, {
            "task_id": task_id,
            "role": "channel",
            "instruction": prompt[:200],
        })

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
                await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                    "task_id": task_id,
                    "error": result.error,
                })
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
                await self.event_bus.emit(Events.SUBAGENT_CANCELLED, {
                    "task_id": task_id,
                })
        except Exception as e:
            result = SubAgentResult(
                task_id=task_id,
                role_name="channel",
                status=SubAgentStatus.FAILED,
                error=f"Unexpected error: {e}",
            )
            with contextlib.suppress(Exception):
                await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                    "task_id": task_id,
                    "error": result.error,
                })
        finally:
            self._running_tasks.pop(task_id, None)
            self._task_nesting_depths.pop(task_id, None)

        self._results[task_id] = result

        # Prune old results to prevent unbounded memory growth
        if len(self._results) > self._max_results:
            excess = len(self._results) - self._max_results
            for old_key in list(self._results)[:excess]:
                del self._results[old_key]

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

        await self.event_bus.emit(Events.SUBAGENT_STARTED, {
            "task_id": task_id,
            "role": "channel",
        })

        try:
            # Prefer SDK path when available
            if self.sdk_service is not None:
                response_text = await self._execute_via_sdk(
                    prompt, task_id, session,
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

            await self.event_bus.emit(Events.SUBAGENT_COMPLETED, {
                "task_id": task_id,
                "role": "channel",
                "duration_ms": duration_ms,
            })

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
            await self.event_bus.emit(Events.SUBAGENT_FAILED, {
                "task_id": task_id,
                "error": str(e),
            })
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
        self, role: SubAgentRole, nesting_depth: int = 0,
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
        """Scoped registries cannot enable tools in the parent."""
        logger.debug("scoped_registry_enable_noop", tool=name)

    def disable_tool(self, name: str) -> None:
        """Scoped registries cannot disable tools in the parent."""
        logger.debug("scoped_registry_disable_noop", tool=name)

    def unregister_tool(self, name: str) -> None:
        """Scoped registries cannot unregister tools from the parent."""
        logger.debug("scoped_registry_unregister_noop", tool=name)
