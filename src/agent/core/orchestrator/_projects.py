"""Project pipeline methods for SubAgentOrchestrator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import uuid4 as _uuid4

import structlog

from agent.core.events import Events
from agent.core.orchestrator._scoped_registry import ScopedToolRegistry
from agent.core.subagent import (
    Project,
    ProjectResult,
    ProjectStage,
    ProjectStageResult,
    QualityReport,
    ReplanDecision,
    SubAgentResult,
    SubAgentStatus,
    SubAgentTask,
)

if TYPE_CHECKING:
    from agent.core.orchestrator._core import SubAgentOrchestrator

logger = structlog.get_logger(__name__)


class PipelineController:
    """Controls pipeline execution with pause/resume/replan capability."""

    def __init__(
        self,
        orchestrator: SubAgentOrchestrator,
        project: Project,
        event_bus: Any,
        llm: Any = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.project = project
        self.event_bus = event_bus
        self.llm = llm  # For evaluation/replanning. None = linear mode
        self.current_stage_index = 0
        self.completed_stages: dict[str, ProjectStageResult] = {}
        self.accumulated_context = ""
        self.plan_history: list[list[str]] = []

    async def run(
        self,
        instruction: str,
        context: str = "",
        parent_session_id: str = "",
    ) -> ProjectResult:
        """Main execution loop with dynamic re-planning.

        For each stage: run it, then (if LLM available) evaluate progress.
        Based on the decision: continue, replan, skip, or abort.
        If no LLM: default to continue (backward compatible).

        Args:
            instruction: The task instruction for the project.
            context: Initial context.
            parent_session_id: Parent session ID.

        Returns:
            ProjectResult with all stage results and final output.
        """
        import time as _time

        start = _time.monotonic()
        project_name = self.project.name
        self.accumulated_context = context

        # Record initial plan
        runnable = self._runnable_stages()
        self.plan_history.append([s.name for s in runnable])

        stage_results: list[ProjectStageResult] = []
        all_stage_errors: list[str] = []
        total_feedback_iterations = 0

        while self.current_stage_index < len(runnable):
            stage = runnable[self.current_stage_index]

            # Run the stage via the orchestrator's existing helper
            if stage.mode == "discussion" and stage.discussion:
                stage_result, has_failure = await self.orchestrator._run_discussion_stage(
                    project_name,
                    stage,
                    instruction,
                    self.accumulated_context,
                    parent_session_id,
                )
            else:
                stage_result, has_failure = await self.orchestrator._run_stage(
                    project_name,
                    stage,
                    instruction,
                    self.accumulated_context,
                    parent_session_id,
                )

            stage_results.append(stage_result)
            self.completed_stages[stage.name] = stage_result

            # Collect errors
            for r in stage_result.results:
                if r.status == SubAgentStatus.FAILED and r.error:
                    all_stage_errors.append(f"[{stage.name}/{r.role_name}] {r.error}")

            # Handle resolution errors
            if has_failure and stage_result.results and stage_result.results[0].error:
                first_error = stage_result.results[0].error
                if first_error.startswith("RESOLUTION_ERROR:"):
                    duration_ms = int((_time.monotonic() - start) * 1000)
                    await self.event_bus.emit(
                        Events.PROJECT_FAILED,
                        {
                            "project": project_name,
                            "stage": stage.name,
                            "error": first_error,
                        },
                    )
                    return ProjectResult(
                        project_name=project_name,
                        status=SubAgentStatus.FAILED,
                        stages=stage_results,
                        error=first_error,
                        duration_ms=duration_ms,
                        stage_errors=all_stage_errors,
                        plan_history=self.plan_history,
                    )

            combined = stage_result.combined_output

            # Stop pipeline on stage failure without feedback
            if has_failure and not stage.feedback:
                total_duration = int((_time.monotonic() - start) * 1000)
                failure_msg = f"Stage '{stage.name}' had agent failures"

                await self.event_bus.emit(
                    Events.PROJECT_FAILED,
                    {
                        "project": project_name,
                        "stage": stage.name,
                        "error": failure_msg,
                    },
                )

                return ProjectResult(
                    project_name=project_name,
                    status=SubAgentStatus.FAILED,
                    stages=stage_results,
                    final_output=combined,
                    duration_ms=total_duration,
                    error=failure_msg,
                    stage_errors=all_stage_errors,
                    plan_history=self.plan_history,
                )

            # --- Feedback loop ---
            if stage.feedback:
                fix_stage = self.orchestrator._find_stage_by_name(
                    self.project,
                    stage.feedback.fix_stage,
                )
                if fix_stage is None:
                    logger.warning(
                        "feedback_fix_stage_not_found",
                        fix_stage=stage.feedback.fix_stage,
                    )
                else:
                    passed, eval_summary = await self.orchestrator._evaluate_review_output(
                        combined,
                    )

                    if not passed:
                        await self.event_bus.emit(
                            Events.PROJECT_FEEDBACK_STARTED,
                            {
                                "project": project_name,
                                "review_stage": stage.name,
                                "fix_stage": stage.feedback.fix_stage,
                                "max_retries": stage.feedback.max_retries,
                                "issues": eval_summary,
                            },
                        )

                        feedback_passed = False
                        for iteration in range(
                            1,
                            stage.feedback.max_retries + 1,
                        ):
                            total_feedback_iterations += 1

                            await self.event_bus.emit(
                                Events.PROJECT_FEEDBACK_ITERATION,
                                {
                                    "project": project_name,
                                    "iteration": iteration,
                                    "max_retries": (stage.feedback.max_retries),
                                    "issues": eval_summary,
                                },
                            )

                            # Build fix context with review feedback
                            fix_context = self.accumulated_context
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
                            fix_result, fix_failed = await self.orchestrator._run_stage(
                                project_name,
                                fix_stage,
                                instruction,
                                fix_context,
                                parent_session_id,
                                feedback_iteration=iteration,
                            )
                            stage_results.append(fix_result)

                            if fix_failed:
                                eval_summary = f"Fix stage failed on iteration " f"{iteration}"
                                break

                            # Build re-review context
                            review_context = self.accumulated_context
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
                            review_result, review_failed = await self.orchestrator._run_stage(
                                project_name,
                                stage,
                                instruction,
                                review_context,
                                parent_session_id,
                                feedback_iteration=iteration,
                            )
                            stage_results.append(review_result)
                            combined = review_result.combined_output

                            if review_failed:
                                break

                            # Re-evaluate
                            passed, eval_summary = await self.orchestrator._evaluate_review_output(
                                combined,
                            )
                            if passed:
                                await self.event_bus.emit(
                                    Events.PROJECT_FEEDBACK_PASSED,
                                    {
                                        "project": project_name,
                                        "iteration": iteration,
                                        "summary": eval_summary,
                                    },
                                )
                                feedback_passed = True
                                break

                        if not feedback_passed:
                            await self.event_bus.emit(
                                Events.PROJECT_FEEDBACK_EXHAUSTED,
                                {
                                    "project": project_name,
                                    "review_stage": stage.name,
                                    "max_retries": (stage.feedback.max_retries),
                                    "last_issues": eval_summary,
                                },
                            )
                            total_duration = int((_time.monotonic() - start) * 1000)
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
                                feedback_iterations=(total_feedback_iterations),
                                stage_errors=all_stage_errors,
                                plan_history=self.plan_history,
                            )

            # --- Quality gate evaluation for feedback stages ---
            if stage.feedback and self.llm is not None:
                try:
                    qg_action, qg_report = await self.orchestrator._evaluate_with_quality_gate(
                        stage_name=stage.name,
                        stage_output=combined,
                        instruction=instruction,
                    )
                    if not hasattr(self, "_quality_reports"):
                        self._quality_reports: dict[str, dict] = {}
                    self._quality_reports[stage.name] = {
                        "passed": qg_report.passed,
                        "confidence": qg_report.confidence,
                        "issues": [
                            {
                                "severity": i.severity,
                                "category": i.category,
                                "description": i.description,
                                "suggested_fix": i.suggested_fix,
                            }
                            for i in qg_report.issues
                        ],
                        "summary": qg_report.summary,
                        "recommended_action": (qg_report.recommended_action),
                    }
                except Exception as e:
                    logger.warning(
                        "quality_gate_evaluation_error",
                        stage=stage.name,
                        error=str(e),
                    )
                    if not hasattr(self, "_quality_reports"):
                        self._quality_reports = {}
                    self._quality_reports[stage.name] = {
                        "passed": True,
                        "confidence": 1.0,
                        "issues": [],
                        "summary": f"Evaluation error: {e}",
                        "recommended_action": "proceed",
                    }

            # --- Evaluate progress (dynamic re-planning) ---
            decision = await self._evaluate_progress(
                stage,
                stage_result,
                instruction,
                runnable,
            )

            if decision.action == "abort":
                total_duration = int((_time.monotonic() - start) * 1000)
                abort_msg = f"Pipeline aborted after stage '{stage.name}': " f"{decision.reason}"
                logger.warning(
                    "pipeline_aborted",
                    project=project_name,
                    stage=stage.name,
                    reason=decision.reason,
                )
                await self.event_bus.emit(
                    Events.PROJECT_FAILED,
                    {
                        "project": project_name,
                        "stage": stage.name,
                        "error": abort_msg,
                    },
                )
                return ProjectResult(
                    project_name=project_name,
                    status=SubAgentStatus.FAILED,
                    stages=stage_results,
                    final_output=combined,
                    duration_ms=total_duration,
                    error=abort_msg,
                    stage_errors=all_stage_errors,
                    plan_history=self.plan_history,
                )

            if decision.action == "replan":
                restart_idx = max(
                    0,
                    min(
                        decision.restart_from,
                        len(runnable) - 1,
                    ),
                )
                logger.info(
                    "pipeline_replanning",
                    project=project_name,
                    reason=decision.reason,
                    restart_from=restart_idx,
                )
                new_stages = await self._generate_new_plan(
                    decision.reason,
                    self.accumulated_context,
                )
                runnable = new_stages
                self.plan_history.append([s.name for s in runnable])
                self.current_stage_index = restart_idx

                await self.event_bus.emit(
                    Events.PIPELINE_REPLANNED,
                    {
                        "project": project_name,
                        "reason": decision.reason,
                        "restart_from": restart_idx,
                        "new_stages": [s.name for s in runnable],
                    },
                )
                continue

            if decision.action == "skip_to":
                target_idx = max(
                    self.current_stage_index + 1,
                    min(decision.target_stage, len(runnable)),
                )
                skipped = runnable[self.current_stage_index + 1 : target_idx]
                for skipped_stage in skipped:
                    logger.info(
                        "pipeline_stage_skipped",
                        project=project_name,
                        stage=skipped_stage.name,
                        reason=decision.reason,
                    )
                    await self.event_bus.emit(
                        Events.PIPELINE_STAGE_SKIPPED,
                        {
                            "project": project_name,
                            "stage": skipped_stage.name,
                            "reason": decision.reason,
                        },
                    )
                self.current_stage_index = target_idx
            else:
                # "continue" (default)
                self.current_stage_index += 1

            # Accumulate context for next stage
            if self.accumulated_context:
                self.accumulated_context += f"\n\n--- Previous stage: {stage.name} ---\n\n"
                self.accumulated_context += combined
            else:
                self.accumulated_context = f"--- Previous stage: {stage.name} ---\n\n{combined}"

            # Cap accumulated context
            _max_context = 50_000
            if len(self.accumulated_context) > _max_context:
                self.accumulated_context = (
                    "... (earlier context truncated) ...\n\n"
                    + self.accumulated_context[-_max_context:]
                )

        # Final result
        total_duration = int((_time.monotonic() - start) * 1000)
        if stage_results:
            final_output = stage_results[-1].combined_output
            final_status = SubAgentStatus.COMPLETED
        elif self.project.stages:
            final_output = ""
            final_status = SubAgentStatus.FAILED
        else:
            final_output = ""
            final_status = SubAgentStatus.COMPLETED

        # Attach quality reports if any were collected
        quality_reports = getattr(self, "_quality_reports", {})

        return ProjectResult(
            project_name=project_name,
            status=final_status,
            stages=stage_results,
            final_output=final_output,
            duration_ms=total_duration,
            feedback_iterations=total_feedback_iterations,
            stage_errors=all_stage_errors,
            plan_history=self.plan_history,
            quality_reports=quality_reports,
        )

    async def _evaluate_progress(
        self,
        stage: ProjectStage,
        result: ProjectStageResult,
        instruction: str,
        runnable_stages: list[ProjectStage],
    ) -> ReplanDecision:
        """Evaluate progress after a stage and decide next action.

        If no LLM is available, defaults to "continue" (backward compat).

        Returns:
            ReplanDecision with the chosen action.
        """
        if self.llm is None:
            return ReplanDecision(action="continue")

        remaining = [s.name for s in runnable_stages[self.current_stage_index + 1 :]]

        eval_prompt = (
            "You are evaluating progress of a software project pipeline.\n"
            f'Stage "{stage.name}" just completed with output:\n'
            f"{result.combined_output[:3000]}\n\n"
            f"Original instruction: {instruction[:1000]}\n"
            f"Stages remaining: {remaining}\n\n"
            "Should we:\n"
            '1. "continue" - proceed to next stage\n'
            '2. "replan" - the design/approach needs revision, '
            "restart from stage N\n"
            '3. "skip_to" - skip intermediate stages, '
            "go directly to stage N\n"
            '4. "abort" - fundamental issue, stop the project\n\n'
            "Respond with JSON: "
            '{"action": "...", "reason": "...", '
            '"restart_from": 0, "target_stage": 0}'
        )

        try:
            from agent.llm.provider import LLMResponse

            resp: LLMResponse = await self.llm.completion(
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=300,
            )
            response_text = resp.content.strip()

            # Try to extract JSON from the response
            parsed = json.loads(response_text)
            return ReplanDecision(
                action=parsed.get("action", "continue"),
                reason=parsed.get("reason", ""),
                restart_from=parsed.get("restart_from", 0),
                target_stage=parsed.get("target_stage", 0),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "replan_evaluation_parse_failure",
                stage=stage.name,
            )
            return ReplanDecision(
                action="continue",
                reason="Failed to parse evaluation response",
            )
        except Exception as e:
            logger.warning(
                "replan_evaluation_error",
                stage=stage.name,
                error=str(e),
            )
            return ReplanDecision(
                action="continue",
                reason=f"Evaluation error: {e}",
            )

    async def _generate_new_plan(
        self,
        reason: str,
        context: str,
    ) -> list[ProjectStage]:
        """Generate a revised set of stages after a replan decision.

        For now, returns the original stages. A full implementation would
        use the LLM to generate new YAML-based stage definitions.

        Args:
            reason: Why the replan was triggered.
            context: Accumulated pipeline context.

        Returns:
            List of ProjectStage objects for the revised plan.
        """
        logger.info(
            "pipeline_generate_new_plan",
            project=self.project.name,
            reason=reason,
        )
        # Return the original runnable stages (stub for future LLM-based
        # plan generation).
        return self._runnable_stages()

    def _runnable_stages(self) -> list[ProjectStage]:
        """Return stages that participate in normal flow.

        Filters out feedback-target stages, which are only triggered
        by feedback loops.
        """
        return [s for s in self.project.stages if not s.feedback_target]


def _find_stage_by_name(
    self: SubAgentOrchestrator,
    project: Project,
    name: str,
) -> ProjectStage | None:
    """Find a stage in a project by name."""
    return next((s for s in project.stages if s.name == name), None)


async def _run_stage(
    self: SubAgentOrchestrator,
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

    await self.event_bus.emit(
        Events.PROJECT_STAGE_STARTED,
        {
            "project": project_name,
            "stage": stage.name,
            "agents": len(stage.agents),
        },
    )

    # Resolve agent refs to SubAgentTasks
    tasks: list[SubAgentTask] = []
    resolution_error: str | None = None

    for agent_ref in stage.agents:
        team = self.teams.get(agent_ref.team)
        if not team:
            resolution_error = (
                f"RESOLUTION_ERROR: Stage '{stage.name}': " f"team '{agent_ref.team}' not found"
            )
            break

        role = next(
            (r for r in team.roles if r.name == agent_ref.role),
            None,
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

        tasks.append(
            SubAgentTask(
                role=role,
                instruction=instruction,
                context=stage_context,
                parent_session_id=parent_session_id,
            )
        )

    if resolution_error:
        stage_result = ProjectStageResult(
            stage_name=stage.name,
            results=[
                SubAgentResult(
                    task_id="",
                    role_name="",
                    status=SubAgentStatus.FAILED,
                    error=resolution_error,
                )
            ],
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
            stage_outputs.append(f"[{r.role_name}]:\n{r.output}")
        elif r.status == SubAgentStatus.FAILED:
            has_failure = True
            stage_outputs.append(f"[{r.role_name}] FAILED: {r.error}")

    combined = "\n\n---\n\n".join(stage_outputs)

    stage_result = ProjectStageResult(
        stage_name=stage.name,
        results=results,
        combined_output=combined,
        duration_ms=stage_duration,
        feedback_iteration=feedback_iteration,
    )

    await self.event_bus.emit(
        Events.PROJECT_STAGE_COMPLETED,
        {
            "project": project_name,
            "stage": stage.name,
            "agents_completed": sum(1 for r in results if r.status == SubAgentStatus.COMPLETED),
            "agents_failed": sum(1 for r in results if r.status == SubAgentStatus.FAILED),
            "duration_ms": stage_duration,
        },
    )

    return stage_result, has_failure


async def _evaluate_review_output(
    self: SubAgentOrchestrator,
    review_output: str,
) -> tuple[bool, str]:
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


async def _evaluate_with_quality_gate(
    self: SubAgentOrchestrator,
    stage_name: str,
    stage_output: str,
    instruction: str,
    task_id: str | None = None,
) -> tuple[str, QualityReport]:
    """Evaluate a stage using the structured quality gate.

    Creates a :class:`QualityGate`, calls ``evaluate()``, optionally
    routes issues to the task board, and stores the report.

    Args:
        stage_name: Name of the stage being evaluated.
        stage_output: Combined output from the stage.
        instruction: Original task instruction.
        task_id: Optional task ID for routing issues to the task board.

    Returns:
        Tuple of (recommended_action, QualityReport).
        If the quality gate cannot be created (no LLM), returns
        ``("proceed", <default report>)``.
    """
    from agent.core.quality_gate import QualityGate

    llm = getattr(self.agent_loop, "llm", None)
    if llm is None:
        return "proceed", QualityReport(
            passed=True,
            summary="No LLM available for quality evaluation",
        )

    gate = QualityGate(llm=llm, event_bus=self.event_bus)
    report = await gate.evaluate(
        stage_name=stage_name,
        stage_output=stage_output,
        instruction=instruction,
    )

    # Route issues to task board if available
    if report.issues and self.task_board is not None and task_id:
        await gate.route_issues(report, self.task_board, task_id)

    return report.recommended_action, report


async def run_project(
    self: SubAgentOrchestrator,
    project_name: str,
    instruction: str,
    context: str = "",
    parent_session_id: str = "",
) -> ProjectResult:
    """Run a cross-team project pipeline.

    Executes stages sequentially. Within each stage, all agents run
    in parallel. Each stage's combined output is passed as context
    to the next stage. Stages with feedback config trigger iterative
    fix->verify loops on failure.

    When an LLM is available, uses PipelineController for dynamic
    re-planning (evaluate after each stage, potentially replan/skip/abort).
    Otherwise, runs the linear pipeline for backward compatibility.

    Args:
        project_name: Name of the registered project.
        instruction: The task instruction for the project.
        context: Initial context (added to first stage).
        parent_session_id: Parent session ID.

    Returns:
        ProjectResult with all stage results and final output.
    """

    project = self.projects.get(project_name)
    if not project:
        available = ", ".join(self.projects.keys()) or "none"
        return ProjectResult(
            project_name=project_name,
            status=SubAgentStatus.FAILED,
            error=f"Unknown project: '{project_name}'. Available: {available}",
        )

    # Determine if we have an LLM for dynamic re-planning
    llm = getattr(self.agent_loop, "llm", None)

    # Use PipelineController for dynamic re-planning
    controller = PipelineController(
        orchestrator=self,
        project=project,
        event_bus=self.event_bus,
        llm=llm,
    )

    await self.event_bus.emit(
        Events.PROJECT_STARTED,
        {
            "project": project_name,
            "stages": len(project.stages),
            "instruction": instruction[:200],
        },
    )

    result = await controller.run(
        instruction=instruction,
        context=context,
        parent_session_id=parent_session_id,
    )

    # Emit completion/failure events based on result
    if result.status == SubAgentStatus.COMPLETED:
        await self.event_bus.emit(
            Events.PROJECT_COMPLETED,
            {
                "project": project_name,
                "stages_completed": len(result.stages),
                "duration_ms": result.duration_ms,
            },
        )
    # Note: failure events are already emitted by PipelineController

    return result
