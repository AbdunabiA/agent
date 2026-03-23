"""Intelligent project planner — three-phase decomposition and execution.

Phase 1: RequirementsGatherer — interactive Q&A to produce a ProjectSpec
Phase 2: ProjectPlanner — decomposes spec into micro-tasks with dependency graph
Phase 3: PlanExecutor — executes tasks sequentially with quality gates
"""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import Events
from agent.core.planner_models import (
    ExecutionPlan,
    MicroTask,
    ProjectResult,
    ProjectSpec,
    QualityGateResult,
    TaskStatus,
)

if TYPE_CHECKING:
    from agent.core.events import EventBus
    from agent.core.orchestrator import SubAgentOrchestrator
    from agent.core.role_registry import RoleRegistry
    from agent.core.subagent import SubAgentRole
    from agent.core.working_memory import WorkingMemory
    from agent.llm.claude_sdk import ClaudeSDKService

logger = structlog.get_logger(__name__)

# Type alias for the question callback
QuestionCallback = Callable[[str], Coroutine[Any, Any, str]]

# Maximum retries per micro-task before asking the user
_MAX_TASK_RETRIES = 3


def _extract_json_block(text: str) -> str:
    """Extract JSON from a response that may contain markdown fences."""
    # Try to find ```json ... ``` block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.find("```", start)
        if end == -1:
            # No closing fence — take everything after opening
            return text[start:].strip()
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        # Skip optional language tag on same line
        if "\n" in text[start : start + 20]:
            start = text.index("\n", start) + 1
        end = text.find("```", start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()
    # Try raw JSON
    # Find first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        return text[first_brace : last_brace + 1]
    return text


def topological_sort(tasks: list[MicroTask]) -> list[list[str]]:
    """Compute execution layers via topological sort.

    Returns a list of layers, where each layer contains task IDs
    that can execute after all previous layers are complete.

    Raises:
        ValueError: If the dependency graph contains a cycle.
    """
    task_map = {t.id: t for t in tasks}
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    dependents: dict[str, list[str]] = {t.id: [] for t in tasks}

    for t in tasks:
        for dep in t.dependencies:
            if dep not in task_map:
                logger.warning(
                    "topological_sort_missing_dep",
                    task=t.id,
                    missing_dep=dep,
                )
                continue
            in_degree[t.id] += 1
            dependents[dep].append(t.id)

    layers: list[list[str]] = []
    remaining = set(in_degree.keys())

    while remaining:
        # Find all nodes with in_degree 0
        layer = [tid for tid in remaining if in_degree[tid] == 0]
        if not layer:
            raise ValueError(f"Dependency cycle detected among tasks: {remaining}")

        layers.append(sorted(layer))

        for tid in layer:
            remaining.remove(tid)
            for dep_id in dependents[tid]:
                in_degree[dep_id] -= 1

    # Update task layer numbers
    for layer_idx, layer_ids in enumerate(layers):
        for tid in layer_ids:
            task_map[tid].layer = layer_idx

    return layers


class RequirementsGatherer:
    """Phase 1: Gather requirements interactively via Q&A.

    Spawns an SDK sub-agent with the requirements_gatherer role
    that asks the user questions and produces a ProjectSpec.
    """

    def __init__(
        self,
        sdk_service: ClaudeSDKService,
        event_bus: EventBus,
        working_memory: WorkingMemory | None = None,
    ) -> None:
        self._sdk = sdk_service
        self._event_bus = event_bus
        self._working_memory = working_memory

    async def gather(
        self,
        initial_instruction: str,
        on_question: QuestionCallback | None = None,
        task_id: str = "requirements",
    ) -> ProjectSpec:
        """Run the requirements gathering phase.

        Args:
            initial_instruction: The user's initial project request.
            on_question: Callback to route questions to the user.
            task_id: Task ID for tracking.

        Returns:
            A structured ProjectSpec.
        """
        from agent.core.orchestrator import ScopedToolRegistry

        prompt = (
            "The user wants to build a project. Gather requirements by asking "
            "clarifying questions, then produce a ProjectSpec JSON.\n\n"
            f"User's request: {initial_instruction}\n\n"
            "When you have enough information, output the final spec inside "
            "a ```json code block. Make sure it follows the ProjectSpec format "
            "with title, description, tech_stack, features, components, "
            "constraints, deployment, and integrations fields."
        )

        # Build empty registry — gatherer only needs to ask questions
        empty_registry = ScopedToolRegistry(
            parent=self._sdk.tool_registry,
            allowed_tools=[],
        )

        result = await self._sdk.run_subagent(
            prompt=prompt,
            task_id=f"req-{task_id}",
            role_persona=self._get_gatherer_persona(),
            scoped_registry=empty_registry,
            max_turns=20,
            on_question=on_question,
        )

        # Parse the ProjectSpec from the response
        spec = self._parse_spec(result)

        # Save to working memory
        if self._working_memory:
            await self._working_memory.save_artifact(
                task_id=task_id,
                role="requirements_gatherer",
                content=json.dumps(spec.to_dict(), indent=2),
                metadata={"label": "project_spec", "title": spec.title},
            )

        await self._event_bus.emit(
            Events.PLAN_CREATED,
            {
                "task_id": task_id,
                "phase": "requirements",
                "spec_title": spec.title,
                "feature_count": len(spec.features),
            },
        )

        return spec

    def _get_gatherer_persona(self) -> str:
        """Return the requirements_gatherer persona."""
        return (
            "You are a senior product consultant and requirements analyst. "
            "Your job is to define a clear, complete project specification. "
            "Ask clarifying questions ONE AT A TIME about tech stack, features, "
            "scope, design, deployment, and integrations. When you have enough "
            "info, produce a ProjectSpec as a JSON code block with fields: "
            "title, description, tech_stack, features (with name, description, "
            "priority, acceptance_criteria), components, constraints, deployment, "
            "integrations."
        )

    def _parse_spec(self, text: str) -> ProjectSpec:
        """Parse a ProjectSpec from LLM output."""
        try:
            json_str = _extract_json_block(text)
            data = json.loads(json_str)
            return ProjectSpec.from_dict(data)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("spec_parse_failed", error=str(e), text=text[:500])
            # Fallback: create a minimal spec from the text
            return ProjectSpec(
                title="Untitled Project",
                description=text[:500],
            )


class ProjectPlanner:
    """Phase 2: Decompose a ProjectSpec into an ExecutionPlan.

    Spawns an SDK sub-agent with the project_planner role that reads
    the spec and available roles, then produces a dependency graph
    of micro-tasks.
    """

    def __init__(
        self,
        sdk_service: ClaudeSDKService,
        event_bus: EventBus,
        role_registry: RoleRegistry | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> None:
        self._sdk = sdk_service
        self._event_bus = event_bus
        self._role_registry = role_registry
        self._working_memory = working_memory

    async def plan(
        self,
        spec: ProjectSpec,
        task_id: str = "planning",
    ) -> ExecutionPlan:
        """Decompose a ProjectSpec into an ExecutionPlan.

        Args:
            spec: The project specification to decompose.
            task_id: Task ID for tracking.

        Returns:
            An ExecutionPlan with micro-tasks and dependency graph.
        """
        from agent.core.orchestrator import ScopedToolRegistry

        # Build context with spec and available roles
        context_parts = [
            "## Project Specification",
            json.dumps(spec.to_dict(), indent=2),
        ]

        if self._role_registry:
            roster = self._role_registry.get_roster_description()
            context_parts.append(
                "\n## Available Agent Roles\n"
                f"{roster}\n\n"
                "Use ONLY role names from this roster."
            )

        prompt = (
            "Decompose this project into 10-30 atomic micro-tasks with a "
            "dependency graph. Output a JSON execution plan.\n\n" + "\n".join(context_parts)
        )

        empty_registry = ScopedToolRegistry(
            parent=self._sdk.tool_registry,
            allowed_tools=[],
        )

        result = await self._sdk.run_subagent(
            prompt=prompt,
            task_id=f"plan-{task_id}",
            role_persona=self._get_planner_persona(),
            scoped_registry=empty_registry,
            max_turns=10,
        )

        # Parse the ExecutionPlan
        plan = self._parse_plan(result)

        # Validate and fix
        self._validate_plan(plan)

        # Save to working memory
        if self._working_memory:
            await self._working_memory.save_artifact(
                task_id=task_id,
                role="project_planner",
                content=json.dumps(plan.to_dict(), indent=2),
                metadata={"label": "execution_plan", "task_count": len(plan.tasks)},
            )

        await self._event_bus.emit(
            Events.PLAN_CREATED,
            {
                "task_id": task_id,
                "phase": "planning",
                "task_count": len(plan.tasks),
                "layer_count": len(plan.execution_order),
            },
        )

        return plan

    def _get_planner_persona(self) -> str:
        """Return the project_planner persona."""
        return (
            "You are a senior technical project manager and software architect. "
            "Decompose the project into atomic micro-tasks with dependencies. "
            "Output a JSON plan with tasks (id, title, description, role, "
            "dependencies, acceptance_criteria, layer) and execution_order "
            "(list of layers, each a list of task IDs). Use only role names "
            "from the available roster. Keep tasks small and specific."
        )

    def _parse_plan(self, text: str) -> ExecutionPlan:
        """Parse an ExecutionPlan from LLM output."""
        try:
            json_str = _extract_json_block(text)
            data = json.loads(json_str)
            plan = ExecutionPlan.from_dict(data)
            return plan
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("plan_parse_failed", error=str(e), text=text[:500])
            return ExecutionPlan()

    def _validate_plan(self, plan: ExecutionPlan) -> None:
        """Validate and fix the execution plan.

        - Ensures DAG is acyclic (via topological sort)
        - Removes references to non-existent dependencies
        - Validates roles exist in registry
        - Rebuilds execution_order from topological sort
        """
        if not plan.tasks:
            return

        # Remove invalid dependency references
        task_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            task.dependencies = [d for d in task.dependencies if d in task_ids]

        # Validate roles
        if self._role_registry:
            for task in plan.tasks:
                if self._role_registry.get_role(task.role) is None:
                    logger.warning(
                        "plan_unknown_role",
                        task_id=task.id,
                        role=task.role,
                    )

        # Compute topological sort (also validates acyclicity)
        try:
            layers = topological_sort(plan.tasks)
            plan.execution_order = layers
        except ValueError as e:
            logger.error("plan_cycle_detected", error=str(e))
            # Fallback: flatten all tasks into one layer
            plan.execution_order = [[t.id for t in plan.tasks]]


class PlanExecutor:
    """Phase 3: Execute an ExecutionPlan task by task with quality gates.

    Walks the dependency graph, spawning workers for each ready task
    and running quality gate reviews after each one.
    """

    def __init__(
        self,
        orchestrator: SubAgentOrchestrator,
        sdk_service: ClaudeSDKService,
        event_bus: EventBus,
        role_registry: RoleRegistry | None = None,
        working_memory: WorkingMemory | None = None,
        on_question: QuestionCallback | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._sdk = sdk_service
        self._event_bus = event_bus
        self._role_registry = role_registry
        self._working_memory = working_memory
        self._on_question = on_question

    async def execute(
        self,
        plan: ExecutionPlan,
        spec: ProjectSpec,
        task_id: str = "execution",
    ) -> ProjectResult:
        """Execute the plan task by task.

        Args:
            plan: The execution plan to run.
            spec: The project specification (for context).
            task_id: Parent task ID for tracking.

        Returns:
            ProjectResult with overall status and details.
        """
        total = len(plan.tasks)
        completed = 0
        failed = 0

        logger.info(
            "plan_execution_started",
            task_id=task_id,
            total_tasks=total,
            layers=len(plan.execution_order),
        )

        while not plan.all_completed():
            ready_tasks = plan.get_next_ready_tasks()
            if not ready_tasks:
                # No more tasks can be started — remaining have unmet deps
                pending = [
                    t for t in plan.tasks if t.status in (TaskStatus.PENDING, TaskStatus.FAILED)
                ]
                if pending:
                    logger.warning(
                        "plan_execution_stuck",
                        pending=[t.id for t in pending],
                    )
                break

            # Execute tasks one at a time (sequential for rate limits)
            for micro_task in ready_tasks:
                completed_so_far = completed + 1

                await self._event_bus.emit(
                    Events.PLAN_TASK_STARTED,
                    {
                        "task_id": task_id,
                        "micro_task_id": micro_task.id,
                        "micro_task_title": micro_task.title,
                        "progress": f"{completed_so_far}/{total}",
                        "role": micro_task.role,
                    },
                )

                micro_task.status = TaskStatus.RUNNING

                success = await self._execute_single_task(
                    micro_task,
                    spec,
                    plan,
                    task_id,
                )

                if success:
                    micro_task.status = TaskStatus.PASSED
                    completed += 1

                    await self._event_bus.emit(
                        Events.PLAN_TASK_COMPLETED,
                        {
                            "task_id": task_id,
                            "micro_task_id": micro_task.id,
                            "micro_task_title": micro_task.title,
                            "progress": f"{completed}/{total}",
                        },
                    )

                    await self._event_bus.emit(
                        Events.PLAN_PROGRESS,
                        {
                            "task_id": task_id,
                            "progress": plan.progress_summary(),
                        },
                    )
                else:
                    micro_task.status = TaskStatus.FAILED
                    failed += 1

                    await self._event_bus.emit(
                        Events.PLAN_TASK_FAILED,
                        {
                            "task_id": task_id,
                            "micro_task_id": micro_task.id,
                            "micro_task_title": micro_task.title,
                            "error": micro_task.error,
                            "progress": f"{completed}/{total}",
                        },
                    )

        # Final result
        success = plan.all_completed()

        await self._event_bus.emit(
            Events.PLAN_COMPLETED,
            {
                "task_id": task_id,
                "success": success,
                "completed": completed,
                "failed": failed,
                "total": total,
            },
        )

        return ProjectResult(
            spec=spec,
            plan=plan,
            success=success,
            summary=plan.progress_summary(),
            tasks_completed=completed,
            tasks_failed=failed,
            total_tasks=total,
        )

    async def _execute_single_task(
        self,
        micro_task: MicroTask,
        spec: ProjectSpec,
        plan: ExecutionPlan,
        parent_task_id: str,
    ) -> bool:
        """Execute a single micro-task with quality gate and retry.

        Returns True if the task passed the quality gate.
        """
        from agent.core.subagent import SubAgentTask

        for attempt in range(1, _MAX_TASK_RETRIES + 1):
            # Build enriched prompt
            prompt = self._build_task_prompt(micro_task, spec, plan, attempt)

            # Resolve the role
            role = self._resolve_role(micro_task)

            task = SubAgentTask(
                role=role,
                instruction=prompt,
                context=f"Project: {spec.title}\nTask: {micro_task.title}",
                timeout_seconds=self._orchestrator.config.subagent_timeout,
            )

            # Execute via orchestrator (reuses retry/timeout logic)
            result = await self._orchestrator.spawn_subagent(task)

            if result.status != "completed":
                micro_task.error = result.error or "Worker failed"
                micro_task.retry_count = attempt
                logger.warning(
                    "micro_task_worker_failed",
                    task_id=micro_task.id,
                    attempt=attempt,
                    error=micro_task.error,
                )
                continue

            micro_task.output = result.output

            # Save output to working memory
            if self._working_memory:
                await self._working_memory.save_finding(
                    task_id=parent_task_id,
                    role=micro_task.role,
                    key=f"task_{micro_task.id}_output",
                    value=result.output[:3000],
                )

            # Quality gate: review acceptance criteria
            gate_result = await self._quality_gate(micro_task, result.output)

            if gate_result.passed:
                logger.info(
                    "micro_task_passed",
                    task_id=micro_task.id,
                    attempt=attempt,
                )
                return True

            # Failed quality gate — retry with feedback
            micro_task.error = gate_result.feedback
            micro_task.retry_count = attempt
            logger.warning(
                "micro_task_quality_gate_failed",
                task_id=micro_task.id,
                attempt=attempt,
                feedback=gate_result.feedback[:200],
            )

        # All retries exhausted
        logger.error(
            "micro_task_exhausted",
            task_id=micro_task.id,
            retries=_MAX_TASK_RETRIES,
        )
        return False

    def _build_task_prompt(
        self,
        micro_task: MicroTask,
        spec: ProjectSpec,
        plan: ExecutionPlan,
        attempt: int,
    ) -> str:
        """Build an enriched prompt for a micro-task worker."""
        parts = [
            f"## Task: {micro_task.title}",
            f"\n{micro_task.description}",
        ]

        if micro_task.acceptance_criteria:
            parts.append("\n## Acceptance Criteria")
            for i, c in enumerate(micro_task.acceptance_criteria, 1):
                parts.append(f"{i}. {c}")

        # Include outputs from dependency tasks
        dep_outputs: list[str] = []
        for dep_id in micro_task.dependencies:
            dep_task = plan.get_task(dep_id)
            if dep_task and dep_task.output:
                dep_outputs.append(f"### {dep_task.title} (completed)\n{dep_task.output[:1500]}")

        if dep_outputs:
            parts.append("\n## Previous Task Outputs")
            parts.extend(dep_outputs)

        # Include project spec summary
        parts.append(f"\n## Project Context\n{spec.summary()}")

        # Retry feedback
        if attempt > 1 and micro_task.error:
            parts.append(
                f"\n## RETRY (attempt {attempt}/{_MAX_TASK_RETRIES})\n"
                f"Previous attempt failed quality review:\n{micro_task.error}\n"
                f"Fix the issues listed above."
            )

        return "\n".join(parts)

    def _resolve_role(self, micro_task: MicroTask) -> SubAgentRole:
        """Resolve the role for a micro-task from the registry."""
        from agent.core.subagent import SubAgentRole

        if self._role_registry:
            role = self._role_registry.get_role(micro_task.role)
            if role:
                return role

        # Fallback to a generic role
        return SubAgentRole(
            name=micro_task.role,
            persona=f"You are a {micro_task.role.replace('_', ' ')}. Complete the task.",
            max_iterations=30,
        )

    async def _quality_gate(
        self,
        micro_task: MicroTask,
        worker_output: str,
    ) -> QualityGateResult:
        """Run a quality gate review on a completed task.

        Uses the orchestrator's existing _evaluate_review_output method.
        """
        review_input = f"Task: {micro_task.title}\n" f"Acceptance Criteria:\n"
        for c in micro_task.acceptance_criteria:
            review_input += f"- {c}\n"
        review_input += f"\nWorker Output:\n{worker_output[:3000]}"

        try:
            passed, feedback = await self._orchestrator._evaluate_review_output(review_input)
            return QualityGateResult(
                passed=passed,
                feedback=feedback,
                reviewer_output=feedback,
            )
        except Exception as e:
            logger.warning("quality_gate_error", error=str(e))
            # On error, pass by default to avoid blocking
            return QualityGateResult(passed=True, feedback="Review skipped (error)")


class ProjectPlannerService:
    """Top-level service that orchestrates all three phases.

    This is the entry point called by the plan_and_build tool.
    """

    def __init__(
        self,
        orchestrator: SubAgentOrchestrator,
        sdk_service: ClaudeSDKService,
        event_bus: EventBus,
        role_registry: RoleRegistry | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._sdk = sdk_service
        self._event_bus = event_bus
        self._role_registry = role_registry
        self._working_memory = working_memory

    async def plan_and_build(
        self,
        instruction: str,
        skip_requirements: bool = False,
        on_question: QuestionCallback | None = None,
        task_id: str | None = None,
        user_id: str | None = None,
    ) -> ProjectResult:
        """Run the full planning pipeline: requirements → plan → execute.

        Args:
            instruction: The user's project request.
            skip_requirements: If True, skip Phase 1 and use instruction as spec.
            on_question: Callback to route questions to the user.
            task_id: Optional task ID override.
            user_id: User ID for notifications.

        Returns:
            ProjectResult with the full execution outcome.
        """
        import uuid

        task_id = task_id or f"plan-{uuid.uuid4().hex[:8]}"

        logger.info(
            "project_planner_started",
            task_id=task_id,
            instruction=instruction[:200],
            skip_requirements=skip_requirements,
        )

        # Phase 1: Requirements
        if skip_requirements:
            spec = ProjectSpec(
                title="Project",
                description=instruction,
            )
        else:
            gatherer = RequirementsGatherer(
                sdk_service=self._sdk,
                event_bus=self._event_bus,
                working_memory=self._working_memory,
            )
            spec = await gatherer.gather(
                initial_instruction=instruction,
                on_question=on_question,
                task_id=task_id,
            )

        logger.info(
            "requirements_gathered",
            task_id=task_id,
            title=spec.title,
            features=len(spec.features),
        )

        # Phase 2: Planning
        planner = ProjectPlanner(
            sdk_service=self._sdk,
            event_bus=self._event_bus,
            role_registry=self._role_registry,
            working_memory=self._working_memory,
        )
        plan = await planner.plan(spec, task_id=task_id)

        if not plan.tasks:
            return ProjectResult(
                spec=spec,
                plan=plan,
                success=False,
                summary="Planning produced no tasks",
            )

        logger.info(
            "plan_created",
            task_id=task_id,
            tasks=len(plan.tasks),
            layers=len(plan.execution_order),
        )

        # Phase 3: Execution
        executor = PlanExecutor(
            orchestrator=self._orchestrator,
            sdk_service=self._sdk,
            event_bus=self._event_bus,
            role_registry=self._role_registry,
            working_memory=self._working_memory,
            on_question=on_question,
        )
        result = await executor.execute(plan, spec, task_id=task_id)

        logger.info(
            "project_planner_completed",
            task_id=task_id,
            success=result.success,
            completed=result.tasks_completed,
            failed=result.tasks_failed,
            total=result.total_tasks,
        )

        return result
