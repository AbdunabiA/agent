"""Integration tests for the project planner pipeline.

Tests RequirementsGatherer, ProjectPlanner, and PlanExecutor
with mocked SDK and orchestrator.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.events import EventBus, Events
from agent.core.planner_models import (
    ExecutionPlan,
    MicroTask,
    ProjectResult,
    ProjectSpec,
    TaskPriority,
    TaskStatus,
)
from agent.core.project_planner import (
    PlanExecutor,
    ProjectPlanner,
    ProjectPlannerService,
    RequirementsGatherer,
    _extract_json_block,
)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_sdk() -> MagicMock:
    sdk = MagicMock()
    sdk.run_subagent = AsyncMock()
    sdk.tool_registry = MagicMock()
    sdk.tool_registry.list_tools.return_value = []
    return sdk


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    orch = MagicMock()
    orch.config = MagicMock()
    orch.config.subagent_timeout = 300
    orch.tool_registry = MagicMock()
    orch.tool_registry.list_tools.return_value = []
    orch.spawn_subagent = AsyncMock()
    orch._evaluate_review_output = AsyncMock(return_value=(True, "PASS: All good"))
    return orch


@pytest.fixture
def mock_working_memory() -> MagicMock:
    wm = MagicMock()
    wm.save_finding = AsyncMock()
    wm.save_artifact = AsyncMock(return_value=1)
    wm.get_context_for_role = AsyncMock(return_value="")
    return wm


@pytest.fixture
def mock_role_registry() -> MagicMock:
    from agent.core.subagent import SubAgentRole

    registry = MagicMock()
    registry.get_role.return_value = SubAgentRole(
        name="backend_developer",
        persona="You are a backend developer.",
        max_iterations=10,
    )
    registry.get_roster_description.return_value = (
        "  engineering/backend_developer — a backend developer\n"
        "  engineering/frontend_developer — a frontend developer"
    )
    return registry


class TestExtractJsonBlock:
    """Test JSON extraction from LLM output."""

    def test_markdown_fenced(self) -> None:
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert json.loads(_extract_json_block(text)) == {"key": "value"}

    def test_bare_json(self) -> None:
        text = '{"key": "value"}'
        assert json.loads(_extract_json_block(text)) == {"key": "value"}

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is the result: {"a": 1} done.'
        assert json.loads(_extract_json_block(text)) == {"a": 1}

    def test_generic_fence(self) -> None:
        text = '```\n{"x": 42}\n```'
        assert json.loads(_extract_json_block(text)) == {"x": 42}

    def test_no_json_at_all(self) -> None:
        """When there are no braces, returns raw text (caller handles parse error)."""
        text = "Just plain text with no JSON"
        result = _extract_json_block(text)
        assert result == text

    def test_nested_braces(self) -> None:
        text = '{"outer": {"inner": 1}}'
        assert json.loads(_extract_json_block(text)) == {"outer": {"inner": 1}}

    def test_multiple_json_blocks_picks_fenced(self) -> None:
        """When both fenced and bare JSON exist, fenced wins."""
        text = 'Ignored: {"bad": true}\n```json\n{"good": true}\n```'
        assert json.loads(_extract_json_block(text)) == {"good": True}

    def test_fenced_with_language_tag(self) -> None:
        text = '```javascript\n{"lang": "js"}\n```'
        result = _extract_json_block(text)
        assert json.loads(result) == {"lang": "js"}

    def test_unclosed_json_fence(self) -> None:
        """Unclosed ```json fence should not raise ValueError."""
        text = '```json\n{"key": "value"}'
        result = _extract_json_block(text)
        assert json.loads(result) == {"key": "value"}

    def test_unclosed_generic_fence(self) -> None:
        """Unclosed ``` fence should not raise ValueError."""
        text = '```\n{"key": "value"}'
        result = _extract_json_block(text)
        assert json.loads(result) == {"key": "value"}


class TestRequirementsGatherer:
    """Test Phase 1: Requirements gathering."""

    @pytest.mark.asyncio
    async def test_gather_produces_spec(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_working_memory: MagicMock,
    ) -> None:
        spec_json = json.dumps(
            {
                "title": "TaskFlow",
                "description": "Task management SaaS",
                "tech_stack": ["FastAPI", "React"],
                "features": [
                    {
                        "name": "Auth",
                        "description": "User authentication",
                        "priority": "critical",
                        "acceptance_criteria": ["Users can register"],
                    },
                ],
                "components": ["backend", "frontend"],
                "constraints": [],
                "deployment": "Docker",
                "integrations": [],
            }
        )
        mock_sdk.run_subagent.return_value = f"```json\n{spec_json}\n```"

        gatherer = RequirementsGatherer(
            sdk_service=mock_sdk,
            event_bus=event_bus,
            working_memory=mock_working_memory,
        )
        spec = await gatherer.gather("Build a task management app")

        assert spec.title == "TaskFlow"
        assert len(spec.features) == 1
        assert spec.features[0].priority == TaskPriority.CRITICAL
        mock_working_memory.save_artifact.assert_called_once()

    @pytest.mark.asyncio
    async def test_gather_with_question_callback(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
    ) -> None:
        spec_json = json.dumps({"title": "App", "description": "An app"})
        mock_sdk.run_subagent.return_value = spec_json

        on_question = AsyncMock(return_value="React")

        gatherer = RequirementsGatherer(
            sdk_service=mock_sdk,
            event_bus=event_bus,
        )
        spec = await gatherer.gather(
            "Build an app",
            on_question=on_question,
        )

        assert spec.title == "App"
        # on_question is passed to run_subagent
        call_kwargs = mock_sdk.run_subagent.call_args.kwargs
        assert call_kwargs.get("on_question") is on_question

    @pytest.mark.asyncio
    async def test_gather_fallback_on_parse_error(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
    ) -> None:
        mock_sdk.run_subagent.return_value = "Not valid JSON at all"

        gatherer = RequirementsGatherer(
            sdk_service=mock_sdk,
            event_bus=event_bus,
        )
        spec = await gatherer.gather("Build something")

        # Should fall back to a minimal spec
        assert spec.title == "Untitled Project"


class TestProjectPlanner:
    """Test Phase 2: Project decomposition."""

    @pytest.mark.asyncio
    async def test_plan_produces_tasks(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "mt-001",
                        "title": "Create project structure",
                        "description": "Set up directories",
                        "role": "backend_developer",
                        "dependencies": [],
                        "acceptance_criteria": ["Dirs exist"],
                        "layer": 0,
                    },
                    {
                        "id": "mt-002",
                        "title": "Implement auth",
                        "description": "JWT auth",
                        "role": "backend_developer",
                        "dependencies": ["mt-001"],
                        "acceptance_criteria": ["Login works"],
                        "layer": 1,
                    },
                ],
                "execution_order": [["mt-001"], ["mt-002"]],
            }
        )
        mock_sdk.run_subagent.return_value = f"```json\n{plan_json}\n```"

        planner = ProjectPlanner(
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )

        spec = ProjectSpec(title="Test", description="Test project")
        plan = await planner.plan(spec)

        assert len(plan.tasks) == 2
        assert plan.tasks[0].id == "mt-001"
        assert plan.tasks[1].dependencies == ["mt-001"]
        assert len(plan.execution_order) == 2

    @pytest.mark.asyncio
    async def test_plan_removes_invalid_deps(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
    ) -> None:
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "mt-001",
                        "title": "Task",
                        "description": "",
                        "role": "backend_developer",
                        "dependencies": ["nonexistent"],
                        "acceptance_criteria": [],
                        "layer": 0,
                    },
                ],
                "execution_order": [["mt-001"]],
            }
        )
        mock_sdk.run_subagent.return_value = plan_json

        planner = ProjectPlanner(
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
        )

        spec = ProjectSpec(title="Test", description="Test")
        plan = await planner.plan(spec)

        # Invalid dep should be removed
        assert plan.tasks[0].dependencies == []


class TestPlanExecutor:
    """Test Phase 3: Plan execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_plan(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-test",
            role_name="backend_developer",
            status=SubAgentStatus.COMPLETED,
            output="Created the project structure successfully.",
        )

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Create structure",
                    description="Set up dirs",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Dirs exist"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="Test", description="Test project")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )

        result = await executor.execute(plan, spec)

        assert result.success
        assert result.tasks_completed == 1
        assert result.tasks_failed == 0
        assert plan.tasks[0].status == TaskStatus.PASSED

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        call_count = 0

        async def mock_spawn(task: Any) -> SubAgentResult:
            nonlocal call_count
            call_count += 1
            return SubAgentResult(
                task_id=f"sa-{call_count}",
                role_name="backend_developer",
                status=SubAgentStatus.COMPLETED,
                output=f"Task {call_count} done",
            )

        mock_orchestrator.spawn_subagent.side_effect = mock_spawn

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Setup",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Done"],
                ),
                MicroTask(
                    id="mt-002",
                    title="Build",
                    description="",
                    role="backend_developer",
                    dependencies=["mt-001"],
                    acceptance_criteria=["Done"],
                ),
            ],
            execution_order=[["mt-001"], ["mt-002"]],
        )
        spec = ProjectSpec(title="Test", description="Test")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )

        result = await executor.execute(plan, spec)

        assert result.success
        assert result.tasks_completed == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_worker_failure_and_retry(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        call_count = 0

        async def mock_spawn(task: Any) -> SubAgentResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SubAgentResult(
                    task_id="sa-fail",
                    role_name="dev",
                    status=SubAgentStatus.FAILED,
                    error="Timeout",
                )
            return SubAgentResult(
                task_id="sa-ok",
                role_name="dev",
                status=SubAgentStatus.COMPLETED,
                output="Done on retry",
            )

        mock_orchestrator.spawn_subagent.side_effect = mock_spawn

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Task",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Done"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="Test", description="Test")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )

        result = await executor.execute(plan, spec)

        assert result.success
        assert call_count == 2  # first failed, second succeeded


class TestRequirementsGathererEdgeCases:
    """Edge cases for requirements gathering."""

    @pytest.mark.asyncio
    async def test_gather_without_working_memory(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
    ) -> None:
        """Working memory is optional — should not fail when None."""
        spec_json = json.dumps({"title": "App", "description": "desc"})
        mock_sdk.run_subagent.return_value = spec_json

        gatherer = RequirementsGatherer(
            sdk_service=mock_sdk,
            event_bus=event_bus,
            working_memory=None,
        )
        spec = await gatherer.gather("Build something")
        assert spec.title == "App"

    @pytest.mark.asyncio
    async def test_gather_emits_plan_created_event(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
    ) -> None:
        spec_json = json.dumps({"title": "E", "description": "D", "features": []})
        mock_sdk.run_subagent.return_value = spec_json

        events_received: list[dict] = []
        event_bus.on(Events.PLAN_CREATED, lambda d: events_received.append(d))

        gatherer = RequirementsGatherer(sdk_service=mock_sdk, event_bus=event_bus)
        await gatherer.gather("Build it")

        assert len(events_received) == 1
        assert events_received[0]["phase"] == "requirements"


class TestProjectPlannerEdgeCases:
    """Edge cases for project planner."""

    @pytest.mark.asyncio
    async def test_plan_without_role_registry(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
    ) -> None:
        """Planner should work even without a role registry."""
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "mt-001",
                        "title": "T",
                        "description": "",
                        "role": "dev",
                        "dependencies": [],
                        "acceptance_criteria": [],
                        "layer": 0,
                    },
                ],
                "execution_order": [["mt-001"]],
            }
        )
        mock_sdk.run_subagent.return_value = plan_json

        planner = ProjectPlanner(
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=None,
        )
        spec = ProjectSpec(title="T", description="D")
        plan = await planner.plan(spec)

        assert len(plan.tasks) == 1
        # Prompt should not contain roster section
        call_args = mock_sdk.run_subagent.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs["prompt"]
        assert "Available Agent Roles" not in prompt

    @pytest.mark.asyncio
    async def test_plan_with_cyclic_deps_flattened(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
    ) -> None:
        """When LLM returns cyclic deps, validate_plan should flatten to one layer."""
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "a",
                        "title": "A",
                        "description": "",
                        "role": "backend_developer",
                        "dependencies": ["b"],
                        "acceptance_criteria": [],
                        "layer": 0,
                    },
                    {
                        "id": "b",
                        "title": "B",
                        "description": "",
                        "role": "backend_developer",
                        "dependencies": ["a"],
                        "acceptance_criteria": [],
                        "layer": 0,
                    },
                ],
                "execution_order": [["a", "b"]],
            }
        )
        mock_sdk.run_subagent.return_value = plan_json

        planner = ProjectPlanner(
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
        )
        spec = ProjectSpec(title="T", description="D")
        plan = await planner.plan(spec)

        # Cycle should be detected, fallback to flat layer
        assert len(plan.execution_order) == 1
        assert sorted(plan.execution_order[0]) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_plan_parse_failure_returns_empty(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
    ) -> None:
        mock_sdk.run_subagent.return_value = "Completely invalid output"

        planner = ProjectPlanner(sdk_service=mock_sdk, event_bus=event_bus)
        spec = ProjectSpec(title="T", description="D")
        plan = await planner.plan(spec)

        assert len(plan.tasks) == 0

    @pytest.mark.asyncio
    async def test_plan_emits_plan_created_event(
        self,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
    ) -> None:
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "mt-001",
                        "title": "T",
                        "description": "",
                        "role": "backend_developer",
                        "dependencies": [],
                        "acceptance_criteria": [],
                        "layer": 0,
                    },
                ],
                "execution_order": [["mt-001"]],
            }
        )
        mock_sdk.run_subagent.return_value = plan_json

        events_received: list[dict] = []
        event_bus.on(Events.PLAN_CREATED, lambda d: events_received.append(d))

        planner = ProjectPlanner(
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
        )
        spec = ProjectSpec(title="T", description="D")
        await planner.plan(spec)

        assert len(events_received) == 1
        assert events_received[0]["phase"] == "planning"
        assert events_received[0]["task_count"] == 1


class TestPlanExecutorEdgeCases:
    """Edge cases for plan execution."""

    @pytest.mark.asyncio
    async def test_execute_empty_plan(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
    ) -> None:
        """Empty plan = vacuous success."""
        plan = ExecutionPlan(tasks=[], execution_order=[])
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
        )
        result = await executor.execute(plan, spec)

        assert result.success
        assert result.tasks_completed == 0
        assert result.total_tasks == 0

    @pytest.mark.asyncio
    async def test_execute_all_retries_exhausted(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """Worker fails all 3 attempts → task fails permanently."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-fail",
            role_name="dev",
            status=SubAgentStatus.FAILED,
            error="Always fails",
        )

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Doomed",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Never"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )
        result = await executor.execute(plan, spec)

        assert not result.success
        assert result.tasks_failed == 1
        assert result.tasks_completed == 0
        assert plan.tasks[0].status == TaskStatus.FAILED
        # Should have attempted 3 times
        assert mock_orchestrator.spawn_subagent.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_quality_gate_fail_then_pass(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """Quality gate fails first attempt, passes second."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-ok",
            role_name="dev",
            status=SubAgentStatus.COMPLETED,
            output="Some output",
        )

        call_count = 0

        async def mock_evaluate(review_output: str) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False, "FAIL: Missing error handling"
            return True, "PASS: All good"

        mock_orchestrator._evaluate_review_output.side_effect = mock_evaluate

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Task",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Has error handling"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )
        result = await executor.execute(plan, spec)

        assert result.success
        assert mock_orchestrator.spawn_subagent.call_count == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_quality_gate_exception_auto_passes(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """When quality gate throws an exception, task passes by default."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-ok",
            role_name="dev",
            status=SubAgentStatus.COMPLETED,
            output="Output",
        )
        mock_orchestrator._evaluate_review_output.side_effect = RuntimeError("LLM down")

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Task",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Something"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )
        result = await executor.execute(plan, spec)

        # Auto-pass on quality gate error
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_stuck_when_dep_fails(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """When a task fails, downstream tasks with unmet deps can't run."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-fail",
            role_name="dev",
            status=SubAgentStatus.FAILED,
            error="Broken",
        )

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Foundation",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Done"],
                ),
                MicroTask(
                    id="mt-002",
                    title="Build on top",
                    description="",
                    role="backend_developer",
                    dependencies=["mt-001"],
                    acceptance_criteria=["Done"],
                ),
            ],
            execution_order=[["mt-001"], ["mt-002"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )
        result = await executor.execute(plan, spec)

        assert not result.success
        assert result.tasks_failed == 1
        assert result.tasks_completed == 0
        # mt-002 should still be PENDING (never started)
        assert plan.tasks[1].status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_resolve_role_unknown_uses_fallback(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_working_memory: MagicMock,
    ) -> None:
        """When role_registry is None, a generic fallback role is used."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-ok",
            role_name="unknown_role",
            status=SubAgentStatus.COMPLETED,
            output="Done",
        )

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Task",
                    description="",
                    role="unknown_role",
                    dependencies=[],
                    acceptance_criteria=["Done"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=None,  # No registry
            working_memory=mock_working_memory,
        )
        result = await executor.execute(plan, spec)

        assert result.success
        # Verify the fallback role was used
        task_arg = mock_orchestrator.spawn_subagent.call_args.args[0]
        assert task_arg.role.name == "unknown_role"
        assert "unknown role" in task_arg.role.persona

    @pytest.mark.asyncio
    async def test_build_task_prompt_includes_dep_outputs(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """Worker prompt for task with deps includes previous task outputs."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        call_prompts: list[str] = []

        async def capture_spawn(task: Any) -> SubAgentResult:
            call_prompts.append(task.instruction)
            return SubAgentResult(
                task_id="sa-ok",
                role_name="dev",
                status=SubAgentStatus.COMPLETED,
                output="Result from this task",
            )

        mock_orchestrator.spawn_subagent.side_effect = capture_spawn

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Setup DB",
                    description="Create schema",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Schema created"],
                ),
                MicroTask(
                    id="mt-002",
                    title="Build API",
                    description="Create endpoints",
                    role="backend_developer",
                    dependencies=["mt-001"],
                    acceptance_criteria=["Endpoints work"],
                ),
            ],
            execution_order=[["mt-001"], ["mt-002"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )
        await executor.execute(plan, spec)

        assert len(call_prompts) == 2
        # Second task's prompt should include output from first
        assert "Result from this task" in call_prompts[1]
        assert "Previous Task Outputs" in call_prompts[1]

    @pytest.mark.asyncio
    async def test_build_task_prompt_includes_retry_feedback(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """On retry, prompt includes feedback from the failed quality gate."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        call_prompts: list[str] = []

        async def capture_spawn(task: Any) -> SubAgentResult:
            call_prompts.append(task.instruction)
            return SubAgentResult(
                task_id="sa-ok",
                role_name="dev",
                status=SubAgentStatus.COMPLETED,
                output="Output",
            )

        mock_orchestrator.spawn_subagent.side_effect = capture_spawn

        gate_count = 0

        async def mock_gate(review: str) -> tuple[bool, str]:
            nonlocal gate_count
            gate_count += 1
            if gate_count == 1:
                return False, "FAIL: Missing validation"
            return True, "PASS: OK"

        mock_orchestrator._evaluate_review_output.side_effect = mock_gate

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Task",
                    description="Do it",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Validated"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )
        await executor.execute(plan, spec)

        assert len(call_prompts) == 2
        # Second prompt should contain retry feedback
        assert "RETRY" in call_prompts[1]
        assert "Missing validation" in call_prompts[1]

    @pytest.mark.asyncio
    async def test_execute_emits_all_events(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """Verify all expected events are emitted during execution."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-ok",
            role_name="dev",
            status=SubAgentStatus.COMPLETED,
            output="Done",
        )

        events_by_type: dict[str, list[dict]] = {}

        async def capture_event(event_name: str, data: dict) -> None:
            events_by_type.setdefault(event_name, []).append(data)

        # Subscribe to all plan events
        for event_name in [
            Events.PLAN_TASK_STARTED,
            Events.PLAN_TASK_COMPLETED,
            Events.PLAN_PROGRESS,
            Events.PLAN_COMPLETED,
        ]:
            event_bus.on(
                event_name,
                lambda data, en=event_name: events_by_type.setdefault(en, []).append(data),
            )

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Task",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Done"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )
        await executor.execute(plan, spec)

        assert Events.PLAN_TASK_STARTED in events_by_type
        assert Events.PLAN_TASK_COMPLETED in events_by_type
        assert Events.PLAN_PROGRESS in events_by_type
        assert Events.PLAN_COMPLETED in events_by_type
        assert events_by_type[Events.PLAN_COMPLETED][0]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_without_working_memory(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
    ) -> None:
        """Executor works without working memory (just skips save)."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-ok",
            role_name="dev",
            status=SubAgentStatus.COMPLETED,
            output="Done",
        )

        plan = ExecutionPlan(
            tasks=[
                MicroTask(
                    id="mt-001",
                    title="Task",
                    description="",
                    role="backend_developer",
                    dependencies=[],
                    acceptance_criteria=["Done"],
                ),
            ],
            execution_order=[["mt-001"]],
        )
        spec = ProjectSpec(title="T", description="D")

        executor = PlanExecutor(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=None,
        )
        result = await executor.execute(plan, spec)

        assert result.success


class TestProjectPlannerService:
    """Test the full pipeline."""

    @pytest.mark.asyncio
    async def test_plan_and_build_skip_requirements(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        # Planner returns a simple plan
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "mt-001",
                        "title": "Do the thing",
                        "description": "Do it",
                        "role": "backend_developer",
                        "dependencies": [],
                        "acceptance_criteria": ["It's done"],
                        "layer": 0,
                    },
                ],
                "execution_order": [["mt-001"]],
            }
        )
        mock_sdk.run_subagent.return_value = plan_json

        # Worker succeeds
        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-1",
            role_name="backend_developer",
            status=SubAgentStatus.COMPLETED,
            output="All done!",
        )

        service = ProjectPlannerService(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )

        result = await service.plan_and_build(
            instruction="Build a todo app",
            skip_requirements=True,
        )

        assert result.success
        assert result.tasks_completed == 1
        assert result.total_tasks == 1

    @pytest.mark.asyncio
    async def test_plan_and_build_with_requirements(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """Full pipeline with requirements gathering."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        # SDK returns spec for gatherer, then plan for planner
        spec_json = json.dumps(
            {
                "title": "Todo App",
                "description": "A task tracker",
                "tech_stack": ["Python"],
                "features": [],
            }
        )
        plan_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "mt-001",
                        "title": "Setup",
                        "description": "",
                        "role": "backend_developer",
                        "dependencies": [],
                        "acceptance_criteria": ["Done"],
                        "layer": 0,
                    },
                ],
                "execution_order": [["mt-001"]],
            }
        )
        mock_sdk.run_subagent.side_effect = [spec_json, plan_json]

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-1",
            role_name="backend_developer",
            status=SubAgentStatus.COMPLETED,
            output="Done!",
        )

        service = ProjectPlannerService(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )

        result = await service.plan_and_build(
            instruction="Build a todo app",
            skip_requirements=False,
        )

        assert result.success
        assert result.spec is not None
        assert result.spec.title == "Todo App"
        assert mock_sdk.run_subagent.call_count == 2

    @pytest.mark.asyncio
    async def test_plan_and_build_empty_plan_returns_failure(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
        mock_working_memory: MagicMock,
    ) -> None:
        """When planner returns empty plan, should return failure."""
        mock_sdk.run_subagent.return_value = "No valid JSON here"

        service = ProjectPlannerService(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
            working_memory=mock_working_memory,
        )

        result = await service.plan_and_build(
            instruction="Build something",
            skip_requirements=True,
        )

        assert not result.success
        assert "no tasks" in result.summary.lower()
        # Worker should never be called
        mock_orchestrator.spawn_subagent.assert_not_called()

    @pytest.mark.asyncio
    async def test_plan_and_build_generates_task_id(
        self,
        mock_orchestrator: MagicMock,
        mock_sdk: MagicMock,
        event_bus: EventBus,
        mock_role_registry: MagicMock,
    ) -> None:
        """task_id is auto-generated when not provided."""
        plan_json = json.dumps({"tasks": [], "execution_order": []})
        mock_sdk.run_subagent.return_value = plan_json

        service = ProjectPlannerService(
            orchestrator=mock_orchestrator,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            role_registry=mock_role_registry,
        )

        result = await service.plan_and_build(
            instruction="Build it",
            skip_requirements=True,
        )

        # Should not crash; returns failure due to empty plan
        assert not result.success


class TestPlannerTools:
    """Test the plan_and_build tool registration and edge cases."""

    def test_get_planner_service_uninitialized(self) -> None:
        """Should raise RuntimeError when service not set."""
        # Save and clear
        import agent.tools.builtins.planner_tools as mod
        from agent.tools.builtins.planner_tools import (
            get_planner_service,
        )

        original = mod._global_planner_service
        mod._global_planner_service = None

        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                get_planner_service()
        finally:
            mod._global_planner_service = original

    def test_set_and_get_planner_service(self) -> None:
        import agent.tools.builtins.planner_tools as mod
        from agent.tools.builtins.planner_tools import get_planner_service, set_planner_service

        original = mod._global_planner_service

        mock_service = MagicMock()
        set_planner_service(mock_service)

        try:
            assert get_planner_service() is mock_service
        finally:
            mod._global_planner_service = original

    @pytest.mark.asyncio
    async def test_plan_and_build_tool_success(self) -> None:
        import agent.tools.builtins.planner_tools as mod
        from agent.tools.builtins.planner_tools import plan_and_build_tool

        original = mod._global_planner_service

        mock_service = MagicMock()
        mock_service.plan_and_build = AsyncMock(
            return_value=ProjectResult(
                spec=ProjectSpec(title="App", description=""),
                success=True,
                tasks_completed=3,
                total_tasks=3,
                summary="3/3 done",
            )
        )
        mod._global_planner_service = mock_service

        try:
            result = await plan_and_build_tool("Build an app")
            assert "completed successfully" in result
            assert "3/3" in result
        finally:
            mod._global_planner_service = original

    @pytest.mark.asyncio
    async def test_plan_and_build_tool_partial_failure(self) -> None:
        import agent.tools.builtins.planner_tools as mod
        from agent.tools.builtins.planner_tools import plan_and_build_tool

        original = mod._global_planner_service

        mock_service = MagicMock()
        mock_service.plan_and_build = AsyncMock(
            return_value=ProjectResult(
                success=False,
                tasks_completed=2,
                tasks_failed=1,
                total_tasks=3,
                summary="2/3 done, 0 running, 1 failed",
            )
        )
        mod._global_planner_service = mock_service

        try:
            result = await plan_and_build_tool("Build something")
            assert "partially completed" in result
            assert "1 failed" in result
        finally:
            mod._global_planner_service = original

    @pytest.mark.asyncio
    async def test_plan_and_build_tool_exception(self) -> None:
        import agent.tools.builtins.planner_tools as mod
        from agent.tools.builtins.planner_tools import plan_and_build_tool

        original = mod._global_planner_service

        mock_service = MagicMock()
        mock_service.plan_and_build = AsyncMock(side_effect=RuntimeError("SDK down"))
        mod._global_planner_service = mock_service

        try:
            result = await plan_and_build_tool("Build it")
            assert "failed" in result.lower()
            assert "SDK down" in result
        finally:
            mod._global_planner_service = original
