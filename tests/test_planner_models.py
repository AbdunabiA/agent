"""Tests for planner_models — data models and topological sort."""

from __future__ import annotations

import pytest

from agent.core.planner_models import (
    ExecutionPlan,
    FeatureSpec,
    MicroTask,
    ProjectResult,
    ProjectSpec,
    QualityGateResult,
    TaskPriority,
    TaskStatus,
)
from agent.core.project_planner import topological_sort


class TestProjectSpec:
    """ProjectSpec serialization and deserialization."""

    def test_to_dict_round_trip(self) -> None:
        spec = ProjectSpec(
            title="Test App",
            description="A test application",
            tech_stack=["Python", "React"],
            features=[
                FeatureSpec(
                    name="Auth",
                    description="User login",
                    priority=TaskPriority.CRITICAL,
                    acceptance_criteria=["Users can log in"],
                ),
            ],
            components=["backend", "frontend"],
            constraints=["Must be fast"],
            deployment="Docker",
            integrations=["Stripe"],
        )

        data = spec.to_dict()
        restored = ProjectSpec.from_dict(data)

        assert restored.title == "Test App"
        assert restored.tech_stack == ["Python", "React"]
        assert len(restored.features) == 1
        assert restored.features[0].name == "Auth"
        assert restored.features[0].priority == TaskPriority.CRITICAL
        assert restored.components == ["backend", "frontend"]

    def test_from_dict_defaults(self) -> None:
        spec = ProjectSpec.from_dict({"title": "Minimal"})
        assert spec.title == "Minimal"
        assert spec.tech_stack == []
        assert spec.features == []

    def test_summary(self) -> None:
        spec = ProjectSpec(
            title="My Project",
            description="Does things",
            tech_stack=["Go"],
            features=[
                FeatureSpec(name="F1", description="Feature one", priority=TaskPriority.HIGH),
            ],
        )
        summary = spec.summary()
        assert "My Project" in summary
        assert "Go" in summary
        assert "F1" in summary


class TestMicroTask:
    """MicroTask serialization."""

    def test_to_dict_round_trip(self) -> None:
        task = MicroTask(
            id="mt-001",
            title="Create structure",
            description="Set up dirs",
            role="fullstack_developer",
            dependencies=[],
            acceptance_criteria=["Dirs exist"],
            layer=0,
        )

        data = task.to_dict()
        restored = MicroTask.from_dict(data)

        assert restored.id == "mt-001"
        assert restored.role == "fullstack_developer"
        assert restored.status == TaskStatus.PENDING
        assert restored.layer == 0


class TestExecutionPlan:
    """ExecutionPlan — get_next_ready_tasks, all_completed, progress."""

    def _make_plan(self) -> ExecutionPlan:
        return ExecutionPlan(
            tasks=[
                MicroTask(id="a", title="A", description="", role="dev", dependencies=[]),
                MicroTask(id="b", title="B", description="", role="dev", dependencies=["a"]),
                MicroTask(id="c", title="C", description="", role="dev", dependencies=["a"]),
                MicroTask(id="d", title="D", description="", role="dev", dependencies=["b", "c"]),
            ],
            execution_order=[["a"], ["b", "c"], ["d"]],
        )

    def test_get_next_ready_initial(self) -> None:
        plan = self._make_plan()
        ready = plan.get_next_ready_tasks()
        assert [t.id for t in ready] == ["a"]

    def test_get_next_ready_after_a(self) -> None:
        plan = self._make_plan()
        plan.tasks[0].status = TaskStatus.PASSED
        ready = plan.get_next_ready_tasks()
        assert sorted(t.id for t in ready) == ["b", "c"]

    def test_get_next_ready_after_ab(self) -> None:
        plan = self._make_plan()
        plan.tasks[0].status = TaskStatus.PASSED
        plan.tasks[1].status = TaskStatus.PASSED
        ready = plan.get_next_ready_tasks()
        # d still needs c
        assert [t.id for t in ready] == ["c"]

    def test_all_completed(self) -> None:
        plan = self._make_plan()
        assert not plan.all_completed()
        for t in plan.tasks:
            t.status = TaskStatus.PASSED
        assert plan.all_completed()

    def test_progress_summary(self) -> None:
        plan = self._make_plan()
        plan.tasks[0].status = TaskStatus.PASSED
        plan.tasks[1].status = TaskStatus.RUNNING
        summary = plan.progress_summary()
        assert "1/4 done" in summary
        assert "1 running" in summary

    def test_get_task(self) -> None:
        plan = self._make_plan()
        assert plan.get_task("b") is not None
        assert plan.get_task("z") is None

    def test_to_dict_round_trip(self) -> None:
        plan = self._make_plan()
        data = plan.to_dict()
        restored = ExecutionPlan.from_dict(data)
        assert len(restored.tasks) == 4
        assert restored.execution_order == [["a"], ["b", "c"], ["d"]]


class TestTopologicalSort:
    """topological_sort — layers and cycle detection."""

    def test_linear_chain(self) -> None:
        tasks = [
            MicroTask(id="1", title="", description="", role="dev", dependencies=[]),
            MicroTask(id="2", title="", description="", role="dev", dependencies=["1"]),
            MicroTask(id="3", title="", description="", role="dev", dependencies=["2"]),
        ]
        layers = topological_sort(tasks)
        assert layers == [["1"], ["2"], ["3"]]
        assert tasks[0].layer == 0
        assert tasks[1].layer == 1
        assert tasks[2].layer == 2

    def test_diamond(self) -> None:
        tasks = [
            MicroTask(id="a", title="", description="", role="dev", dependencies=[]),
            MicroTask(id="b", title="", description="", role="dev", dependencies=["a"]),
            MicroTask(id="c", title="", description="", role="dev", dependencies=["a"]),
            MicroTask(id="d", title="", description="", role="dev", dependencies=["b", "c"]),
        ]
        layers = topological_sort(tasks)
        assert layers[0] == ["a"]
        assert sorted(layers[1]) == ["b", "c"]
        assert layers[2] == ["d"]

    def test_cycle_detection(self) -> None:
        tasks = [
            MicroTask(id="x", title="", description="", role="dev", dependencies=["y"]),
            MicroTask(id="y", title="", description="", role="dev", dependencies=["x"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(tasks)

    def test_missing_dep_skipped(self) -> None:
        tasks = [
            MicroTask(id="a", title="", description="", role="dev", dependencies=["nonexistent"]),
        ]
        # Should not raise — missing deps are logged and skipped
        layers = topological_sort(tasks)
        assert layers == [["a"]]

    def test_no_tasks(self) -> None:
        layers = topological_sort([])
        assert layers == []

    def test_parallel_independent(self) -> None:
        tasks = [
            MicroTask(id="a", title="", description="", role="dev", dependencies=[]),
            MicroTask(id="b", title="", description="", role="dev", dependencies=[]),
            MicroTask(id="c", title="", description="", role="dev", dependencies=[]),
        ]
        layers = topological_sort(tasks)
        assert len(layers) == 1
        assert sorted(layers[0]) == ["a", "b", "c"]


class TestExecutionPlanEmpty:
    """ExecutionPlan edge cases with empty or minimal plans."""

    def test_empty_plan_get_next_ready(self) -> None:
        plan = ExecutionPlan()
        assert plan.get_next_ready_tasks() == []

    def test_empty_plan_all_completed(self) -> None:
        # Vacuously true — no tasks means all tasks have passed
        plan = ExecutionPlan()
        assert plan.all_completed()

    def test_empty_plan_progress_summary(self) -> None:
        plan = ExecutionPlan()
        summary = plan.progress_summary()
        assert "0/0 done" in summary

    def test_get_next_ready_skips_running(self) -> None:
        plan = ExecutionPlan(
            tasks=[
                MicroTask(id="a", title="A", description="", role="dev", dependencies=[]),
                MicroTask(id="b", title="B", description="", role="dev", dependencies=[]),
            ],
        )
        plan.tasks[0].status = TaskStatus.RUNNING
        ready = plan.get_next_ready_tasks()
        # Only b is PENDING; a is RUNNING so not returned
        assert [t.id for t in ready] == ["b"]

    def test_get_next_ready_skips_failed(self) -> None:
        plan = ExecutionPlan(
            tasks=[
                MicroTask(id="a", title="A", description="", role="dev", dependencies=[]),
                MicroTask(id="b", title="B", description="", role="dev", dependencies=[]),
            ],
        )
        plan.tasks[0].status = TaskStatus.FAILED
        ready = plan.get_next_ready_tasks()
        assert [t.id for t in ready] == ["b"]

    def test_stuck_when_dep_failed(self) -> None:
        """When a dependency fails, downstream tasks can never become ready."""
        plan = ExecutionPlan(
            tasks=[
                MicroTask(id="a", title="A", description="", role="dev", dependencies=[]),
                MicroTask(id="b", title="B", description="", role="dev", dependencies=["a"]),
            ],
        )
        plan.tasks[0].status = TaskStatus.FAILED
        # b depends on a which is FAILED (not PASSED) — b stays PENDING but not ready
        ready = plan.get_next_ready_tasks()
        assert ready == []

    def test_all_completed_false_when_failed(self) -> None:
        plan = ExecutionPlan(
            tasks=[
                MicroTask(id="a", title="A", description="", role="dev", dependencies=[]),
            ],
        )
        plan.tasks[0].status = TaskStatus.FAILED
        assert not plan.all_completed()

    def test_progress_summary_all_statuses(self) -> None:
        plan = ExecutionPlan(
            tasks=[
                MicroTask(id="a", title="A", description="", role="dev"),
                MicroTask(id="b", title="B", description="", role="dev"),
                MicroTask(id="c", title="C", description="", role="dev"),
                MicroTask(id="d", title="D", description="", role="dev"),
            ],
        )
        plan.tasks[0].status = TaskStatus.PASSED
        plan.tasks[1].status = TaskStatus.RUNNING
        plan.tasks[2].status = TaskStatus.FAILED
        # d stays PENDING
        summary = plan.progress_summary()
        assert "1/4 done" in summary
        assert "1 running" in summary
        assert "1 failed" in summary


class TestProjectSpecEdgeCases:
    """Additional ProjectSpec edge cases."""

    def test_from_dict_completely_empty(self) -> None:
        spec = ProjectSpec.from_dict({})
        assert spec.title == ""
        assert spec.description == ""
        assert spec.features == []
        assert spec.deployment == ""

    def test_summary_minimal_spec(self) -> None:
        spec = ProjectSpec(title="X", description="Y")
        summary = spec.summary()
        assert "X" in summary
        assert "Y" in summary
        # No optional sections
        assert "Tech Stack" not in summary
        assert "Components" not in summary
        assert "Constraints" not in summary
        assert "Deployment" not in summary
        assert "Features" not in summary

    def test_summary_with_constraints_and_deployment(self) -> None:
        spec = ProjectSpec(
            title="P",
            description="D",
            constraints=["Fast", "Cheap"],
            deployment="K8s",
        )
        summary = spec.summary()
        assert "Fast; Cheap" in summary
        assert "K8s" in summary


class TestMicroTaskEdgeCases:
    """Additional MicroTask edge cases."""

    def test_from_dict_missing_optional_fields(self) -> None:
        task = MicroTask.from_dict(
            {
                "id": "x",
                "title": "T",
                "role": "dev",
            }
        )
        assert task.description == ""
        assert task.dependencies == []
        assert task.acceptance_criteria == []
        assert task.status == TaskStatus.PENDING
        assert task.layer == 0

    def test_from_dict_preserves_status(self) -> None:
        task = MicroTask.from_dict(
            {
                "id": "x",
                "title": "T",
                "role": "dev",
                "status": "passed",
            }
        )
        assert task.status == TaskStatus.PASSED


class TestTopologicalSortEdgeCases:
    """Additional topological sort edge cases."""

    def test_self_loop(self) -> None:
        tasks = [
            MicroTask(id="a", title="", description="", role="dev", dependencies=["a"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(tasks)

    def test_three_node_cycle(self) -> None:
        tasks = [
            MicroTask(id="a", title="", description="", role="dev", dependencies=["c"]),
            MicroTask(id="b", title="", description="", role="dev", dependencies=["a"]),
            MicroTask(id="c", title="", description="", role="dev", dependencies=["b"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(tasks)

    def test_deep_chain(self) -> None:
        """10-node linear chain."""
        tasks = []
        for i in range(10):
            deps = [str(i - 1)] if i > 0 else []
            tasks.append(
                MicroTask(id=str(i), title="", description="", role="dev", dependencies=deps)
            )
        layers = topological_sort(tasks)
        assert len(layers) == 10
        for i, layer in enumerate(layers):
            assert layer == [str(i)]

    def test_single_task(self) -> None:
        tasks = [MicroTask(id="only", title="", description="", role="dev", dependencies=[])]
        layers = topological_sort(tasks)
        assert layers == [["only"]]
        assert tasks[0].layer == 0

    def test_wide_then_narrow(self) -> None:
        """3 independent tasks funneling into 1."""
        tasks = [
            MicroTask(id="a", title="", description="", role="dev", dependencies=[]),
            MicroTask(id="b", title="", description="", role="dev", dependencies=[]),
            MicroTask(id="c", title="", description="", role="dev", dependencies=[]),
            MicroTask(id="d", title="", description="", role="dev", dependencies=["a", "b", "c"]),
        ]
        layers = topological_sort(tasks)
        assert len(layers) == 2
        assert sorted(layers[0]) == ["a", "b", "c"]
        assert layers[1] == ["d"]


class TestQualityGateResult:
    """QualityGateResult basic usage."""

    def test_pass(self) -> None:
        result = QualityGateResult(passed=True, feedback="All good")
        assert result.passed

    def test_fail(self) -> None:
        result = QualityGateResult(passed=False, feedback="Missing tests")
        assert not result.passed

    def test_with_reviewer_output(self) -> None:
        result = QualityGateResult(
            passed=True, feedback="OK", reviewer_output="PASS: all criteria met"
        )
        assert result.reviewer_output == "PASS: all criteria met"


class TestProjectResult:
    """ProjectResult basic usage."""

    def test_defaults(self) -> None:
        result = ProjectResult()
        assert not result.success
        assert result.tasks_completed == 0

    def test_with_spec_and_plan(self) -> None:
        spec = ProjectSpec(title="T", description="D")
        plan = ExecutionPlan(tasks=[])
        result = ProjectResult(
            spec=spec,
            plan=plan,
            success=True,
            tasks_completed=5,
            tasks_failed=1,
            total_tasks=6,
        )
        assert result.spec is spec
        assert result.plan is plan
        assert result.tasks_failed == 1
