"""Data models for the intelligent project planner system.

Defines the structured types used by RequirementsGatherer, ProjectPlanner,
and PlanExecutor to decompose complex projects into micro-tasks with
dependency graphs and quality gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    """Status of a micro-task in the execution plan."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(StrEnum):
    """Priority level for features and tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FeatureSpec:
    """Individual feature in the project specification."""

    name: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class ProjectSpec:
    """Requirements output from the RequirementsGatherer.

    Contains all information needed for the ProjectPlanner to
    decompose the project into micro-tasks.
    """

    title: str
    description: str
    tech_stack: list[str] = field(default_factory=list)
    features: list[FeatureSpec] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    deployment: str = ""
    integrations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "title": self.title,
            "description": self.description,
            "tech_stack": self.tech_stack,
            "features": [
                {
                    "name": f.name,
                    "description": f.description,
                    "priority": f.priority.value,
                    "acceptance_criteria": f.acceptance_criteria,
                }
                for f in self.features
            ],
            "components": self.components,
            "constraints": self.constraints,
            "deployment": self.deployment,
            "integrations": self.integrations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectSpec:
        """Deserialize from a dict."""
        features = [
            FeatureSpec(
                name=f["name"],
                description=f["description"],
                priority=TaskPriority(f.get("priority", "medium")),
                acceptance_criteria=f.get("acceptance_criteria", []),
            )
            for f in data.get("features", [])
        ]
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            tech_stack=data.get("tech_stack", []),
            features=features,
            components=data.get("components", []),
            constraints=data.get("constraints", []),
            deployment=data.get("deployment", ""),
            integrations=data.get("integrations", []),
        )

    def summary(self) -> str:
        """Return a human-readable summary for user approval."""
        lines = [
            f"**{self.title}**",
            f"\n{self.description}",
        ]
        if self.tech_stack:
            lines.append(f"\nTech Stack: {', '.join(self.tech_stack)}")
        if self.components:
            lines.append(f"Components: {', '.join(self.components)}")
        if self.features:
            lines.append(f"\nFeatures ({len(self.features)}):")
            for f in self.features:
                lines.append(f"  - [{f.priority.value}] {f.name}: {f.description}")
        if self.constraints:
            lines.append(f"\nConstraints: {'; '.join(self.constraints)}")
        if self.deployment:
            lines.append(f"Deployment: {self.deployment}")
        return "\n".join(lines)


@dataclass
class MicroTask:
    """Atomic work unit in the execution plan."""

    id: str
    title: str
    description: str
    role: str  # role_name from RoleRegistry
    dependencies: list[str] = field(default_factory=list)  # list of MicroTask.id
    acceptance_criteria: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    error: str = ""
    retry_count: int = 0
    layer: int = 0  # execution layer (0 = no deps, 1 = deps on layer 0, etc.)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "role": self.role,
            "dependencies": self.dependencies,
            "acceptance_criteria": self.acceptance_criteria,
            "status": self.status.value,
            "layer": self.layer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MicroTask:
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            role=data["role"],
            dependencies=data.get("dependencies", []),
            acceptance_criteria=data.get("acceptance_criteria", []),
            status=TaskStatus(data.get("status", "pending")),
            layer=data.get("layer", 0),
        )


@dataclass
class ExecutionPlan:
    """The full work breakdown structure with dependency graph."""

    tasks: list[MicroTask] = field(default_factory=list)
    execution_order: list[list[str]] = field(default_factory=list)  # layers of task IDs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "execution_order": self.execution_order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        """Deserialize from dict."""
        tasks = [MicroTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            tasks=tasks,
            execution_order=data.get("execution_order", []),
        )

    def get_task(self, task_id: str) -> MicroTask | None:
        """Look up a task by ID."""
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def get_next_ready_tasks(self) -> list[MicroTask]:
        """Return tasks whose dependencies are all completed (PASSED)."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.PASSED}
        ready = []
        for t in self.tasks:
            if t.status != TaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in t.dependencies):
                ready.append(t)
        return ready

    def all_completed(self) -> bool:
        """Check if all tasks have passed."""
        return all(t.status == TaskStatus.PASSED for t in self.tasks)

    def progress_summary(self) -> str:
        """Return a compact progress string."""
        total = len(self.tasks)
        passed = sum(1 for t in self.tasks if t.status == TaskStatus.PASSED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        running = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)
        return f"{passed}/{total} done, {running} running, {failed} failed"


@dataclass
class QualityGateResult:
    """Outcome of a quality gate review."""

    passed: bool
    feedback: str = ""
    reviewer_output: str = ""


@dataclass
class ProjectResult:
    """Final result of the entire plan execution."""

    spec: ProjectSpec | None = None
    plan: ExecutionPlan | None = None
    success: bool = False
    summary: str = ""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tasks: int = 0
