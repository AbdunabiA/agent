"""Sub-agent data models for the orchestration system.

Defines roles, tasks, results, and teams for the multi-agent pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


@dataclass
class ControllerWorkOrder:
    """A work order from the main agent to the controller."""

    order_id: str = field(default_factory=lambda: f"wo-{uuid4().hex[:8]}")
    instruction: str = ""
    context: str = ""
    priority: int = 0
    user_id: str = ""


@dataclass
class ControllerDirective:
    """A control command from main agent to controller."""

    order_id: str = ""
    command: str = ""  # "stop", "pause", "redirect"
    details: str = ""


@dataclass
class ControllerTaskState:
    """Tracks the state of a controller-managed work order."""

    order_id: str = ""
    status: str = "pending"  # pending, planning, executing, completed, failed, cancelled
    user_id: str = ""
    worker_task_ids: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""


@dataclass
class FeedbackConfig:
    """Configuration for iterative feedback loops in project pipelines.

    When a review/verify stage fails, the pipeline loops back to the
    fix_stage, re-runs it with feedback, then re-runs the review stage.
    """

    fix_stage: str
    max_retries: int = 3


class DelegationMode(StrEnum):
    """Mode for inter-agent delegation."""

    SYNC = "sync"
    ASYNC = "async"


class SubAgentStatus(StrEnum):
    """Lifecycle status of a sub-agent task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubAgentRole:
    """Definition of a sub-agent's persona and constraints."""

    name: str
    persona: str = "You are a helpful assistant."
    model: str | None = None  # Override parent model
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    max_iterations: int = 5
    max_tokens_budget: int | None = None


@dataclass
class SubAgentTask:
    """A task to be executed by a sub-agent."""

    role: SubAgentRole
    instruction: str
    context: str = ""
    parent_session_id: str = ""
    priority: int = 0
    task_id: str = field(default_factory=lambda: str(uuid4())[:12])
    nesting_depth: int = 0
    status: SubAgentStatus = SubAgentStatus.PENDING


@dataclass
class SubAgentResult:
    """Result from a sub-agent's execution."""

    task_id: str
    role_name: str
    status: SubAgentStatus
    output: str = ""
    error: str | None = None
    token_usage: int = 0
    duration_ms: int = 0
    tool_calls_made: int = 0
    iterations: int = 0


@dataclass
class AgentTeam:
    """Pre-defined team of sub-agent roles for common workflows."""

    name: str
    description: str
    roles: list[SubAgentRole] = field(default_factory=list)


@dataclass
class ConsultRequest:
    """A request from one agent to consult another."""

    requesting_agent_id: str
    requesting_role: str
    target_team: str
    target_role: str
    question: str
    context: str = ""
    request_id: str = field(default_factory=lambda: str(uuid4())[:8])


@dataclass
class ConsultResponse:
    """Response from a consulted agent."""

    request_id: str
    target_role: str
    status: SubAgentStatus
    answer: str = ""
    error: str | None = None
    token_usage: int = 0
    duration_ms: int = 0


@dataclass
class DelegationRequest:
    """A delegation request from one agent to a specialist."""

    delegating_agent_id: str
    delegating_role: str
    target_team: str
    target_role: str
    instruction: str
    context: str = ""
    mode: DelegationMode = DelegationMode.SYNC
    request_id: str = field(default_factory=lambda: str(uuid4())[:8])


@dataclass
class DelegationResult:
    """Result from a delegated task."""

    request_id: str
    target_role: str
    status: SubAgentStatus
    output: str = ""
    error: str | None = None
    task_id: str = ""
    token_usage: int = 0
    duration_ms: int = 0


@dataclass
class ProjectAgentRef:
    """Reference to an agent within a project stage.

    Points to a team + role by name rather than embedding the full role
    definition, so projects compose from existing teams.
    """

    team: str
    role: str


@dataclass
class DiscussionConfig:
    """Configuration for discussion-mode stages."""

    rounds: int = 3
    moderator: ProjectAgentRef | None = None
    consensus_required: bool = False


@dataclass
class DiscussionRound:
    """Record of a single discussion round."""

    round_number: int
    contributions: list[SubAgentResult] = field(default_factory=list)
    moderator_summary: str = ""
    consensus_reached: bool = False


@dataclass
class ProjectStage:
    """A single stage in a project pipeline.

    All agents within a stage run in parallel. The stage's combined output
    is passed as context to the next stage.
    """

    name: str
    agents: list[ProjectAgentRef] = field(default_factory=list)
    parallel: bool = True  # reserved for future sequential-within-stage
    feedback: FeedbackConfig | None = None
    feedback_target: bool = False  # skipped in normal flow, only triggered by loops
    mode: str = "standard"  # "standard" or "discussion"
    discussion: DiscussionConfig | None = None


@dataclass
class Project:
    """A cross-team workflow with sequential stages.

    Each stage runs its agents in parallel, collects outputs, and feeds
    them as context into the next stage — like a dev pipeline.
    """

    name: str
    description: str
    stages: list[ProjectStage] = field(default_factory=list)


@dataclass
class ProjectStageResult:
    """Result from a single project stage."""

    stage_name: str
    results: list[SubAgentResult] = field(default_factory=list)
    combined_output: str = ""
    duration_ms: int = 0
    feedback_iteration: int = 0  # 0 = first run, 1+ = retry


@dataclass
class ProjectResult:
    """Result from a complete project run."""

    project_name: str
    status: SubAgentStatus
    stages: list[ProjectStageResult] = field(default_factory=list)
    final_output: str = ""
    duration_ms: int = 0
    error: str | None = None
    feedback_iterations: int = 0  # total iterations across all stages
