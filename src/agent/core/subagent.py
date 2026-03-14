"""Sub-agent data models for the orchestration system.

Defines roles, tasks, results, and teams for the multi-agent pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


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
    task_id: str = field(default_factory=lambda: str(uuid4())[:8])


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
