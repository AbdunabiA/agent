"""FastAPI dependency injection — access shared state from request.app.state.

All components are initialized once in the application factory (create_app)
and stored on app.state. These dependency functions provide typed access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from agent.config import AgentConfig
    from agent.core.agent_loop import AgentLoop
    from agent.core.audit import AuditLog
    from agent.core.cost_tracker import CostTracker
    from agent.core.events import EventBus
    from agent.core.heartbeat import HeartbeatDaemon
    from agent.core.scheduler import TaskScheduler
    from agent.core.session import SessionStore
    from agent.memory.soul import SoulLoader
    from agent.memory.store import FactStore
    from agent.memory.vectors import VectorStore
    from agent.skills.manager import SkillManager
    from agent.tools.registry import ToolRegistry
    from agent.voice.pipeline import VoicePipeline
    from agent.workspaces.manager import WorkspaceManager


def get_agent_loop(request: Request) -> AgentLoop:
    """Get the agent loop from app state."""
    return request.app.state.agent_loop


def get_session_store(request: Request) -> SessionStore:
    """Get the session store from app state."""
    return request.app.state.session_store


def get_audit_log(request: Request) -> AuditLog:
    """Get the audit log from app state."""
    return request.app.state.audit


def get_event_bus(request: Request) -> EventBus:
    """Get the event bus from app state."""
    return request.app.state.event_bus


def get_tool_registry(request: Request) -> ToolRegistry:
    """Get the tool registry from app state."""
    return request.app.state.tool_registry


def get_heartbeat(request: Request) -> HeartbeatDaemon | None:
    """Get the heartbeat daemon from app state (may be None)."""
    return getattr(request.app.state, "heartbeat", None)


def get_config(request: Request) -> AgentConfig:
    """Get the agent config from app state."""
    return request.app.state.config


def get_fact_store(request: Request) -> FactStore | None:
    """Get the fact store from app state (may be None)."""
    return getattr(request.app.state, "fact_store", None)


def get_vector_store(request: Request) -> VectorStore | None:
    """Get the vector store from app state (may be None)."""
    return getattr(request.app.state, "vector_store", None)


def get_soul_loader(request: Request) -> SoulLoader | None:
    """Get the soul loader from app state (may be None)."""
    return getattr(request.app.state, "soul_loader", None)


def get_cost_tracker(request: Request) -> CostTracker | None:
    """Get the cost tracker from app state (may be None)."""
    return getattr(request.app.state, "cost_tracker", None)


def get_task_scheduler(request: Request) -> TaskScheduler | None:
    """Get the task scheduler from app state (may be None)."""
    return getattr(request.app.state, "task_scheduler", None)


def get_skill_manager(request: Request) -> SkillManager | None:
    """Get the skill manager from app state (may be None)."""
    return getattr(request.app.state, "skill_manager", None)


def get_workspace_manager(request: Request) -> WorkspaceManager | None:
    """Get the workspace manager from app state (may be None)."""
    return getattr(request.app.state, "workspace_manager", None)


def get_voice_pipeline(request: Request) -> VoicePipeline | None:
    """Get the voice pipeline from app state (may be None)."""
    return getattr(request.app.state, "voice_pipeline", None)
