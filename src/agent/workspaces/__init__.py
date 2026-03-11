"""Multi-agent workspace system.

Provides isolated workspaces with per-workspace config, memory, and soul.
Phase 7B adds routing, delegation, and shared memory.
"""

from agent.workspaces.config import (
    ResolvedWorkspace,
    RoutingConfig,
    RoutingRuleConfig,
    WorkspaceConfig,
    WorkspacesSection,
)
from agent.workspaces.delegation import WorkspaceDelegator, register_delegation_tools
from agent.workspaces.isolation import WorkspaceIsolation
from agent.workspaces.manager import (
    WorkspaceExistsError,
    WorkspaceManager,
    WorkspaceNotFoundError,
)
from agent.workspaces.router import RoutingRule, WorkspaceRouter
from agent.workspaces.shared_memory import SharedMemoryLayer, register_shared_memory_tools

__all__ = [
    "ResolvedWorkspace",
    "RoutingConfig",
    "RoutingRule",
    "RoutingRuleConfig",
    "SharedMemoryLayer",
    "WorkspaceConfig",
    "WorkspaceDelegator",
    "WorkspaceExistsError",
    "WorkspaceIsolation",
    "WorkspaceManager",
    "WorkspaceNotFoundError",
    "WorkspaceRouter",
    "WorkspacesSection",
    "register_delegation_tools",
    "register_shared_memory_tools",
]
