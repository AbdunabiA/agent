"""Workspace configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class WorkspaceConfig(BaseModel):
    """Configuration for a single workspace.

    Loaded from workspaces/<name>/config.yaml.
    Fields here OVERRIDE the corresponding fields in the global AgentConfig.
    Fields not specified inherit from global config.
    """

    name: str
    display_name: str = ""
    description: str = ""

    # Override sections (all optional — None means inherit from global)
    default_model: str | None = None
    fallback_model: str | None = None
    persona: str | None = None
    heartbeat_interval: str | None = None
    max_iterations: int | None = None

    # Workspace-specific settings
    enabled_skills: list[str] | None = None
    disabled_skills: list[str] | None = None
    enabled_tools: list[str] | None = None
    disabled_tools: list[str] | None = None

    # Channel bindings
    telegram_enabled: bool | None = None
    telegram_users: list[int] | None = None
    webchat_enabled: bool | None = None


class RoutingRuleConfig(BaseModel):
    """A single routing rule from agent.yaml."""

    channel: str = "*"
    workspace: str = "default"
    user_id: str | None = None
    pattern: str | None = None


class RoutingConfig(BaseModel):
    """Routing section inside workspaces config."""

    default: str = "default"
    rules: list[RoutingRuleConfig] = []


class WorkspacesSection(BaseModel):
    """Top-level workspaces config in agent.yaml."""

    directory: str = "workspaces"
    default: str = "default"
    auto_create_default: bool = True
    routing: RoutingConfig = RoutingConfig()


@dataclass
class ResolvedWorkspace:
    """A fully resolved workspace with all paths determined.

    Created by merging: global config <- workspace config.yaml overrides.
    """

    name: str
    display_name: str
    description: str
    root_dir: Path
    data_dir: Path
    soul_path: Path
    heartbeat_path: Path
    config: WorkspaceConfig

    def get_db_path(self) -> str:
        """Get the SQLite database path for this workspace."""
        return str(self.data_dir / "agent.db")

    def get_chromadb_path(self) -> str:
        """Get the ChromaDB directory for this workspace."""
        return str(self.data_dir / "memory")
