"""Workspace resource isolation.

Each workspace gets its own:
- SQLite database (data/agent.db)
- ChromaDB collection (data/memory/)
- File backups (data/backups/)
- Session data (data/sessions/)
- Notes (data/notes/)
- soul.md
- HEARTBEAT.md

Shared across all workspaces:
- LLM provider (API keys are global)
- Tool registry (tools are global, but can be filtered per workspace)
- Skills (global, but can be enabled/disabled per workspace)
- Gateway (single port, routes to active workspace)
"""

from __future__ import annotations

from typing import Any

import structlog

from agent.workspaces.config import ResolvedWorkspace

logger = structlog.get_logger(__name__)


class WorkspaceIsolation:
    """Applies workspace isolation to agent components."""

    def __init__(self, workspace: ResolvedWorkspace) -> None:
        self.workspace = workspace

    def get_database_config(self) -> dict[str, Any]:
        """Get database configuration for this workspace."""
        return {
            "db_path": self.workspace.get_db_path(),
        }

    def get_vector_config(self) -> dict[str, Any]:
        """Get ChromaDB configuration for this workspace."""
        return {
            "persist_directory": self.workspace.get_chromadb_path(),
        }

    def get_soul_path(self) -> str:
        """Get the soul.md path for this workspace."""
        return str(self.workspace.soul_path)

    def get_heartbeat_path(self) -> str:
        """Get HEARTBEAT.md path for this workspace."""
        return str(self.workspace.heartbeat_path)

    def get_backup_dir(self) -> str:
        """Get backup directory for this workspace."""
        return str(self.workspace.data_dir / "backups")

    def filter_tools(self, all_tools: list[str]) -> list[str]:
        """Filter tool list based on workspace config.

        If enabled_tools is set: only those tools are available.
        If disabled_tools is set: those tools are removed.
        """
        ws = self.workspace.config

        if ws.enabled_tools is not None:
            return [t for t in all_tools if t in ws.enabled_tools]

        if ws.disabled_tools is not None:
            return [t for t in all_tools if t not in ws.disabled_tools]

        return all_tools

    def filter_skills(self, all_skills: list[str]) -> list[str]:
        """Filter skills based on workspace config."""
        ws = self.workspace.config

        if ws.enabled_skills is not None:
            return [s for s in all_skills if s in ws.enabled_skills]

        if ws.disabled_skills is not None:
            return [s for s in all_skills if s not in ws.disabled_skills]

        return all_skills
