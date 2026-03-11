"""Workspace lifecycle management."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import structlog
import yaml

from agent.config import AgentConfig
from agent.workspaces.config import ResolvedWorkspace, WorkspaceConfig

logger = structlog.get_logger(__name__)


class WorkspaceNotFoundError(Exception):
    """Raised when a workspace directory does not exist."""


class WorkspaceExistsError(Exception):
    """Raised when trying to create a workspace that already exists."""


class WorkspaceManager:
    """Discovers, creates, switches, and resolves workspaces."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspaces_dir = Path(config.workspaces.directory)
        self._active: ResolvedWorkspace | None = None

    def discover(self) -> list[str]:
        """List all workspace names found in the workspaces directory."""
        if not self.workspaces_dir.exists():
            return []
        return sorted([
            d.name for d in self.workspaces_dir.iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        ])

    def resolve(self, name: str) -> ResolvedWorkspace:
        """Load and resolve a workspace by name.

        Steps:
        1. Check workspace directory exists
        2. Load config.yaml if present (otherwise empty overrides)
        3. Ensure data/ subdirectories exist
        4. Ensure soul.md exists (create default if not)
        5. Return ResolvedWorkspace with all paths resolved
        """
        ws_dir = self.workspaces_dir / name

        if not ws_dir.exists():
            raise WorkspaceNotFoundError(
                f"Workspace '{name}' not found at {ws_dir}"
            )

        # Load workspace config
        config_file = ws_dir / "config.yaml"
        ws_config = WorkspaceConfig(name=name)
        if config_file.exists():
            with open(config_file) as f:
                raw = yaml.safe_load(f) or {}
            ws_config = WorkspaceConfig(name=name, **raw)

        # Ensure directories
        data_dir = ws_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "memory").mkdir(exist_ok=True)
        (data_dir / "backups").mkdir(exist_ok=True)
        (data_dir / "sessions").mkdir(exist_ok=True)
        (data_dir / "notes").mkdir(exist_ok=True)

        # Ensure soul.md
        soul_path = ws_dir / "soul.md"
        if not soul_path.exists():
            soul_path.write_text(
                f"# {ws_config.display_name or name}\n\n"
                f"You are Agent running in the '{name}' workspace.\n"
            )

        heartbeat_path = ws_dir / "HEARTBEAT.md"

        return ResolvedWorkspace(
            name=name,
            display_name=ws_config.display_name or name,
            description=ws_config.description,
            root_dir=ws_dir,
            data_dir=data_dir,
            soul_path=soul_path,
            heartbeat_path=heartbeat_path,
            config=ws_config,
        )

    def create(
        self,
        name: str,
        display_name: str = "",
        description: str = "",
        clone_from: str | None = None,
    ) -> ResolvedWorkspace:
        """Create a new workspace.

        Args:
            name: Workspace identifier (directory name). Lowercase, no spaces.
            display_name: Human-readable name.
            description: What this workspace is for.
            clone_from: If set, copy config/soul from existing workspace.
        """
        ws_dir = self.workspaces_dir / name

        if ws_dir.exists():
            raise WorkspaceExistsError(f"Workspace '{name}' already exists")

        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Invalid workspace name: '{name}'. "
                "Use alphanumeric, hyphens, underscores."
            )

        ws_dir.mkdir(parents=True)

        if clone_from:
            source = self.workspaces_dir / clone_from
            if source.exists():
                for filename in ["config.yaml", "soul.md", "HEARTBEAT.md"]:
                    src = source / filename
                    if src.exists():
                        shutil.copy2(src, ws_dir / filename)

        # Write config.yaml only if not already present (e.g. from clone)
        if not (ws_dir / "config.yaml").exists():
            config_data: dict[str, str] = {
                "display_name": display_name or name,
                "description": description,
            }
            with open(ws_dir / "config.yaml", "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)

        logger.info("workspace_created", workspace=name, clone_from=clone_from)
        return self.resolve(name)

    def delete(self, name: str, *, confirm: bool = False) -> bool:
        """Delete a workspace and all its data.

        Requires confirm=True as safety check (data loss is irreversible).
        Cannot delete the 'default' workspace.
        """
        if name == "default":
            raise ValueError("Cannot delete the 'default' workspace")
        if not confirm:
            raise ValueError(
                "Pass confirm=True to delete a workspace (data loss is irreversible)"
            )

        ws_dir = self.workspaces_dir / name
        if ws_dir.exists():
            shutil.rmtree(ws_dir)
            logger.info("workspace_deleted", workspace=name)
            return True
        return False

    def get_active(self) -> ResolvedWorkspace:
        """Get the currently active workspace."""
        if self._active is None:
            self._active = self.resolve(self.config.workspaces.default)
        return self._active

    def switch(self, name: str) -> ResolvedWorkspace:
        """Switch to a different workspace.

        NOTE: In Phase 7A, switching requires restart.
        Phase 7B will support hot-switching via channel routing.
        """
        self._active = self.resolve(name)
        logger.info("workspace_switched", workspace=name)
        return self._active

    def apply_overrides(
        self, base_config: AgentConfig, workspace: ResolvedWorkspace,
    ) -> AgentConfig:
        """Apply workspace config overrides to the global config.

        Creates a modified copy of AgentConfig with workspace-specific values.
        Only overrides fields that are explicitly set in the workspace config.
        """
        config = copy.deepcopy(base_config)
        ws = workspace.config

        # Model overrides
        if ws.default_model is not None:
            config.models.default = ws.default_model
        if ws.fallback_model is not None:
            config.models.fallback = ws.fallback_model

        # Agent overrides
        if ws.persona is not None:
            config.agent.persona = ws.persona
        if ws.heartbeat_interval is not None:
            config.agent.heartbeat_interval = ws.heartbeat_interval
        if ws.max_iterations is not None:
            config.agent.max_iterations = ws.max_iterations

        # Skills overrides
        if ws.enabled_skills is not None:
            config.skills.enabled = ws.enabled_skills
        if ws.disabled_skills is not None:
            config.skills.disabled = ws.disabled_skills

        # Channel overrides
        if ws.telegram_enabled is not None:
            config.channels.telegram.enabled = ws.telegram_enabled
        if ws.telegram_users is not None:
            config.channels.telegram.allowed_users = ws.telegram_users
        if ws.webchat_enabled is not None:
            config.channels.webchat.enabled = ws.webchat_enabled

        # Override memory paths to workspace-specific
        config.memory.db_path = workspace.get_db_path()
        config.memory.markdown_dir = str(workspace.data_dir / "memory") + "/"
        config.memory.soul_path = str(workspace.soul_path)

        return config

    def ensure_default(self) -> None:
        """Create the default workspace if it doesn't exist."""
        default_dir = self.workspaces_dir / "default"
        if not default_dir.exists() and self.config.workspaces.auto_create_default:
            self.create(
                "default",
                display_name="Default",
                description="Default workspace",
            )
            logger.info("default_workspace_created")
