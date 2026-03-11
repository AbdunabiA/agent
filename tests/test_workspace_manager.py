"""Tests for WorkspaceManager."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent.config import AgentConfig, WorkspacesSection
from agent.workspaces.manager import (
    WorkspaceExistsError,
    WorkspaceManager,
    WorkspaceNotFoundError,
)


@pytest.fixture
def ws_config(tmp_path: Path) -> AgentConfig:
    """AgentConfig with workspaces directory in tmp_path."""
    return AgentConfig(
        workspaces=WorkspacesSection(directory=str(tmp_path / "workspaces")),
    )


@pytest.fixture
def manager(ws_config: AgentConfig) -> WorkspaceManager:
    return WorkspaceManager(ws_config)


class TestDiscover:
    def test_empty_dir(self, manager: WorkspaceManager) -> None:
        """discover returns empty list when directory doesn't exist."""
        assert manager.discover() == []

    def test_finds_workspaces(self, manager: WorkspaceManager) -> None:
        """discover finds all workspace directories."""
        base = manager.workspaces_dir
        base.mkdir(parents=True)
        (base / "alpha").mkdir()
        (base / "beta").mkdir()
        (base / ".hidden").mkdir()
        (base / "_internal").mkdir()
        assert manager.discover() == ["alpha", "beta"]


class TestResolve:
    def test_not_found(self, manager: WorkspaceManager) -> None:
        """resolve raises WorkspaceNotFoundError for missing workspace."""
        with pytest.raises(WorkspaceNotFoundError, match="nonexistent"):
            manager.resolve("nonexistent")

    def test_basic_resolve(self, manager: WorkspaceManager) -> None:
        """resolve loads workspace and creates data directories."""
        ws_dir = manager.workspaces_dir / "test"
        ws_dir.mkdir(parents=True)

        ws = manager.resolve("test")
        assert ws.name == "test"
        assert ws.display_name == "test"
        assert (ws.data_dir / "memory").exists()
        assert (ws.data_dir / "backups").exists()
        assert (ws.data_dir / "sessions").exists()
        assert (ws.data_dir / "notes").exists()

    def test_creates_soul_if_missing(self, manager: WorkspaceManager) -> None:
        """resolve creates default soul.md if missing."""
        ws_dir = manager.workspaces_dir / "test"
        ws_dir.mkdir(parents=True)

        ws = manager.resolve("test")
        assert ws.soul_path.exists()
        content = ws.soul_path.read_text()
        assert "test" in content

    def test_loads_config_yaml(self, manager: WorkspaceManager) -> None:
        """resolve loads config.yaml when present."""
        ws_dir = manager.workspaces_dir / "work"
        ws_dir.mkdir(parents=True)
        config_data = {
            "display_name": "Work Assistant",
            "description": "For coding",
            "default_model": "claude-opus-4-6",
            "max_iterations": 20,
        }
        with open(ws_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        ws = manager.resolve("work")
        assert ws.display_name == "Work Assistant"
        assert ws.config.default_model == "claude-opus-4-6"
        assert ws.config.max_iterations == 20

    def test_preserves_existing_soul(self, manager: WorkspaceManager) -> None:
        """resolve does not overwrite existing soul.md."""
        ws_dir = manager.workspaces_dir / "test"
        ws_dir.mkdir(parents=True)
        soul = ws_dir / "soul.md"
        soul.write_text("Custom soul content")

        ws = manager.resolve("test")
        assert ws.soul_path.read_text() == "Custom soul content"


class TestCreate:
    def test_basic_create(self, manager: WorkspaceManager) -> None:
        """create makes workspace directory with config.yaml + soul.md."""
        ws = manager.create("mywork", display_name="My Work", description="Testing")
        assert ws.name == "mywork"
        assert ws.display_name == "My Work"
        assert (ws.root_dir / "config.yaml").exists()
        assert ws.soul_path.exists()
        assert ws.data_dir.exists()

    def test_create_with_clone(self, manager: WorkspaceManager) -> None:
        """create with clone copies config and soul from source."""
        # Create source
        source = manager.create("source", display_name="Source")
        source.soul_path.write_text("Source soul")
        with open(source.root_dir / "config.yaml", "w") as f:
            yaml.dump({"display_name": "Source", "default_model": "gpt-4o"}, f)

        # Clone
        cloned = manager.create("cloned", clone_from="source")
        cloned_config = yaml.safe_load((cloned.root_dir / "config.yaml").read_text())
        assert cloned_config.get("default_model") == "gpt-4o"
        assert cloned.soul_path.read_text() == "Source soul"

    def test_create_existing_raises(self, manager: WorkspaceManager) -> None:
        """create raises WorkspaceExistsError for existing workspace."""
        manager.create("existing")
        with pytest.raises(WorkspaceExistsError, match="existing"):
            manager.create("existing")

    def test_invalid_name(self, manager: WorkspaceManager) -> None:
        """create rejects invalid names (spaces, special chars)."""
        with pytest.raises(ValueError, match="Invalid workspace name"):
            manager.create("has spaces")
        with pytest.raises(ValueError, match="Invalid workspace name"):
            manager.create("bad!name")

    def test_valid_names(self, manager: WorkspaceManager) -> None:
        """create accepts alphanumeric, hyphens, underscores."""
        ws1 = manager.create("my-workspace")
        assert ws1.name == "my-workspace"
        ws2 = manager.create("my_workspace")
        assert ws2.name == "my_workspace"
        ws3 = manager.create("workspace123")
        assert ws3.name == "workspace123"


class TestDelete:
    def test_delete_workspace(self, manager: WorkspaceManager) -> None:
        """delete removes workspace directory."""
        manager.create("to-delete")
        assert manager.delete("to-delete", confirm=True)
        assert "to-delete" not in manager.discover()

    def test_delete_default_raises(self, manager: WorkspaceManager) -> None:
        """delete refuses to delete 'default' workspace."""
        manager.create("default", display_name="Default")
        with pytest.raises(ValueError, match="Cannot delete"):
            manager.delete("default", confirm=True)

    def test_delete_requires_confirm(self, manager: WorkspaceManager) -> None:
        """delete requires confirm=True."""
        manager.create("to-delete")
        with pytest.raises(ValueError, match="confirm=True"):
            manager.delete("to-delete")

    def test_delete_nonexistent(self, manager: WorkspaceManager) -> None:
        """delete returns False for nonexistent workspace."""
        assert not manager.delete("nonexistent", confirm=True)


class TestSwitch:
    def test_switch_changes_active(self, manager: WorkspaceManager) -> None:
        """switch changes the active workspace."""
        manager.create("alpha")
        manager.create("beta")
        ws = manager.switch("beta")
        assert ws.name == "beta"
        assert manager.get_active().name == "beta"

    def test_switch_nonexistent_raises(self, manager: WorkspaceManager) -> None:
        """switch raises for nonexistent workspace."""
        with pytest.raises(WorkspaceNotFoundError):
            manager.switch("nonexistent")


class TestApplyOverrides:
    def test_overrides_model(self, manager: WorkspaceManager, ws_config: AgentConfig) -> None:
        """apply_overrides correctly overrides model."""
        manager.create("work")
        ws_dir = manager.workspaces_dir / "work"
        with open(ws_dir / "config.yaml", "w") as f:
            yaml.dump({"display_name": "work", "default_model": "gpt-4o-mini"}, f)

        ws = manager.resolve("work")
        new_cfg = manager.apply_overrides(ws_config, ws)
        assert new_cfg.models.default == "gpt-4o-mini"

    def test_preserves_unset_values(
        self, manager: WorkspaceManager, ws_config: AgentConfig,
    ) -> None:
        """apply_overrides preserves global values when workspace field is None."""
        manager.create("minimal")
        ws = manager.resolve("minimal")
        original_model = ws_config.models.default

        new_cfg = manager.apply_overrides(ws_config, ws)
        assert new_cfg.models.default == original_model

    def test_overrides_memory_paths(
        self, manager: WorkspaceManager, ws_config: AgentConfig,
    ) -> None:
        """apply_overrides sets workspace-specific memory paths."""
        manager.create("work")
        ws = manager.resolve("work")
        new_cfg = manager.apply_overrides(ws_config, ws)
        assert "work" in new_cfg.memory.db_path
        assert "work" in new_cfg.memory.markdown_dir
        assert "work" in (new_cfg.memory.soul_path or "")

    def test_does_not_mutate_original(
        self, manager: WorkspaceManager, ws_config: AgentConfig,
    ) -> None:
        """apply_overrides does not mutate the base config."""
        manager.create("work")
        ws_dir = manager.workspaces_dir / "work"
        with open(ws_dir / "config.yaml", "w") as f:
            yaml.dump({"display_name": "work", "default_model": "gpt-4o-mini"}, f)

        ws = manager.resolve("work")
        original_model = ws_config.models.default
        manager.apply_overrides(ws_config, ws)
        assert ws_config.models.default == original_model


class TestEnsureDefault:
    def test_creates_default_workspace(self, manager: WorkspaceManager) -> None:
        """ensure_default creates the 'default' workspace."""
        manager.ensure_default()
        assert "default" in manager.discover()

    def test_idempotent(self, manager: WorkspaceManager) -> None:
        """ensure_default is safe to call multiple times."""
        manager.ensure_default()
        manager.ensure_default()
        assert manager.discover().count("default") == 1

    def test_respects_auto_create_flag(self, tmp_path: Path) -> None:
        """ensure_default does nothing when auto_create_default is False."""
        cfg = AgentConfig(
            workspaces=WorkspacesSection(
                directory=str(tmp_path / "workspaces"),
                auto_create_default=False,
            ),
        )
        manager = WorkspaceManager(cfg)
        manager.ensure_default()
        assert manager.discover() == []
