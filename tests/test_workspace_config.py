"""Tests for workspace configuration models."""

from __future__ import annotations

from pathlib import Path

from agent.config import AgentConfig, WorkspacesSection
from agent.workspaces.config import ResolvedWorkspace, WorkspaceConfig


class TestWorkspaceConfig:
    """Tests for WorkspaceConfig model."""

    def test_minimal_config(self) -> None:
        """WorkspaceConfig requires only a name."""
        wc = WorkspaceConfig(name="test")
        assert wc.name == "test"
        assert wc.display_name == ""
        assert wc.description == ""

    def test_all_fields(self) -> None:
        """WorkspaceConfig loads all fields correctly."""
        wc = WorkspaceConfig(
            name="work",
            display_name="Work Assistant",
            description="For coding",
            default_model="claude-opus-4-6",
            fallback_model="gpt-4o",
            persona="You are a coder.",
            heartbeat_interval="1h",
            max_iterations=15,
            enabled_skills=["github-monitor"],
            disabled_skills=["weather"],
            enabled_tools=["shell", "filesystem"],
            disabled_tools=["browser_navigate"],
            telegram_enabled=True,
            telegram_users=[12345],
            webchat_enabled=False,
        )
        assert wc.default_model == "claude-opus-4-6"
        assert wc.max_iterations == 15
        assert wc.enabled_skills == ["github-monitor"]
        assert wc.telegram_users == [12345]

    def test_none_means_inherit(self) -> None:
        """Fields not specified default to None (inherit from global)."""
        wc = WorkspaceConfig(name="test")
        assert wc.default_model is None
        assert wc.fallback_model is None
        assert wc.persona is None
        assert wc.heartbeat_interval is None
        assert wc.max_iterations is None
        assert wc.enabled_skills is None
        assert wc.disabled_skills is None
        assert wc.enabled_tools is None
        assert wc.disabled_tools is None
        assert wc.telegram_enabled is None
        assert wc.telegram_users is None
        assert wc.webchat_enabled is None


class TestWorkspacesSection:
    """Tests for WorkspacesSection model."""

    def test_defaults(self) -> None:
        """WorkspacesSection has sensible defaults."""
        ws = WorkspacesSection()
        assert ws.directory == "workspaces"
        assert ws.default == "default"
        assert ws.auto_create_default is True

    def test_custom_values(self) -> None:
        ws = WorkspacesSection(directory="my-workspaces", default="work", auto_create_default=False)
        assert ws.directory == "my-workspaces"
        assert ws.default == "work"
        assert ws.auto_create_default is False

    def test_in_agent_config(self) -> None:
        """WorkspacesSection is accessible in AgentConfig."""
        cfg = AgentConfig()
        assert cfg.workspaces.directory == "workspaces"
        assert cfg.workspaces.default == "default"


class TestResolvedWorkspace:
    """Tests for ResolvedWorkspace dataclass."""

    def test_db_path(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        ws = ResolvedWorkspace(
            name="test",
            display_name="Test",
            description="",
            root_dir=tmp_path,
            data_dir=data_dir,
            soul_path=tmp_path / "soul.md",
            heartbeat_path=tmp_path / "HEARTBEAT.md",
            config=WorkspaceConfig(name="test"),
        )
        assert ws.get_db_path() == str(data_dir / "agent.db")

    def test_chromadb_path(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        ws = ResolvedWorkspace(
            name="test",
            display_name="Test",
            description="",
            root_dir=tmp_path,
            data_dir=data_dir,
            soul_path=tmp_path / "soul.md",
            heartbeat_path=tmp_path / "HEARTBEAT.md",
            config=WorkspaceConfig(name="test"),
        )
        assert ws.get_chromadb_path() == str(data_dir / "memory")
