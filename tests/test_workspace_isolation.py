"""Tests for WorkspaceIsolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.workspaces.config import ResolvedWorkspace, WorkspaceConfig
from agent.workspaces.isolation import WorkspaceIsolation


@pytest.fixture
def workspace(tmp_path: Path) -> ResolvedWorkspace:
    """Create a test ResolvedWorkspace."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "memory").mkdir()
    (data_dir / "backups").mkdir()
    return ResolvedWorkspace(
        name="test",
        display_name="Test",
        description="Test workspace",
        root_dir=tmp_path,
        data_dir=data_dir,
        soul_path=tmp_path / "soul.md",
        heartbeat_path=tmp_path / "HEARTBEAT.md",
        config=WorkspaceConfig(name="test"),
    )


@pytest.fixture
def isolation(workspace: ResolvedWorkspace) -> WorkspaceIsolation:
    return WorkspaceIsolation(workspace)


class TestDatabaseConfig:
    def test_returns_workspace_db_path(self, isolation: WorkspaceIsolation) -> None:
        cfg = isolation.get_database_config()
        assert "db_path" in cfg
        assert cfg["db_path"].endswith("agent.db")
        assert "data" in cfg["db_path"]


class TestVectorConfig:
    def test_returns_workspace_vector_path(self, isolation: WorkspaceIsolation) -> None:
        cfg = isolation.get_vector_config()
        assert "persist_directory" in cfg
        assert "memory" in cfg["persist_directory"]


class TestSoulPath:
    def test_returns_workspace_soul(self, isolation: WorkspaceIsolation) -> None:
        path = isolation.get_soul_path()
        assert path.endswith("soul.md")


class TestHeartbeatPath:
    def test_returns_workspace_heartbeat(self, isolation: WorkspaceIsolation) -> None:
        path = isolation.get_heartbeat_path()
        assert path.endswith("HEARTBEAT.md")


class TestBackupDir:
    def test_returns_workspace_backups(self, isolation: WorkspaceIsolation) -> None:
        path = isolation.get_backup_dir()
        assert "backups" in path


class TestFilterTools:
    def test_no_overrides_returns_all(self, isolation: WorkspaceIsolation) -> None:
        """No tool overrides returns all tools."""
        tools = ["shell", "filesystem", "browser", "http"]
        assert isolation.filter_tools(tools) == tools

    def test_enabled_tools_filters(self, tmp_path: Path) -> None:
        """enabled_tools returns only those tools."""
        ws = _make_workspace(tmp_path, enabled_tools=["shell", "filesystem"])
        iso = WorkspaceIsolation(ws)
        result = iso.filter_tools(["shell", "filesystem", "browser", "http"])
        assert result == ["shell", "filesystem"]

    def test_disabled_tools_removes(self, tmp_path: Path) -> None:
        """disabled_tools removes specified tools."""
        ws = _make_workspace(tmp_path, disabled_tools=["browser"])
        iso = WorkspaceIsolation(ws)
        result = iso.filter_tools(["shell", "filesystem", "browser", "http"])
        assert result == ["shell", "filesystem", "http"]

    def test_enabled_takes_priority(self, tmp_path: Path) -> None:
        """enabled_tools is checked before disabled_tools."""
        ws = _make_workspace(
            tmp_path,
            enabled_tools=["shell"],
            disabled_tools=["filesystem"],
        )
        iso = WorkspaceIsolation(ws)
        result = iso.filter_tools(["shell", "filesystem", "browser"])
        assert result == ["shell"]


class TestFilterSkills:
    def test_no_overrides_returns_all(self, isolation: WorkspaceIsolation) -> None:
        skills = ["weather", "reminder", "notes"]
        assert isolation.filter_skills(skills) == skills

    def test_enabled_skills_filters(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path, enabled_skills=["weather"])
        iso = WorkspaceIsolation(ws)
        result = iso.filter_skills(["weather", "reminder", "notes"])
        assert result == ["weather"]

    def test_disabled_skills_removes(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path, disabled_skills=["notes"])
        iso = WorkspaceIsolation(ws)
        result = iso.filter_skills(["weather", "reminder", "notes"])
        assert result == ["weather", "reminder"]


def _make_workspace(
    tmp_path: Path,
    *,
    enabled_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
    enabled_skills: list[str] | None = None,
    disabled_skills: list[str] | None = None,
) -> ResolvedWorkspace:
    """Helper to create a workspace with specific config overrides."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return ResolvedWorkspace(
        name="test",
        display_name="Test",
        description="",
        root_dir=tmp_path,
        data_dir=data_dir,
        soul_path=tmp_path / "soul.md",
        heartbeat_path=tmp_path / "HEARTBEAT.md",
        config=WorkspaceConfig(
            name="test",
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enabled_skills=enabled_skills,
            disabled_skills=disabled_skills,
        ),
    )
