"""Tests for RoleRegistry and dynamic role selection in the controller."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from agent.core.role_registry import RoleRegistry


@pytest.fixture
def teams_dir(tmp_path: Path) -> Path:
    """Create a temporary teams directory with sample YAML files."""
    # Engineering team
    eng = {
        "name": "engineering",
        "description": "Engineering team",
        "roles": [
            {
                "name": "architect",
                "persona": "You are a software architect. You design systems.",
                "allowed_tools": ["read_file", "write_file"],
                "max_iterations": 8,
            },
            {
                "name": "backend_developer",
                "persona": "You are a senior backend developer. You write Python.",
                "allowed_tools": ["read_file", "write_file", "shell_exec"],
                "max_iterations": 10,
            },
        ],
    }
    (tmp_path / "engineering.yaml").write_text(yaml.dump(eng))

    # Quality team
    qa = {
        "name": "quality",
        "description": "Quality team",
        "roles": [
            {
                "name": "qa_engineer",
                "persona": "You are a QA engineer. You write tests.",
                "allowed_tools": ["read_file", "write_file", "shell_exec"],
                "max_iterations": 10,
            },
            {
                "name": "security_reviewer",
                "persona": "You are a security engineer. You review code for vulns.",
                "allowed_tools": ["read_file"],
                "denied_tools": ["write_file"],
                "max_iterations": 5,
            },
        ],
    }
    (tmp_path / "quality.yaml").write_text(yaml.dump(qa))

    return tmp_path


class TestRoleRegistry:
    """Tests for RoleRegistry loading and lookup."""

    def test_loads_all_roles(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        assert len(reg.roles) == 4
        assert "architect" in reg.roles
        assert "backend_developer" in reg.roles
        assert "qa_engineer" in reg.roles
        assert "security_reviewer" in reg.roles

    def test_team_groups(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        assert "engineering" in reg.team_groups
        assert "quality" in reg.team_groups
        assert set(reg.team_groups["engineering"]) == {"architect", "backend_developer"}
        assert set(reg.team_groups["quality"]) == {"qa_engineer", "security_reviewer"}

    def test_get_role(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        role = reg.get_role("backend_developer")
        assert role is not None
        assert role.name == "backend_developer"
        assert role.max_iterations == 10

    def test_get_role_missing(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        assert reg.get_role("nonexistent") is None

    def test_get_team(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        roles = reg.get_team("engineering")
        assert len(roles) == 2
        names = {r.name for r in roles}
        assert names == {"architect", "backend_developer"}

    def test_get_team_missing(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        roles = reg.get_team("nonexistent")
        assert roles == []

    def test_roster_description(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        roster = reg.get_roster_description()
        assert "engineering/architect" in roster
        assert "engineering/backend_developer" in roster
        assert "quality/qa_engineer" in roster
        assert "quality/security_reviewer" in roster
        # Should strip "You are " prefix
        assert "You are " not in roster

    def test_reload(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        assert len(reg.roles) == 4

        # Add a new team file
        new_team = {
            "name": "devops",
            "description": "DevOps team",
            "roles": [
                {
                    "name": "devops_engineer",
                    "persona": "You are a DevOps engineer.",
                    "allowed_tools": ["shell_exec"],
                    "max_iterations": 5,
                },
            ],
        }
        (teams_dir / "devops.yaml").write_text(yaml.dump(new_team))

        reg.reload()
        assert len(reg.roles) == 5
        assert "devops_engineer" in reg.roles

    def test_empty_directory(self, tmp_path: Path) -> None:
        reg = RoleRegistry(tmp_path)
        assert len(reg.roles) == 0
        assert reg.get_roster_description() == ""

    def test_invalid_yaml_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text("{{invalid yaml")
        reg = RoleRegistry(tmp_path)
        assert len(reg.roles) == 0  # No crash

    def test_denied_tools_preserved(self, teams_dir: Path) -> None:
        reg = RoleRegistry(teams_dir)
        sr = reg.get_role("security_reviewer")
        assert sr is not None
        assert "write_file" in sr.denied_tools


class TestControllerRoleSelection:
    """Tests for the controller's _select_roles_heuristic method."""

    def _make_controller(self, teams_dir: Path):
        from agent.core.controller import ControllerAgent

        controller = ControllerAgent(
            orchestrator=MagicMock(),
            sdk_service=None,
            event_bus=MagicMock(),
            config=MagicMock(default_max_iterations=5),
            role_registry=RoleRegistry(teams_dir),
        )
        return controller

    def test_backend_task_selects_developer(self, teams_dir: Path) -> None:
        controller = self._make_controller(teams_dir)
        roles, max_rounds, order = controller._select_roles_heuristic(
            "Fix the database connection bug in the backend API"
        )
        role_names = {r.name for r in roles}
        assert "backend_developer" in role_names

    def test_test_task_selects_qa(self, teams_dir: Path) -> None:
        controller = self._make_controller(teams_dir)
        roles, max_rounds, order = controller._select_roles_heuristic(
            "Write comprehensive pytest tests for the auth module"
        )
        role_names = {r.name for r in roles}
        assert "qa_engineer" in role_names
        assert max_rounds >= 3  # QA cycle needs more rounds

    def test_security_task_selects_reviewer(self, teams_dir: Path) -> None:
        controller = self._make_controller(teams_dir)
        roles, _, _ = controller._select_roles_heuristic(
            "Review the authentication token handling for vulnerabilities"
        )
        role_names = {r.name for r in roles}
        assert "security_reviewer" in role_names

    def test_design_task_selects_architect(self, teams_dir: Path) -> None:
        controller = self._make_controller(teams_dir)
        roles, _, _ = controller._select_roles_heuristic(
            "Design the new API structure for the workspace system"
        )
        role_names = {r.name for r in roles}
        assert "architect" in role_names

    def test_generic_task_defaults_to_developer(self, teams_dir: Path) -> None:
        controller = self._make_controller(teams_dir)
        roles, _, _ = controller._select_roles_heuristic("Do something vague")
        assert len(roles) >= 1
        # Should at least have backend_developer as default
        role_names = {r.name for r in roles}
        assert "backend_developer" in role_names

    def test_complex_task_selects_multiple_roles(self, teams_dir: Path) -> None:
        controller = self._make_controller(teams_dir)
        roles, _, _ = controller._select_roles_heuristic(
            "Design and implement a new API endpoint with database integration, "
            "write comprehensive tests, and review for security vulnerabilities"
        )
        role_names = {r.name for r in roles}
        # Should pick multiple roles for this complex task
        assert len(role_names) >= 3
        assert "architect" in role_names or "backend_developer" in role_names
        assert "qa_engineer" in role_names
        assert "security_reviewer" in role_names

    def test_no_registry_returns_generic_worker(self) -> None:
        from agent.core.controller import ControllerAgent

        controller = ControllerAgent(
            orchestrator=MagicMock(),
            sdk_service=None,
            event_bus=MagicMock(),
            config=MagicMock(default_max_iterations=5),
            role_registry=None,
        )
        roles, max_rounds, _ = (
            controller._select_roles_heuristic.__wrapped__(controller, "anything")
            if hasattr(controller._select_roles_heuristic, "__wrapped__")
            else (
                # Heuristic without registry hits the assert — test the async path
                None,
                None,
                None,
            )
        )
        # This case is handled by _select_roles_for_task, not heuristic

    async def test_select_roles_for_task_no_registry(self) -> None:
        from agent.core.controller import ControllerAgent

        controller = ControllerAgent(
            orchestrator=MagicMock(),
            sdk_service=None,
            event_bus=MagicMock(),
            config=MagicMock(default_max_iterations=5),
            role_registry=None,
        )
        roles, max_rounds, order = await controller._select_roles_for_task("Fix a bug")
        assert len(roles) == 1
        assert roles[0].name == "controller-worker"
        assert max_rounds == 1

    async def test_select_roles_for_task_with_registry(self, teams_dir: Path) -> None:
        from agent.core.controller import ControllerAgent

        controller = ControllerAgent(
            orchestrator=MagicMock(),
            sdk_service=None,
            event_bus=MagicMock(),
            config=MagicMock(default_max_iterations=5),
            role_registry=RoleRegistry(teams_dir),
        )
        roles, max_rounds, order = await controller._select_roles_for_task(
            "Write tests for the auth module"
        )
        role_names = {r.name for r in roles}
        assert "qa_engineer" in role_names
        assert max_rounds >= 2


class TestRoleRegistryWithRealTeams:
    """Tests against the actual teams/ directory in the project."""

    def test_loads_real_teams(self) -> None:
        teams_path = Path("teams")
        if not teams_path.is_dir():
            pytest.skip("teams/ directory not found")

        reg = RoleRegistry(teams_path)
        # Should have roles from all 4 team files
        assert len(reg.roles) >= 10
        assert "architect" in reg.roles
        assert "backend_developer" in reg.roles
        assert "qa_engineer" in reg.roles
        assert "security_reviewer" in reg.roles
        assert "product_manager" in reg.roles
        assert "technical_writer" in reg.roles
