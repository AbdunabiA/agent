"""Role registry — flat index of all roles from all teams for dynamic selection.

The controller uses this to pick only the roles actually needed for a task,
instead of loading a full fixed team every time.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from agent.core.subagent import SubAgentRole

logger = structlog.get_logger(__name__)


class RoleRegistry:
    """Flat index of all agent roles loaded from teams/*.yaml.

    Builds two indexes:
    - ``roles``: role_name → SubAgentRole (flat lookup)
    - ``team_groups``: team_name → [role_names] (group lookup)

    Args:
        teams_dir: Path to the teams/ directory containing YAML files.
    """

    def __init__(self, teams_dir: str | Path) -> None:
        self._teams_dir = Path(teams_dir)
        self.roles: dict[str, SubAgentRole] = {}
        self.team_groups: dict[str, list[str]] = {}
        self._role_team: dict[str, str] = {}  # role_name → team_name
        self._load_all()

    def _load_all(self) -> None:
        """Load all YAML team files and build indexes."""
        from agent.teams.loader import config_to_team, discover_team_files, parse_team_file

        self.roles.clear()
        self.team_groups.clear()
        self._role_team.clear()

        files = discover_team_files(self._teams_dir)
        for path in files:
            try:
                configs = parse_team_file(path)
                for cfg in configs:
                    team = config_to_team(cfg)
                    role_names: list[str] = []
                    for role in team.roles:
                        self.roles[role.name] = role
                        self._role_team[role.name] = team.name
                        role_names.append(role.name)
                    self.team_groups[team.name] = role_names
            except Exception as e:
                logger.warning(
                    "role_registry_load_failed",
                    path=str(path),
                    error=str(e),
                )

        logger.info(
            "role_registry_loaded",
            roles=len(self.roles),
            teams=len(self.team_groups),
        )

    def get_role(self, name: str) -> SubAgentRole | None:
        """Look up a role by name.

        Args:
            name: Role name (e.g. 'backend_developer').

        Returns:
            The SubAgentRole, or None if not found.
        """
        return self.roles.get(name)

    def get_team(self, team_name: str) -> list[SubAgentRole]:
        """Get all roles for a named team.

        Args:
            team_name: Team name (e.g. 'engineering').

        Returns:
            List of SubAgentRole instances, empty if team not found.
        """
        role_names = self.team_groups.get(team_name, [])
        return [self.roles[n] for n in role_names if n in self.roles]

    def get_roster_description(self) -> str:
        """Build a compact text roster for the controller's LLM prompt.

        Returns:
            Formatted string listing all available roles with their
            team and a one-line summary of their persona.
        """
        lines: list[str] = []
        for team_name, role_names in sorted(self.team_groups.items()):
            for rn in role_names:
                role = self.roles.get(rn)
                if not role:
                    continue
                # Extract first sentence of persona for compact description
                summary = role.persona.strip().split("\n")[0].strip()
                if summary.startswith("You are "):
                    summary = summary[8:]  # strip "You are "
                # Truncate long summaries
                if len(summary) > 100:
                    summary = summary[:97] + "..."
                lines.append(f"  {team_name}/{rn} — {summary}")
        return "\n".join(lines)

    def reload(self) -> None:
        """Hot-reload: clear indexes and re-scan the teams directory.

        Safe to call at any time — the registry is rebuilt atomically.
        """
        logger.info("role_registry_reloading")
        self._load_all()
