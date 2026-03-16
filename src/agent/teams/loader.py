"""Team loader — discovers and parses team YAML files from the teams/ directory.

Scans a directory for `*.yaml` / `*.yml` files, validates each against the
``AgentTeamConfig`` Pydantic model, and converts them into ``AgentTeam``
dataclass instances ready for the orchestrator.

Teams loaded from files are merged with any inline teams defined in
``agent.yaml`` — file-based teams take precedence on name collision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml

from agent.config import AgentTeamConfig, AgentTeamRoleConfig
from agent.core.subagent import (
    AgentTeam,
    DiscussionConfig,
    FeedbackConfig,
    Project,
    ProjectAgentRef,
    ProjectStage,
    SubAgentRole,
)

logger = structlog.get_logger(__name__)


class TeamLoadError(Exception):
    """Raised when a team YAML file cannot be loaded or validated."""


def discover_team_files(teams_dir: str | Path) -> list[Path]:
    """Find all YAML files in the teams directory.

    Args:
        teams_dir: Path to the teams directory.

    Returns:
        Sorted list of YAML file paths.
    """
    teams_path = Path(teams_dir)
    if not teams_path.is_dir():
        logger.debug("teams_dir_not_found", path=str(teams_path))
        return []

    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml"):
        files.extend(teams_path.glob(pattern))

    # Sort by filename for deterministic ordering
    files.sort(key=lambda p: p.stem)
    return files


def parse_team_file(path: Path) -> list[AgentTeamConfig]:
    """Parse a team YAML file into config models.

    A file can contain a single team definition (dict) or multiple
    teams (list of dicts).

    Args:
        path: Path to the YAML file.

    Returns:
        List of AgentTeamConfig models.

    Raises:
        TeamLoadError: If the file cannot be parsed or validated.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise TeamLoadError(f"Cannot read {path}: {e}") from e

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise TeamLoadError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        return []

    # Normalize to list
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise TeamLoadError(
            f"Expected dict or list in {path}, got {type(data).__name__}"
        )

    configs: list[AgentTeamConfig] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise TeamLoadError(
                f"Item {i} in {path} is not a dict: {type(item).__name__}"
            )
        try:
            config = _parse_team_dict(item, source=str(path))
            configs.append(config)
        except Exception as e:
            raise TeamLoadError(f"Invalid team definition in {path}: {e}") from e

    return configs


def _parse_team_dict(data: dict[str, Any], source: str = "") -> AgentTeamConfig:
    """Parse a single team dict into an AgentTeamConfig.

    Args:
        data: Raw dict from YAML.
        source: Source file path for error messages.

    Returns:
        Validated AgentTeamConfig.
    """
    # Parse roles if present
    roles_data = data.get("roles", [])
    roles = []
    for r in roles_data:
        if isinstance(r, dict):
            roles.append(AgentTeamRoleConfig(**r))
        else:
            raise TeamLoadError(f"Role must be a dict, got {type(r).__name__} in {source}")

    return AgentTeamConfig(
        name=data.get("name", Path(source).stem if source else "unnamed"),
        description=data.get("description", ""),
        roles=roles,
    )


def config_to_team(config: AgentTeamConfig) -> AgentTeam:
    """Convert a Pydantic config model to an AgentTeam dataclass.

    Args:
        config: The team config to convert.

    Returns:
        AgentTeam instance ready for the orchestrator.
    """
    roles = [
        SubAgentRole(
            name=r.name,
            persona=r.persona,
            model=r.model,
            allowed_tools=r.allowed_tools,
            denied_tools=r.denied_tools,
            max_iterations=r.max_iterations,
        )
        for r in config.roles
    ]
    return AgentTeam(
        name=config.name,
        description=config.description,
        roles=roles,
    )


def load_teams_from_directory(teams_dir: str | Path) -> list[AgentTeam]:
    """Discover and load all teams from a directory.

    Args:
        teams_dir: Path to the teams directory.

    Returns:
        List of AgentTeam instances.
    """
    files = discover_team_files(teams_dir)
    if not files:
        return []

    teams: list[AgentTeam] = []
    for path in files:
        try:
            configs = parse_team_file(path)
            for cfg in configs:
                teams.append(config_to_team(cfg))
                logger.info(
                    "team_loaded",
                    name=cfg.name,
                    roles=len(cfg.roles),
                    source=str(path),
                )
        except TeamLoadError as e:
            logger.warning("team_load_failed", path=str(path), error=str(e))

    return teams


def merge_teams(
    file_teams: list[AgentTeam],
    config_teams: list[AgentTeam],
) -> list[AgentTeam]:
    """Merge file-based teams with inline config teams.

    File-based teams take precedence on name collision.

    Args:
        file_teams: Teams loaded from YAML files.
        config_teams: Teams from agent.yaml orchestration.teams.

    Returns:
        Merged list with no duplicate names.
    """
    seen: dict[str, AgentTeam] = {}

    # Config teams first (lower priority)
    for team in config_teams:
        seen[team.name] = team

    # File teams override
    for team in file_teams:
        if team.name in seen:
            logger.info(
                "team_override",
                name=team.name,
                msg="File-based team overrides inline config",
            )
        seen[team.name] = team

    return list(seen.values())


# ---------------------------------------------------------------------------
# Project loading
# ---------------------------------------------------------------------------


def discover_project_files(teams_dir: str | Path) -> list[Path]:
    """Find all YAML files in the teams/projects/ subdirectory.

    Args:
        teams_dir: Path to the teams directory.

    Returns:
        Sorted list of project YAML file paths.
    """
    projects_path = Path(teams_dir) / "projects"
    if not projects_path.is_dir():
        return []

    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml"):
        files.extend(projects_path.glob(pattern))

    files.sort(key=lambda p: p.stem)
    return files


def parse_project_file(path: Path) -> list[Project]:
    """Parse a project YAML file into Project dataclasses.

    A file can contain a single project (dict) or multiple (list).

    Args:
        path: Path to the YAML file.

    Returns:
        List of Project instances.

    Raises:
        TeamLoadError: If the file cannot be parsed or validated.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise TeamLoadError(f"Cannot read {path}: {e}") from e

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise TeamLoadError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        return []

    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise TeamLoadError(
            f"Expected dict or list in {path}, got {type(data).__name__}"
        )

    projects: list[Project] = []
    for item in items:
        if not isinstance(item, dict):
            raise TeamLoadError(f"Project must be a dict in {path}")
        projects.append(_parse_project_dict(item, source=str(path)))

    return projects


def _parse_project_dict(data: dict[str, Any], source: str = "") -> Project:
    """Parse a single project dict into a Project dataclass."""
    name = data.get("name", Path(source).stem if source else "unnamed")
    description = data.get("description", "")

    stages: list[ProjectStage] = []
    for stage_data in data.get("stages", []):
        if not isinstance(stage_data, dict):
            raise TeamLoadError(f"Stage must be a dict in {source}")

        agents: list[ProjectAgentRef] = []
        for agent_data in stage_data.get("agents", []):
            if not isinstance(agent_data, dict):
                raise TeamLoadError(f"Agent ref must be a dict in {source}")
            if "team" not in agent_data or "role" not in agent_data:
                raise TeamLoadError(
                    f"Agent ref must have 'team' and 'role' fields in {source}"
                )
            agents.append(ProjectAgentRef(
                team=agent_data["team"],
                role=agent_data["role"],
            ))

        # Parse feedback config
        feedback: FeedbackConfig | None = None
        feedback_data = stage_data.get("feedback")
        if feedback_data is not None:
            if not isinstance(feedback_data, dict):
                raise TeamLoadError(
                    f"Stage feedback must be a dict in {source}"
                )
            if "fix_stage" not in feedback_data:
                raise TeamLoadError(
                    f"Stage feedback must have 'fix_stage' field in {source}"
                )
            feedback = FeedbackConfig(
                fix_stage=feedback_data["fix_stage"],
                max_retries=feedback_data.get("max_retries", 3),
            )

        feedback_target = stage_data.get("feedback_target", False)

        # Parse discussion mode
        mode = stage_data.get("mode", "standard")
        if mode not in ("standard", "discussion"):
            raise TeamLoadError(
                f"Invalid stage mode '{mode}' in {source}. "
                f"Must be 'standard' or 'discussion'."
            )

        discussion: DiscussionConfig | None = None
        discussion_data = stage_data.get("discussion")
        if discussion_data is not None:
            if not isinstance(discussion_data, dict):
                raise TeamLoadError(
                    f"Stage discussion must be a dict in {source}"
                )
            moderator: ProjectAgentRef | None = None
            mod_data = discussion_data.get("moderator")
            if mod_data is not None:
                if not isinstance(mod_data, dict):
                    raise TeamLoadError(
                        f"Discussion moderator must be a dict in {source}"
                    )
                if "team" not in mod_data or "role" not in mod_data:
                    raise TeamLoadError(
                        f"Discussion moderator must have 'team' and 'role' "
                        f"fields in {source}"
                    )
                moderator = ProjectAgentRef(
                    team=mod_data["team"],
                    role=mod_data["role"],
                )
            discussion = DiscussionConfig(
                rounds=discussion_data.get("rounds", 3),
                moderator=moderator,
                consensus_required=discussion_data.get(
                    "consensus_required", False,
                ),
            )

        if mode == "discussion" and discussion is None:
            raise TeamLoadError(
                f"Stage mode is 'discussion' but no discussion config "
                f"provided in {source}"
            )

        stages.append(ProjectStage(
            name=stage_data.get("name", f"stage_{len(stages)}"),
            agents=agents,
            parallel=stage_data.get("parallel", True),
            feedback=feedback,
            feedback_target=feedback_target,
            mode=mode,
            discussion=discussion,
        ))

    # Validate feedback stage references
    stage_names = {s.name for s in stages}
    for stage in stages:
        if stage.feedback and stage.feedback.fix_stage not in stage_names:
            raise TeamLoadError(
                f"Feedback fix_stage '{stage.feedback.fix_stage}' not found "
                f"in project '{name}'. Available stages: "
                f"{', '.join(stage_names)} (in {source})"
            )

    return Project(name=name, description=description, stages=stages)


def load_projects_from_directory(teams_dir: str | Path) -> list[Project]:
    """Discover and load all projects from teams/projects/.

    Args:
        teams_dir: Path to the teams directory.

    Returns:
        List of Project instances.
    """
    files = discover_project_files(teams_dir)
    if not files:
        return []

    projects: list[Project] = []
    for path in files:
        try:
            parsed = parse_project_file(path)
            for proj in parsed:
                projects.append(proj)
                logger.info(
                    "project_loaded",
                    name=proj.name,
                    stages=len(proj.stages),
                    source=str(path),
                )
        except TeamLoadError as e:
            logger.warning("project_load_failed", path=str(path), error=str(e))

    return projects
