"""Tests for teams/ folder discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.core.subagent import AgentTeam, SubAgentRole
from agent.teams.loader import (
    TeamLoadError,
    config_to_team,
    discover_team_files,
    load_teams_from_directory,
    merge_teams,
    parse_project_file,
    parse_team_file,
)

# ---------------------------------------------------------------------------
# discover_team_files
# ---------------------------------------------------------------------------


class TestDiscoverTeamFiles:
    def test_finds_yaml_files(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.yaml").write_text("name: alpha")
        (tmp_path / "beta.yml").write_text("name: beta")
        (tmp_path / "readme.txt").write_text("not a team")

        files = discover_team_files(tmp_path)

        names = [f.name for f in files]
        assert "alpha.yaml" in names
        assert "beta.yml" in names
        assert "readme.txt" not in names

    def test_returns_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "zulu.yaml").write_text("name: zulu")
        (tmp_path / "alpha.yaml").write_text("name: alpha")

        files = discover_team_files(tmp_path)

        assert files[0].stem == "alpha"
        assert files[1].stem == "zulu"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        files = discover_team_files(tmp_path / "nonexistent")
        assert files == []

    def test_returns_empty_for_empty_dir(self, tmp_path: Path) -> None:
        files = discover_team_files(tmp_path)
        assert files == []


# ---------------------------------------------------------------------------
# parse_team_file
# ---------------------------------------------------------------------------


class TestParseTeamFile:
    def test_parses_single_team(self, tmp_path: Path) -> None:
        f = tmp_path / "eng.yaml"
        f.write_text(
            "name: engineering\n"
            "description: Eng team\n"
            "roles:\n"
            "  - name: backend\n"
            "    persona: You write backend code.\n"
            "    allowed_tools: [read_file, write_file]\n"
            "    max_iterations: 8\n"
        )

        configs = parse_team_file(f)

        assert len(configs) == 1
        assert configs[0].name == "engineering"
        assert configs[0].description == "Eng team"
        assert len(configs[0].roles) == 1
        assert configs[0].roles[0].name == "backend"
        assert configs[0].roles[0].max_iterations == 8
        assert "read_file" in configs[0].roles[0].allowed_tools

    def test_parses_multiple_teams_in_one_file(self, tmp_path: Path) -> None:
        f = tmp_path / "multi.yaml"
        f.write_text(
            "- name: team_a\n"
            "  description: A\n"
            "  roles:\n"
            "    - name: agent_a\n"
            "- name: team_b\n"
            "  description: B\n"
            "  roles:\n"
            "    - name: agent_b\n"
        )

        configs = parse_team_file(f)

        assert len(configs) == 2
        assert configs[0].name == "team_a"
        assert configs[1].name == "team_b"

    def test_parses_team_with_multiple_roles(self, tmp_path: Path) -> None:
        f = tmp_path / "big.yaml"
        f.write_text(
            "name: big_team\n"
            "description: Many roles\n"
            "roles:\n"
            "  - name: alpha\n"
            "    persona: First\n"
            "  - name: beta\n"
            "    persona: Second\n"
            "  - name: gamma\n"
            "    persona: Third\n"
        )

        configs = parse_team_file(f)

        assert len(configs) == 1
        assert len(configs[0].roles) == 3
        role_names = [r.name for r in configs[0].roles]
        assert role_names == ["alpha", "beta", "gamma"]

    def test_defaults_name_to_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "my_team.yaml"
        f.write_text("description: No name field\nroles: []\n")

        configs = parse_team_file(f)

        assert configs[0].name == "my_team"

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("")

        configs = parse_team_file(f)
        assert configs == []

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(":\n  :\n    - [invalid")

        with pytest.raises(TeamLoadError, match="Invalid YAML"):
            parse_team_file(f)

    def test_non_dict_top_level_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "string.yaml"
        f.write_text('"just a string"')

        with pytest.raises(TeamLoadError, match="Expected dict or list"):
            parse_team_file(f)

    def test_role_not_dict_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad_role.yaml"
        f.write_text("name: bad\n" "roles:\n" "  - just a string\n")

        with pytest.raises(TeamLoadError, match="Role must be a dict"):
            parse_team_file(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "nope.yaml"

        with pytest.raises(TeamLoadError, match="Cannot read"):
            parse_team_file(f)

    def test_role_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "defaults.yaml"
        f.write_text("name: minimal\nroles:\n  - name: worker\n")

        configs = parse_team_file(f)
        role = configs[0].roles[0]

        assert role.persona == "You are a helpful assistant."
        assert role.allowed_tools == []
        assert role.denied_tools == []
        assert role.max_iterations == 5

    def test_denied_tools_parsed(self, tmp_path: Path) -> None:
        f = tmp_path / "denied.yaml"
        f.write_text(
            "name: restricted\n"
            "roles:\n"
            "  - name: reviewer\n"
            "    denied_tools: [write_file, shell_exec]\n"
        )

        configs = parse_team_file(f)
        assert configs[0].roles[0].denied_tools == ["write_file", "shell_exec"]


# ---------------------------------------------------------------------------
# config_to_team
# ---------------------------------------------------------------------------


class TestConfigToTeam:
    def test_converts_to_dataclass(self) -> None:
        from agent.config import AgentTeamConfig, AgentTeamRoleConfig

        cfg = AgentTeamConfig(
            name="test_team",
            description="A test",
            roles=[
                AgentTeamRoleConfig(
                    name="worker",
                    persona="You work.",
                    allowed_tools=["read_file"],
                    denied_tools=["shell_exec"],
                    max_iterations=3,
                ),
            ],
        )

        team = config_to_team(cfg)

        assert isinstance(team, AgentTeam)
        assert team.name == "test_team"
        assert team.description == "A test"
        assert len(team.roles) == 1
        assert isinstance(team.roles[0], SubAgentRole)
        assert team.roles[0].name == "worker"
        assert team.roles[0].allowed_tools == ["read_file"]
        assert team.roles[0].denied_tools == ["shell_exec"]
        assert team.roles[0].max_iterations == 3


# ---------------------------------------------------------------------------
# load_teams_from_directory
# ---------------------------------------------------------------------------


class TestLoadTeamsFromDirectory:
    def test_loads_all_valid_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("name: alpha\ndescription: A\nroles:\n  - name: a1\n")
        (tmp_path / "b.yaml").write_text("name: beta\ndescription: B\nroles:\n  - name: b1\n")

        teams = load_teams_from_directory(tmp_path)

        names = {t.name for t in teams}
        assert names == {"alpha", "beta"}

    def test_skips_invalid_files(self, tmp_path: Path) -> None:
        (tmp_path / "good.yaml").write_text("name: good\nroles:\n  - name: g1\n")
        (tmp_path / "bad.yaml").write_text(":\n  [invalid")

        teams = load_teams_from_directory(tmp_path)

        assert len(teams) == 1
        assert teams[0].name == "good"

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        teams = load_teams_from_directory(tmp_path / "nope")
        assert teams == []

    def test_multi_team_file(self, tmp_path: Path) -> None:
        (tmp_path / "multi.yaml").write_text(
            "- name: x\n  roles:\n    - name: x1\n" "- name: y\n  roles:\n    - name: y1\n"
        )

        teams = load_teams_from_directory(tmp_path)

        names = {t.name for t in teams}
        assert names == {"x", "y"}


# ---------------------------------------------------------------------------
# merge_teams
# ---------------------------------------------------------------------------


class TestMergeTeams:
    def _team(self, name: str) -> AgentTeam:
        return AgentTeam(name=name, description=f"{name} team")

    def test_no_overlap(self) -> None:
        file_teams = [self._team("a")]
        config_teams = [self._team("b")]

        merged = merge_teams(file_teams, config_teams)

        names = {t.name for t in merged}
        assert names == {"a", "b"}

    def test_file_overrides_config(self) -> None:
        file_team = AgentTeam(name="x", description="from file")
        config_team = AgentTeam(name="x", description="from config")

        merged = merge_teams([file_team], [config_team])

        assert len(merged) == 1
        assert merged[0].description == "from file"

    def test_empty_inputs(self) -> None:
        assert merge_teams([], []) == []

    def test_only_file_teams(self) -> None:
        merged = merge_teams([self._team("a"), self._team("b")], [])
        assert len(merged) == 2

    def test_only_config_teams(self) -> None:
        merged = merge_teams([], [self._team("a")])
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# Integration: load real teams/ directory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feedback config parsing
# ---------------------------------------------------------------------------


class TestFeedbackConfigParsing:
    def test_parse_feedback_config(self, tmp_path: Path) -> None:
        """YAML with feedback config parsed correctly."""
        f = tmp_path / "fb.yaml"
        f.write_text(
            "name: fb_proj\n"
            "stages:\n"
            "  - name: fix\n"
            "    feedback_target: true\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: dev\n"
            "  - name: verify\n"
            "    agents:\n"
            "      - team: qa\n"
            "        role: tester\n"
            "    feedback:\n"
            "      fix_stage: fix\n"
            "      max_retries: 5\n"
        )

        projects = parse_project_file(f)

        assert len(projects) == 1
        proj = projects[0]
        verify_stage = proj.stages[1]
        assert verify_stage.feedback is not None
        assert verify_stage.feedback.fix_stage == "fix"
        assert verify_stage.feedback.max_retries == 5

    def test_parse_feedback_target(self, tmp_path: Path) -> None:
        """Boolean feedback_target parsed, defaults to False."""
        f = tmp_path / "target.yaml"
        f.write_text(
            "name: target_proj\n"
            "stages:\n"
            "  - name: fix\n"
            "    feedback_target: true\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: dev\n"
            "  - name: review\n"
            "    agents:\n"
            "      - team: qa\n"
            "        role: tester\n"
        )

        projects = parse_project_file(f)

        assert projects[0].stages[0].feedback_target is True
        assert projects[0].stages[1].feedback_target is False

    def test_feedback_missing_fix_stage_errors(self, tmp_path: Path) -> None:
        """Validation catches missing fix_stage field."""
        f = tmp_path / "no_fix.yaml"
        f.write_text(
            "name: no_fix\n"
            "stages:\n"
            "  - name: verify\n"
            "    agents:\n"
            "      - team: qa\n"
            "        role: tester\n"
            "    feedback:\n"
            "      max_retries: 3\n"
        )

        with pytest.raises(TeamLoadError, match="fix_stage"):
            parse_project_file(f)

    def test_feedback_invalid_stage_ref_errors(self, tmp_path: Path) -> None:
        """Validation catches nonexistent stage reference."""
        f = tmp_path / "bad_ref.yaml"
        f.write_text(
            "name: bad_ref\n"
            "stages:\n"
            "  - name: verify\n"
            "    agents:\n"
            "      - team: qa\n"
            "        role: tester\n"
            "    feedback:\n"
            "      fix_stage: nonexistent\n"
        )

        with pytest.raises(TeamLoadError, match="nonexistent.*not found"):
            parse_project_file(f)


class TestDiscussionConfigParsing:
    def test_parse_discussion_config(self, tmp_path: Path) -> None:
        """Discussion fields parsed correctly."""
        f = tmp_path / "disc.yaml"
        f.write_text(
            "name: disc_proj\n"
            "stages:\n"
            "  - name: review\n"
            "    mode: discussion\n"
            "    discussion:\n"
            "      rounds: 4\n"
            "      consensus_required: true\n"
            "      moderator:\n"
            "        team: product\n"
            "        role: pm\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: dev\n"
        )

        projects = parse_project_file(f)

        stage = projects[0].stages[0]
        assert stage.mode == "discussion"
        assert stage.discussion is not None
        assert stage.discussion.rounds == 4
        assert stage.discussion.consensus_required is True
        assert stage.discussion.moderator is not None
        assert stage.discussion.moderator.team == "product"
        assert stage.discussion.moderator.role == "pm"

    def test_parse_discussion_mode_without_config_errors(self, tmp_path: Path) -> None:
        """mode=discussion without discussion config raises error."""
        f = tmp_path / "bad_disc.yaml"
        f.write_text(
            "name: bad_disc\n"
            "stages:\n"
            "  - name: review\n"
            "    mode: discussion\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: dev\n"
        )

        with pytest.raises(TeamLoadError, match="discussion"):
            parse_project_file(f)

    def test_parse_discussion_moderator_ref(self, tmp_path: Path) -> None:
        """Moderator team/role parsed as ProjectAgentRef."""
        f = tmp_path / "mod.yaml"
        f.write_text(
            "name: mod_proj\n"
            "stages:\n"
            "  - name: review\n"
            "    mode: discussion\n"
            "    discussion:\n"
            "      rounds: 2\n"
            "      moderator:\n"
            "        team: management\n"
            "        role: lead\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: dev\n"
        )

        projects = parse_project_file(f)

        mod = projects[0].stages[0].discussion.moderator
        assert mod is not None
        assert mod.team == "management"
        assert mod.role == "lead"


class TestRealTeamsDirectory:
    """Smoke test against the actual teams/ directory in the project."""

    def test_loads_project_teams(self) -> None:
        project_teams_dir = Path(__file__).parent.parent / "teams"
        if not project_teams_dir.is_dir():
            pytest.skip("teams/ directory not found")

        teams = load_teams_from_directory(project_teams_dir)

        assert len(teams) >= 1
        for team in teams:
            assert team.name
            assert len(team.roles) >= 1
            for role in team.roles:
                assert role.name
                assert role.persona
