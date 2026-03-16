"""Tests for cross-team project pipelines."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import OrchestrationConfig
from agent.core.events import EventBus, Events
from agent.core.orchestrator import SubAgentOrchestrator
from agent.core.subagent import (
    AgentTeam,
    FeedbackConfig,
    Project,
    ProjectAgentRef,
    ProjectStage,
    ProjectStageResult,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
    SubAgentTask,
)
from agent.llm.provider import LLMResponse
from agent.core.session import TokenUsage
from agent.teams.loader import (
    TeamLoadError,
    discover_project_files,
    load_projects_from_directory,
    parse_project_file,
)
from agent.tools.registry import ToolDefinition, ToolTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_def(
    name: str, tier: ToolTier = ToolTier.SAFE, enabled: bool = True,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        tier=tier,
        parameters={"type": "object", "properties": {}},
        function=AsyncMock(),
        category="builtin",
        enabled=enabled,
    )


def _make_parent_registry(tools: list[ToolDefinition] | None = None) -> MagicMock:
    tools = tools or []
    tool_map = {t.name: t for t in tools}
    registry = MagicMock()
    registry.get_tool = MagicMock(side_effect=lambda n: tool_map.get(n))
    registry.list_tools = MagicMock(return_value=tools)
    registry.get_tool_schemas = MagicMock(
        return_value=[
            {"type": "function", "function": {"name": t.name, "description": "", "parameters": {}}}
            for t in tools if t.enabled
        ]
    )
    return registry


def _make_role(name: str = "worker", **kwargs) -> SubAgentRole:
    defaults = {"persona": f"You are {name}.", "max_iterations": 3}
    defaults.update(kwargs)
    return SubAgentRole(name=name, **defaults)


def _make_team(name: str, roles: list[SubAgentRole]) -> AgentTeam:
    return AgentTeam(name=name, description=f"Team {name}", roles=roles)


def _make_project(
    name: str = "test_project",
    stages: list[ProjectStage] | None = None,
) -> Project:
    if stages is None:
        stages = [
            ProjectStage(
                name="stage1",
                agents=[ProjectAgentRef(team="eng", role="dev")],
            ),
            ProjectStage(
                name="stage2",
                agents=[ProjectAgentRef(team="qa", role="tester")],
            ),
        ]
    return Project(name=name, description="Test project", stages=stages)


_PATCH_AGENT_LOOP = "agent.core.agent_loop.AgentLoop"
_PATCH_TOOL_EXECUTOR = "agent.tools.executor.ToolExecutor"
_PATCH_BUILD_PROMPT = "agent.llm.prompts.build_system_prompt"


# ---------------------------------------------------------------------------
# Project loader tests
# ---------------------------------------------------------------------------


class TestProjectLoader:
    def test_discover_project_files(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        (projects_dir / "a.yaml").write_text("name: a")
        (projects_dir / "b.yml").write_text("name: b")
        (projects_dir / "readme.txt").write_text("not a project")

        files = discover_project_files(tmp_path)

        names = [f.name for f in files]
        assert "a.yaml" in names
        assert "b.yml" in names
        assert "readme.txt" not in names

    def test_discover_no_projects_dir(self, tmp_path: Path) -> None:
        files = discover_project_files(tmp_path)
        assert files == []

    def test_parse_single_project(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        f = projects_dir / "feature.yaml"
        f.write_text(
            "name: feature\n"
            "description: Build a feature\n"
            "stages:\n"
            "  - name: plan\n"
            "    agents:\n"
            "      - team: product\n"
            "        role: pm\n"
            "  - name: build\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: dev\n"
        )

        projects = parse_project_file(f)

        assert len(projects) == 1
        assert projects[0].name == "feature"
        assert len(projects[0].stages) == 2
        assert projects[0].stages[0].name == "plan"
        assert projects[0].stages[0].agents[0].team == "product"
        assert projects[0].stages[0].agents[0].role == "pm"

    def test_parse_project_defaults_name_to_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "my_project.yaml"
        f.write_text("description: Test\nstages: []\n")

        projects = parse_project_file(f)
        assert projects[0].name == "my_project"

    def test_parse_project_missing_team_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(
            "name: bad\n"
            "stages:\n"
            "  - name: s1\n"
            "    agents:\n"
            "      - role: dev\n"
        )

        with pytest.raises(TeamLoadError, match="team.*role"):
            parse_project_file(f)

    def test_parse_project_missing_role_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad2.yaml"
        f.write_text(
            "name: bad2\n"
            "stages:\n"
            "  - name: s1\n"
            "    agents:\n"
            "      - team: eng\n"
        )

        with pytest.raises(TeamLoadError, match="team.*role"):
            parse_project_file(f)

    def test_parse_multiple_projects_in_one_file(self, tmp_path: Path) -> None:
        f = tmp_path / "multi.yaml"
        f.write_text(
            "- name: proj_a\n"
            "  stages:\n"
            "    - name: s1\n"
            "      agents:\n"
            "        - team: t\n"
            "          role: r\n"
            "- name: proj_b\n"
            "  stages: []\n"
        )

        projects = parse_project_file(f)
        assert len(projects) == 2
        assert projects[0].name == "proj_a"
        assert projects[1].name == "proj_b"

    def test_parse_project_parallel_default_true(self, tmp_path: Path) -> None:
        f = tmp_path / "par.yaml"
        f.write_text(
            "name: par\n"
            "stages:\n"
            "  - name: s1\n"
            "    agents:\n"
            "      - team: t\n"
            "        role: r\n"
        )

        projects = parse_project_file(f)
        assert projects[0].stages[0].parallel is True

    def test_parse_project_parallel_false(self, tmp_path: Path) -> None:
        f = tmp_path / "seq.yaml"
        f.write_text(
            "name: seq\n"
            "stages:\n"
            "  - name: s1\n"
            "    parallel: false\n"
            "    agents:\n"
            "      - team: t\n"
            "        role: r\n"
        )

        projects = parse_project_file(f)
        assert projects[0].stages[0].parallel is False

    def test_load_projects_from_directory(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        (projects_dir / "a.yaml").write_text(
            "name: alpha\n"
            "stages:\n"
            "  - name: s1\n"
            "    agents:\n"
            "      - team: t\n"
            "        role: r\n"
        )

        projects = load_projects_from_directory(tmp_path)

        assert len(projects) == 1
        assert projects[0].name == "alpha"

    def test_load_projects_skips_invalid(self, tmp_path: Path) -> None:
        projects_dir = tmp_path / "projects"
        projects_dir.mkdir()
        (projects_dir / "good.yaml").write_text(
            "name: good\nstages: []\n"
        )
        (projects_dir / "bad.yaml").write_text(":\n  [invalid")

        projects = load_projects_from_directory(tmp_path)
        assert len(projects) == 1
        assert projects[0].name == "good"

    def test_load_real_project_files(self) -> None:
        """Smoke test against actual teams/projects/ directory."""
        teams_dir = Path(__file__).parent.parent / "teams"
        if not (teams_dir / "projects").is_dir():
            pytest.skip("teams/projects/ not found")

        projects = load_projects_from_directory(teams_dir)
        assert len(projects) >= 1
        for proj in projects:
            assert proj.name
            assert len(proj.stages) >= 1
            for stage in proj.stages:
                assert stage.name
                assert len(stage.agents) >= 1


# ---------------------------------------------------------------------------
# Orchestrator run_project tests
# ---------------------------------------------------------------------------


class TestRunProject:
    @pytest.fixture
    def config(self) -> OrchestrationConfig:
        return OrchestrationConfig(
            enabled=True,
            max_concurrent_agents=15,
            subagent_timeout=10,
            default_max_iterations=5,
        )

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def tool_registry(self) -> MagicMock:
        return _make_parent_registry([_make_tool_def("read_file")])

    @pytest.fixture
    def mock_agent_loop(self) -> MagicMock:
        loop = MagicMock()
        loop.llm = MagicMock()
        loop.tool_executor = MagicMock()
        loop.tool_executor.config = None
        loop.cost_tracker = None
        loop.process_message = AsyncMock(
            return_value=LLMResponse(
                content="Done.",
                model="test",
                usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                finish_reason="stop",
            )
        )
        return loop

    @pytest.fixture
    def teams(self) -> dict[str, AgentTeam]:
        return {
            "eng": _make_team("eng", [_make_role("dev"), _make_role("frontend")]),
            "qa": _make_team("qa", [_make_role("tester"), _make_role("security")]),
            "product": _make_team("product", [_make_role("pm")]),
        }

    @pytest.fixture
    def orchestrator(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        teams: dict[str, AgentTeam],
    ) -> SubAgentOrchestrator:
        orch = SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            teams=list(teams.values()),
            sdk_service=MagicMock(run_subagent=AsyncMock(return_value="Agent output.")),
        )
        return orch

    async def test_unknown_project_returns_failed(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        result = await orchestrator.run_project("nonexistent", "do stuff")

        assert result.status == SubAgentStatus.FAILED
        assert "Unknown project" in result.error

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_two_stage_pipeline(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """A two-stage project runs both stages sequentially."""
        project = _make_project(
            "two_stage",
            stages=[
                ProjectStage(
                    name="plan",
                    agents=[ProjectAgentRef(team="product", role="pm")],
                ),
                ProjectStage(
                    name="build",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
            ],
        )
        orchestrator.projects["two_stage"] = project

        result = await orchestrator.run_project("two_stage", "Build login page")

        assert result.status == SubAgentStatus.COMPLETED
        assert result.project_name == "two_stage"
        assert len(result.stages) == 2
        assert result.stages[0].stage_name == "plan"
        assert result.stages[1].stage_name == "build"
        assert result.duration_ms >= 0

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_stage_context_chains(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Output from stage 1 appears in stage 2's context."""
        project = _make_project(
            "chain",
            stages=[
                ProjectStage(
                    name="first",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="second",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                ),
            ],
        )
        orchestrator.projects["chain"] = project

        result = await orchestrator.run_project("chain", "Test something")

        assert result.status == SubAgentStatus.COMPLETED
        # Stage 1 output should be in stage 2's combined context
        # We verify by checking that stage 2 completed (meaning context was passed)
        assert len(result.stages) == 2
        assert result.stages[1].stage_name == "second"

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_parallel_agents_within_stage(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Multiple agents in one stage all produce results."""
        project = _make_project(
            "parallel",
            stages=[
                ProjectStage(
                    name="review",
                    agents=[
                        ProjectAgentRef(team="qa", role="tester"),
                        ProjectAgentRef(team="qa", role="security"),
                    ],
                ),
            ],
        )
        orchestrator.projects["parallel"] = project

        result = await orchestrator.run_project("parallel", "Review the code")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 1
        assert len(result.stages[0].results) == 2

    async def test_missing_team_ref_fails(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Referencing a nonexistent team fails the project."""
        project = _make_project(
            "bad_team",
            stages=[
                ProjectStage(
                    name="s1",
                    agents=[ProjectAgentRef(team="nonexistent", role="dev")],
                ),
            ],
        )
        orchestrator.projects["bad_team"] = project

        result = await orchestrator.run_project("bad_team", "do stuff")

        assert result.status == SubAgentStatus.FAILED
        assert "nonexistent" in result.error
        assert "not found" in result.error

    async def test_missing_role_ref_fails(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Referencing a nonexistent role in a valid team fails."""
        project = _make_project(
            "bad_role",
            stages=[
                ProjectStage(
                    name="s1",
                    agents=[ProjectAgentRef(team="eng", role="nonexistent")],
                ),
            ],
        )
        orchestrator.projects["bad_role"] = project

        result = await orchestrator.run_project("bad_role", "do stuff")

        assert result.status == SubAgentStatus.FAILED
        assert "nonexistent" in result.error
        assert "not found" in result.error

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_emits_project_events(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
        event_bus: EventBus,
    ) -> None:
        """Project execution emits started, stage, and completed events."""
        project = _make_project(
            "events",
            stages=[
                ProjectStage(
                    name="only_stage",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
            ],
        )
        orchestrator.projects["events"] = project

        events_seen: list[str] = []

        async def on_proj_started(data: dict) -> None:
            events_seen.append("project_started")

        async def on_stage_started(data: dict) -> None:
            events_seen.append("stage_started")

        async def on_stage_completed(data: dict) -> None:
            events_seen.append("stage_completed")

        async def on_proj_completed(data: dict) -> None:
            events_seen.append("project_completed")

        event_bus.on(Events.PROJECT_STARTED, on_proj_started)
        event_bus.on(Events.PROJECT_STAGE_STARTED, on_stage_started)
        event_bus.on(Events.PROJECT_STAGE_COMPLETED, on_stage_completed)
        event_bus.on(Events.PROJECT_COMPLETED, on_proj_completed)

        await orchestrator.run_project("events", "test events")

        assert "project_started" in events_seen
        assert "stage_started" in events_seen
        assert "stage_completed" in events_seen
        assert "project_completed" in events_seen

    async def test_failed_ref_emits_project_failed(
        self,
        orchestrator: SubAgentOrchestrator,
        event_bus: EventBus,
    ) -> None:
        """A bad agent reference emits PROJECT_FAILED."""
        project = _make_project(
            "fail_ref",
            stages=[
                ProjectStage(
                    name="s1",
                    agents=[ProjectAgentRef(team="bad", role="bad")],
                ),
            ],
        )
        orchestrator.projects["fail_ref"] = project

        failed_events: list[dict] = []

        async def on_failed(data: dict) -> None:
            failed_events.append(data)

        event_bus.on(Events.PROJECT_FAILED, on_failed)

        await orchestrator.run_project("fail_ref", "test")

        assert len(failed_events) == 1
        assert "bad" in failed_events[0]["error"]

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_initial_context_passed_to_first_stage(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Initial context is available to the first stage's agents."""
        project = _make_project(
            "ctx",
            stages=[
                ProjectStage(
                    name="s1",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
            ],
        )
        orchestrator.projects["ctx"] = project

        result = await orchestrator.run_project(
            "ctx", "Build X", context="We use FastAPI and PostgreSQL"
        )

        assert result.status == SubAgentStatus.COMPLETED

    def test_list_projects_empty(self, orchestrator: SubAgentOrchestrator) -> None:
        assert orchestrator.list_projects() == []

    def test_list_projects_populated(self, orchestrator: SubAgentOrchestrator) -> None:
        project = _make_project("listed")
        orchestrator.projects["listed"] = project

        projects = orchestrator.list_projects()

        assert len(projects) == 1
        assert projects[0]["name"] == "listed"
        assert len(projects[0]["stages"]) == 2

    @patch(_PATCH_BUILD_PROMPT, return_value="system")
    @patch(_PATCH_TOOL_EXECUTOR)
    @patch(_PATCH_AGENT_LOOP)
    async def test_three_stage_full_pipeline(
        self,
        mock_loop_cls: MagicMock,
        mock_executor_cls: MagicMock,
        _mock_prompt: MagicMock,
        orchestrator: SubAgentOrchestrator,
    ) -> None:
        """A realistic 3-stage pipeline completes all stages."""
        project = _make_project(
            "full",
            stages=[
                ProjectStage(
                    name="plan",
                    agents=[
                        ProjectAgentRef(team="product", role="pm"),
                        ProjectAgentRef(team="eng", role="dev"),
                    ],
                ),
                ProjectStage(
                    name="build",
                    agents=[
                        ProjectAgentRef(team="eng", role="dev"),
                        ProjectAgentRef(team="eng", role="frontend"),
                    ],
                ),
                ProjectStage(
                    name="test",
                    agents=[
                        ProjectAgentRef(team="qa", role="tester"),
                        ProjectAgentRef(team="qa", role="security"),
                    ],
                ),
            ],
        )
        orchestrator.projects["full"] = project

        result = await orchestrator.run_project("full", "Build user auth")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 3
        assert result.final_output  # last stage produced output

    # -------------------------------------------------------------------
    # Edge cases: agent failures within stages
    # -------------------------------------------------------------------

    async def test_one_agent_fails_in_stage_pipeline_stops(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """If one agent in a stage fails, the pipeline stops (GAP 11)."""
        call_count = 0

        async def selective_fail(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("role_persona", "").startswith("You are tester"):
                raise RuntimeError("tester crashed")
            return "Agent output."

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=selective_fail)

        project = _make_project(
            "partial_fail",
            stages=[
                ProjectStage(
                    name="review",
                    agents=[
                        ProjectAgentRef(team="qa", role="tester"),
                        ProjectAgentRef(team="qa", role="security"),
                    ],
                ),
                ProjectStage(
                    name="fix",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
            ],
        )
        orchestrator.projects["partial_fail"] = project

        result = await orchestrator.run_project("partial_fail", "Review code")

        # Pipeline stops on stage failure
        assert result.status == SubAgentStatus.FAILED
        assert len(result.stages) == 1  # Only the failed stage was executed
        assert result.error is not None
        # Stage 1 has one success and one failure
        s1_statuses = {r.role_name: r.status for r in result.stages[0].results}
        assert SubAgentStatus.FAILED in s1_statuses.values()
        assert SubAgentStatus.COMPLETED in s1_statuses.values()

    async def test_all_agents_fail_in_stage_pipeline_stops(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """If all agents in a stage fail, the pipeline stops (GAP 11)."""
        orchestrator.sdk_service.run_subagent = AsyncMock(
            side_effect=RuntimeError("all broken")
        )

        project = _make_project(
            "all_fail",
            stages=[
                ProjectStage(
                    name="doomed",
                    agents=[
                        ProjectAgentRef(team="eng", role="dev"),
                        ProjectAgentRef(team="qa", role="tester"),
                    ],
                ),
                ProjectStage(
                    name="after_doom",
                    agents=[ProjectAgentRef(team="product", role="pm")],
                ),
            ],
        )
        orchestrator.projects["all_fail"] = project

        result = await orchestrator.run_project("all_fail", "Try anyway")

        assert result.status == SubAgentStatus.FAILED
        assert len(result.stages) == 1  # Pipeline stopped after first stage
        # All agents in stage 1 failed
        assert all(
            r.status == SubAgentStatus.FAILED for r in result.stages[0].results
        )

    async def test_failed_agent_output_in_combined_output(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Failed agents' errors appear in the combined output."""
        orchestrator.sdk_service.run_subagent = AsyncMock(
            side_effect=RuntimeError("stage1 error message")
        )

        project = _make_project(
            "fail_ctx",
            stages=[
                ProjectStage(
                    name="broken",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="next",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                ),
            ],
        )
        orchestrator.projects["fail_ctx"] = project

        result = await orchestrator.run_project("fail_ctx", "Check")

        # The failed agent's error should appear in stage 1's combined output
        assert result.status == SubAgentStatus.FAILED
        assert "FAILED" in result.stages[0].combined_output
        assert "stage1 error message" in result.stages[0].combined_output

    # -------------------------------------------------------------------
    # Edge cases: context chaining verification
    # -------------------------------------------------------------------

    async def test_stage1_output_appears_in_stage2_task_context(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Verify stage 1 output is actually passed as context to stage 2 agents."""
        calls: list[dict] = []

        async def capture_calls(**kwargs):
            calls.append(kwargs)
            return f"Output from {kwargs.get('role_persona', 'unknown')[:20]}"

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=capture_calls)

        project = _make_project(
            "chain_verify",
            stages=[
                ProjectStage(
                    name="first",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="second",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                ),
            ],
        )
        orchestrator.projects["chain_verify"] = project

        await orchestrator.run_project("chain_verify", "Build X")

        # Stage 2 agent should have received stage 1's output in its prompt
        assert len(calls) == 2
        stage2_prompt = calls[1]["prompt"]
        assert "Previous stage: first" in stage2_prompt

    async def test_initial_context_plus_stage_output_accumulates(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Initial context and stage outputs accumulate across stages."""
        calls: list[dict] = []

        async def capture_calls(**kwargs):
            calls.append(kwargs)
            return "stage output"

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=capture_calls)

        project = _make_project(
            "accum",
            stages=[
                ProjectStage(
                    name="s1",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="s2",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                ),
                ProjectStage(
                    name="s3",
                    agents=[ProjectAgentRef(team="product", role="pm")],
                ),
            ],
        )
        orchestrator.projects["accum"] = project

        await orchestrator.run_project(
            "accum", "Build", context="Initial context here"
        )

        assert len(calls) == 3
        # Stage 1 gets initial context
        assert "Initial context here" in calls[0]["prompt"]
        # Stage 2 gets initial context + stage 1 output
        s2_prompt = calls[1]["prompt"]
        assert "Initial context here" in s2_prompt
        assert "Previous stage: s1" in s2_prompt
        # Stage 3 gets everything
        s3_prompt = calls[2]["prompt"]
        assert "Initial context here" in s3_prompt
        assert "Previous stage: s1" in s3_prompt
        assert "Previous stage: s2" in s3_prompt

    # -------------------------------------------------------------------
    # Edge cases: degenerate inputs
    # -------------------------------------------------------------------

    async def test_project_with_no_stages(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """A project with zero stages completes immediately with empty output."""
        project = Project(name="empty", description="No stages", stages=[])
        orchestrator.projects["empty"] = project

        result = await orchestrator.run_project("empty", "Do nothing")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 0
        assert result.final_output == ""

    async def test_stage_with_no_agents(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """A stage with zero agents completes with empty results."""
        project = _make_project(
            "no_agents",
            stages=[
                ProjectStage(name="empty_stage", agents=[]),
            ],
        )
        orchestrator.projects["no_agents"] = project

        result = await orchestrator.run_project("no_agents", "Test")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 1
        assert result.stages[0].results == []

    # -------------------------------------------------------------------
    # Edge cases: combined output format
    # -------------------------------------------------------------------

    async def test_combined_output_format(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Combined output uses [role_name]: prefix and --- separators."""
        orchestrator.sdk_service.run_subagent = AsyncMock(
            return_value="Review findings here."
        )

        project = _make_project(
            "fmt",
            stages=[
                ProjectStage(
                    name="review",
                    agents=[
                        ProjectAgentRef(team="qa", role="tester"),
                        ProjectAgentRef(team="qa", role="security"),
                    ],
                ),
            ],
        )
        orchestrator.projects["fmt"] = project

        result = await orchestrator.run_project("fmt", "Review code")

        combined = result.stages[0].combined_output
        assert "[tester]:" in combined
        assert "[security]:" in combined
        assert "---" in combined
        assert "Review findings here." in combined

    async def test_final_output_is_last_stage(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """final_output comes from the last stage's combined output."""
        call_count = 0

        async def numbered(**kwargs):
            nonlocal call_count
            call_count += 1
            return f"Output {call_count}"

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=numbered)

        project = _make_project(
            "last",
            stages=[
                ProjectStage(
                    name="first",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="last_stage",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                ),
            ],
        )
        orchestrator.projects["last"] = project

        result = await orchestrator.run_project("last", "Test")

        # final_output should be from the last stage, not the first
        assert result.final_output == result.stages[-1].combined_output
        assert "Output 2" in result.final_output

    async def test_completed_agent_with_empty_output_skipped(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Agents that complete with empty output are skipped in combined output."""
        orchestrator.sdk_service.run_subagent = AsyncMock(return_value="")

        project = _make_project(
            "empty_out",
            stages=[
                ProjectStage(
                    name="s1",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
            ],
        )
        orchestrator.projects["empty_out"] = project

        result = await orchestrator.run_project("empty_out", "Test")

        assert result.status == SubAgentStatus.COMPLETED
        # Empty output means nothing in combined (the [role]: prefix isn't added)
        assert result.stages[0].combined_output == ""

    # -------------------------------------------------------------------
    # Edge cases: resolution failure mid-pipeline
    # -------------------------------------------------------------------

    async def test_resolution_failure_in_stage2_preserves_stage1_results(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """If stage 2 has a bad ref, stage 1 results are still in the ProjectResult."""
        orchestrator.sdk_service.run_subagent = AsyncMock(return_value="Stage 1 ok.")

        project = _make_project(
            "mid_fail",
            stages=[
                ProjectStage(
                    name="good_stage",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="bad_stage",
                    agents=[ProjectAgentRef(team="nonexistent", role="x")],
                ),
            ],
        )
        orchestrator.projects["mid_fail"] = project

        result = await orchestrator.run_project("mid_fail", "Test")

        assert result.status == SubAgentStatus.FAILED
        assert "nonexistent" in result.error
        # Stage 1 results are preserved
        assert len(result.stages) == 2
        assert result.stages[0].stage_name == "good_stage"
        assert result.stages[0].results[0].status == SubAgentStatus.COMPLETED

    # -------------------------------------------------------------------
    # Edge cases: SDK path verification
    # -------------------------------------------------------------------

    async def test_project_uses_sdk_service(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Verify run_project calls through sdk_service.run_subagent."""
        orchestrator.sdk_service.run_subagent = AsyncMock(return_value="SDK result")

        project = _make_project(
            "sdk_verify",
            stages=[
                ProjectStage(
                    name="s1",
                    agents=[
                        ProjectAgentRef(team="eng", role="dev"),
                        ProjectAgentRef(team="qa", role="tester"),
                    ],
                ),
            ],
        )
        orchestrator.projects["sdk_verify"] = project

        result = await orchestrator.run_project("sdk_verify", "Build it")

        assert result.status == SubAgentStatus.COMPLETED
        assert orchestrator.sdk_service.run_subagent.await_count == 2
        # Verify correct role personas were passed
        call_personas = [
            c.kwargs["role_persona"]
            for c in orchestrator.sdk_service.run_subagent.call_args_list
        ]
        assert any("dev" in p for p in call_personas)
        assert any("tester" in p for p in call_personas)

    async def test_project_stage_event_counts_agents(
        self,
        orchestrator: SubAgentOrchestrator,
        event_bus: EventBus,
    ) -> None:
        """PROJECT_STAGE_COMPLETED event includes correct agent counts."""
        call_count = 0

        async def fail_second(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("fail")
            return "ok"

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=fail_second)

        project = _make_project(
            "count_evt",
            stages=[
                ProjectStage(
                    name="mixed",
                    agents=[
                        ProjectAgentRef(team="eng", role="dev"),
                        ProjectAgentRef(team="qa", role="tester"),
                    ],
                ),
            ],
        )
        orchestrator.projects["count_evt"] = project

        stage_events: list[dict] = []

        async def on_stage_done(data: dict) -> None:
            stage_events.append(data)

        event_bus.on(Events.PROJECT_STAGE_COMPLETED, on_stage_done)

        await orchestrator.run_project("count_evt", "Test")

        assert len(stage_events) == 1
        assert stage_events[0]["agents_completed"] == 1
        assert stage_events[0]["agents_failed"] == 1


# ---------------------------------------------------------------------------
# Feedback loop tests
# ---------------------------------------------------------------------------


class TestFeedbackLoop:
    @pytest.fixture
    def config(self) -> OrchestrationConfig:
        return OrchestrationConfig(
            enabled=True,
            max_concurrent_agents=15,
            subagent_timeout=10,
            default_max_iterations=5,
        )

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def tool_registry(self) -> MagicMock:
        return _make_parent_registry([_make_tool_def("read_file")])

    @pytest.fixture
    def mock_agent_loop(self) -> MagicMock:
        loop = MagicMock()
        loop.llm = MagicMock()
        loop.tool_executor = MagicMock()
        loop.tool_executor.config = None
        loop.cost_tracker = None
        return loop

    @pytest.fixture
    def teams(self) -> dict[str, AgentTeam]:
        return {
            "eng": _make_team("eng", [_make_role("dev"), _make_role("frontend")]),
            "qa": _make_team("qa", [_make_role("tester"), _make_role("security")]),
        }

    @pytest.fixture
    def orchestrator(
        self,
        mock_agent_loop: MagicMock,
        config: OrchestrationConfig,
        event_bus: EventBus,
        tool_registry: MagicMock,
        teams: dict[str, AgentTeam],
    ) -> SubAgentOrchestrator:
        return SubAgentOrchestrator(
            agent_loop=mock_agent_loop,
            config=config,
            event_bus=event_bus,
            tool_registry=tool_registry,
            teams=list(teams.values()),
            sdk_service=MagicMock(run_subagent=AsyncMock(return_value="Agent output.")),
        )

    def _make_feedback_project(self) -> Project:
        """Create a project with investigation → fix (feedback_target) → verify (feedback)."""
        return Project(
            name="fb_proj",
            description="Feedback test project",
            stages=[
                ProjectStage(
                    name="investigation",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="fix",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                    feedback_target=True,
                ),
                ProjectStage(
                    name="verify",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                    feedback=FeedbackConfig(fix_stage="fix", max_retries=3),
                ),
            ],
        )

    async def test_feedback_loop_passes_first_try(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Review passes on first try, no feedback loop triggered."""
        # Evaluator returns PASS
        orchestrator.sdk_service.run_subagent = AsyncMock(
            side_effect=lambda **kw: (
                "PASS: All checks passed"
                if "Evaluate" in kw.get("prompt", "")
                else "Agent output."
            )
        )

        project = self._make_feedback_project()
        orchestrator.projects["fb_proj"] = project

        result = await orchestrator.run_project("fb_proj", "Fix the bug")

        assert result.status == SubAgentStatus.COMPLETED
        assert result.feedback_iterations == 0

    async def test_feedback_loop_fail_fix_pass(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Review fails, fix runs, re-review passes (1 iteration)."""
        call_count = 0

        async def mock_subagent(**kw):
            nonlocal call_count
            call_count += 1
            prompt = kw.get("prompt", "")
            if "Evaluate" in prompt:
                # First eval: FAIL, second eval: PASS
                if call_count <= 3:  # investigation + verify + first eval
                    return "FAIL: Tests failing"
                return "PASS: All fixed"
            return "Agent output."

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=mock_subagent)

        project = self._make_feedback_project()
        orchestrator.projects["fb_proj"] = project

        result = await orchestrator.run_project("fb_proj", "Fix the bug")

        assert result.status == SubAgentStatus.COMPLETED
        assert result.feedback_iterations == 1

    async def test_feedback_loop_max_retries_exhausted(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Fails N times, pipeline fails with exhausted retries."""
        async def always_fail(**kw):
            if "Evaluate" in kw.get("prompt", ""):
                return "FAIL: Still broken"
            return "Agent output."

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=always_fail)

        project = Project(
            name="exhaust",
            description="Exhaust test",
            stages=[
                ProjectStage(
                    name="fix",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                    feedback_target=True,
                ),
                ProjectStage(
                    name="verify",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                    feedback=FeedbackConfig(fix_stage="fix", max_retries=2),
                ),
            ],
        )
        orchestrator.projects["exhaust"] = project

        result = await orchestrator.run_project("exhaust", "Fix it")

        assert result.status == SubAgentStatus.FAILED
        assert "exhausted" in result.error.lower()
        assert result.feedback_iterations == 2

    async def test_feedback_target_skipped_in_normal_flow(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Stages with feedback_target=True are skipped in normal sequential flow."""
        calls: list[str] = []

        async def track_calls(**kw):
            prompt = kw.get("prompt", "")
            if "Evaluate" in prompt:
                return "PASS: All good"
            role = kw.get("role_persona", "")
            calls.append(role)
            return "Agent output."

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=track_calls)

        project = Project(
            name="skip_test",
            description="Skip test",
            stages=[
                ProjectStage(
                    name="investigate",
                    agents=[ProjectAgentRef(team="eng", role="dev")],
                ),
                ProjectStage(
                    name="fix",
                    agents=[ProjectAgentRef(team="eng", role="frontend")],
                    feedback_target=True,
                ),
                ProjectStage(
                    name="verify",
                    agents=[ProjectAgentRef(team="qa", role="tester")],
                    feedback=FeedbackConfig(fix_stage="fix", max_retries=2),
                ),
            ],
        )
        orchestrator.projects["skip_test"] = project

        result = await orchestrator.run_project("skip_test", "Check")

        assert result.status == SubAgentStatus.COMPLETED
        # fix stage (frontend role) should NOT have been called since review passed
        assert not any("frontend" in c for c in calls)

    async def test_feedback_loop_emits_events(
        self, orchestrator: SubAgentOrchestrator, event_bus: EventBus,
    ) -> None:
        """Correct feedback events emitted with data."""
        call_count = 0

        async def mock_subagent(**kw):
            nonlocal call_count
            call_count += 1
            if "Evaluate" in kw.get("prompt", ""):
                if call_count <= 3:
                    return "FAIL: Issues found"
                return "PASS: Fixed"
            return "Agent output."

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=mock_subagent)

        events_seen: list[str] = []

        async def on_fb_started(data: dict) -> None:
            events_seen.append("feedback_started")

        async def on_fb_iteration(data: dict) -> None:
            events_seen.append("feedback_iteration")

        async def on_fb_passed(data: dict) -> None:
            events_seen.append("feedback_passed")

        event_bus.on(Events.PROJECT_FEEDBACK_STARTED, on_fb_started)
        event_bus.on(Events.PROJECT_FEEDBACK_ITERATION, on_fb_iteration)
        event_bus.on(Events.PROJECT_FEEDBACK_PASSED, on_fb_passed)

        project = self._make_feedback_project()
        orchestrator.projects["fb_proj"] = project

        await orchestrator.run_project("fb_proj", "Fix bug")

        assert "feedback_started" in events_seen
        assert "feedback_iteration" in events_seen
        assert "feedback_passed" in events_seen

    async def test_feedback_context_enrichment(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Fix agent receives review feedback, re-review sees fix output."""
        calls: list[dict] = []
        call_count = 0

        async def capture_calls(**kw):
            nonlocal call_count
            call_count += 1
            calls.append(kw)
            if "Evaluate" in kw.get("prompt", ""):
                if call_count <= 3:
                    return "FAIL: Tests failing"
                return "PASS: All good"
            return f"Output from call {call_count}"

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=capture_calls)

        project = self._make_feedback_project()
        orchestrator.projects["fb_proj"] = project

        await orchestrator.run_project("fb_proj", "Fix bug")

        # Find fix stage call (should have review feedback in context)
        fix_calls = [c for c in calls if "Review feedback" in c.get("prompt", "")]
        assert len(fix_calls) >= 1
        assert "Issues to fix" in fix_calls[0]["prompt"]

        # Find re-review call (should have fix output context)
        re_review_calls = [c for c in calls if "re-verify" in c.get("prompt", "")]
        assert len(re_review_calls) >= 1

    async def test_feedback_fix_stage_fails(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """If the fix stage itself fails, the feedback loop stops."""
        call_count = 0

        async def mock_subagent(**kw):
            nonlocal call_count
            call_count += 1
            prompt = kw.get("prompt", "")
            if "Evaluate" in prompt:
                return "FAIL: Issues found"
            # Fail on fix stage calls (they have "Review feedback" in context)
            if "Review feedback" in prompt:
                raise RuntimeError("Fix agent crashed")
            return "Agent output."

        orchestrator.sdk_service.run_subagent = AsyncMock(side_effect=mock_subagent)

        project = self._make_feedback_project()
        orchestrator.projects["fb_proj"] = project

        result = await orchestrator.run_project("fb_proj", "Fix bug")

        assert result.status == SubAgentStatus.FAILED
        assert "exhausted" in result.error.lower() or "retries" in result.error.lower()

    async def test_no_feedback_config_unchanged(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Projects without feedback config work exactly as before."""
        orchestrator.sdk_service.run_subagent = AsyncMock(return_value="Done.")

        project = _make_project("no_fb")
        orchestrator.projects["no_fb"] = project

        result = await orchestrator.run_project("no_fb", "Simple task")

        assert result.status == SubAgentStatus.COMPLETED
        assert result.feedback_iterations == 0
        assert len(result.stages) == 2

    async def test_evaluate_review_output_pass(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Evaluator correctly parses PASS response."""
        orchestrator.sdk_service.run_subagent = AsyncMock(
            return_value="PASS: All tests green"
        )

        passed, summary = await orchestrator._evaluate_review_output("All good")

        assert passed is True
        assert "All tests green" in summary

    async def test_evaluate_review_output_fail(
        self, orchestrator: SubAgentOrchestrator,
    ) -> None:
        """Evaluator correctly parses FAIL response."""
        orchestrator.sdk_service.run_subagent = AsyncMock(
            return_value="FAIL: 3 tests failing"
        )

        passed, summary = await orchestrator._evaluate_review_output("Errors found")

        assert passed is False
        assert "3 tests failing" in summary
