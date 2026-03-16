"""Tests for discussion rounds (Feature 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import OrchestrationConfig
from agent.core.events import EventBus, Events
from agent.core.orchestrator import SubAgentOrchestrator
from agent.core.subagent import (
    AgentTeam,
    DiscussionConfig,
    Project,
    ProjectAgentRef,
    ProjectStage,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
    SubAgentTask,
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
    registry.get_tool.side_effect = lambda name: tool_map.get(name)
    registry.get_tool_schemas.return_value = [
        {"function": {"name": t.name}} for t in tools if t.enabled
    ]
    registry.list_tools.return_value = tools
    return registry


def _make_orchestrator(
    teams: list[AgentTeam] | None = None,
) -> SubAgentOrchestrator:
    agent_loop = MagicMock()
    agent_loop.llm = MagicMock()
    agent_loop.tool_executor = MagicMock()
    config = OrchestrationConfig(
        enabled=True, max_concurrent_agents=10, subagent_timeout=60,
    )
    event_bus = EventBus()
    registry = _make_parent_registry()
    return SubAgentOrchestrator(
        agent_loop=agent_loop,
        config=config,
        event_bus=event_bus,
        tool_registry=registry,
        teams=teams,
    )


def _spawn_counter() -> tuple[list[int], AsyncMock]:
    """Create a spawn_subagent mock that counts calls and returns numbered output."""
    call_count = [0]

    async def mock_spawn(task: SubAgentTask) -> SubAgentResult:
        call_count[0] += 1
        return SubAgentResult(
            task_id=f"t{call_count[0]}",
            role_name=task.role.name,
            status=SubAgentStatus.COMPLETED,
            output=f"Output from {task.role.name} (call {call_count[0]})",
        )

    return call_count, AsyncMock(side_effect=mock_spawn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscussionRounds:
    @pytest.mark.asyncio
    async def test_discussion_basic_rounds(self) -> None:
        """Agents take turns over N rounds."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        r2 = SubAgentRole(name="bob", persona="Bob.")
        team = AgentTeam(name="eng", description="Eng", roles=[r1, r2])
        orch = _make_orchestrator(teams=[team])

        stage = ProjectStage(
            name="discuss",
            agents=[
                ProjectAgentRef(team="eng", role="alice"),
                ProjectAgentRef(team="eng", role="bob"),
            ],
            mode="discussion",
            discussion=DiscussionConfig(rounds=2),
        )

        call_count, mock_spawn = _spawn_counter()

        with patch.object(orch, "spawn_subagent", mock_spawn):
            result, has_failure = await orch._run_discussion_stage(
                "test_proj", stage, "Discuss the design.", "", "",
            )

        assert not has_failure
        # 2 rounds × 2 agents = 4 calls
        assert call_count[0] == 4
        assert "[Round 1]" in result.combined_output
        assert "[Round 2]" in result.combined_output

    @pytest.mark.asyncio
    async def test_discussion_context_accumulates(self) -> None:
        """Each round sees previous outputs in context."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        team = AgentTeam(name="eng", description="Eng", roles=[r1])
        orch = _make_orchestrator(teams=[team])

        stage = ProjectStage(
            name="discuss",
            agents=[ProjectAgentRef(team="eng", role="alice")],
            mode="discussion",
            discussion=DiscussionConfig(rounds=2),
        )

        contexts_seen: list[str] = []

        async def capture_context(task: SubAgentTask) -> SubAgentResult:
            contexts_seen.append(task.context)
            return SubAgentResult(
                task_id="t1",
                role_name="alice",
                status=SubAgentStatus.COMPLETED,
                output="Round contribution",
            )

        with patch.object(orch, "spawn_subagent", side_effect=capture_context):
            await orch._run_discussion_stage(
                "test_proj", stage, "Discuss.", "", "",
            )

        # Second call should see first round's output in context
        assert len(contexts_seen) == 2
        assert "Round contribution" in contexts_seen[1]

    @pytest.mark.asyncio
    async def test_discussion_sequential_within_round(self) -> None:
        """Agents run sequentially in rounds after the first (round 1 is parallel)."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        r2 = SubAgentRole(name="bob", persona="Bob.")
        team = AgentTeam(name="eng", description="Eng", roles=[r1, r2])
        orch = _make_orchestrator(teams=[team])

        stage = ProjectStage(
            name="discuss",
            agents=[
                ProjectAgentRef(team="eng", role="alice"),
                ProjectAgentRef(team="eng", role="bob"),
            ],
            mode="discussion",
            discussion=DiscussionConfig(rounds=2),
        )

        execution_order: list[str] = []

        async def track_order(task: SubAgentTask) -> SubAgentResult:
            execution_order.append(task.role.name)
            return SubAgentResult(
                task_id="t1",
                role_name=task.role.name,
                status=SubAgentStatus.COMPLETED,
                output=f"From {task.role.name}",
            )

        with patch.object(orch, "spawn_subagent", side_effect=track_order):
            await orch._run_discussion_stage(
                "test_proj", stage, "Discuss.", "", "",
            )

        # Round 1: parallel (alice, bob in any order), Round 2: sequential
        assert len(execution_order) == 4
        # Round 2 agents should be sequential: alice then bob
        assert execution_order[2:] == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_discussion_with_moderator(self) -> None:
        """Moderator summarizes each round."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        mod_role = SubAgentRole(name="pm", persona="Product Manager.")
        eng_team = AgentTeam(name="eng", description="Eng", roles=[r1])
        pm_team = AgentTeam(name="product", description="Product", roles=[mod_role])
        orch = _make_orchestrator(teams=[eng_team, pm_team])

        stage = ProjectStage(
            name="discuss",
            agents=[ProjectAgentRef(team="eng", role="alice")],
            mode="discussion",
            discussion=DiscussionConfig(
                rounds=1,
                moderator=ProjectAgentRef(team="product", role="pm"),
            ),
        )

        call_count, mock_spawn = _spawn_counter()

        with patch.object(orch, "spawn_subagent", mock_spawn):
            result, _ = await orch._run_discussion_stage(
                "test_proj", stage, "Discuss.", "", "",
            )

        # 1 agent + 1 moderator = 2 calls
        assert call_count[0] == 2
        assert "[Moderator" in result.combined_output

    @pytest.mark.asyncio
    async def test_discussion_consensus_early_stop(self) -> None:
        """Consensus reached before max rounds stops discussion."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        mod_role = SubAgentRole(name="pm", persona="PM.")
        eng_team = AgentTeam(name="eng", description="Eng", roles=[r1])
        pm_team = AgentTeam(name="product", description="Product", roles=[mod_role])
        orch = _make_orchestrator(teams=[eng_team, pm_team])

        stage = ProjectStage(
            name="discuss",
            agents=[ProjectAgentRef(team="eng", role="alice")],
            mode="discussion",
            discussion=DiscussionConfig(
                rounds=5,
                moderator=ProjectAgentRef(team="product", role="pm"),
                consensus_required=True,
            ),
        )

        call_num = [0]

        async def consensus_on_first(task: SubAgentTask) -> SubAgentResult:
            call_num[0] += 1
            output = "Some contribution"
            if task.role.name == "pm":
                output = "CONSENSUS: Everyone agrees on approach A."
            return SubAgentResult(
                task_id=f"t{call_num[0]}",
                role_name=task.role.name,
                status=SubAgentStatus.COMPLETED,
                output=output,
            )

        with patch.object(orch, "spawn_subagent", side_effect=consensus_on_first):
            result, _ = await orch._run_discussion_stage(
                "test_proj", stage, "Discuss.", "", "",
            )

        # Should stop after round 1: 1 agent + 1 moderator = 2 calls
        assert call_num[0] == 2

    @pytest.mark.asyncio
    async def test_discussion_no_consensus_exhausts_rounds(self) -> None:
        """All rounds run when no consensus reached."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        mod_role = SubAgentRole(name="pm", persona="PM.")
        eng_team = AgentTeam(name="eng", description="Eng", roles=[r1])
        pm_team = AgentTeam(name="product", description="Product", roles=[mod_role])
        orch = _make_orchestrator(teams=[eng_team, pm_team])

        stage = ProjectStage(
            name="discuss",
            agents=[ProjectAgentRef(team="eng", role="alice")],
            mode="discussion",
            discussion=DiscussionConfig(
                rounds=3,
                moderator=ProjectAgentRef(team="product", role="pm"),
                consensus_required=True,
            ),
        )

        call_num = [0]

        async def no_consensus(task: SubAgentTask) -> SubAgentResult:
            call_num[0] += 1
            output = "Some contribution"
            if task.role.name == "pm":
                output = "CONTINUE: Still debating."
            return SubAgentResult(
                task_id=f"t{call_num[0]}",
                role_name=task.role.name,
                status=SubAgentStatus.COMPLETED,
                output=output,
            )

        with patch.object(orch, "spawn_subagent", side_effect=no_consensus):
            result, _ = await orch._run_discussion_stage(
                "test_proj", stage, "Discuss.", "", "",
            )

        # 3 rounds × (1 agent + 1 moderator) = 6 calls
        assert call_num[0] == 6

    @pytest.mark.asyncio
    async def test_discussion_single_round(self) -> None:
        """rounds=1 works correctly."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        team = AgentTeam(name="eng", description="Eng", roles=[r1])
        orch = _make_orchestrator(teams=[team])

        stage = ProjectStage(
            name="discuss",
            agents=[ProjectAgentRef(team="eng", role="alice")],
            mode="discussion",
            discussion=DiscussionConfig(rounds=1),
        )

        call_count, mock_spawn = _spawn_counter()

        with patch.object(orch, "spawn_subagent", mock_spawn):
            result, has_failure = await orch._run_discussion_stage(
                "test_proj", stage, "Discuss.", "", "",
            )

        assert not has_failure
        assert call_count[0] == 1
        assert "[Round 1]" in result.combined_output

    @pytest.mark.asyncio
    async def test_discussion_yaml_parsing(self, tmp_path) -> None:
        """YAML with discussion config parses correctly."""
        from agent.teams.loader import parse_project_file

        f = tmp_path / "disc.yaml"
        f.write_text(
            "name: disc_proj\n"
            "stages:\n"
            "  - name: review\n"
            "    mode: discussion\n"
            "    discussion:\n"
            "      rounds: 3\n"
            "      consensus_required: true\n"
            "      moderator:\n"
            "        team: product\n"
            "        role: pm\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: alice\n"
            "      - team: eng\n"
            "        role: bob\n"
        )

        projects = parse_project_file(f)

        assert len(projects) == 1
        stage = projects[0].stages[0]
        assert stage.mode == "discussion"
        assert stage.discussion is not None
        assert stage.discussion.rounds == 3
        assert stage.discussion.consensus_required is True
        assert stage.discussion.moderator is not None
        assert stage.discussion.moderator.team == "product"
        assert stage.discussion.moderator.role == "pm"

    @pytest.mark.asyncio
    async def test_discussion_yaml_backward_compatible(self, tmp_path) -> None:
        """Existing YAML without mode field defaults to 'standard'."""
        from agent.teams.loader import parse_project_file

        f = tmp_path / "old.yaml"
        f.write_text(
            "name: old_proj\n"
            "stages:\n"
            "  - name: build\n"
            "    agents:\n"
            "      - team: eng\n"
            "        role: dev\n"
        )

        projects = parse_project_file(f)

        stage = projects[0].stages[0]
        assert stage.mode == "standard"
        assert stage.discussion is None

    @pytest.mark.asyncio
    async def test_discussion_events_emitted(self) -> None:
        """Correct event sequence emitted during discussion."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        team = AgentTeam(name="eng", description="Eng", roles=[r1])
        orch = _make_orchestrator(teams=[team])

        stage = ProjectStage(
            name="discuss",
            agents=[ProjectAgentRef(team="eng", role="alice")],
            mode="discussion",
            discussion=DiscussionConfig(rounds=2),
        )

        events_received: list[str] = []

        async def capture(name: str):
            events_received.append(name)

        orch.event_bus.on(
            Events.DISCUSSION_STARTED,
            lambda d: capture("started"),
        )
        orch.event_bus.on(
            Events.DISCUSSION_ROUND_COMPLETED,
            lambda d: capture("round_completed"),
        )
        orch.event_bus.on(
            Events.DISCUSSION_COMPLETED,
            lambda d: capture("completed"),
        )

        call_count, mock_spawn = _spawn_counter()

        with patch.object(orch, "spawn_subagent", mock_spawn):
            await orch._run_discussion_stage(
                "test_proj", stage, "Discuss.", "", "",
            )

        assert "started" in events_received
        assert events_received.count("round_completed") == 2
        assert "completed" in events_received


class TestDiscussionProjectIntegration:
    """Test discussion stages within run_project."""

    @pytest.mark.asyncio
    async def test_run_project_routes_discussion_stage(self) -> None:
        """run_project correctly routes discussion-mode stages."""
        r1 = SubAgentRole(name="alice", persona="Alice.")
        team = AgentTeam(name="eng", description="Eng", roles=[r1])
        orch = _make_orchestrator(teams=[team])

        project = Project(
            name="test_proj",
            description="Test project",
            stages=[
                ProjectStage(
                    name="discuss",
                    agents=[ProjectAgentRef(team="eng", role="alice")],
                    mode="discussion",
                    discussion=DiscussionConfig(rounds=1),
                ),
            ],
        )
        orch.projects["test_proj"] = project

        call_count, mock_spawn = _spawn_counter()

        with patch.object(orch, "spawn_subagent", mock_spawn):
            result = await orch.run_project("test_proj", "Do something.")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 1
        assert "[Round 1]" in result.stages[0].combined_output
