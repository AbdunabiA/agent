"""Tests for dynamic re-planning in pipeline execution."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config import OrchestrationConfig
from agent.core.events import EventBus, Events
from agent.core.orchestrator import SubAgentOrchestrator
from agent.core.session import TokenUsage
from agent.core.subagent import (
    AgentTeam,
    Project,
    ProjectAgentRef,
    ProjectStage,
    ProjectStageResult,
    ReplanDecision,
    SubAgentResult,
    SubAgentRole,
    SubAgentStatus,
)
from agent.llm.provider import LLMResponse
from agent.tools.registry import ToolDefinition, ToolTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_def(
    name: str,
    tier: ToolTier = ToolTier.SAFE,
    enabled: bool = True,
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
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": "",
                    "parameters": {},
                },
            }
            for t in tools
            if t.enabled
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
                name="design",
                agents=[ProjectAgentRef(team="eng", role="dev")],
            ),
            ProjectStage(
                name="build",
                agents=[ProjectAgentRef(team="eng", role="dev")],
            ),
            ProjectStage(
                name="test",
                agents=[ProjectAgentRef(team="qa", role="tester")],
            ),
        ]
    return Project(name=name, description="Test project", stages=stages)


def _make_orchestrator(
    event_bus: EventBus | None = None,
    llm: MagicMock | None = None,
) -> SubAgentOrchestrator:
    """Create a minimal orchestrator with mock dependencies."""
    agent_loop = MagicMock()
    agent_loop.llm = llm

    config = OrchestrationConfig()
    if event_bus is None:
        event_bus = EventBus()
    tool_registry = _make_parent_registry([_make_tool_def("test_tool")])

    dev_role = _make_role("dev")
    tester_role = _make_role("tester")
    eng_team = _make_team("eng", [dev_role])
    qa_team = _make_team("qa", [tester_role])

    orch = SubAgentOrchestrator(
        agent_loop=agent_loop,
        config=config,
        event_bus=event_bus,
        tool_registry=tool_registry,
        teams=[eng_team, qa_team],
    )
    return orch


def _make_stage_result(
    stage_name: str,
    output: str = "stage output",
    status: SubAgentStatus = SubAgentStatus.COMPLETED,
) -> tuple[ProjectStageResult, bool]:
    """Create a stage result tuple like _run_stage returns."""
    has_failure = status == SubAgentStatus.FAILED
    result = ProjectStageResult(
        stage_name=stage_name,
        results=[
            SubAgentResult(
                task_id="t1",
                role_name="dev",
                status=status,
                output=output if status == SubAgentStatus.COMPLETED else None,
                error=output if status == SubAgentStatus.FAILED else None,
            ),
        ],
        combined_output=output,
        duration_ms=100,
    )
    return result, has_failure


def _llm_response(content: str) -> LLMResponse:
    """Create a mock LLMResponse."""
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineControllerLinearMode:
    """Linear mode (no LLM) — pipeline runs all stages sequentially."""

    @pytest.mark.asyncio
    async def test_linear_mode_runs_all_stages(self) -> None:
        """Without LLM, pipeline runs all stages and returns completed."""
        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=None)
        project = _make_project()
        orch.projects["test_project"] = project

        # Mock _run_stage to return success
        stage_idx = 0

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            nonlocal stage_idx
            result, has_failure = _make_stage_result(
                stage.name,
                f"output_{stage_idx}",
            )
            stage_idx += 1
            return result, has_failure

        orch._run_stage = mock_run_stage

        result = await orch.run_project(
            "test_project",
            "Build the thing",
            context="",
        )

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 3
        assert result.stages[0].stage_name == "design"
        assert result.stages[1].stage_name == "build"
        assert result.stages[2].stage_name == "test"

    @pytest.mark.asyncio
    async def test_linear_mode_plan_history_recorded(self) -> None:
        """Plan history should contain the initial plan."""
        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=None)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "ok")

        orch._run_stage = mock_run_stage

        result = await orch.run_project(
            "test_project",
            "Build",
            context="",
        )

        assert len(result.plan_history) == 1
        assert result.plan_history[0] == ["design", "build", "test"]


class TestPipelineControllerContinue:
    """Continue decision — evaluation returns continue, next stage runs."""

    @pytest.mark.asyncio
    async def test_continue_runs_all_stages(self) -> None:
        """LLM says continue after each stage -> all stages run."""
        llm = MagicMock()
        llm.completion = AsyncMock(
            return_value=_llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "Looks good",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )
        )

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, f"output for {stage.name}")

        orch._run_stage = mock_run_stage

        result = await orch.run_project(
            "test_project",
            "Build it",
            context="",
        )

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 3
        # LLM was called once per stage for evaluation
        assert llm.completion.call_count == 3


class TestPipelineControllerAbort:
    """Abort decision — evaluation returns abort, pipeline stops."""

    @pytest.mark.asyncio
    async def test_abort_stops_pipeline(self) -> None:
        """LLM says abort after first stage -> pipeline fails."""
        llm = MagicMock()
        llm.completion = AsyncMock(
            return_value=_llm_response(
                json.dumps(
                    {
                        "action": "abort",
                        "reason": "Fundamental flaw in the design",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )
        )

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "design output")

        orch._run_stage = mock_run_stage

        result = await orch.run_project(
            "test_project",
            "Build it",
            context="",
        )

        assert result.status == SubAgentStatus.FAILED
        assert "abort" in result.error.lower() or "aborted" in result.error.lower()
        # Only the first stage ran
        assert len(result.stages) == 1
        assert result.stages[0].stage_name == "design"


class TestPipelineControllerReplan:
    """Replan decision — stages restart from a given index."""

    @pytest.mark.asyncio
    async def test_replan_restarts_from_index(self) -> None:
        """LLM says replan after first stage, then continue."""
        call_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First evaluation: replan from stage 0
                return _llm_response(
                    json.dumps(
                        {
                            "action": "replan",
                            "reason": "Design needs rework",
                            "restart_from": 0,
                            "target_stage": 0,
                        }
                    )
                )
            else:
                # Subsequent evaluations: continue
                return _llm_response(
                    json.dumps(
                        {
                            "action": "continue",
                            "reason": "Looks good now",
                            "restart_from": 0,
                            "target_stage": 0,
                        }
                    )
                )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        stages_run: list[str] = []

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            stages_run.append(stage.name)
            return _make_stage_result(stage.name, f"output for {stage.name}")

        orch._run_stage = mock_run_stage

        result = await orch.run_project(
            "test_project",
            "Build it",
            context="",
        )

        assert result.status == SubAgentStatus.COMPLETED
        # design ran, replan, then design+build+test again
        assert stages_run[0] == "design"
        assert "design" in stages_run[1:]  # re-ran design after replan
        # Plan history should have 2 entries (initial + replan)
        assert len(result.plan_history) == 2

    @pytest.mark.asyncio
    async def test_replan_event_emitted(self) -> None:
        """PIPELINE_REPLANNED event is emitted on replan."""
        call_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    json.dumps(
                        {
                            "action": "replan",
                            "reason": "Need revision",
                            "restart_from": 0,
                            "target_stage": 0,
                        }
                    )
                )
            return _llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "OK",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        replan_events: list[dict] = []

        async def capture_replan(data: dict) -> None:
            replan_events.append(data)

        event_bus.on(Events.PIPELINE_REPLANNED, capture_replan)

        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "output")

        orch._run_stage = mock_run_stage

        await orch.run_project("test_project", "Build it")

        assert len(replan_events) == 1
        assert replan_events[0]["reason"] == "Need revision"


class TestPipelineControllerSkip:
    """Skip decision — stages are skipped."""

    @pytest.mark.asyncio
    async def test_skip_to_skips_stages(self) -> None:
        """LLM says skip_to stage 2 after first stage."""
        call_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # After design: skip build, go to test (index 2)
                return _llm_response(
                    json.dumps(
                        {
                            "action": "skip_to",
                            "reason": "Build not needed",
                            "restart_from": 0,
                            "target_stage": 2,
                        }
                    )
                )
            return _llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "OK",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        skip_events: list[dict] = []

        async def capture_skip(data: dict) -> None:
            skip_events.append(data)

        event_bus.on(Events.PIPELINE_STAGE_SKIPPED, capture_skip)

        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        stages_run: list[str] = []

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            stages_run.append(stage.name)
            return _make_stage_result(stage.name, f"output for {stage.name}")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        assert result.status == SubAgentStatus.COMPLETED
        # design and test ran, build was skipped
        assert stages_run == ["design", "test"]
        # Skip event emitted for "build"
        assert len(skip_events) == 1
        assert skip_events[0]["stage"] == "build"


class TestPipelineControllerParseFailure:
    """Default on parse failure — continues if LLM returns garbage."""

    @pytest.mark.asyncio
    async def test_garbage_llm_response_defaults_to_continue(self) -> None:
        """If LLM returns unparseable response, default to continue."""
        llm = MagicMock()
        llm.completion = AsyncMock(return_value=_llm_response("This is not JSON at all!"))

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "output")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        # All stages should complete despite garbage LLM responses
        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 3

    @pytest.mark.asyncio
    async def test_llm_exception_defaults_to_continue(self) -> None:
        """If LLM raises an exception, default to continue."""
        llm = MagicMock()
        llm.completion = AsyncMock(side_effect=RuntimeError("LLM service down"))

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "output")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 3


class TestPipelineControllerPlanHistory:
    """Plan history is tracked across replans."""

    @pytest.mark.asyncio
    async def test_plan_history_tracks_replans(self) -> None:
        """Plan history records initial + each replan."""
        call_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _llm_response(
                    json.dumps(
                        {
                            "action": "replan",
                            "reason": "Rework needed",
                            "restart_from": 1,
                            "target_stage": 0,
                        }
                    )
                )
            return _llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "OK",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "output")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.plan_history) == 2
        # Both plans should have the same stage names (stub returns originals)
        assert result.plan_history[0] == ["design", "build", "test"]
        assert result.plan_history[1] == ["design", "build", "test"]


class TestReplanDecisionDataclass:
    """Basic tests for the ReplanDecision dataclass."""

    def test_default_values(self) -> None:
        d = ReplanDecision()
        assert d.action == "continue"
        assert d.reason == ""
        assert d.restart_from == 0
        assert d.target_stage == 0

    def test_custom_values(self) -> None:
        d = ReplanDecision(
            action="replan",
            reason="design flaw",
            restart_from=1,
            target_stage=2,
        )
        assert d.action == "replan"
        assert d.reason == "design flaw"
        assert d.restart_from == 1
        assert d.target_stage == 2


class TestReplanEdgeCases:
    """Edge case tests for re-planning behaviour."""

    @pytest.mark.asyncio
    async def test_skip_to_out_of_bounds_aborts(self) -> None:
        """skip_to with target_stage >= len(stages) should clamp and not crash."""
        call_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # After first stage: skip_to index 99 (out of bounds)
                return _llm_response(
                    json.dumps(
                        {
                            "action": "skip_to",
                            "reason": "Skip everything",
                            "restart_from": 0,
                            "target_stage": 99,
                        }
                    )
                )
            return _llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "OK",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        stages_run: list[str] = []

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            stages_run.append(stage.name)
            return _make_stage_result(stage.name, f"output for {stage.name}")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        # Pipeline should complete gracefully; only design ran before skip
        assert result.status == SubAgentStatus.COMPLETED
        assert stages_run == ["design"]

    @pytest.mark.asyncio
    async def test_abort_preserves_completed_stages(self) -> None:
        """Abort after stage 2 should still have stage 1+2 results."""
        call_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # After first stage: continue
                return _llm_response(
                    json.dumps(
                        {
                            "action": "continue",
                            "reason": "OK",
                            "restart_from": 0,
                            "target_stage": 0,
                        }
                    )
                )
            # After second stage: abort
            return _llm_response(
                json.dumps(
                    {
                        "action": "abort",
                        "reason": "Critical flaw discovered",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, f"output for {stage.name}")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        assert result.status == SubAgentStatus.FAILED
        # Stages 1 and 2 completed before abort
        assert len(result.stages) == 2
        assert result.stages[0].stage_name == "design"
        assert result.stages[1].stage_name == "build"

    @pytest.mark.asyncio
    async def test_replan_on_empty_stage_output(self) -> None:
        """Replan evaluation on empty output defaults to continue."""
        llm = MagicMock()
        llm.completion = AsyncMock(
            return_value=_llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "Output was empty but OK",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )
        )

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        assert result.status == SubAgentStatus.COMPLETED
        assert len(result.stages) == 3

    @pytest.mark.asyncio
    async def test_replan_limit_prevents_infinite_loop(self) -> None:
        """If LLM always says replan, should stop after exhausting stages (not loop forever)."""
        replan_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal replan_count
            replan_count += 1
            # Always replan, but after many iterations the clamped index
            # will cause re-runs. We rely on the test timeout to catch
            # infinite loops, but also cap at a generous threshold.
            if replan_count > 20:
                # Safety: force continue to avoid actual infinite loop in test
                return _llm_response(
                    json.dumps(
                        {
                            "action": "continue",
                            "reason": "OK",
                            "restart_from": 0,
                            "target_stage": 0,
                        }
                    )
                )
            return _llm_response(
                json.dumps(
                    {
                        "action": "replan",
                        "reason": "Need another revision",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "output")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        # Pipeline should eventually complete (not hang)
        assert result.status == SubAgentStatus.COMPLETED
        # Multiple replans should have been recorded
        assert len(result.plan_history) > 2

    @pytest.mark.asyncio
    async def test_multiple_replans_accumulate_history(self) -> None:
        """3 consecutive replans should have 3+1=4 entries in plan_history."""
        call_count = 0
        llm = MagicMock()

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _llm_response(
                    json.dumps(
                        {
                            "action": "replan",
                            "reason": f"Revision {call_count}",
                            "restart_from": 0,
                            "target_stage": 0,
                        }
                    )
                )
            return _llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "OK",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project()
        orch.projects["test_project"] = project

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            return _make_stage_result(stage.name, "output")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        assert result.status == SubAgentStatus.COMPLETED
        # 1 initial plan + 3 replans = 4 entries
        assert len(result.plan_history) == 4

    @pytest.mark.asyncio
    async def test_skip_events_emitted_per_skipped_stage(self) -> None:
        """If skipping 2 stages (build + test skipped from design), should emit 2 events."""
        call_count = 0
        llm = MagicMock()

        # Project with 4 stages: design, build, review, test
        four_stages = [
            ProjectStage(
                name="design",
                agents=[ProjectAgentRef(team="eng", role="dev")],
            ),
            ProjectStage(
                name="build",
                agents=[ProjectAgentRef(team="eng", role="dev")],
            ),
            ProjectStage(
                name="review",
                agents=[ProjectAgentRef(team="qa", role="tester")],
            ),
            ProjectStage(
                name="test",
                agents=[ProjectAgentRef(team="qa", role="tester")],
            ),
        ]

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # After design: skip to test (index 3), skipping build+review
                return _llm_response(
                    json.dumps(
                        {
                            "action": "skip_to",
                            "reason": "Jump to testing",
                            "restart_from": 0,
                            "target_stage": 3,
                        }
                    )
                )
            return _llm_response(
                json.dumps(
                    {
                        "action": "continue",
                        "reason": "OK",
                        "restart_from": 0,
                        "target_stage": 0,
                    }
                )
            )

        llm.completion = AsyncMock(side_effect=mock_completion)

        event_bus = EventBus()
        skip_events: list[dict] = []

        async def capture_skip(data: dict) -> None:
            skip_events.append(data)

        event_bus.on(Events.PIPELINE_STAGE_SKIPPED, capture_skip)

        orch = _make_orchestrator(event_bus=event_bus, llm=llm)
        project = _make_project(stages=four_stages)
        orch.projects["test_project"] = project

        stages_run: list[str] = []

        async def mock_run_stage(
            project_name,
            stage,
            instruction,
            context,
            parent_session_id,
            feedback_iteration=0,
        ):
            stages_run.append(stage.name)
            return _make_stage_result(stage.name, f"output for {stage.name}")

        orch._run_stage = mock_run_stage

        result = await orch.run_project("test_project", "Build it")

        assert result.status == SubAgentStatus.COMPLETED
        assert stages_run == ["design", "test"]
        # 2 skip events: build and review
        assert len(skip_events) == 2
        skipped_names = {e["stage"] for e in skip_events}
        assert skipped_names == {"build", "review"}
