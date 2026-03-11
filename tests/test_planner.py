"""Tests for the planning engine."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.config import AgentPersonaConfig
from agent.core.planner import Plan, Planner, PlanStatus, PlanStep
from agent.core.session import Session


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()
    return llm


@pytest.fixture
def planner(mock_llm: AsyncMock) -> Planner:
    config = AgentPersonaConfig()
    return Planner(llm=mock_llm, config=config)


@pytest.fixture
def session() -> Session:
    return Session()


class TestShouldPlan:
    """Tests for should_plan heuristic."""

    async def test_simple_question_no_plan(
        self, planner: Planner, session: Session
    ) -> None:
        """Simple questions should not need planning."""
        result = await planner.should_plan("What time is it?", session)
        assert result is False

    async def test_greeting_no_plan(self, planner: Planner, session: Session) -> None:
        """Greetings should not need planning."""
        result = await planner.should_plan("Hello there", session)
        assert result is False

    async def test_short_message_no_plan(
        self, planner: Planner, session: Session
    ) -> None:
        """Short messages should not need planning."""
        result = await planner.should_plan("list files", session)
        assert result is False

    async def test_complex_request_needs_plan(
        self, planner: Planner, session: Session
    ) -> None:
        """Complex multi-step requests should need planning."""
        result = await planner.should_plan(
            "Create a new FastAPI project, write the main app, add tests, and then deploy it",
            session,
        )
        assert result is True

    async def test_setup_request_needs_plan(
        self, planner: Planner, session: Session
    ) -> None:
        """Setup requests should need planning."""
        result = await planner.should_plan(
            "Set up a Python project with tests and CI",
            session,
        )
        assert result is True

    async def test_and_then_pattern(
        self, planner: Planner, session: Session
    ) -> None:
        """'and then' pattern should trigger planning."""
        result = await planner.should_plan(
            "Create the config file and then run the tests",
            session,
        )
        assert result is True


class TestCreatePlan:
    """Tests for plan creation."""

    async def test_create_plan_parses_steps(
        self, planner: Planner, mock_llm: AsyncMock, session: Session
    ) -> None:
        """create_plan should parse LLM response into Plan with steps."""
        from agent.core.session import TokenUsage

        mock_llm.completion.return_value = AsyncMock(
            content="1. Create project directory\n2. Write main.py\n3. Run tests",
            usage=TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70),
            model="test-model",
            tool_calls=None,
        )

        plan = await planner.create_plan("Build a Python project", session)

        assert plan.goal == "Build a Python project"
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Create project directory"
        assert plan.steps[1].description == "Write main.py"
        assert plan.steps[2].description == "Run tests"
        assert plan.status == PlanStatus.IN_PROGRESS


class TestPlan:
    """Tests for Plan dataclass."""

    def test_progress_string(self) -> None:
        plan = Plan(
            goal="Test",
            steps=[
                PlanStep(index=1, description="Step 1", status=PlanStatus.COMPLETED),
                PlanStep(index=2, description="Step 2", status=PlanStatus.COMPLETED),
                PlanStep(index=3, description="Step 3", status=PlanStatus.PENDING),
            ],
        )
        assert plan.progress == "[2/3]"

    def test_context_string_format(self) -> None:
        plan = Plan(
            goal="Deploy project",
            steps=[
                PlanStep(index=1, description="Create dir", status=PlanStatus.COMPLETED),
                PlanStep(index=2, description="Write code", status=PlanStatus.IN_PROGRESS),
                PlanStep(index=3, description="Run tests", status=PlanStatus.PENDING),
            ],
        )
        ctx = plan.to_context_string()
        assert "Deploy project" in ctx
        assert "[DONE]" in ctx
        assert "[CURRENT]" in ctx
        assert "[PENDING]" in ctx

    def test_empty_plan(self) -> None:
        plan = Plan(goal="Empty", steps=[])
        assert plan.progress == "[0/0]"


class TestReplan:
    """Tests for replanning after failure."""

    async def test_replan_preserves_completed(
        self, planner: Planner, mock_llm: AsyncMock, session: Session
    ) -> None:
        """Replan should keep completed steps and replace remaining."""
        from agent.core.session import TokenUsage

        original_plan = Plan(
            goal="Build project",
            steps=[
                PlanStep(index=1, description="Step 1", status=PlanStatus.COMPLETED),
                PlanStep(index=2, description="Step 2", status=PlanStatus.FAILED),
                PlanStep(index=3, description="Step 3", status=PlanStatus.PENDING),
            ],
            current_step=1,
        )

        mock_llm.completion.return_value = AsyncMock(
            content="1. Fix the error\n2. Continue with step 3",
            usage=TokenUsage(input_tokens=50, output_tokens=20, total_tokens=70),
            model="test-model",
            tool_calls=None,
        )

        new_plan = await planner.replan(original_plan, "Step 2 failed with error X")

        # First step should be preserved
        assert new_plan.steps[0].status == PlanStatus.COMPLETED
        # New steps appended
        assert len(new_plan.steps) >= 2
