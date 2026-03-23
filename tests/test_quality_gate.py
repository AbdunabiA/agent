"""Tests for the quality gate evaluation system."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.events import EventBus, Events
from agent.core.quality_gate import _ISSUE_ROLE_MAP, QualityGate
from agent.core.session import TokenUsage
from agent.core.subagent import (
    QualityIssue,
    QualityReport,
)
from agent.llm.provider import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(content: str) -> LLMResponse:
    """Build an LLMResponse with the given content."""
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(10, 20, 30),
    )


def _make_quality_json(
    passed: bool = True,
    confidence: float = 0.9,
    issues: list[dict] | None = None,
    summary: str = "Looks good",
    recommended_action: str = "proceed",
) -> str:
    """Build a JSON string matching the quality gate prompt format."""
    return json.dumps(
        {
            "passed": passed,
            "confidence": confidence,
            "issues": issues or [],
            "summary": summary,
            "recommended_action": recommended_action,
        }
    )


def _make_mock_llm(response_content: str) -> MagicMock:
    """Create a mock LLMProvider that returns the given content."""
    llm = MagicMock()
    llm.completion = AsyncMock(
        return_value=_make_llm_response(response_content),
    )
    return llm


# ---------------------------------------------------------------------------
# QualityGate.evaluate
# ---------------------------------------------------------------------------


class TestQualityGateEvaluate:
    """Tests for QualityGate.evaluate()."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_quality_report_with_issues(self) -> None:
        """evaluate() should parse LLM JSON into a QualityReport with issues."""
        issues = [
            {
                "severity": "major",
                "category": "security",
                "description": "SQL injection risk",
                "suggested_fix": "Use parameterized queries",
            },
            {
                "severity": "minor",
                "category": "style",
                "description": "Inconsistent naming",
                "suggested_fix": "Use snake_case",
            },
        ]
        response = _make_quality_json(
            passed=False,
            confidence=0.85,
            issues=issues,
            summary="Security issue found",
            recommended_action="fix_and_retry",
        )
        llm = _make_mock_llm(response)
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate(
            stage_name="review",
            stage_output="some output",
            instruction="build a web app",
        )

        assert isinstance(report, QualityReport)
        assert report.passed is False
        assert report.confidence == 0.85
        assert len(report.issues) == 2
        assert report.issues[0].severity == "major"
        assert report.issues[0].category == "security"
        assert report.issues[0].description == "SQL injection risk"
        assert report.issues[1].severity == "minor"
        assert report.summary == "Security issue found"
        assert report.recommended_action == "fix_and_retry"

    @pytest.mark.asyncio
    async def test_evaluate_passing_report(self) -> None:
        """evaluate() should return a passing report when LLM says passed."""
        response = _make_quality_json(passed=True, summary="All good")
        llm = _make_mock_llm(response)
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate(
            stage_name="test",
            stage_output="all tests passed",
            instruction="run tests",
        )

        assert report.passed is True
        assert report.issues == []
        assert report.summary == "All good"

    @pytest.mark.asyncio
    async def test_evaluate_defaults_to_passed_on_llm_failure(self) -> None:
        """evaluate() should default to passed=True if the LLM call fails."""
        llm = MagicMock()
        llm.completion = AsyncMock(side_effect=RuntimeError("LLM down"))
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate(
            stage_name="review",
            stage_output="output",
            instruction="build app",
        )

        assert report.passed is True
        assert "Evaluation error" in report.summary

    @pytest.mark.asyncio
    async def test_evaluate_defaults_to_passed_on_parse_failure(self) -> None:
        """evaluate() should default to passed=True if JSON parsing fails."""
        llm = _make_mock_llm("This is not JSON at all.")
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate(
            stage_name="review",
            stage_output="output",
            instruction="build app",
        )

        assert report.passed is True
        assert "Could not parse" in report.summary

    @pytest.mark.asyncio
    async def test_evaluate_handles_markdown_fenced_json(self) -> None:
        """evaluate() should strip markdown code fences from JSON response."""
        raw = _make_quality_json(passed=False, summary="Issue found")
        fenced = f"```json\n{raw}\n```"
        llm = _make_mock_llm(fenced)
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate(
            stage_name="review",
            stage_output="output",
            instruction="build app",
        )

        assert report.passed is False
        assert report.summary == "Issue found"


# ---------------------------------------------------------------------------
# QualityGate events
# ---------------------------------------------------------------------------


class TestQualityGateEvents:
    """Tests for QUALITY_GATE_PASSED/FAILED event emission."""

    @pytest.mark.asyncio
    async def test_emits_passed_event_on_pass(self) -> None:
        """QUALITY_GATE_PASSED should be emitted when report.passed is True."""
        response = _make_quality_json(passed=True, summary="All good")
        llm = _make_mock_llm(response)
        bus = EventBus()
        received: list[dict] = []
        bus.on(Events.QUALITY_GATE_PASSED, AsyncMock(side_effect=lambda d: received.append(d)))

        gate = QualityGate(llm=llm, event_bus=bus)
        await gate.evaluate("stage1", "output", "instruction")

        assert len(received) == 1
        assert received[0]["stage"] == "stage1"
        assert received[0]["summary"] == "All good"

    @pytest.mark.asyncio
    async def test_emits_failed_event_on_fail(self) -> None:
        """QUALITY_GATE_FAILED should be emitted when report.passed is False."""
        issues = [
            {
                "severity": "critical",
                "category": "security",
                "description": "vuln",
                "suggested_fix": "fix",
            }
        ]
        response = _make_quality_json(
            passed=False,
            summary="Failed",
            issues=issues,
            recommended_action="abort",
        )
        llm = _make_mock_llm(response)
        bus = EventBus()
        received: list[dict] = []
        bus.on(Events.QUALITY_GATE_FAILED, AsyncMock(side_effect=lambda d: received.append(d)))

        gate = QualityGate(llm=llm, event_bus=bus)
        await gate.evaluate("stage1", "output", "instruction")

        assert len(received) == 1
        assert received[0]["stage"] == "stage1"
        assert received[0]["issues"] == 1
        assert received[0]["recommended_action"] == "abort"

    @pytest.mark.asyncio
    async def test_no_failed_event_on_pass(self) -> None:
        """QUALITY_GATE_FAILED should NOT be emitted when report passes."""
        response = _make_quality_json(passed=True)
        llm = _make_mock_llm(response)
        bus = EventBus()
        failed_events: list = []
        bus.on(Events.QUALITY_GATE_FAILED, AsyncMock(side_effect=lambda d: failed_events.append(d)))

        gate = QualityGate(llm=llm, event_bus=bus)
        await gate.evaluate("stage1", "output", "instruction")

        assert len(failed_events) == 0


# ---------------------------------------------------------------------------
# QualityGate.route_issues
# ---------------------------------------------------------------------------


class TestQualityGateRouteIssues:
    """Tests for QualityGate.route_issues()."""

    @pytest.mark.asyncio
    async def test_route_issues_creates_correct_tickets(self) -> None:
        """route_issues() should create a task board ticket per issue."""
        report = QualityReport(
            passed=False,
            issues=[
                QualityIssue(severity="critical", category="security", description="SQL injection"),
                QualityIssue(severity="major", category="performance", description="Slow query"),
                QualityIssue(severity="minor", category="correctness", description="Off-by-one"),
                QualityIssue(severity="suggestion", category="style", description="Naming"),
            ],
        )

        task_board = AsyncMock()
        task_board.post_task = AsyncMock(side_effect=["tkt-1", "tkt-2", "tkt-3", "tkt-4"])
        bus = EventBus()
        llm = _make_mock_llm("")  # not used for route_issues

        gate = QualityGate(llm=llm, event_bus=bus)
        ticket_ids = await gate.route_issues(report, task_board, "task-123")

        assert ticket_ids == ["tkt-1", "tkt-2", "tkt-3", "tkt-4"]
        assert task_board.post_task.call_count == 4

        # Verify role routing
        calls = task_board.post_task.call_args_list
        assert calls[0].kwargs["to_role"] == "security_reviewer"
        assert calls[1].kwargs["to_role"] == "performance_engineer"
        assert calls[2].kwargs["to_role"] == "backend_developer"
        assert calls[3].kwargs["to_role"] == "code_reviewer"

    @pytest.mark.asyncio
    async def test_route_issues_critical_gets_blocker_priority(self) -> None:
        """Critical issues should be posted with blocker priority."""
        report = QualityReport(
            passed=False,
            issues=[
                QualityIssue(severity="critical", category="security", description="Critical bug"),
                QualityIssue(severity="minor", category="style", description="Naming issue"),
            ],
        )

        task_board = AsyncMock()
        task_board.post_task = AsyncMock(side_effect=["tkt-1", "tkt-2"])
        bus = EventBus()
        llm = _make_mock_llm("")

        gate = QualityGate(llm=llm, event_bus=bus)
        await gate.route_issues(report, task_board, "task-456")

        calls = task_board.post_task.call_args_list
        assert calls[0].kwargs["priority"] == "blocker"
        assert calls[1].kwargs["priority"] == "normal"

    @pytest.mark.asyncio
    async def test_route_issues_emits_routed_event(self) -> None:
        """QUALITY_ISSUES_ROUTED should be emitted after routing."""
        report = QualityReport(
            passed=False,
            issues=[
                QualityIssue(severity="major", category="correctness", description="Bug"),
            ],
        )

        task_board = AsyncMock()
        task_board.post_task = AsyncMock(return_value="tkt-99")
        bus = EventBus()
        routed_events: list[dict] = []
        bus.on(
            Events.QUALITY_ISSUES_ROUTED,
            AsyncMock(side_effect=lambda d: routed_events.append(d)),
        )
        llm = _make_mock_llm("")

        gate = QualityGate(llm=llm, event_bus=bus)
        await gate.route_issues(report, task_board, "task-789")

        assert len(routed_events) == 1
        assert routed_events[0]["task_id"] == "task-789"
        assert routed_events[0]["tickets_created"] == 1
        assert routed_events[0]["ticket_ids"] == ["tkt-99"]

    @pytest.mark.asyncio
    async def test_route_issues_no_event_when_no_issues(self) -> None:
        """No QUALITY_ISSUES_ROUTED event if the report has no issues."""
        report = QualityReport(passed=True, issues=[])

        task_board = AsyncMock()
        bus = EventBus()
        routed_events: list = []
        bus.on(
            Events.QUALITY_ISSUES_ROUTED,
            AsyncMock(side_effect=lambda d: routed_events.append(d)),
        )
        llm = _make_mock_llm("")

        gate = QualityGate(llm=llm, event_bus=bus)
        tickets = await gate.route_issues(report, task_board, "task-000")

        assert tickets == []
        assert len(routed_events) == 0
        assert task_board.post_task.call_count == 0


# ---------------------------------------------------------------------------
# _parse_response edge cases
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for QualityGate._parse_response static method."""

    def test_parse_valid_json(self) -> None:
        text = _make_quality_json(passed=False, confidence=0.7, summary="Issues")
        report = QualityGate._parse_response(text)
        assert report.passed is False
        assert report.confidence == 0.7

    def test_parse_json_embedded_in_text(self) -> None:
        text = 'Here is the evaluation:\n{"passed": false, "summary": "bad"}\nEnd.'
        report = QualityGate._parse_response(text)
        assert report.passed is False
        assert report.summary == "bad"

    def test_parse_garbage_defaults_to_passed(self) -> None:
        report = QualityGate._parse_response("totally unparseable garbage")
        assert report.passed is True
        assert "Could not parse" in report.summary

    def test_parse_with_code_fence(self) -> None:
        raw = _make_quality_json(passed=True, summary="OK")
        text = f"```json\n{raw}\n```"
        report = QualityGate._parse_response(text)
        assert report.passed is True
        assert report.summary == "OK"


# ---------------------------------------------------------------------------
# Integration: _evaluate_with_quality_gate on SubAgentOrchestrator
# ---------------------------------------------------------------------------


class TestEvaluateWithQualityGate:
    """Tests for the _evaluate_with_quality_gate orchestrator method."""

    def _make_orchestrator(
        self,
        llm: MagicMock | None = None,
        task_board: MagicMock | None = None,
    ) -> MagicMock:
        """Build a minimal mock orchestrator with the real method bound."""
        from agent.core.orchestrator._projects import _evaluate_with_quality_gate

        orch = MagicMock()
        orch.agent_loop = MagicMock()
        orch.agent_loop.llm = llm
        orch.event_bus = EventBus()
        orch.task_board = task_board

        # Bind the real async function as a method
        import types

        orch._evaluate_with_quality_gate = types.MethodType(
            _evaluate_with_quality_gate,
            orch,
        )
        return orch

    @pytest.mark.asyncio
    async def test_returns_proceed_when_no_llm(self) -> None:
        """Should return 'proceed' and a default report if no LLM available."""
        orch = self._make_orchestrator(llm=None)
        action, report = await orch._evaluate_with_quality_gate(
            stage_name="test",
            stage_output="output",
            instruction="build",
        )

        assert action == "proceed"
        assert report.passed is True

    @pytest.mark.asyncio
    async def test_returns_quality_gate_action(self) -> None:
        """Should return the recommended_action from the quality gate."""
        response = _make_quality_json(
            passed=False,
            recommended_action="fix_and_retry",
            summary="Issues found",
            issues=[
                {
                    "severity": "major",
                    "category": "correctness",
                    "description": "Bug",
                    "suggested_fix": "Fix it",
                }
            ],
        )
        llm = _make_mock_llm(response)
        orch = self._make_orchestrator(llm=llm)

        action, report = await orch._evaluate_with_quality_gate(
            stage_name="review",
            stage_output="output",
            instruction="build",
        )

        assert action == "fix_and_retry"
        assert report.passed is False
        assert len(report.issues) == 1

    @pytest.mark.asyncio
    async def test_routes_issues_to_task_board(self) -> None:
        """Should route issues to task board when available."""
        response = _make_quality_json(
            passed=False,
            issues=[
                {
                    "severity": "major",
                    "category": "security",
                    "description": "Vuln",
                    "suggested_fix": "Patch",
                }
            ],
        )
        llm = _make_mock_llm(response)
        task_board = AsyncMock()
        task_board.post_task = AsyncMock(return_value="tkt-abc")
        orch = self._make_orchestrator(llm=llm, task_board=task_board)

        await orch._evaluate_with_quality_gate(
            stage_name="review",
            stage_output="output",
            instruction="build",
            task_id="parent-task-1",
        )

        assert task_board.post_task.call_count == 1
        assert task_board.post_task.call_args.kwargs["to_role"] == "security_reviewer"

    @pytest.mark.asyncio
    async def test_no_routing_without_task_id(self) -> None:
        """Should skip routing if task_id is not provided."""
        response = _make_quality_json(
            passed=False,
            issues=[
                {
                    "severity": "minor",
                    "category": "style",
                    "description": "Naming",
                    "suggested_fix": "Rename",
                }
            ],
        )
        llm = _make_mock_llm(response)
        task_board = AsyncMock()
        orch = self._make_orchestrator(llm=llm, task_board=task_board)

        await orch._evaluate_with_quality_gate(
            stage_name="review",
            stage_output="output",
            instruction="build",
            # No task_id
        )

        assert task_board.post_task.call_count == 0


# ---------------------------------------------------------------------------
# Issue role mapping coverage
# ---------------------------------------------------------------------------


class TestIssueRoleMap:
    """Verify the _ISSUE_ROLE_MAP covers the expected categories."""

    def test_all_categories_mapped(self) -> None:
        expected = {"security", "performance", "correctness", "style"}
        assert set(_ISSUE_ROLE_MAP.keys()) == expected

    def test_security_maps_to_security_reviewer(self) -> None:
        assert _ISSUE_ROLE_MAP["security"] == "security_reviewer"

    def test_performance_maps_to_performance_engineer(self) -> None:
        assert _ISSUE_ROLE_MAP["performance"] == "performance_engineer"

    def test_correctness_maps_to_backend_developer(self) -> None:
        assert _ISSUE_ROLE_MAP["correctness"] == "backend_developer"

    def test_style_maps_to_code_reviewer(self) -> None:
        assert _ISSUE_ROLE_MAP["style"] == "code_reviewer"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestQualityGateEdgeCases:
    """Edge case tests for quality gate evaluation and routing."""

    @pytest.mark.asyncio
    async def test_empty_stage_output(self) -> None:
        """Quality gate on empty output should default to passed."""
        response = _make_quality_json(passed=True, summary="Nothing to evaluate")
        llm = _make_mock_llm(response)
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate(
            stage_name="build",
            stage_output="",
            instruction="build app",
        )

        assert report.passed is True
        # Verify the LLM was still called
        llm.completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_very_large_stage_output(self) -> None:
        """Stage output > 5000 chars should be truncated in LLM prompt."""
        response = _make_quality_json(passed=True, summary="OK")
        llm = _make_mock_llm(response)
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        large_output = "A" * 10000
        report = await gate.evaluate(
            stage_name="build",
            stage_output=large_output,
            instruction="build app",
        )

        assert report.passed is True
        # Verify the prompt sent to LLM had truncated output
        call_args = llm.completion.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        # The source truncates to 5000 chars: stage_output[:5000]
        assert "A" * 5000 in prompt_content
        assert "A" * 10000 not in prompt_content

    @pytest.mark.asyncio
    async def test_unknown_issue_category_uses_default_role(self) -> None:
        """Issue with unknown category should route to backend_developer."""
        report = QualityReport(
            passed=False,
            issues=[
                QualityIssue(
                    severity="major",
                    category="unknown_category",
                    description="Something odd",
                ),
            ],
        )

        task_board = AsyncMock()
        task_board.post_task = AsyncMock(return_value="tkt-1")
        bus = EventBus()
        llm = _make_mock_llm("")

        gate = QualityGate(llm=llm, event_bus=bus)
        ticket_ids = await gate.route_issues(report, task_board, "task-unk")

        assert ticket_ids == ["tkt-1"]
        call_args = task_board.post_task.call_args
        assert call_args.kwargs["to_role"] == "backend_developer"

    @pytest.mark.asyncio
    async def test_passed_false_with_zero_issues(self) -> None:
        """Report with passed=False but no issues should still emit FAILED event."""
        response = _make_quality_json(
            passed=False,
            issues=[],
            summary="Failed but no specific issues",
            recommended_action="redesign",
        )
        llm = _make_mock_llm(response)
        bus = EventBus()
        failed_events: list[dict] = []
        bus.on(
            Events.QUALITY_GATE_FAILED,
            AsyncMock(side_effect=lambda d: failed_events.append(d)),
        )

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate("review", "output", "instruction")

        assert report.passed is False
        assert report.issues == []
        assert len(failed_events) == 1
        assert failed_events[0]["issues"] == 0
        assert failed_events[0]["recommended_action"] == "redesign"

    @pytest.mark.asyncio
    async def test_route_issues_without_task_board(self) -> None:
        """If task_board methods raise, route_issues should propagate but not crash internally."""
        report = QualityReport(
            passed=False,
            issues=[
                QualityIssue(
                    severity="minor",
                    category="style",
                    description="Naming issue",
                ),
            ],
        )

        # A task_board that raises on post_task
        task_board = AsyncMock()
        task_board.post_task = AsyncMock(side_effect=RuntimeError("board unavailable"))
        bus = EventBus()
        llm = _make_mock_llm("")

        gate = QualityGate(llm=llm, event_bus=bus)

        with pytest.raises(RuntimeError, match="board unavailable"):
            await gate.route_issues(report, task_board, "task-err")

    @pytest.mark.asyncio
    async def test_multiple_issues_different_categories(self) -> None:
        """Multiple issues route to different roles correctly."""
        report = QualityReport(
            passed=False,
            issues=[
                QualityIssue(severity="critical", category="security", description="XSS"),
                QualityIssue(severity="major", category="performance", description="Slow"),
                QualityIssue(severity="minor", category="correctness", description="Bug"),
                QualityIssue(severity="suggestion", category="style", description="Format"),
            ],
        )

        task_board = AsyncMock()
        task_board.post_task = AsyncMock(
            side_effect=["tkt-1", "tkt-2", "tkt-3", "tkt-4"],
        )
        bus = EventBus()
        llm = _make_mock_llm("")

        gate = QualityGate(llm=llm, event_bus=bus)
        ticket_ids = await gate.route_issues(report, task_board, "task-multi")

        assert len(ticket_ids) == 4
        calls = task_board.post_task.call_args_list
        roles_assigned = [c.kwargs["to_role"] for c in calls]
        assert roles_assigned == [
            "security_reviewer",
            "performance_engineer",
            "backend_developer",
            "code_reviewer",
        ]

    @pytest.mark.asyncio
    async def test_confidence_value_in_report(self) -> None:
        """Confidence value (0.0-1.0) should be preserved in report."""
        response = _make_quality_json(
            passed=True,
            confidence=0.42,
            summary="Uncertain",
        )
        llm = _make_mock_llm(response)
        bus = EventBus()

        gate = QualityGate(llm=llm, event_bus=bus)
        report = await gate.evaluate("test", "output", "instruction")

        assert report.confidence == 0.42
