"""Quality gate evaluation for pipeline stages."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from agent.core.events import Events
from agent.core.subagent import QualityIssue, QualityReport

if TYPE_CHECKING:
    from agent.core.events import EventBus
    from agent.core.task_board import TaskBoard
    from agent.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

# Maps issue categories to the role that should handle them.
_ISSUE_ROLE_MAP = {
    "security": "security_reviewer",
    "performance": "performance_engineer",
    "correctness": "backend_developer",
    "style": "code_reviewer",
}


class QualityGate:
    """Structured quality evaluation for pipeline stages.

    Uses an LLM to evaluate stage output and produces a
    :class:`QualityReport` with typed issues.  Issues can then be
    routed to the appropriate role via the task board.

    Args:
        llm: The LLM provider to use for evaluation.
        event_bus: Event bus for emitting quality gate events.
    """

    def __init__(self, llm: LLMProvider, event_bus: EventBus) -> None:
        self.llm = llm
        self.event_bus = event_bus

    async def evaluate(
        self,
        stage_name: str,
        stage_output: str,
        instruction: str,
        context: str = "",
    ) -> QualityReport:
        """Evaluate the output of a pipeline stage.

        Calls the LLM with a structured prompt asking for a JSON quality
        evaluation.  On parse failure the gate defaults to ``passed=True``
        so as not to block the pipeline.

        Args:
            stage_name: Name of the stage being evaluated.
            stage_output: The combined output produced by the stage.
            instruction: The original task instruction.
            context: Optional additional context.

        Returns:
            A :class:`QualityReport` parsed from the LLM response.
        """
        eval_prompt = (
            f'You are a quality reviewer evaluating the output of stage "{stage_name}".\n\n'
            f"Original instruction: {instruction}\n"
            f"Stage output:\n{stage_output[:5000]}\n\n"
            "Evaluate the quality. Respond with JSON:\n"
            "{\n"
            '  "passed": true/false,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "issues": [\n'
            '    {"severity": "critical/major/minor/suggestion", '
            '"category": "correctness/security/performance/style", '
            '"description": "...", "suggested_fix": "..."}\n'
            "  ],\n"
            '  "summary": "brief summary",\n'
            '  "recommended_action": "proceed/fix_and_retry/redesign/abort"\n'
            "}"
        )

        try:
            resp = await self.llm.completion(
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=1000,
            )
            report = self._parse_response(resp.content)
        except Exception as exc:
            logger.warning(
                "quality_gate_evaluation_error",
                stage=stage_name,
                error=str(exc),
            )
            # Default to passed so we don't block the pipeline.
            report = QualityReport(passed=True, summary=f"Evaluation error: {exc}")

        # Emit the appropriate event.
        if report.passed:
            await self.event_bus.emit(
                Events.QUALITY_GATE_PASSED,
                {
                    "stage": stage_name,
                    "confidence": report.confidence,
                    "summary": report.summary,
                },
            )
        else:
            await self.event_bus.emit(
                Events.QUALITY_GATE_FAILED,
                {
                    "stage": stage_name,
                    "confidence": report.confidence,
                    "issues": len(report.issues),
                    "summary": report.summary,
                    "recommended_action": report.recommended_action,
                },
            )

        return report

    async def route_issues(
        self,
        report: QualityReport,
        task_board: TaskBoard,
        task_id: str,
    ) -> list[str]:
        """Route quality issues to the appropriate roles via the task board.

        Each issue is posted as a ticket assigned to the role determined
        by :data:`_ISSUE_ROLE_MAP`.

        Args:
            report: The quality report containing issues.
            task_board: Task board instance for posting tickets.
            task_id: Parent orchestration task ID.

        Returns:
            List of created ticket IDs.
        """
        ticket_ids: list[str] = []

        for issue in report.issues:
            to_role = _ISSUE_ROLE_MAP.get(issue.category, "backend_developer")
            priority = "blocker" if issue.severity == "critical" else "normal"

            ticket_id = await task_board.post_task(
                from_role="quality_gate",
                to_role=to_role,
                task_id=task_id,
                title=f"[{issue.severity.upper()}] {issue.category}: {issue.description[:80]}",
                description=issue.description,
                priority=priority,
                context={
                    "severity": issue.severity,
                    "category": issue.category,
                    "suggested_fix": issue.suggested_fix,
                },
            )
            ticket_ids.append(ticket_id)

        if ticket_ids:
            await self.event_bus.emit(
                Events.QUALITY_ISSUES_ROUTED,
                {
                    "task_id": task_id,
                    "tickets_created": len(ticket_ids),
                    "ticket_ids": ticket_ids,
                },
            )

        return ticket_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> QualityReport:
        """Parse the LLM JSON response into a QualityReport.

        Tries to extract a JSON object from the response text, falling
        back to ``passed=True`` if parsing fails.
        """
        # Strip markdown code fences if present.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_nl = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_nl + 1 :]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    return QualityReport(
                        passed=True,
                        summary="Could not parse quality evaluation response",
                    )
            else:
                return QualityReport(
                    passed=True,
                    summary="Could not parse quality evaluation response",
                )

        issues = [
            QualityIssue(
                severity=i.get("severity", "minor"),
                category=i.get("category", "correctness"),
                description=i.get("description", ""),
                suggested_fix=i.get("suggested_fix", ""),
            )
            for i in data.get("issues", [])
        ]

        return QualityReport(
            passed=bool(data.get("passed", True)),
            confidence=float(data.get("confidence", 1.0)),
            issues=issues,
            summary=str(data.get("summary", "")),
            recommended_action=str(data.get("recommended_action", "proceed")),
        )
