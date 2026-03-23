"""Planning engine for complex multi-step tasks.

Detects when a request needs planning (vs simple single-step),
generates a plan via LLM, and tracks execution progress.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.config import AgentPersonaConfig
    from agent.core.session import Session
    from agent.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class PlanStatus(StrEnum):
    """Status of a plan or plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


@dataclass
class PlanStep:
    """A single step in a plan."""

    index: int
    description: str
    status: PlanStatus = PlanStatus.PENDING
    result: str | None = None
    error: str | None = None


@dataclass
class Plan:
    """An execution plan for a complex task."""

    goal: str
    steps: list[PlanStep]
    status: PlanStatus = PlanStatus.PENDING
    current_step: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def progress(self) -> str:
        """Return progress string like '[3/7]'."""
        completed = sum(1 for s in self.steps if s.status == PlanStatus.COMPLETED)
        return f"[{completed}/{len(self.steps)}]"

    def to_context_string(self) -> str:
        """Format the plan for injection into LLM context.

        Returns a formatted string showing the plan with status indicators.
        """
        lines = [f"Current Plan: {self.goal}", f"{self.progress} Steps:"]

        for step in self.steps:
            if step.status == PlanStatus.COMPLETED:
                icon = "[DONE]"
            elif step.status == PlanStatus.IN_PROGRESS:
                icon = "[CURRENT]"
            elif step.status == PlanStatus.FAILED:
                icon = "[FAILED]"
            else:
                icon = "[PENDING]"

            line = f"  {icon} {step.index}. {step.description}"
            if step.status == PlanStatus.IN_PROGRESS:
                line += "  <-- CURRENT"
            lines.append(line)

        return "\n".join(lines)


# Heuristic patterns that suggest a request needs planning
_PLAN_PATTERNS = [
    r"\band\s+then\b",
    r"\bstep\s*\d",
    r"\bfirst\b.*\bthen\b",
    r"\bset\s+up\b",
    r"\bcreate\s+a\s+project\b",
    r"\bdeploy\b",
    r"\bmigrate\b",
    r"\binstall\s+and\s+configure\b",
    r"\bbuild\s+and\b",
    r"\bfollow\s+these\s+steps\b",
    r"^\d+\.\s",  # Numbered list
]

# Patterns that suggest a simple request (no planning needed)
_SIMPLE_PATTERNS = [
    r"^(what|how|why|when|where|who|which)\s",  # Questions
    r"^(list|show|display|print|tell|explain)\s",  # Simple queries
    r"^(hi|hello|hey|thanks|thank|ok|okay)\b",  # Greetings
]


class Planner:
    """Planning engine for complex multi-step tasks.

    Detects when a request needs planning (vs simple single-step),
    generates a plan via LLM, and tracks execution progress.
    """

    def __init__(self, llm: LLMProvider | None, config: AgentPersonaConfig) -> None:
        self.llm = llm
        self.config = config
        self._active_plan: Plan | None = None
        self._plan_patterns = [re.compile(p, re.IGNORECASE) for p in _PLAN_PATTERNS]
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]

    async def should_plan(self, user_message: str, session: Session) -> bool:
        """Determine if this request needs a plan.

        Uses heuristics: messages with multiple action verbs, numbered lists,
        setup/deploy/migrate keywords -> likely needs plan.
        Simple questions or single-step tasks -> no plan needed.

        Args:
            user_message: The user's message.
            session: Current conversation session.

        Returns:
            True if planning is recommended.
        """
        # Check simple patterns first (no plan needed)
        for pattern in self._simple_patterns:
            if pattern.search(user_message):
                return False

        # Short messages rarely need planning
        if len(user_message.split()) < 5:
            return False

        # Check planning patterns
        matches = sum(1 for p in self._plan_patterns if p.search(user_message))
        if matches >= 1:
            return True

        # Count action verbs as a signal
        action_verbs = [
            "create",
            "write",
            "build",
            "setup",
            "install",
            "configure",
            "deploy",
            "run",
            "test",
            "update",
            "delete",
            "move",
            "copy",
        ]
        verb_count = sum(1 for verb in action_verbs if verb in user_message.lower())
        return verb_count >= 2

    async def create_plan(self, goal: str, session: Session) -> Plan:
        """Ask the LLM to create an execution plan.

        Args:
            goal: The goal to plan for.
            session: Current conversation session for context.

        Returns:
            A Plan with numbered PlanSteps.
        """
        planning_prompt = (
            "You are a planning assistant. Break this goal into numbered steps. "
            "Each step should be a single, concrete action. "
            "Return ONLY a numbered list, no other text.\n\n"
            f"Goal: {goal}"
        )

        # Include some session context if available
        history = session.get_history(max_messages=5)
        context_messages = [
            {"role": "system", "content": planning_prompt},
        ]
        if history:
            context_messages.append(
                {
                    "role": "user",
                    "content": f"Context from conversation:\n{history[-1].get('content', '')}",
                }
            )
        context_messages.append({"role": "user", "content": goal})

        if self.llm is not None:
            response = await self.llm.completion(
                messages=context_messages,
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.content
        else:
            from agent.llm.fallback import llm_complete

            system_prompt = ""
            prompt_parts = []
            for msg in context_messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    prompt_parts.append(msg["content"])
            content = await llm_complete(
                prompt="\n\n".join(prompt_parts),
                system=system_prompt,
                temperature=0.3,
                max_tokens=1024,
            )

        # Parse the numbered list
        steps = self._parse_steps(content)

        plan = Plan(
            goal=goal,
            steps=steps,
            status=PlanStatus.IN_PROGRESS,
        )
        self._active_plan = plan

        logger.info("plan_created", goal=goal, step_count=len(steps))
        return plan

    async def replan(self, plan: Plan, error: str) -> Plan:
        """Replan when a step fails.

        Provides the LLM with the original plan, progress, and error,
        then asks for an updated plan from the failed step onward.

        Args:
            plan: The current plan that failed.
            error: The error that occurred.

        Returns:
            Updated Plan with revised steps.
        """
        replan_prompt = (
            "A step in the plan failed. Update the remaining steps.\n\n"
            f"Original plan:\n{plan.to_context_string()}\n\n"
            f"Error at step {plan.current_step + 1}: {error}\n\n"
            "Provide updated numbered steps for the remaining work. "
            "Return ONLY a numbered list, no other text."
        )

        if self.llm is not None:
            response = await self.llm.completion(
                messages=[
                    {"role": "system", "content": "You are a planning assistant."},
                    {"role": "user", "content": replan_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.content
        else:
            from agent.llm.fallback import llm_complete

            content = await llm_complete(
                prompt=replan_prompt,
                system="You are a planning assistant.",
                temperature=0.3,
                max_tokens=1024,
            )

        new_steps = self._parse_steps(content)

        # Preserve completed steps, replace remaining
        completed_steps = [s for s in plan.steps if s.status == PlanStatus.COMPLETED]
        for i, step in enumerate(new_steps):
            step.index = len(completed_steps) + i + 1

        updated_plan = Plan(
            goal=plan.goal,
            steps=completed_steps + new_steps,
            status=PlanStatus.IN_PROGRESS,
            current_step=len(completed_steps),
        )
        self._active_plan = updated_plan

        logger.info("plan_replanned", goal=plan.goal, new_step_count=len(new_steps))
        return updated_plan

    def get_active_plan(self) -> Plan | None:
        """Get the currently active plan, if any.

        Returns:
            The active Plan or None.
        """
        return self._active_plan

    def clear_plan(self) -> None:
        """Clear the active plan."""
        self._active_plan = None

    def _parse_steps(self, text: str) -> list[PlanStep]:
        """Parse numbered steps from LLM text output.

        Args:
            text: LLM response with numbered steps.

        Returns:
            List of PlanStep objects.
        """
        steps: list[PlanStep] = []
        lines = text.strip().split("\n")

        step_pattern = re.compile(r"^\s*\d+[\.\)]\s*(.+)")

        for line in lines:
            match = step_pattern.match(line.strip())
            if match:
                description = match.group(1).strip()
                if description:
                    steps.append(
                        PlanStep(
                            index=len(steps) + 1,
                            description=description,
                        )
                    )

        # If no numbered steps found, split by newlines
        if not steps:
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith("#"):
                    steps.append(PlanStep(index=i + 1, description=line))

        return steps
