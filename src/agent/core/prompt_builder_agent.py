"""Prompt builder agent — crafts detailed task briefs before worker spawns.

Spawns a lightweight, read-only sub-agent that reads the codebase and
WorkingMemory to produce a maximally detailed, contextual prompt for the
incoming worker.  Falls back to a static template on any failure.

The builder NEVER raises — it always returns a usable prompt string.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.subagent import SubAgentRole

if TYPE_CHECKING:
    from agent.core.role_registry import RoleRegistry
    from agent.core.working_memory import WorkingMemory
    from agent.llm.claude_sdk import ClaudeSDKService
    from agent.observability.tracer import AgentTracer

logger = structlog.get_logger(__name__)

# Default builder role if teams/internal/prompt_builder.yaml is unavailable
_DEFAULT_BUILDER_ROLE = SubAgentRole(
    name="prompt_builder",
    persona=(
        "You are a senior technical lead preparing task briefs for AI agents. "
        "Your ONLY job is to prepare — never implement. "
        "Read relevant files, understand the codebase context, then produce "
        "a structured brief with sections: Task, Why this task exists, "
        "Read these files first, What the team has done so far, "
        "Similar patterns to follow, Requirements, Acceptance criteria, "
        "Constraints, When done."
    ),
    allowed_tools=["file_read", "list_directory", "find_files", "memory_search"],
    denied_tools=[
        "file_write",
        "shell_exec",
        "python_exec",
        "report_bug",
        "assign_task",
        "ask_team",
        "consult_agent",
        "spawn_subagent",
        "spawn_team",
    ],
    max_iterations=6,
    model="claude-haiku-4-5-20251001",
)

_BUILDER_TIMEOUT = 60.0  # seconds


class PromptBuilderAgent:
    """Spawns a temporary read-only agent to craft detailed worker briefs.

    Called by the orchestrator before every worker spawn.  If the builder
    fails, times out, or produces an empty result, a static fallback
    template is returned.

    Args:
        sdk_service: The Claude SDK service for spawning the builder.
        working_memory: Shared working memory for team context.
        tracer: Agent tracer for observability spans.
        role_registry: Registry to load the builder role from YAML.
    """

    def __init__(
        self,
        sdk_service: ClaudeSDKService,
        working_memory: WorkingMemory | None = None,
        tracer: AgentTracer | None = None,
        role_registry: RoleRegistry | None = None,
    ) -> None:
        self._sdk = sdk_service
        self._working_memory = working_memory
        self._tracer = tracer
        self._builder_role = self._resolve_builder_role(role_registry)

    def _resolve_builder_role(
        self,
        registry: RoleRegistry | None,
    ) -> SubAgentRole:
        """Load the prompt_builder role from registry, falling back to default."""
        if registry is not None:
            role = registry.get_role("prompt_builder")
            if role is not None:
                return role
        return _DEFAULT_BUILDER_ROLE

    async def build_prompt(
        self,
        worker_role: SubAgentRole,
        task_id: str,
        task_description: str,
        ticket: dict[str, Any] | None = None,
    ) -> str:
        """Build a detailed prompt for a worker agent.

        Never raises — always returns a usable prompt string.

        Args:
            worker_role: The worker role that will receive the brief.
            task_id: Parent orchestration task ID.
            task_description: The task/instruction text.
            ticket: Optional ticket context from TaskBoard.

        Returns:
            Detailed prompt string to prepend to the worker's system prompt.
        """
        # 1. Get team context from WorkingMemory
        context = ""
        if self._working_memory is not None:
            try:
                context = await self._working_memory.get_context_for_role(
                    task_id,
                    worker_role.name,
                )
            except Exception as e:
                logger.warning(
                    "prompt_builder_context_failed",
                    task_id=task_id,
                    error=str(e),
                )

        # 2. Build input for builder agent
        ticket_context = json.dumps(ticket or {}, indent=2)
        builder_input = (
            f"Prepare a detailed task brief for role: {worker_role.name}\n"
            f"Available tools: {', '.join(worker_role.allowed_tools or [])}\n\n"
            f"TASK: {task_description}\n"
            f"TICKET CONTEXT: {ticket_context}\n"
            f"TEAM PROGRESS (WorkingMemory):\n"
            f"{context or 'No prior context — this may be the first agent to run.'}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Read relevant project files to understand the codebase\n"
            f"2. Write a DETAILED brief (at least 500 words) with these sections:\n"
            f"   - Task: What needs to be done\n"
            f"   - Codebase context: Key files, structure, patterns found\n"
            f"   - Requirements: Specific things to check/do\n"
            f"   - Acceptance criteria: How to know the task is done\n"
            f"3. The brief must give the worker enough context to do the "
            f"job WITHOUT needing to re-read files\n"
            f"4. Be specific — include file paths, function names, line numbers"
        )

        # 3. Spawn builder with hard timeout — never block main flow
        try:
            from agent.core.orchestrator import ScopedToolRegistry

            scoped_registry = ScopedToolRegistry(
                parent=self._sdk.tool_registry,
                allowed_tools=set(self._builder_role.allowed_tools),
                denied_tools=set(self._builder_role.denied_tools),
                exclude_dangerous=True,
            )

            result = await asyncio.wait_for(
                self._sdk.run_subagent(
                    prompt=builder_input,
                    task_id=f"{task_id}-pb",
                    role_persona=self._builder_role.persona,
                    scoped_registry=scoped_registry,
                    model=self._builder_role.model,
                    max_turns=self._builder_role.max_iterations,
                    nesting_depth=1,  # counts against nesting limit
                    denied_builtins={
                        "Write",
                        "Edit",
                        "Bash",
                        "NotebookEdit",
                    },
                ),
                timeout=_BUILDER_TIMEOUT,
            )

            if result and len(result) > 10:
                # Detect SDK errors propagated as result text
                if result.strip().startswith("[SDK_ERROR]"):
                    logger.warning(
                        "prompt_builder_sdk_error",
                        task_id=task_id,
                        role=worker_role.name,
                        error=result[:200],
                    )
                    # Fall through to return None (caller uses fallback)
                else:
                    logger.info(
                        "prompt_builder_success",
                        task_id=task_id,
                        role=worker_role.name,
                        brief_len=len(result),
                    )
                    return result

        except TimeoutError:
            logger.warning(
                "prompt_builder_timeout",
                task_id=task_id,
                role=worker_role.name,
                timeout=_BUILDER_TIMEOUT,
            )
        except Exception as e:
            logger.warning(
                "prompt_builder_failed",
                task_id=task_id,
                role=worker_role.name,
                error=str(e),
            )

        # 4. Fallback — always works, no LLM needed
        return self._fallback_template(task_description, context, ticket)

    def _fallback_template(
        self,
        description: str,
        context: str,
        ticket: dict[str, Any] | None,
    ) -> str:
        """Static fallback template when the builder agent is unavailable."""
        ticket_section = ""
        if ticket:
            ticket_section = f"\n## Ticket Context\n" f"{json.dumps(ticket, indent=2)}\n"

        return (
            f"## Task\n{description}\n"
            f"\n## Team Context\n{context or 'No prior context.'}\n"
            f"{ticket_section}"
            f"\n## When Done\n"
            f"Call complete_my_task() with a summary of what you changed."
        )
