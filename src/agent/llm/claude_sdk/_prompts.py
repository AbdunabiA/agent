"""System prompt building for Claude SDK service."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.llm.claude_sdk._core import ClaudeSDKService


def _compute_prompt_fingerprint(
    self: ClaudeSDKService,
    facts: list[Any] | None = None,
) -> str:
    """Hash soul + facts into a fingerprint for drift detection."""
    parts: list[str] = []
    if self.soul_loader:
        parts.append(self.soul_loader.content or "")
    if facts:
        parts.extend(f"{f.key}={f.value}" for f in facts)
    raw = "\n".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_system_prompt(
    self: ClaudeSDKService,
    facts: list[Any] | None = None,
    vector_results: list[Any] | None = None,
    channel: str | None = None,
) -> str | None:
    """Build system prompt from soul.md + memory context.

    Mirrors the format used by agent.core.context.build_messages().
    Wraps the soul content with identity override instructions so the
    agent adopts the persona defined in soul.md rather than defaulting
    to Claude/Anthropic identity.

    Args:
        facts: Known facts to inject.
        vector_results: Semantic search results to inject.
        channel: Channel name (e.g. "telegram", "webchat", "cli").
    """
    parts: list[str] = []

    # Identity override — must come first
    parts.append(
        "CRITICAL IDENTITY INSTRUCTIONS:\n"
        "You must fully adopt the persona described below. "
        "You are NOT Claude and you are NOT made by Anthropic. "
        "When asked who you are, what you are, or who made you, "
        "answer ONLY according to the persona below. "
        "Never mention Claude, Anthropic, or any other AI company "
        "as your creator or identity. You are the agent described below, "
        "running locally on the user's machine."
    )

    # Soul personality
    if self.soul_loader:
        soul_content = self.soul_loader.content
        if soul_content:
            parts.append(soul_content)

    # Inject known facts
    if facts:
        fact_lines = [f"- {f.key}: {f.value}" for f in facts]
        parts.append("KNOWN FACTS ABOUT THE USER:\n" + "\n".join(fact_lines))

    # Inject related past conversations
    if vector_results:
        vr_lines = []
        for vr in vector_results:
            score_pct = int(vr.score * 100)
            vr_lines.append(f"[Relevance: {score_pct}%] {vr.text}")
        parts.append("RELATED PAST CONVERSATIONS:\n" + "\n---\n".join(vr_lines))

    # Orchestration mandate — only when orchestration is enabled
    from agent.config import get_config

    _cfg = get_config()
    if _cfg.orchestration.enabled:
        parts.append(
            "WORK DELEGATION RULE:\n"
            "You MUST delegate all substantive work (coding, file creation, "
            "research, commands, multi-step tasks) to sub-agents. You are the "
            "orchestrator — stay free for conversation and coordination. "
            "Only handle simple conversational replies and clarifying questions yourself.\n\n"
            "CHOOSING THE RIGHT WORKFLOW:\n"
            '- Vague/broad requests ("build me an app", "create a website"): '
            "First ask the user clarifying questions (purpose, features, tech stack, "
            "scope). Once clear, use run_project build_app.\n"
            "- Bug reports: use run_project bug_fix.\n"
            "- Code review requests: use run_project code_review.\n"
            "- Complex features with clear requirements: use run_project full_feature.\n"
            "- Specific single tasks: use spawn_subagent with the right role.\n"
            "- Multiple independent tasks: use spawn_parallel_agents.\n"
            "Use list_projects to see all available pipelines.\n\n"
            "DISCOVERY:\n"
            "When a request is too vague, YOU ask clarifying questions directly — "
            "do NOT delegate discovery. Gather requirements, confirm scope, "
            "then launch the right project or sub-agent with full context."
        )

    # Runtime context (channel awareness, capabilities)
    from agent.llm.prompts import build_runtime_context

    runtime_ctx = build_runtime_context(
        channel=channel or "cli",
        model_name=self.model,
        has_memory=self.fact_store is not None,
        has_orchestration=_cfg.orchestration.enabled,
    )
    parts.append(runtime_ctx)

    return "\n\n".join(parts) if parts else None


def _build_subagent_prompt(
    self: ClaudeSDKService,
    role_persona: str,
    task_context: str = "",
    nesting_depth: int = 0,
) -> str:
    """Build a focused system prompt for sub-agents.

    Unlike the full system prompt, this omits identity override,
    orchestration mandate, memory facts, and soul.md. Sub-agents
    are workers — they should execute, not delegate.

    Args:
        role_persona: The sub-agent's role persona description.
        task_context: Optional task context to include.
        nesting_depth: How deep in the sub-agent chain this agent is.
    """
    parts: list[str] = []

    # Role persona as primary identity
    parts.append(f"ROLE:\n{role_persona}")

    # Worker instruction — prevent recursive delegation
    delegation_note = ""
    if nesting_depth >= 1:
        delegation_note = (
            f" You are a nested sub-agent (depth {nesting_depth}). "
            "You MUST NOT use consult_agent, delegate_to_specialist, "
            "spawn_subagent, or any orchestration tools."
        )
    parts.append(
        "INSTRUCTIONS:\n"
        "You are a focused worker agent. Complete the task directly using "
        "the tools available to you. Do NOT delegate work to sub-agents. "
        "Do NOT spawn additional agents. Execute the task yourself." + delegation_note
    )

    if task_context:
        parts.append(f"CONTEXT:\n{task_context}")

    return "\n\n".join(parts)
