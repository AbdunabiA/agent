"""Context window manager.

Assembles LLM messages that fit within a model's context window.
Injects memory context (facts + vector results) into the system prompt.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.core.session import Session
    from agent.memory.models import Fact
    from agent.memory.vectors import VectorResult

logger = structlog.get_logger(__name__)

# Token limits for known model families (by prefix)
MODEL_LIMITS: dict[str, int] = {
    "claude-3-5": 200_000,
    "claude-3": 200_000,
    "claude-sonnet": 200_000,
    "claude-opus": 200_000,
    "claude-haiku": 200_000,
    "claude": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_385,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-pro": 32_000,
    "gemini": 32_000,
    "o1": 128_000,
    "o3": 200_000,
}

# Default limit for unknown models (e.g. Ollama local models)
_DEFAULT_LIMIT = 8_000

# Tokens reserved for the model's response
_RESPONSE_RESERVATION = 4_096


def get_model_limit(model: str) -> int:
    """Get the context window token limit for a model.

    Matches model names by longest prefix first.

    Args:
        model: Model identifier string (e.g. 'claude-3-5-sonnet-20241022').

    Returns:
        Token limit for the model.
    """
    model_lower = model.lower()
    # Sort prefixes by length descending so longer prefixes match first
    for prefix in sorted(MODEL_LIMITS, key=len, reverse=True):
        if model_lower.startswith(prefix):
            return MODEL_LIMITS[prefix]
    return _DEFAULT_LIMIT


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using ~4 chars per token heuristic.

    Args:
        text: Input text string.

    Returns:
        Estimated number of tokens.
    """
    return len(text) // 4


def _sanitize_tool_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove orphaned tool messages that have no matching assistant tool_call.

    Gemini (via LiteLLM) requires every tool-role message to reference a
    tool_call_id that exists in a preceding assistant message's tool_calls.
    If a session has stale/corrupted history, this strips the orphans.
    """
    # Collect all valid tool_call IDs from assistant messages
    valid_tool_call_ids: set[str] = set()
    for msg in history:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                valid_tool_call_ids.add(tc_id)

    # Filter: keep non-tool messages, and tool messages with valid IDs
    cleaned: list[dict[str, Any]] = []
    for msg in history:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id not in valid_tool_call_ids:
                logger.debug("dropped_orphan_tool_message", tool_call_id=tc_id[:60])
                continue
        cleaned.append(msg)

    return cleaned


def build_messages(
    session: Session,
    system_prompt: str,
    plan: Any | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    facts: list[Fact] | None = None,
    vector_results: list[VectorResult] | None = None,
) -> list[dict[str, Any]]:
    """Build a messages list that fits within the model's context window.

    Priority order (never trimmed → trimmed):
    1. System prompt (always included)
    2. Memory context: facts + vector results (always included if present)
    3. Plan context (always included if active)
    4. Tool schemas token cost (accounted for but passed separately)
    5. Conversation history (trimmed oldest-first to fit)

    Response reservation of 4096 tokens is always applied.

    Args:
        session: Current conversation session.
        system_prompt: The system prompt string.
        plan: Active plan object (must have .to_context_string()), or None.
        tool_schemas: Tool definitions (counted for budget but not in messages).
        model: Model identifier for context limit lookup.
        facts: Relevant facts to inject into system prompt.
        vector_results: Semantic search results to inject into system prompt.

    Returns:
        List of message dicts for the LLM API.
    """
    limit = get_model_limit(model)
    budget = limit - _RESPONSE_RESERVATION

    # Build system message
    system_content = system_prompt

    # Inject memory context
    if facts:
        facts_text = "\n".join(f"- {f.key}: {f.value}" for f in facts)
        system_content += f"\n\nKNOWN FACTS ABOUT THE USER:\n{facts_text}"

    if vector_results:
        vectors_text = "\n---\n".join(
            f"[Relevance: {int(vr.score * 100)}%] {vr.text}"
            for vr in vector_results
        )
        system_content += f"\n\nRELATED PAST CONVERSATIONS:\n{vectors_text}"

    if plan and hasattr(plan, "to_context_string"):
        system_content += f"\n\nACTIVE PLAN:\n{plan.to_context_string()}"

    system_tokens = estimate_tokens(system_content)
    budget -= system_tokens

    # Account for tool schemas token cost
    if tool_schemas:
        schemas_text = json.dumps(tool_schemas)
        budget -= estimate_tokens(schemas_text)

    # Build result starting with system message
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content}
    ]

    # Get history and trim to fit budget (keep newest messages)
    history = session.get_history()
    if not history:
        return messages

    # Sanitize: ensure every tool message has a matching assistant tool_call.
    # Gemini requires strict pairing — orphaned tool messages crash LiteLLM.
    history = _sanitize_tool_messages(history)

    # Calculate tokens for each history message
    msg_tokens: list[int] = []
    for msg in history:
        content = msg.get("content", "")
        tokens = estimate_tokens(str(content))
        # Tool calls add extra tokens
        if msg.get("tool_calls"):
            tokens += estimate_tokens(json.dumps(msg["tool_calls"]))
        msg_tokens.append(tokens)

    # Include messages from newest to oldest until budget exhausted
    total_history_tokens = sum(msg_tokens)

    if total_history_tokens <= budget:
        # All history fits
        messages.extend(history)
    else:
        # Trim oldest messages
        trimmed_history: list[dict[str, Any]] = []
        remaining_budget = budget

        for msg, tokens in zip(
            reversed(history), reversed(msg_tokens), strict=True
        ):
            if tokens <= remaining_budget:
                trimmed_history.insert(0, msg)
                remaining_budget -= tokens
            else:
                break

        messages.extend(trimmed_history)

    return messages
