"""Universal LLM completion with automatic backend fallback.

Provides a single async function that tries LiteLLM first,
then falls back to the Claude SDK when LiteLLM is unavailable.
All components that need LLM completions should use this instead
of calling ``self.llm.completion()`` directly.

Usage::

    from agent.llm.fallback import llm_complete

    result = await llm_complete(
        prompt="Extract facts from this text...",
        system="You are a fact extraction system.",
        temperature=0.2,
        max_tokens=1024,
    )
    print(result)  # str — the LLM's response text
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.llm.claude_sdk import ClaudeSDKService
    from agent.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

# Module-level references set during startup
_llm_provider: LLMProvider | None = None
_sdk_service: ClaudeSDKService | None = None


def configure(
    llm: LLMProvider | None = None,
    sdk_service: ClaudeSDKService | None = None,
) -> None:
    """Set the LLM backends (called once during startup).

    Args:
        llm: LiteLLM provider (preferred).
        sdk_service: Claude SDK service (fallback).
    """
    global _llm_provider, _sdk_service
    _llm_provider = llm
    _sdk_service = sdk_service
    logger.info(
        "llm_fallback_configured",
        has_litellm=llm is not None,
        has_sdk=sdk_service is not None,
    )


def is_available() -> bool:
    """Check if at least one LLM backend is configured."""
    return _llm_provider is not None or _sdk_service is not None


async def llm_complete(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Run a single LLM completion with automatic backend fallback.

    Tries LiteLLM first (fast, supports all models). If unavailable,
    falls back to Claude SDK (spawns a short-lived subprocess).

    Args:
        prompt: The user/task prompt.
        system: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        The LLM's response text.

    Raises:
        RuntimeError: If no LLM backend is available.
    """
    # Path 1: LiteLLM (preferred — fast, in-process)
    if _llm_provider is not None:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await _llm_provider.completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content

    # Path 2: Claude SDK (fallback — spawns subprocess)
    if _sdk_service is not None:
        return await _complete_via_sdk(prompt, system)

    raise RuntimeError("No LLM backend available. Configure an API key or use claude-sdk backend.")


async def _complete_via_sdk(prompt: str, system: str = "") -> str:
    """Run a completion via the Claude SDK.

    Uses run_subagent with an empty tool registry for a pure LLM call
    (no tools, no MCP server, just prompt → response).
    """
    assert _sdk_service is not None

    from agent.core.orchestrator import ScopedToolRegistry

    # Empty registry — no tools needed for pure completion
    empty_registry = ScopedToolRegistry(
        parent=_sdk_service.tool_registry,
        allowed_tools=[],
    )

    full_prompt = prompt
    role_persona = system or "You are a helpful assistant. Respond concisely."

    import uuid

    result = await _sdk_service.run_subagent(
        prompt=full_prompt,
        task_id=f"completion-{uuid.uuid4().hex[:8]}",
        role_persona=role_persona,
        scoped_registry=empty_registry,
        max_turns=1,
    )

    # Strip SDK error markers if present
    if result.startswith("[SDK_ERROR]"):
        raise RuntimeError(result)

    return result
