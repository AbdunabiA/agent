"""Claude Agent SDK integration — use Claude Code via local subscription.

Delegates to Claude Code's own agent loop using your Max plan subscription
instead of requiring API keys. The SDK reads auth from ~/.claude/.

Usage:
    sdk = ClaudeSDKService(working_dir="/path/to/project")
    async for event in sdk.run_task_stream("Fix the bug in main.py"):
        print(event.content)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.core.cost_tracker import CostTracker
    from agent.core.events import EventBus
    from agent.memory.extraction import FactExtractor
    from agent.memory.soul import SoulLoader
    from agent.memory.store import FactStore
    from agent.memory.vectors import VectorStore
    from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

# Lazy SDK availability check
_SDK_AVAILABLE: bool | None = None
_SDK_PATCHED: bool = False


def _patch_sdk_permission_protocol() -> None:
    """Patch claude-code-sdk to use the new permission response format.

    SDK v0.0.25 sends {allow: true, input: {...}} but Claude Code CLI v2.1+
    expects {behavior: "allow", updatedInput: {...}}. This monkey-patch
    fixes the protocol mismatch until the SDK is updated.
    """
    global _SDK_PATCHED
    if _SDK_PATCHED:
        return

    try:
        from claude_code_sdk._internal.query import Query
        from claude_code_sdk.types import PermissionResultAllow, PermissionResultDeny

        _original_handle = Query._handle_control_request

        async def _patched_handle_control_request(
            self: Any,
            request: dict[str, Any],
        ) -> None:
            import json as _json

            request_id = request.get("request_id", "")
            request_data = request["request"]
            subtype = request_data["subtype"]

            if subtype != "can_use_tool":
                # Delegate non-permission requests to original handler
                return await _original_handle(self, request)

            try:
                from claude_code_sdk.types import (
                    SDKControlPermissionRequest,
                    SDKControlResponse,
                    ToolPermissionContext,
                )

                permission_request: SDKControlPermissionRequest = request_data  # type: ignore[assignment]

                if not self.can_use_tool:
                    raise Exception("canUseTool callback is not provided")

                context = ToolPermissionContext(
                    signal=None,
                    suggestions=permission_request.get("permission_suggestions", []) or [],
                )

                response = await self.can_use_tool(
                    permission_request["tool_name"],
                    permission_request["input"],
                    context,
                )

                # New protocol format expected by Claude Code CLI v2.1+
                response_data: dict[str, Any]
                if isinstance(response, PermissionResultAllow):
                    response_data = {"behavior": "allow"}
                    if response.updated_input is not None:
                        response_data["updatedInput"] = response.updated_input
                elif isinstance(response, PermissionResultDeny):
                    response_data = {
                        "behavior": "deny",
                        "message": response.message or "Permission denied",
                    }
                else:
                    raise TypeError(
                        f"Tool permission callback must return PermissionResult, "
                        f"got {type(response)}"
                    )

                success_response: SDKControlResponse = {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": request_id,
                        "response": response_data,
                    },
                }
                await self.transport.write(_json.dumps(success_response) + "\n")

            except Exception as e:
                error_response: SDKControlResponse = {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": request_id,
                        "error": str(e),
                    },
                }
                await self.transport.write(_json.dumps(error_response) + "\n")

        Query._handle_control_request = _patched_handle_control_request  # type: ignore[assignment]
        _SDK_PATCHED = True
        logger.info("sdk_permission_protocol_patched")
    except Exception as e:
        logger.warning("sdk_permission_patch_failed", error=str(e))


def sdk_available() -> bool:
    """Check if claude-agent-sdk is installed."""
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is None:
        try:
            import claude_code_sdk  # noqa: F401

            _SDK_AVAILABLE = True
            _patch_sdk_permission_protocol()
        except ImportError:
            _SDK_AVAILABLE = False
    return _SDK_AVAILABLE


class SDKTaskStatus(StrEnum):
    """Status of an SDK task."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_PERMISSION = "waiting_permission"
    WAITING_ANSWER = "waiting_answer"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SDKStreamEvent:
    """An event emitted during SDK task execution."""

    type: str  # "text", "thinking", "tool_use", "tool_result", "result", "error"
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SDKTaskResult:
    """Final result from an SDK task."""

    success: bool
    output: str
    session_id: str | None = None
    cost_usd: float = 0.0
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


# Type alias for the permission callback used in CLI mode
PermissionCallback = Any  # Callable[[str, str, dict], Awaitable[bool]]
QuestionCallback = Any  # Callable[[str, list[str]], Awaitable[str]]


class ClaudeSDKService:
    """Run tasks via Claude Agent SDK using local Claude subscription.

    This service wraps the claude-agent-sdk to execute prompts through
    Claude Code's agent loop. Authentication uses your local ~/.claude/
    session — no API key required.

    Args:
        working_dir: Default working directory for Claude Code.
        max_turns: Maximum agent iterations per task.
        permission_mode: SDK permission mode (None, "acceptEdits", "bypassPermissions").
        model: Override model (e.g. "claude-sonnet-4-6").
    """

    def __init__(
        self,
        working_dir: str = ".",
        max_turns: int = 50,
        permission_mode: str | None = None,
        model: str | None = None,
        claude_auth_dir: str = "~/.claude",
        tool_registry: ToolRegistry | None = None,
        soul_loader: SoulLoader | None = None,
        fact_store: FactStore | None = None,
        vector_store: VectorStore | None = None,
        fact_extractor: FactExtractor | None = None,
        cost_tracker: CostTracker | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        if not sdk_available():
            raise ImportError(
                "claude-agent-sdk is not installed. " "Install with: pip install claude-agent-sdk"
            )
        self.working_dir = working_dir
        self.max_turns = max_turns
        self.permission_mode = permission_mode
        self.model = model
        self.claude_auth_dir = str(Path(claude_auth_dir).expanduser())
        self.tool_registry = tool_registry
        self.soul_loader = soul_loader
        self.fact_store = fact_store
        self.vector_store = vector_store
        self.fact_extractor = fact_extractor
        self.cost_tracker = cost_tracker
        self.event_bus = event_bus

        # Per-task state
        self._clients: dict[str, Any] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._permission_events: dict[str, asyncio.Event] = {}
        self._permission_responses: dict[str, bool] = {}
        self._question_events: dict[str, asyncio.Event] = {}
        self._question_responses: dict[str, str] = {}
        self._status: dict[str, SDKTaskStatus] = {}

        # Issue 3: Idle timeout tracking
        self._last_activity: dict[str, float] = {}
        self._idle_timeout: int = 1800  # overridden by config in startup
        self._reaper_task: asyncio.Task[None] | None = None

        # Track fire-and-forget background tasks for clean shutdown
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Issue 4: Resume on crash — last known session_id per task
        self._last_session_ids: dict[str, str] = {}

        # Issue 2: MCP tool generation tracking
        self._tool_generations: dict[str, int] = {}

        # Issue 1: System prompt fingerprint tracking
        self._prompt_fingerprints: dict[str, str] = {}
        self._cached_facts: dict[str, list[Any]] = {}

        # Sub-agent execution metrics (tool calls, iterations per task_id)
        self._subagent_metrics: dict[str, dict[str, int | float | None]] = {}

        # Per-task lock to serialize concurrent queries on the same client
        self._query_locks: dict[str, asyncio.Lock] = {}

    def check_available_sync(self) -> tuple[bool, str]:
        """Check if SDK is installed and Claude Code is authenticated (sync)."""
        if not sdk_available():
            return False, "claude-agent-sdk not installed"

        claude_dir = Path(self.claude_auth_dir)
        if not claude_dir.exists():
            return False, (
                f"Claude Code not authenticated ({claude_dir} not found). "
                "Run 'claude auth login' first."
            )

        return True, f"Claude Agent SDK ready (auth: {claude_dir})"

    async def check_available(self) -> tuple[bool, str]:
        """Check if SDK is installed and Claude Code is authenticated."""
        return self.check_available_sync()

    # ------------------------------------------------------------------
    # Issue 3: Idle timeout reaper
    # ------------------------------------------------------------------

    async def start_reaper(self) -> None:
        """Start background task that disconnects idle SDK clients."""
        if self._reaper_task is not None:
            return
        self._reaper_task = asyncio.create_task(self._reap_idle_clients())
        logger.info("sdk_reaper_started", idle_timeout=self._idle_timeout)

    async def stop_reaper(self) -> None:
        """Stop the idle-client reaper and cancel background tasks."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None
            logger.info("sdk_reaper_stopped")
        # Cancel any lingering background tasks (fact extraction, event emission)
        for bg in list(self._background_tasks):
            bg.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

    async def _reap_idle_clients(self) -> None:
        """Periodically disconnect clients that have been idle too long."""
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            idle_ids = [
                tid
                for tid, last in self._last_activity.items()
                if (now - last) > self._idle_timeout and tid in self._clients
            ]
            for tid in idle_ids:
                logger.info(
                    "sdk_reaping_idle_client",
                    task_id=tid,
                    idle_secs=int(now - self._last_activity[tid]),
                )
                await self.disconnect_client(tid)

    # ------------------------------------------------------------------
    # Issue 1: System prompt fingerprint — bound from _prompts.py
    # ------------------------------------------------------------------
    # _compute_prompt_fingerprint — bound below

    async def run_task_stream(
        self,
        prompt: str,
        *,
        task_id: str = "default",
        session_id: str | None = None,
        working_dir: str | None = None,
        on_permission: PermissionCallback | None = None,
        on_question: QuestionCallback | None = None,
        channel: str | None = None,
    ) -> AsyncGenerator[SDKStreamEvent, None]:
        """Run a task via SDK and stream events.

        If a session_id is provided and resumption fails, automatically
        retries with a fresh session (no resume).

        Args:
            prompt: The task/question for Claude Code.
            task_id: Unique task identifier for HITL tracking.
            session_id: Optional session ID to resume a previous conversation.
            working_dir: Override working directory for this task.
            on_permission: Async callback for tool permission requests.
                Signature: (tool_name, details, tool_input) -> bool
            on_question: Async callback for questions from Claude.
                Signature: (question_text, options) -> str
            channel: Channel name (e.g. "telegram", "webchat") for context.

        Yields:
            SDKStreamEvent objects as they occur.
        """
        # Try with resume first; on any error retry once without resume,
        # since session corruption or stale state can cause various failures.
        # Buffer events so we don't yield text from a failed attempt that
        # would get duplicated when the retry also yields text.
        if session_id is not None:
            buffered: list[SDKStreamEvent] = []
            needs_retry = False
            async for event in self._run_task_impl(
                prompt,
                task_id=task_id,
                session_id=session_id,
                working_dir=working_dir,
                on_permission=on_permission,
                on_question=on_question,
                channel=channel,
            ):
                if event.type == "error":
                    logger.warning(
                        "sdk_resume_failed_retrying",
                        task_id=task_id,
                        session_id=session_id[:16] + "...",
                        error=event.content[:120] if event.content else None,
                    )
                    needs_retry = True
                    break
                buffered.append(event)

            if needs_retry:
                # Discard buffered events from the failed attempt
                async for retry_event in self._run_task_impl(
                    prompt,
                    task_id=task_id,
                    session_id=None,
                    working_dir=working_dir,
                    on_permission=on_permission,
                    on_question=on_question,
                    channel=channel,
                ):
                    yield retry_event
            else:
                for event in buffered:
                    yield event
        else:
            async for event in self._run_task_impl(
                prompt,
                task_id=task_id,
                session_id=None,
                working_dir=working_dir,
                on_permission=on_permission,
                on_question=on_question,
                channel=channel,
            ):
                yield event

    async def run_subagent(
        self,
        prompt: str,
        *,
        task_id: str,
        role_persona: str,
        scoped_registry: Any,
        model: str | None = None,
        max_turns: int = 10,
        task_context: str = "",
        tool_executor: Any | None = None,
        nesting_depth: int = 0,
    ) -> str:
        """Run a sub-agent task via a temporary SDK client.

        Creates a short-lived ClaudeSDKClient with a scoped tool registry
        and focused sub-agent prompt. The client is disconnected after use.

        Args:
            prompt: The task instruction for the sub-agent.
            task_id: Unique task identifier.
            role_persona: The sub-agent's persona description.
            scoped_registry: Filtered tool registry for this sub-agent.
            model: Optional model override.
            max_turns: Maximum agent turns (default 10).
            task_context: Optional context to include in the prompt.
            tool_executor: Optional ToolExecutor for safety routing
                through permissions, guardrails, and audit.

        Returns:
            The sub-agent's text response.
        """
        import contextlib as _contextlib
        import os as _os
        from pathlib import Path as _Path

        from claude_code_sdk import (
            AssistantMessage,
            ClaudeCodeOptions,
            ClaudeSDKClient,
            ResultMessage,
        )
        from claude_code_sdk.types import (
            PermissionResultAllow,
            PermissionResultDeny,
        )

        logger.info(
            "sdk_subagent_starting",
            task_id=task_id,
            role=role_persona[:80],
            model=model or self.model,
        )

        # Build scoped MCP server (route through executor for audit/safety)
        mcp_servers: dict[str, Any] = {}
        mcp_result = self._build_mcp_server(
            registry=scoped_registry,
            tool_executor=tool_executor,
            session_id=f"subagent:{task_id}",
        )
        if mcp_result:
            server_config, _ = mcp_result
            mcp_servers["agent-tools"] = server_config

        # Collect safe tool names from scoped registry for auto-approval
        from agent.tools.registry import ToolTier

        agent_safe_tools: set[str] = set()
        all_agent_tools: set[str] = set()
        if scoped_registry is not None:
            for td in scoped_registry.list_tools():
                all_agent_tools.add(td.name)
                if td.enabled and td.tier == ToolTier.SAFE:
                    agent_safe_tools.add(td.name)

        # Build sub-agent system prompt
        system_prompt = self._build_subagent_prompt(
            role_persona=role_persona,
            task_context=task_context,
            nesting_depth=nesting_depth,
        )

        # Permission callback — auto-approve safe tools + Claude Code read tools
        async def can_use_tool(
            tool_name: str,
            tool_input: dict[str, Any],
            context: Any,
        ) -> Any:
            safe_tools = {
                "Read",
                "Glob",
                "Grep",
                "WebFetch",
                "WebSearch",
                "LS",
                "TaskGet",
                "TaskList",
                "TaskOutput",
                "ToolSearch",
            }
            if tool_name in safe_tools or tool_name in agent_safe_tools:
                return PermissionResultAllow(updated_input=tool_input)

            # Auto-approve all tools from scoped registry (already filtered)
            if tool_name in all_agent_tools:
                return PermissionResultAllow(updated_input=tool_input)

            return PermissionResultDeny(message="Tool not in sub-agent scope", interrupt=False)

        resolved_cwd = str(_Path(_os.path.expanduser(self.working_dir)).resolve())  # noqa: ASYNC240

        options_kwargs: dict[str, Any] = {
            "cwd": resolved_cwd,
            "max_turns": max_turns,
            "model": model or self.model,
            "permission_mode": self.permission_mode,
            "can_use_tool": can_use_tool,
            "system_prompt": system_prompt,
            "env": {
                "GIT_TERMINAL_PROMPT": "0",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            },
        }
        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers

        # Ensure conflicting env vars are not inherited by the subprocess.
        # Instead of mutating os.environ under a lock, pass a clean env dict
        # via ClaudeCodeOptions.env so parallel sub-agents can start concurrently.
        # Note: ANTHROPIC_API_KEY is NOT blanked — the subprocess needs it for auth.
        # Only blank vars that control Claude Code CLI behavior to avoid conflicts.
        _env_overrides = dict(options_kwargs.get("env", {}))
        for key in (
            "CLAUDECODE",
            "CLAUDE_CODE_SSE_PORT",
            "CLAUDE_CODE_ENTRYPOINT",
        ):
            if key in _os.environ and key not in _env_overrides:
                _env_overrides[key] = ""
        options_kwargs["env"] = _env_overrides

        options = ClaudeCodeOptions(**options_kwargs)

        client: Any = None
        try:
            client = ClaudeSDKClient(options=options)
            await client.connect()
            logger.info(
                "sdk_subagent_connected",
                task_id=task_id,
                model=model or self.model,
            )

            # Send query and collect response
            await client.query(prompt)

            accumulated = ""
            last_assistant_text = ""
            result_text = ""
            tool_calls = 0
            iterations = 0
            total_cost: float | None = None
            async for message in self._safe_receive(client):
                if isinstance(message, AssistantMessage):
                    iterations += 1
                    # Track only the latest assistant text block — earlier
                    # blocks are intermediate reasoning between tool calls.
                    current_text = ""
                    for block in message.content:
                        if hasattr(block, "text"):
                            current_text += block.text
                        elif hasattr(block, "tool_use_id") or hasattr(block, "name"):
                            tool_calls += 1
                    if current_text:
                        last_assistant_text = current_text
                    accumulated += current_text
                elif isinstance(message, ResultMessage):
                    # ResultMessage contains authoritative metrics
                    iterations = message.num_turns
                    total_cost = message.total_cost_usd
                    if message.result:
                        result_text = message.result

            # Prefer ResultMessage.result (authoritative final output),
            # then last assistant text, then full accumulated text.
            final_text = result_text or last_assistant_text or accumulated
            if not final_text:
                final_text = "[No response from sub-agent]"

            # Verify URLs in output (shared with LiteLLM path)
            if "http" in final_text:
                try:
                    from agent.utils.url_check import check_urls_in_output

                    final_text = await check_urls_in_output(final_text)
                except Exception:
                    pass  # Don't fail sub-agent on URL check errors

            # Store metrics for the orchestrator to read
            self._subagent_metrics[task_id] = {
                "tool_calls": tool_calls,
                "iterations": iterations,
                "total_cost_usd": total_cost,
            }

            logger.info(
                "sdk_subagent_completed",
                task_id=task_id,
                response_len=len(final_text),
                tool_calls=tool_calls,
                iterations=iterations,
            )

            return final_text

        except Exception as e:
            logger.error(
                "sdk_subagent_error",
                task_id=task_id,
                error=str(e),
            )
            raise

        finally:
            if client is not None:
                with _contextlib.suppress(Exception):
                    await client.disconnect()
                logger.info("sdk_subagent_disconnected", task_id=task_id)

    async def _query_memory(
        self,
        user_message: str,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        """Query fact store and vector store for relevant context.

        Returns:
            Tuple of (facts, vector_results), either can be None.
        """
        facts = None
        vectors = None

        async def _get_facts() -> list[Any] | None:
            if not self.fact_store:
                return None
            try:
                return await self.fact_store.get_relevant(limit=15)
            except Exception as e:
                logger.debug("sdk_fact_query_failed", error=str(e))
                return None

        async def _get_vectors() -> list[Any] | None:
            if not self.vector_store:
                return None
            try:
                return await self.vector_store.search(user_message, limit=5)
            except Exception as e:
                logger.debug("sdk_vector_query_failed", error=str(e))
                return None

        facts, vectors = await asyncio.gather(_get_facts(), _get_vectors())
        return facts, vectors

    # _build_system_prompt — bound from _prompts.py below
    # _build_subagent_prompt — bound from _prompts.py below
    # _safe_receive — bound from _client.py below
    # _safe_extract_facts — defined here (small, uses self)

    async def _safe_extract_facts(
        self,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Extract facts from a user+assistant exchange (fire-and-forget)."""
        try:
            await self.fact_extractor.extract_from_messages(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response},
                ]
            )
        except Exception as e:
            logger.debug("sdk_fact_extraction_failed", error=str(e))

    # _build_mcp_server — bound from _mcp.py below
    # _ensure_client — bound from _client.py below
    # disconnect_client — bound from _client.py below
    # _get_query_lock — bound from _client.py below

    async def _run_task_impl(
        self,
        prompt: str,
        *,
        task_id: str = "default",
        session_id: str | None = None,
        working_dir: str | None = None,
        on_permission: PermissionCallback | None = None,
        on_question: QuestionCallback | None = None,
        channel: str | None = None,
    ) -> AsyncGenerator[SDKStreamEvent, None]:
        """Send a message to a persistent SDK client and stream the response.

        Uses a per-task lock to prevent concurrent queries on the same client,
        which would cause out-of-order responses.
        """
        # Serialize queries per task — the SDK client cannot handle concurrent
        # query()+receive_response() calls on the same connection.
        lock = self._get_query_lock(task_id)
        async with lock:
            async for event in self._run_task_locked(
                prompt,
                task_id=task_id,
                session_id=session_id,
                working_dir=working_dir,
                on_permission=on_permission,
                on_question=on_question,
                channel=channel,
            ):
                yield event

    async def _run_task_locked(
        self,
        prompt: str,
        *,
        task_id: str = "default",
        session_id: str | None = None,
        working_dir: str | None = None,
        on_permission: PermissionCallback | None = None,
        on_question: QuestionCallback | None = None,
        channel: str | None = None,
    ) -> AsyncGenerator[SDKStreamEvent, None]:
        """Inner implementation — must be called under _query_locks[task_id]."""
        from claude_code_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
        )

        self._status[task_id] = SDKTaskStatus.RUNNING

        try:
            client = await self._ensure_client(
                task_id,
                on_permission=on_permission,
                on_question=on_question,
                session_id=session_id,
                channel=channel,
            )

            # Issue 3: Record activity
            self._last_activity[task_id] = time.monotonic()

            logger.info(
                "sdk_query",
                task_id=task_id,
                prompt_len=len(prompt),
            )

            await client.query(prompt)

            result_text = ""
            num_turns = 0
            msg_count = 0
            t_query = time.monotonic()

            async for message in self._safe_receive(client):
                msg_count += 1
                msg_type = type(message).__name__
                logger.debug(
                    "sdk_message_received",
                    task_id=task_id,
                    msg_type=msg_type,
                    msg_num=msg_count,
                    elapsed_ms=int((time.monotonic() - t_query) * 1000),
                )

                cancel_event = self._cancel_events.get(task_id)
                if cancel_event and cancel_event.is_set():
                    with contextlib.suppress(Exception):
                        await client.interrupt()
                    break

                if message is None:
                    continue

                if isinstance(message, AssistantMessage):
                    # Subagent messages have parent_tool_use_id set —
                    # their text is internal and should not be part of
                    # the final response sent to the user.
                    is_subagent = (
                        getattr(
                            message,
                            "parent_tool_use_id",
                            None,
                        )
                        is not None
                    )

                    for block in message.content:
                        if isinstance(block, TextBlock):
                            if not is_subagent:
                                result_text += block.text
                            yield SDKStreamEvent(
                                type="text",
                                content=block.text,
                                data={"subagent": is_subagent},
                            )
                        elif isinstance(block, ThinkingBlock):
                            yield SDKStreamEvent(type="thinking", content=block.thinking)
                        elif isinstance(block, ToolUseBlock):
                            yield SDKStreamEvent(
                                type="tool_use",
                                content=f"Using {block.name}",
                                data={
                                    "tool": block.name,
                                    "input": block.input,
                                    "id": block.id,
                                    "subagent": is_subagent,
                                },
                            )

                elif isinstance(message, ResultMessage):
                    # Issue 3: Record activity
                    self._last_activity[task_id] = time.monotonic()

                    # Issue 4: Store session_id for future resume
                    if message.session_id:
                        self._last_session_ids[task_id] = message.session_id

                    num_turns = getattr(message, "num_turns", 0)
                    usage = getattr(message, "usage", {}) or {}

                    in_tokens = usage.get("input_tokens", 0)
                    out_tokens = usage.get("output_tokens", 0)

                    # Record cost
                    if self.cost_tracker and (in_tokens or out_tokens):
                        self.cost_tracker.record(
                            model=self.model or "claude-sdk",
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                            channel="sdk",
                            session_id=task_id,
                        )

                    # Fire-and-forget: extract facts (tracked for clean shutdown)
                    if self.fact_extractor and result_text:
                        bg_task = asyncio.create_task(self._safe_extract_facts(prompt, result_text))
                        self._background_tasks.add(bg_task)
                        bg_task.add_done_callback(self._background_tasks.discard)
                        bg_task.add_done_callback(_log_task_exception)

                    # Emit outgoing event (tracked for clean shutdown)
                    if self.event_bus:
                        from agent.core.events import Events

                        emit_task = asyncio.create_task(
                            self.event_bus.emit(
                                Events.MESSAGE_OUTGOING,
                                {
                                    "content": result_text,
                                    "session_id": task_id,
                                    "model": self.model or "claude-sdk",
                                    "num_turns": num_turns,
                                },
                            )
                        )
                        self._background_tasks.add(emit_task)
                        emit_task.add_done_callback(self._background_tasks.discard)
                        emit_task.add_done_callback(_log_task_exception)

                    # Use the longer of result_text (from TextBlocks) and
                    # message.result — the SDK sometimes only fills one.
                    msg_result = getattr(message, "result", "") or ""
                    final_content = (
                        msg_result if len(msg_result) > len(result_text) else result_text
                    )

                    # Verify URLs in output (shared with LiteLLM path)
                    if "http" in final_content:
                        try:
                            from agent.utils.url_check import check_urls_in_output

                            final_content = await check_urls_in_output(
                                final_content,
                            )
                        except Exception:
                            pass

                    yield SDKStreamEvent(
                        type="result",
                        content=final_content,
                        data={
                            "session_id": message.session_id,
                            "cost_usd": getattr(message, "total_cost_usd", 0.0),
                            "num_turns": num_turns,
                            "input_tokens": in_tokens,
                            "output_tokens": out_tokens,
                        },
                    )

            elapsed_ms = int((time.monotonic() - t_query) * 1000)
            logger.info(
                "sdk_query_complete",
                task_id=task_id,
                messages_received=msg_count,
                result_len=len(result_text),
                num_turns=num_turns,
                elapsed_ms=elapsed_ms,
            )
            self._status[task_id] = SDKTaskStatus.COMPLETED

        except Exception as e:
            self._status[task_id] = SDKTaskStatus.FAILED
            logger.error(
                "sdk_task_failed",
                error=str(e),
                error_type=type(e).__name__,
                task_id=task_id,
            )

            # Client is dead — remove it so next message reconnects
            await self.disconnect_client(task_id)

            yield SDKStreamEvent(type="error", content=str(e))

    async def cancel_task(self, task_id: str = "default") -> None:
        """Cancel a running task."""
        event = self._cancel_events.get(task_id)
        if event:
            event.set()

        client = self._clients.get(task_id)
        if client:
            with contextlib.suppress(Exception):
                await client.interrupt()

    def approve_permission(self, task_id: str = "default", *, approved: bool = True) -> None:
        """Respond to a pending permission request (for async HITL)."""
        self._permission_responses[task_id] = approved
        event = self._permission_events.get(task_id)
        if event:
            event.set()

    def answer_question(self, task_id: str = "default", *, answer: str = "") -> None:
        """Respond to a pending question (for async HITL)."""
        self._question_responses[task_id] = answer
        event = self._question_events.get(task_id)
        if event:
            event.set()

    def get_status(self, task_id: str = "default") -> SDKTaskStatus:
        """Get the current status of a task."""
        return self._status.get(task_id, SDKTaskStatus.IDLE)


def _log_task_exception(task: asyncio.Task[Any]) -> None:
    """Callback for fire-and-forget tasks — log exceptions instead of losing them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning("background_task_failed", error=str(exc), task=task.get_name())


def _format_tool_details(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Format tool details for human-readable display."""
    if tool_name == "Bash":
        cmd = tool_input.get("command", "unknown")
        desc = tool_input.get("description", "")
        return f"{desc}\n$ {cmd}" if desc else f"$ {cmd}"
    if tool_name in ("Write", "Edit"):
        path = tool_input.get("file_path", "unknown")
        return f"File: {path}"
    if tool_name == "NotebookEdit":
        return f"Notebook: {tool_input.get('notebook_path', 'unknown')}"
    return str(tool_input)[:300]


# ------------------------------------------------------------------
# Method binding: attach functions from sub-modules to the class
# ------------------------------------------------------------------

from agent.llm.claude_sdk._prompts import (  # noqa: E402
    _build_subagent_prompt,
    _build_system_prompt,
    _compute_prompt_fingerprint,
)

ClaudeSDKService._compute_prompt_fingerprint = _compute_prompt_fingerprint  # type: ignore[assignment]
ClaudeSDKService._build_system_prompt = _build_system_prompt  # type: ignore[assignment]
ClaudeSDKService._build_subagent_prompt = _build_subagent_prompt  # type: ignore[assignment]

from agent.llm.claude_sdk._mcp import _build_mcp_server  # noqa: E402

ClaudeSDKService._build_mcp_server = _build_mcp_server  # type: ignore[assignment]

from agent.llm.claude_sdk._client import (  # noqa: E402
    _ensure_client,
    _get_query_lock,
    _safe_receive,
    disconnect_client,
)

ClaudeSDKService._ensure_client = _ensure_client  # type: ignore[assignment]
ClaudeSDKService.disconnect_client = disconnect_client  # type: ignore[assignment]
ClaudeSDKService._get_query_lock = _get_query_lock  # type: ignore[assignment]
ClaudeSDKService._safe_receive = staticmethod(_safe_receive)  # type: ignore[assignment]
