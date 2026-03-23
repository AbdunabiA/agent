"""Client management for Claude SDK service."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.llm.claude_sdk._core import ClaudeSDKService, PermissionCallback, QuestionCallback

from agent.llm.claude_sdk._core import SDKTaskStatus, _format_tool_details

logger = structlog.get_logger(__name__)


async def _ensure_client(
    self: ClaudeSDKService,
    task_id: str,
    on_permission: PermissionCallback | None = None,
    on_question: QuestionCallback | None = None,
    session_id: str | None = None,
    channel: str | None = None,
) -> Any:
    """Get or create a persistent SDK client for a task/user.

    The client stays connected across messages — no subprocess restart.
    Reconnects if tool generation or system prompt has drifted.
    """
    # Issue 2: Check if tool generation changed → reconnect
    if task_id in self._clients and self.tool_registry:
        current_gen = self.tool_registry._generation
        if self._tool_generations.get(task_id) != current_gen:
            logger.info(
                "sdk_reconnect_tool_change",
                task_id=task_id,
                old_gen=self._tool_generations.get(task_id),
                new_gen=current_gen,
            )
            await self.disconnect_client(task_id)

    # Issue 1: Check if system prompt drifted → reconnect
    # Compare the stored fingerprint (from connect time) against
    # a freshly computed one.  We pass facts=None so the fingerprint
    # reflects the current soul.md content.  When run_task_impl later
    # queries memory it will get up-to-date facts for the new client.
    if task_id in self._clients:
        new_fp = self._compute_prompt_fingerprint(None)
        old_fp = self._prompt_fingerprints.get(task_id)
        if old_fp and old_fp != new_fp:
            logger.info("sdk_reconnect_prompt_drift", task_id=task_id)
            await self.disconnect_client(task_id)

    if task_id in self._clients:
        return self._clients[task_id]

    from claude_code_sdk import (
        ClaudeCodeOptions,
        ClaudeSDKClient,
    )
    from claude_code_sdk.types import (
        PermissionResultAllow,
        PermissionResultDeny,
    )

    work_dir = self.working_dir
    resolved_cwd = str(Path(os.path.expanduser(work_dir)).resolve())  # noqa: ASYNC240

    # Build MCP server for agent tools
    mcp_result = self._build_mcp_server()
    mcp_servers: dict[str, Any] = {}
    if mcp_result:
        server_config, _ = mcp_result
        mcp_servers["agent-tools"] = server_config

    # Resolve agent tool tiers for permission decisions
    from agent.tools.registry import ToolTier

    agent_safe_tools: set[str] = set()
    if self.tool_registry:
        for td in self.tool_registry.list_tools():
            if td.enabled and td.tier == ToolTier.SAFE:
                agent_safe_tools.add(td.name)

    # Permission callback for Claude Code + agent tools
    cancel_event = self._cancel_events.get(task_id, asyncio.Event())
    self._cancel_events[task_id] = cancel_event

    async def can_use_tool(
        tool_name: str,
        tool_input: dict[str, Any],
        context: Any,
    ) -> Any:
        if cancel_event.is_set():
            return PermissionResultDeny(message="Task cancelled by user", interrupt=True)

        safe_tools = {
            "Read",
            "Glob",
            "Grep",
            "WebFetch",
            "WebSearch",
            "LS",
            "Agent",
            "Explore",
            "TaskGet",
            "TaskList",
            "TaskOutput",
            "ToolSearch",
        }
        if tool_name in safe_tools:
            return PermissionResultAllow(updated_input=tool_input)

        if tool_name in agent_safe_tools:
            return PermissionResultAllow(updated_input=tool_input)

        if tool_name == "AskUserQuestion":
            if on_question:
                self._status[task_id] = SDKTaskStatus.WAITING_ANSWER
                questions = tool_input.get("questions", [])
                q_text = questions[0].get("question", "") if questions else ""
                options = (
                    [o.get("label", "") for o in questions[0].get("options", [])]
                    if questions
                    else []
                )
                answer = await on_question(q_text, options)
                self._status[task_id] = SDKTaskStatus.RUNNING
                if questions:
                    questions[0]["answer"] = answer
                return PermissionResultAllow(updated_input=tool_input)
            return PermissionResultAllow(updated_input=tool_input)

        if on_permission:
            self._status[task_id] = SDKTaskStatus.WAITING_PERMISSION
            details = _format_tool_details(tool_name, tool_input)
            approved = await on_permission(tool_name, details, tool_input)
            self._status[task_id] = SDKTaskStatus.RUNNING
            if approved:
                return PermissionResultAllow(updated_input=tool_input)
            return PermissionResultDeny(message="User denied permission", interrupt=False)

        # No permission callback — deny non-safe tools by default
        return PermissionResultDeny(message="No permission handler configured", interrupt=False)

    # Query memory for system prompt
    facts, vector_results = await self._query_memory("")
    self._cached_facts[task_id] = facts or []
    system_prompt = self._build_system_prompt(
        facts,
        vector_results,
        channel=channel,
    )

    # Issue 4: Use stored session_id for resume if not explicitly provided
    resume_id = session_id or self._last_session_ids.get(task_id)

    options_kwargs: dict[str, Any] = {
        "cwd": resolved_cwd,
        "max_turns": self.max_turns,
        "model": self.model,
        "permission_mode": self.permission_mode,
        "can_use_tool": can_use_tool,
        "env": {
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    }
    if mcp_servers:
        options_kwargs["mcp_servers"] = mcp_servers
    if system_prompt:
        options_kwargs["system_prompt"] = system_prompt
    if resume_id:
        options_kwargs["resume"] = resume_id

    # Prevent conflicting env vars from leaking into the subprocess
    # by overriding them in the options.env dict (no process-wide mutation).
    _env_overrides = dict(options_kwargs.get("env", {}))
    # Note: ANTHROPIC_API_KEY is NOT blanked — the subprocess needs it
    # for auth.  Only blank vars that control Claude Code CLI behavior
    # to avoid conflicts (consistent with run_subagent).
    for key in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDECODE",
        "CLAUDE_CODE_SSE_PORT",
        "CLAUDE_CODE_ENTRYPOINT",
    ):
        if key in os.environ and key not in _env_overrides:
            _env_overrides[key] = ""
    options_kwargs["env"] = _env_overrides

    options = ClaudeCodeOptions(**options_kwargs)

    client = ClaudeSDKClient(options=options)
    try:
        await client.connect()
    except Exception:
        with contextlib.suppress(Exception):
            await client.disconnect()
        raise

    self._clients[task_id] = client
    self._last_activity[task_id] = time.monotonic()

    # Store soul-only fingerprint at connect time.  We intentionally
    # exclude facts so drift detection in _get_or_create_client (which
    # doesn't have the prompt to query memory) stays consistent.
    self._prompt_fingerprints[task_id] = self._compute_prompt_fingerprint(None)
    if self.tool_registry:
        self._tool_generations[task_id] = self.tool_registry._generation

    logger.info(
        "sdk_client_connected",
        task_id=task_id,
        cwd=resolved_cwd,
        model=self.model,
        resume=resume_id[:16] + "..." if resume_id else None,
    )

    return client


async def disconnect_client(self: ClaudeSDKService, task_id: str) -> None:
    """Disconnect and remove a persistent client (e.g. on /new command)."""
    client = self._clients.pop(task_id, None)
    if client:
        with contextlib.suppress(Exception):
            await client.disconnect()
        logger.info("sdk_client_disconnected", task_id=task_id)
    self._cancel_events.pop(task_id, None)
    self._permission_events.pop(task_id, None)
    self._permission_responses.pop(task_id, None)
    self._question_events.pop(task_id, None)
    self._question_responses.pop(task_id, None)
    self._status.pop(task_id, None)
    self._last_activity.pop(task_id, None)
    self._tool_generations.pop(task_id, None)
    self._prompt_fingerprints.pop(task_id, None)
    self._cached_facts.pop(task_id, None)
    self._query_locks.pop(task_id, None)
    self._subagent_metrics.pop(task_id, None)
    # Keep _last_session_ids — needed for resume on next connect


def _get_query_lock(self: ClaudeSDKService, task_id: str) -> asyncio.Lock:
    """Get or create a per-task lock to serialize concurrent queries."""
    if task_id not in self._query_locks:
        self._query_locks[task_id] = asyncio.Lock()
    return self._query_locks[task_id]


async def _safe_receive(client: Any) -> AsyncGenerator[Any, None]:
    """Iterate client messages, skipping unknown types and stale results.

    Uses receive_messages() instead of receive_response() so we can
    detect and skip stale ResultMessages that sit in the buffer from
    a previous query cycle.

    Unknown message types (e.g. rate_limit_event) are skipped per-message
    instead of aborting the entire stream, so the actual response is still
    received.
    """
    from claude_code_sdk import AssistantMessage, ResultMessage

    got_assistant = False
    stream = client.receive_messages().__aiter__()

    while True:
        try:
            message = await stream.__anext__()
        except StopAsyncIteration:
            break
        except Exception as e:
            if "Unknown message type" in str(e):
                # Skip unparseable messages (e.g. rate_limit_event)
                # and keep iterating — the real response follows.
                logger.debug("sdk_unknown_message_skipped", error=str(e))
                continue
            raise

        if isinstance(message, AssistantMessage):
            got_assistant = True

        if isinstance(message, ResultMessage):
            if not got_assistant:
                # Stale ResultMessage from a previous query — skip it
                logger.warning(
                    "sdk_stale_result_skipped",
                    session_id=getattr(message, "session_id", None),
                    num_turns=getattr(message, "num_turns", None),
                )
                continue
            # Real result — yield it and stop
            yield message
            return

        yield message
