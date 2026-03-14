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
import hashlib
import os
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


def sdk_available() -> bool:
    """Check if claude-agent-sdk is installed."""
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is None:
        try:
            import claude_code_sdk  # noqa: F401

            _SDK_AVAILABLE = True
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
                "claude-agent-sdk is not installed. "
                "Install with: pip install claude-agent-sdk"
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

        # Issue 4: Resume on crash — last known session_id per task
        self._last_session_ids: dict[str, str] = {}

        # Issue 2: MCP tool generation tracking
        self._tool_generations: dict[str, int] = {}

        # Issue 1: System prompt fingerprint tracking
        self._prompt_fingerprints: dict[str, str] = {}

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
        """Stop the idle-client reaper."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None
            logger.info("sdk_reaper_stopped")

    async def _reap_idle_clients(self) -> None:
        """Periodically disconnect clients that have been idle too long."""
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            idle_ids = [
                tid for tid, last in self._last_activity.items()
                if (now - last) > self._idle_timeout and tid in self._clients
            ]
            for tid in idle_ids:
                logger.info("sdk_reaping_idle_client", task_id=tid,
                            idle_secs=int(now - self._last_activity[tid]))
                await self.disconnect_client(tid)

    # ------------------------------------------------------------------
    # Issue 1: System prompt fingerprint
    # ------------------------------------------------------------------

    def _compute_prompt_fingerprint(
        self,
        facts: list[Any] | None = None,
        vector_results: list[Any] | None = None,
    ) -> str:
        """Hash soul + facts into a fingerprint for drift detection."""
        parts: list[str] = []
        if self.soul_loader:
            parts.append(self.soul_loader.content or "")
        if facts:
            parts.extend(f"{f.key}={f.value}" for f in facts)
        raw = "\n".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def run_task_stream(
        self,
        prompt: str,
        *,
        task_id: str = "default",
        session_id: str | None = None,
        working_dir: str | None = None,
        on_permission: PermissionCallback | None = None,
        on_question: QuestionCallback | None = None,
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

        Yields:
            SDKStreamEvent objects as they occur.
        """
        # Try with resume first; on any error retry once without resume,
        # since session corruption or stale state can cause various failures.
        async for event in self._run_task_impl(
            prompt,
            task_id=task_id,
            session_id=session_id,
            working_dir=working_dir,
            on_permission=on_permission,
            on_question=on_question,
        ):
            if event.type == "error" and session_id is not None:
                # Resume failed — retry with fresh session
                logger.warning(
                    "sdk_resume_failed_retrying",
                    task_id=task_id,
                    session_id=session_id[:16] + "...",
                    error=event.content[:120] if event.content else None,
                )
                async for retry_event in self._run_task_impl(
                    prompt,
                    task_id=task_id,
                    session_id=None,
                    working_dir=working_dir,
                    on_permission=on_permission,
                    on_question=on_question,
                ):
                    yield retry_event
                return
            yield event

    async def _query_memory(
        self, user_message: str,
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

    def _build_system_prompt(
        self,
        facts: list[Any] | None = None,
        vector_results: list[Any] | None = None,
    ) -> str | None:
        """Build system prompt from soul.md + memory context.

        Mirrors the format used by agent.core.context.build_messages().
        """
        parts: list[str] = []

        # Soul personality
        if self.soul_loader:
            soul_content = self.soul_loader.content
            if soul_content:
                parts.append(soul_content)

        # Inject known facts
        if facts:
            fact_lines = [f"- {f.key}: {f.value}" for f in facts]
            parts.append(
                "KNOWN FACTS ABOUT THE USER:\n" + "\n".join(fact_lines)
            )

        # Inject related past conversations
        if vector_results:
            vr_lines = []
            for vr in vector_results:
                score_pct = int(vr.score * 100)
                vr_lines.append(f"[Relevance: {score_pct}%] {vr.text}")
            parts.append(
                "RELATED PAST CONVERSATIONS:\n"
                + "\n---\n".join(vr_lines)
            )

        return "\n\n".join(parts) if parts else None

    @staticmethod
    async def _safe_receive(client: Any) -> AsyncGenerator[Any, None]:
        """Wrap client.receive_response() to skip unknown message types.

        Uses receive_messages() instead of receive_response() so we can
        detect and skip stale ResultMessages that sit in the buffer from
        a previous query cycle.
        """
        from claude_code_sdk import ResultMessage

        got_assistant = False

        async for raw_message in client.receive_messages():
            try:
                from claude_code_sdk._internal.message_parser import parse_message

                message = parse_message(raw_message)
            except Exception as e:
                if "Unknown message type" in str(e):
                    logger.debug("sdk_unknown_message_skipped", error=str(e))
                    continue
                raise

            from claude_code_sdk import AssistantMessage

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

    async def _safe_extract_facts(
        self, user_message: str, assistant_response: str,
    ) -> None:
        """Extract facts from a user+assistant exchange (fire-and-forget)."""
        try:
            await self.fact_extractor.extract_from_messages([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ])
        except Exception as e:
            logger.debug("sdk_fact_extraction_failed", error=str(e))

    def _build_mcp_server(self) -> tuple[Any, list[str]] | None:
        """Build an in-process MCP server exposing agent tools to the SDK.

        Creates SDK-compatible tool wrappers for each enabled tool in the
        registry, bundles them into an MCP server, and returns the server
        config + list of tool names.

        Returns:
            Tuple of (McpSdkServerConfig, tool_name_list) or None if no tools.
        """
        if not self.tool_registry:
            return None

        from claude_code_sdk import create_sdk_mcp_server
        from claude_code_sdk import tool as sdk_tool

        tools = self.tool_registry.list_tools()
        enabled_tools = [t for t in tools if t.enabled]
        if not enabled_tools:
            return None

        sdk_tools = []
        tool_names = []

        for tool_def in enabled_tools:
            # Capture tool_def in closure via default arg
            def _make_handler(td: Any) -> Any:
                async def handler(args: dict[str, Any]) -> dict[str, Any]:
                    from agent.tools.executor import MultimodalToolOutput

                    try:
                        result = await td.function(**args)
                        if isinstance(result, MultimodalToolOutput):
                            content_blocks: list[dict[str, Any]] = [
                                {"type": "text", "text": result.text},
                            ]
                            for img in result.images:
                                content_blocks.append({
                                    "type": "image",
                                    "data": img.base64_data,
                                    "mimeType": img.media_type,
                                })
                            return {"content": content_blocks}
                        text = str(result) if result is not None else ""
                    except Exception as e:
                        text = f"Error: {e}"
                    return {
                        "content": [{"type": "text", "text": text}],
                    }
                return handler

            wrapped = sdk_tool(
                tool_def.name,
                tool_def.description,
                tool_def.parameters,
            )(_make_handler(tool_def))

            sdk_tools.append(wrapped)
            tool_names.append(tool_def.name)

        server = create_sdk_mcp_server(
            name="agent-tools",
            version="1.0.0",
            tools=sdk_tools,
        )

        logger.info("sdk_mcp_server_built", tool_count=len(sdk_tools))
        return server, tool_names

    async def _ensure_client(
        self,
        task_id: str,
        on_permission: PermissionCallback | None = None,
        on_question: QuestionCallback | None = None,
        session_id: str | None = None,
    ) -> Any:
        """Get or create a persistent SDK client for a task/user.

        The client stays connected across messages — no subprocess restart.
        Reconnects if tool generation or system prompt has drifted.
        """
        # Issue 2: Check if tool generation changed → reconnect
        if task_id in self._clients and self.tool_registry:
            current_gen = self.tool_registry._generation
            if self._tool_generations.get(task_id) != current_gen:
                logger.info("sdk_reconnect_tool_change", task_id=task_id,
                            old_gen=self._tool_generations.get(task_id), new_gen=current_gen)
                await self.disconnect_client(task_id)

        # Issue 1: Check if system prompt drifted → reconnect
        if task_id in self._clients:
            facts, _ = await self._query_memory("")
            new_fp = self._compute_prompt_fingerprint(facts)
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
                return PermissionResultDeny(
                    message="Task cancelled by user", interrupt=True
                )

            safe_tools = {
                "Read", "Glob", "Grep", "WebFetch", "WebSearch",
                "LS", "Agent", "Explore", "TaskGet", "TaskList",
                "TaskOutput", "ToolSearch",
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
                return PermissionResultDeny(
                    message="User denied permission", interrupt=False
                )

            return PermissionResultAllow(updated_input=tool_input)

        # Stderr capture
        import io

        class _StderrCapture(io.TextIOBase):
            def write(self, s: str) -> int:
                for line in s.splitlines():
                    safe = line.rstrip().encode("ascii", errors="replace").decode("ascii")
                    if safe:
                        logger.debug("sdk_stderr", line=safe, task_id=task_id)
                return len(s)

        # Query memory for system prompt
        facts, vector_results = await self._query_memory("")
        system_prompt = self._build_system_prompt(facts, vector_results)

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
            "debug_stderr": _StderrCapture(),
        }
        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers
        if system_prompt:
            options_kwargs["system_prompt"] = system_prompt
        if resume_id:
            options_kwargs["resume"] = resume_id

        options = ClaudeCodeOptions(**options_kwargs)

        # Clear env vars that interfere with SDK subprocess
        _removed_env: dict[str, str] = {}
        for key in (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
            "CLAUDECODE", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT",
        ):
            val = os.environ.pop(key, None)
            if val is not None:
                _removed_env[key] = val

        try:
            client = ClaudeSDKClient(options=options)
            await client.connect()
            self._clients[task_id] = client
            self._last_activity[task_id] = time.monotonic()

            # Store fingerprint + tool generation at connect time
            self._prompt_fingerprints[task_id] = self._compute_prompt_fingerprint(facts)
            if self.tool_registry:
                self._tool_generations[task_id] = self.tool_registry._generation

            logger.info(
                "sdk_client_connected",
                task_id=task_id,
                cwd=resolved_cwd,
                model=self.model,
                resume=resume_id[:16] + "..." if resume_id else None,
            )
        finally:
            os.environ.update(_removed_env)

        return client

    async def disconnect_client(self, task_id: str) -> None:
        """Disconnect and remove a persistent client (e.g. on /new command)."""
        client = self._clients.pop(task_id, None)
        if client:
            with contextlib.suppress(Exception):
                await client.disconnect()
            logger.info("sdk_client_disconnected", task_id=task_id)
        self._cancel_events.pop(task_id, None)
        self._status.pop(task_id, None)
        self._last_activity.pop(task_id, None)
        self._tool_generations.pop(task_id, None)
        self._prompt_fingerprints.pop(task_id, None)
        self._query_locks.pop(task_id, None)
        # Keep _last_session_ids — needed for resume on next connect

    def _get_query_lock(self, task_id: str) -> asyncio.Lock:
        """Get or create a per-task lock to serialize concurrent queries."""
        if task_id not in self._query_locks:
            self._query_locks[task_id] = asyncio.Lock()
        return self._query_locks[task_id]

    async def _run_task_impl(
        self,
        prompt: str,
        *,
        task_id: str = "default",
        session_id: str | None = None,
        working_dir: str | None = None,
        on_permission: PermissionCallback | None = None,
        on_question: QuestionCallback | None = None,
    ) -> AsyncGenerator[SDKStreamEvent, None]:
        """Send a message to a persistent SDK client and stream the response.

        Uses a per-task lock to prevent concurrent queries on the same client,
        which would cause out-of-order responses.
        """
        from claude_code_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
        )

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
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result_text += block.text
                            yield SDKStreamEvent(
                                type="text", content=block.text
                            )
                        elif isinstance(block, ThinkingBlock):
                            yield SDKStreamEvent(
                                type="thinking", content=block.thinking
                            )
                        elif isinstance(block, ToolUseBlock):
                            yield SDKStreamEvent(
                                type="tool_use",
                                content=f"Using {block.name}",
                                data={
                                    "tool": block.name,
                                    "input": block.input,
                                    "id": block.id,
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

                    # Fire-and-forget: extract facts
                    if self.fact_extractor and result_text:
                        bg_task = asyncio.create_task(
                            self._safe_extract_facts(prompt, result_text)
                        )
                        bg_task.add_done_callback(_log_task_exception)

                    # Emit outgoing event
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
                        emit_task.add_done_callback(_log_task_exception)

                    yield SDKStreamEvent(
                        type="result",
                        content=getattr(message, "result", "") or result_text,
                        data={
                            "session_id": message.session_id,
                            "cost_usd": getattr(
                                message, "total_cost_usd", 0.0
                            ),
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

    def approve_permission(
        self, task_id: str = "default", *, approved: bool = True
    ) -> None:
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
