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
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Lazy SDK availability check
_SDK_AVAILABLE: bool | None = None


def sdk_available() -> bool:
    """Check if claude-agent-sdk is installed."""
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is None:
        try:
            import claude_agent_sdk  # noqa: F401

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

        # Per-task state
        self._clients: dict[str, Any] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._permission_events: dict[str, asyncio.Event] = {}
        self._permission_responses: dict[str, bool] = {}
        self._question_events: dict[str, asyncio.Event] = {}
        self._question_responses: dict[str, str] = {}
        self._status: dict[str, SDKTaskStatus] = {}

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
        # Try with resume first; on failure retry without resume
        async for event in self._run_task_impl(
            prompt,
            task_id=task_id,
            session_id=session_id,
            working_dir=working_dir,
            on_permission=on_permission,
            on_question=on_question,
        ):
            if (
                event.type == "error"
                and session_id is not None
            ):
                # Resume failed — retry with fresh session
                logger.warning(
                    "sdk_resume_failed_retrying",
                    task_id=task_id,
                    session_id=session_id[:16] + "...",
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
        """Internal implementation of run_task_stream."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
        )
        from claude_agent_sdk.types import (
            PermissionResultAllow,
            PermissionResultDeny,
        )

        cancel_event = asyncio.Event()
        permission_event = asyncio.Event()
        question_event = asyncio.Event()

        self._cancel_events[task_id] = cancel_event
        self._permission_events[task_id] = permission_event
        self._question_events[task_id] = question_event
        self._status[task_id] = SDKTaskStatus.RUNNING

        work_dir = working_dir or self.working_dir

        # --- Permission callback for Claude Code tools ---
        async def can_use_tool(
            tool_name: str,
            tool_input: dict[str, Any],
            context: Any,
        ) -> Any:
            if cancel_event.is_set():
                return PermissionResultDeny(
                    message="Task cancelled by user", interrupt=True
                )

            # Auto-allow safe read-only tools
            safe_tools = {
                "Read", "Glob", "Grep", "WebFetch", "WebSearch",
                "LS", "Agent", "Explore", "TaskGet", "TaskList",
                "TaskOutput", "ToolSearch",
            }
            if tool_name in safe_tools:
                return PermissionResultAllow(updated_input=tool_input)

            # Handle AskUserQuestion
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

            # Dangerous tools — ask permission
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

            # No callback — auto-allow
            return PermissionResultAllow(updated_input=tool_input)

        # --- Execute ---
        stderr_lines: list[str] = []

        def _capture_stderr(line: str) -> None:
            # Sanitize non-ASCII to avoid cp1251 UnicodeEncodeError on Windows
            safe_line = line.rstrip().encode("ascii", errors="replace").decode("ascii")
            stderr_lines.append(safe_line)
            logger.warning("sdk_stderr", line=safe_line, task_id=task_id)

        # Save and remove env vars that interfere with SDK subprocess.
        # The SDK merges os.environ with user env, so we must clear these
        # from os.environ to prevent them leaking into the subprocess.
        _removed_env: dict[str, str] = {}
        for key in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "CLAUDECODE",
            "CLAUDE_CODE_SSE_PORT",
            "CLAUDE_CODE_ENTRYPOINT",
        ):
            val = os.environ.pop(key, None)
            if val is not None:
                _removed_env[key] = val

        try:
            # Resolve working directory to absolute path
            resolved_cwd = str(Path(work_dir).resolve())

            logger.info(
                "sdk_launching",
                task_id=task_id,
                cwd=resolved_cwd,
                model=self.model,
                permission_mode=self.permission_mode,
                resume=session_id[:16] + "..." if session_id else None,
            )

            options = ClaudeAgentOptions(
                cwd=resolved_cwd,
                max_turns=self.max_turns,
                model=self.model,
                permission_mode=self.permission_mode,
                can_use_tool=can_use_tool,
                resume=session_id,
                env={
                    "GIT_TERMINAL_PROMPT": "0",
                    "PYTHONUTF8": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
                stderr=_capture_stderr,
            )

            async with ClaudeSDKClient(options=options) as client:
                self._clients[task_id] = client
                logger.info("sdk_task_started", task_id=task_id, work_dir=resolved_cwd)

                await client.query(prompt)

                result_text = ""
                num_turns = 0

                async for message in client.receive_response():
                    if cancel_event.is_set():
                        with contextlib.suppress(Exception):
                            await client.interrupt()
                        break

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
                        num_turns = getattr(message, "num_turns", 0)
                        usage = getattr(message, "usage", {}) or {}

                        # Detect invalid session (0 turns with resume)
                        if num_turns == 0 and session_id:
                            logger.warning(
                                "sdk_session_invalid",
                                session_id=session_id[:16] + "...",
                                task_id=task_id,
                            )
                            yield SDKStreamEvent(
                                type="error",
                                content="Session expired or invalid",
                            )
                            return

                        yield SDKStreamEvent(
                            type="result",
                            content=getattr(message, "result", "") or result_text,
                            data={
                                "session_id": message.session_id,
                                "cost_usd": getattr(
                                    message, "total_cost_usd", 0.0
                                ),
                                "num_turns": num_turns,
                                "input_tokens": usage.get("input_tokens", 0),
                                "output_tokens": usage.get("output_tokens", 0),
                            },
                        )

            self._status[task_id] = SDKTaskStatus.COMPLETED

        except Exception as e:
            self._status[task_id] = SDKTaskStatus.FAILED
            # Include captured stderr in the error log for debugging
            stderr_text = "\n".join(stderr_lines).strip()
            logger.error(
                "sdk_task_failed",
                error=str(e),
                task_id=task_id,
                stderr=stderr_text or "(no stderr captured)",
            )
            error_msg = str(e)
            if stderr_text:
                error_msg = f"{error_msg}\nStderr: {stderr_text}"
            yield SDKStreamEvent(type="error", content=error_msg)

        finally:
            # Restore removed env vars
            os.environ.update(_removed_env)
            self._clients.pop(task_id, None)
            self._cancel_events.pop(task_id, None)
            self._permission_events.pop(task_id, None)
            self._question_events.pop(task_id, None)

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
