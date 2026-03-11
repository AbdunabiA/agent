"""Self-healing error recovery system.

Classifies errors and determines the best recovery action.
Tracks retry counts to avoid infinite loops.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.core.session import ToolCall
    from agent.tools.executor import ToolExecutor, ToolResult

logger = structlog.get_logger(__name__)


class ErrorCategory(StrEnum):
    """Categories of errors for recovery decisions."""

    TRANSIENT = "transient"  # Network, rate limit, timeout -> retry
    AUTH = "auth"  # Bad credentials -> notify user
    LOGIC = "logic"  # Wrong approach -> try differently
    PERMISSION = "permission"  # Access denied -> request access or notify
    RESOURCE = "resource"  # Disk full, memory -> notify user
    UNKNOWN = "unknown"  # Can't classify -> log and notify


@dataclass
class RecoveryAction:
    """Describes what recovery action to take."""

    action: str  # "retry", "retry_different", "notify_user", "skip", "abort"
    reason: str
    delay_seconds: float = 0
    max_retries: int = 3
    suggestion: str = ""  # Alternative approach to try


class ErrorRecovery:
    """Self-healing error recovery system.

    Classifies errors and determines the best recovery action.
    Tracks retry counts to avoid infinite loops.
    """

    def __init__(self) -> None:
        self._retry_counts: dict[str, int] = {}

    def classify_error(self, error: Exception, tool_name: str) -> ErrorCategory:
        """Classify an error into a category.

        Args:
            error: The exception that occurred.
            tool_name: Name of the tool that failed.

        Returns:
            ErrorCategory for the error.
        """
        error_msg = str(error).lower()

        # Transient errors — retry
        if isinstance(error, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
            return ErrorCategory.TRANSIENT
        if "timeout" in error_msg or "rate limit" in error_msg:
            return ErrorCategory.TRANSIENT
        if "connection" in error_msg or "connect" in error_msg:
            return ErrorCategory.TRANSIENT

        # Auth errors
        if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg:
            return ErrorCategory.AUTH
        if "authentication" in error_msg or "forbidden" in error_msg:
            return ErrorCategory.AUTH

        # Logic errors
        if isinstance(error, (FileNotFoundError, ValueError, KeyError)):
            return ErrorCategory.LOGIC
        if "404" in error_msg or "not found" in error_msg:
            return ErrorCategory.LOGIC

        # Permission errors
        if isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION
        if "permission denied" in error_msg or "access denied" in error_msg:
            return ErrorCategory.PERMISSION
        if isinstance(error, OSError) and hasattr(error, "errno"):
            import errno

            if error.errno == errno.EACCES:
                return ErrorCategory.PERMISSION

        # Resource errors
        if isinstance(error, MemoryError):
            return ErrorCategory.RESOURCE
        if isinstance(error, OSError) and hasattr(error, "errno"):
            import errno

            if error.errno == errno.ENOSPC:
                return ErrorCategory.RESOURCE
        if "disk full" in error_msg or "no space" in error_msg:
            return ErrorCategory.RESOURCE

        return ErrorCategory.UNKNOWN

    def get_recovery_action(
        self, error: Exception, tool_name: str, category: ErrorCategory
    ) -> RecoveryAction:
        """Determine the best recovery action.

        Args:
            error: The exception that occurred.
            tool_name: Name of the tool that failed.
            category: Classified error category.

        Returns:
            RecoveryAction describing what to do.
        """
        current_retries = self._retry_counts.get(tool_name, 0)

        if category == ErrorCategory.TRANSIENT:
            if current_retries < 3:
                delay = 2 ** current_retries  # 1s, 2s, 4s
                self._retry_counts[tool_name] = current_retries + 1
                return RecoveryAction(
                    action="retry",
                    reason=f"Transient error, retrying ({current_retries + 1}/3)",
                    delay_seconds=delay,
                    max_retries=3,
                )
            return RecoveryAction(
                action="notify_user",
                reason=f"Transient error persisted after 3 retries: {error}",
            )

        elif category == ErrorCategory.AUTH:
            return RecoveryAction(
                action="notify_user",
                reason=f"Authentication error: {error}. Check your API keys or credentials.",
            )

        elif category == ErrorCategory.LOGIC:
            if current_retries < 2:
                self._retry_counts[tool_name] = current_retries + 1
                return RecoveryAction(
                    action="retry_different",
                    reason=f"Logic error: {error}. Try a different approach.",
                    suggestion=f"The previous approach failed with: {error}. Try an alternative.",
                    max_retries=2,
                )
            return RecoveryAction(
                action="notify_user",
                reason=f"Logic error after multiple attempts: {error}",
            )

        elif category == ErrorCategory.PERMISSION:
            return RecoveryAction(
                action="notify_user",
                reason=f"Permission denied: {error}. You may need to adjust file permissions.",
                suggestion="Try: chmod/chown or run with appropriate permissions",
            )

        elif category == ErrorCategory.RESOURCE:
            return RecoveryAction(
                action="notify_user",
                reason=f"Resource exhausted: {error}. Check disk space or memory.",
            )

        # UNKNOWN
        return RecoveryAction(
            action="notify_user",
            reason=f"Unexpected error: {error}",
        )

    async def execute_recovery(
        self,
        action: RecoveryAction,
        tool_call: ToolCall,
        executor: ToolExecutor,
        session_id: str,
    ) -> ToolResult | None:
        """Execute the recovery action.

        Args:
            action: The recovery action to execute.
            tool_call: The original tool call that failed.
            executor: Tool executor for retries.
            session_id: Current session ID.

        Returns:
            ToolResult if recovery succeeds (retry worked),
            None if recovery doesn't involve re-execution.
        """
        if action.action == "retry":
            if action.delay_seconds > 0:
                logger.info(
                    "recovery_delay",
                    seconds=action.delay_seconds,
                    reason=action.reason,
                )
                await asyncio.sleep(action.delay_seconds)

            logger.info("recovery_retry", tool=tool_call.name, reason=action.reason)
            try:
                result = await executor.execute(
                    tool_call=tool_call,
                    session_id=session_id,
                    trigger="recovery",
                )
                # Reset retries on success
                self.reset_retries(tool_call.name)
                return result
            except Exception as e:
                logger.warning("recovery_retry_failed", tool=tool_call.name, error=str(e))
                return None

        # For non-retry actions, return None (caller handles notification)
        logger.info("recovery_action", action=action.action, reason=action.reason)
        return None

    def reset_retries(self, tool_name: str) -> None:
        """Reset retry count for a tool (called on success).

        Args:
            tool_name: The tool to reset retries for.
        """
        self._retry_counts.pop(tool_name, None)
