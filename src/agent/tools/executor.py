"""Tool executor — dispatches tool calls to registered tools with safety controls.

Handles permissions, timeouts, output truncation, audit logging, and event emission.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import structlog

from agent.config import ToolsConfig
from agent.core.audit import AuditLog
from agent.core.events import EventBus
from agent.core.guardrails import Guardrails
from agent.core.permissions import PermissionManager
from agent.core.session import ToolCall
from agent.tools.registry import (
    ToolRegistry,
    ToolTier,
    ToolTimeoutError,
)

logger = structlog.get_logger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    tool_call_id: str
    success: bool
    output: str  # Tool output (stdout, return value, etc.)
    error: str | None = None  # Error message if failed
    duration_ms: int = 0  # Execution time
    tier: ToolTier = ToolTier.SAFE
    approved_by: str = "auto"  # "auto", "user", "denied"


class ToolExecutor:
    """Dispatches tool calls to registered tools with safety controls.

    Responsibilities:
    - Look up tool in registry
    - Check permissions (tier-based)
    - Apply resource limits (timeout, output size)
    - Execute the tool function
    - Log to audit trail
    - Return structured result
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: ToolsConfig,
        event_bus: EventBus,
        audit: AuditLog,
        permissions: PermissionManager,
        guardrails: Guardrails,
    ) -> None:
        self.registry = registry
        self.config = config
        self.event_bus = event_bus
        self.audit = audit
        self.permissions = permissions
        self.guardrails = guardrails

    async def execute(
        self,
        tool_call: ToolCall,
        session_id: str,
        trigger: str = "user_message",
    ) -> ToolResult:
        """Execute a single tool call.

        Steps:
        1. Look up tool in registry
        2. Check if tool is enabled
        3. Check permissions (tier)
        4. Apply guardrails (validate input)
        5. Execute with timeout
        6. Validate/truncate output
        7. Log to audit trail
        8. Return ToolResult

        Args:
            tool_call: The tool call from the LLM.
            session_id: Current session identifier.
            trigger: What triggered this execution.

        Returns:
            ToolResult with execution outcome.
        """
        start_time = time.monotonic()

        # 1. Look up tool
        tool_def = self.registry.get_tool(tool_call.name)
        if not tool_def:
            result = ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output=f"[ERROR] Tool not found: {tool_call.name}",
                error=f"Tool '{tool_call.name}' is not registered",
            )
            await self._log_audit(result, tool_call.arguments, session_id, trigger, "error")
            return result

        # 2. Check if enabled
        if not tool_def.enabled:
            result = ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output=f"[ERROR] Tool is disabled: {tool_call.name}",
                error=f"Tool '{tool_call.name}' is disabled",
                tier=tool_def.tier,
            )
            await self._log_audit(result, tool_call.arguments, session_id, trigger, "blocked")
            return result

        # 3. Check permissions
        perm_result = await self.permissions.check_permission(tool_def, tool_call.arguments)
        if not perm_result.approved:
            result = ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output=f"[DENIED] Execution denied: {perm_result.reason}",
                error=perm_result.reason,
                tier=tool_def.tier,
                approved_by="denied",
            )
            await self._log_audit(result, tool_call.arguments, session_id, trigger, "denied")
            return result

        # 4. Apply guardrails
        guardrail_result = self._check_guardrails(tool_call)
        if not guardrail_result.allowed:
            result = ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output=f"[BLOCKED] {guardrail_result.reason}",
                error=guardrail_result.reason,
                tier=tool_def.tier,
                approved_by=perm_result.method,
            )
            await self._log_audit(result, tool_call.arguments, session_id, trigger, "blocked")
            return result

        # 5. Execute with timeout
        default_timeout = 30
        timeout = tool_call.arguments.get("timeout", default_timeout)
        if not isinstance(timeout, int):
            timeout = default_timeout

        try:
            output = await self._execute_with_timeout(tool_def, tool_call.arguments, timeout)
            duration_ms = int((time.monotonic() - start_time) * 1000)

            # 6. Validate/truncate output
            output = self.guardrails.validate_output(output)

            result = ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=True,
                output=output,
                duration_ms=duration_ms,
                tier=tool_def.tier,
                approved_by=perm_result.method,
            )

        except ToolTimeoutError as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output=str(e),
                error=str(e),
                duration_ms=duration_ms,
                tier=tool_def.tier,
                approved_by=perm_result.method,
            )

        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            result = ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output=f"[ERROR] {type(e).__name__}: {e}",
                error=str(e),
                duration_ms=duration_ms,
                tier=tool_def.tier,
                approved_by=perm_result.method,
            )

        # 7. Log to audit trail
        status = "success" if result.success else "error"
        if result.error and "timed out" in result.error:
            status = "timeout"
        await self._log_audit(
            result, tool_call.arguments, session_id, trigger, status
        )

        return result

    async def execute_parallel(
        self,
        tool_calls: list[ToolCall],
        session_id: str,
        trigger: str = "user_message",
    ) -> list[ToolResult]:
        """Execute multiple independent tool calls concurrently.

        Uses asyncio.gather with return_exceptions=True.
        If any tool fails, others still complete.

        Args:
            tool_calls: List of tool calls to execute.
            session_id: Current session identifier.
            trigger: What triggered these executions.

        Returns:
            List of ToolResult objects in the same order as input.
        """
        tasks = [
            self.execute(tc, session_id, trigger)
            for tc in tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: list[ToolResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    ToolResult(
                        tool_name=tool_calls[i].name,
                        tool_call_id=tool_calls[i].id,
                        success=False,
                        output=f"[ERROR] {type(result).__name__}: {result}",
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _execute_with_timeout(
        self, tool_def: object, arguments: dict, timeout: int  # noqa: ASYNC109
    ) -> str:
        """Execute a tool function with a timeout.

        Args:
            tool_def: The tool definition containing the function.
            arguments: Arguments to pass to the function.
            timeout: Timeout in seconds.

        Returns:
            String output from the tool.

        Raises:
            ToolTimeoutError: If execution exceeds timeout.
        """
        from agent.tools.registry import ToolDefinition

        assert isinstance(tool_def, ToolDefinition)

        # Filter arguments to only those the function accepts
        import inspect

        sig = inspect.signature(tool_def.function)
        valid_args = {
            k: v for k, v in arguments.items()
            if k in sig.parameters
        }

        try:
            result = await asyncio.wait_for(
                tool_def.function(**valid_args),
                timeout=timeout,
            )
            return str(result)
        except TimeoutError as e:
            raise ToolTimeoutError(
                f"Tool '{tool_def.name}' timed out after {timeout}s"
            ) from e

    def _check_guardrails(self, tool_call: ToolCall) -> object:
        """Apply guardrail checks based on tool type.

        Args:
            tool_call: The tool call to check.

        Returns:
            GuardrailResult.
        """
        from agent.core.guardrails import GuardrailResult

        # Shell command guardrails
        if tool_call.name == "shell_exec":
            command = tool_call.arguments.get("command", "")
            if isinstance(command, str):
                return self.guardrails.check_command(command)

        # File path guardrails
        if tool_call.name in ("file_read", "file_write", "file_list"):
            path = tool_call.arguments.get("path", ".")
            if isinstance(path, str):
                operation = "read" if tool_call.name == "file_read" else (
                    "write" if tool_call.name == "file_write" else "list"
                )
                return self.guardrails.check_file_path(path, operation)

        # HTTP URL guardrails
        if tool_call.name == "http_request":
            url = tool_call.arguments.get("url", "")
            if isinstance(url, str):
                return self.guardrails.check_url(url)

        return GuardrailResult(allowed=True)

    async def _log_audit(
        self,
        result: ToolResult,
        input_data: dict,
        session_id: str,
        trigger: str,
        status: str,
    ) -> None:
        """Log execution to audit trail.

        Args:
            result: The tool execution result.
            input_data: Input arguments.
            session_id: Session identifier.
            trigger: What triggered the execution.
            status: Execution status string.
        """
        await self.audit.log(
            tool_name=result.tool_name,
            tool_call_id=result.tool_call_id,
            input_data=input_data,
            output=result.output,
            status=status,
            duration_ms=result.duration_ms,
            trigger=trigger,
            session_id=session_id,
            approved_by=result.approved_by,
            error=result.error,
        )
