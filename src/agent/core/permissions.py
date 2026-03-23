"""Tiered permission system for tool execution.

Manages tool execution permissions based on safety tiers:
- SAFE: Always auto-approve (read files, search, memory)
- MODERATE: Auto-approve by default, can require confirmation in config
- DANGEROUS: Always requires user confirmation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

from agent.config import ToolsConfig
from agent.tools.registry import ToolDefinition, ToolTier

if TYPE_CHECKING:
    from agent.channels.base import BaseChannel

logger = structlog.get_logger(__name__)


@dataclass
class PermissionResult:
    """Result of a permission check."""

    approved: bool
    method: str = "auto"  # "auto", "user", "session_approved", "denied"
    reason: str = ""


class PermissionManager:
    """Manages tool execution permissions based on tiers.

    Tiers:
    - SAFE: Always auto-approve (read files, search, memory)
    - MODERATE: Auto-approve by default, can require confirmation in config
    - DANGEROUS: Always requires user confirmation

    In Phase 2 (CLI only), dangerous tools prompt in the terminal.
    In Phase 3+, dangerous tools send approval requests to Telegram/dashboard.
    """

    def __init__(self, config: ToolsConfig) -> None:
        self.config = config
        self._session_approvals: set[str] = set()
        self._approval_channel: BaseChannel | None = None

    def set_approval_channel(self, channel: BaseChannel) -> None:
        """Set the channel used for interactive approval of dangerous tools.

        Args:
            channel: A channel that implements send_approval_request().
        """
        self._approval_channel = channel

    async def check_permission(
        self,
        tool_def: ToolDefinition,
        arguments: dict,
        channel_user_id: str | None = None,
    ) -> PermissionResult:
        """Check if a tool execution is permitted.

        Args:
            tool_def: The tool definition to check.
            arguments: The arguments being passed to the tool.
            channel_user_id: Optional user ID for channel-based approval.

        Returns:
            PermissionResult with .approved, .method, .reason.
        """
        if tool_def.tier == ToolTier.SAFE:
            return PermissionResult(approved=True, method="auto")

        if tool_def.tier == ToolTier.MODERATE:
            # Default: auto-approve moderate tools
            return PermissionResult(approved=True, method="auto")

        if tool_def.tier == ToolTier.DANGEROUS:
            # Check session approvals
            if tool_def.name in self._session_approvals:
                return PermissionResult(approved=True, method="session_approved")

            # Route through channel if available
            if self._approval_channel:
                # If no channel_user_id (e.g. subagent MCP call), try to
                # find a default user from the channel's allowed_users
                effective_user_id = channel_user_id
                if not effective_user_id:
                    effective_user_id = self._get_default_approval_user()
                if effective_user_id:
                    return await self._request_channel_approval(
                        tool_def, arguments, effective_user_id
                    )

            # Fall back to terminal prompt
            return await self._request_approval(tool_def, arguments)

        return PermissionResult(approved=True, method="auto")

    async def _request_approval(
        self, tool_def: ToolDefinition, arguments: dict
    ) -> PermissionResult:
        """Request user approval for a dangerous tool.

        Phase 2: Terminal prompt via input().
        Phase 3+: Will be overridden to use Telegram/dashboard.

        Args:
            tool_def: The tool requesting approval.
            arguments: The arguments for the tool call.

        Returns:
            PermissionResult based on user response.
        """
        # Format the approval request
        args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        prompt_text = (
            f"\n[DANGEROUS] Agent wants to execute: {tool_def.name}({args_str})\n"
            f"[A]pprove / [D]eny / [S]ession (approve all '{tool_def.name}' this session): "
        )

        # Run blocking input in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: input(prompt_text)),
                timeout=120,  # 2 minute approval timeout
            )
        except TimeoutError:
            logger.warning("tool_approval_timeout", tool=tool_def.name)
            return PermissionResult(
                approved=False,
                method="denied",
                reason="Approval timed out after 120 seconds",
            )
        except (EOFError, KeyboardInterrupt):
            return PermissionResult(
                approved=False,
                method="denied",
                reason="User cancelled approval prompt",
            )

        response = response.strip().lower()

        if response in ("a", "approve", "y", "yes"):
            logger.info("tool_approved", tool=tool_def.name, method="user")
            return PermissionResult(approved=True, method="user")
        elif response in ("s", "session"):
            self.approve_for_session(tool_def.name)
            logger.info("tool_session_approved", tool=tool_def.name)
            return PermissionResult(approved=True, method="session_approved")
        else:
            logger.info("tool_denied", tool=tool_def.name)
            return PermissionResult(
                approved=False,
                method="denied",
                reason="User denied execution",
            )

    async def _request_channel_approval(
        self,
        tool_def: ToolDefinition,
        arguments: dict,
        channel_user_id: str,
    ) -> PermissionResult:
        """Request approval through a messaging channel (e.g. Telegram).

        Args:
            tool_def: The tool requesting approval.
            arguments: The arguments for the tool call.
            channel_user_id: The user to ask for approval.

        Returns:
            PermissionResult based on channel response.
        """
        if self._approval_channel is None:
            return PermissionResult(
                approved=False,
                method="denied",
                reason="No approval channel configured",
            )
        request_id = str(uuid4())[:8]

        try:
            approved = await self._approval_channel.send_approval_request(
                channel_user_id=channel_user_id,
                tool_name=tool_def.name,
                arguments=arguments,
                request_id=request_id,
            )
        except Exception as e:
            logger.error("channel_approval_failed", error=str(e))
            return PermissionResult(
                approved=False,
                method="denied",
                reason=f"Channel approval failed: {e}",
            )

        if approved:
            logger.info("tool_channel_approved", tool=tool_def.name)
            return PermissionResult(approved=True, method="user")

        logger.info("tool_channel_denied", tool=tool_def.name)
        return PermissionResult(
            approved=False,
            method="denied",
            reason="User denied via channel",
        )

    def _get_default_approval_user(self) -> str | None:
        """Find a default user ID for channel-based approval.

        When a subagent needs approval but no channel_user_id is in
        context (e.g. MCP tool calls from background workers), look
        for the first allowed user from the approval channel's config.

        Returns:
            User ID string, or None if no default can be determined.
        """
        if not self._approval_channel:
            return None

        # Try allowed_users from channel config
        config = getattr(self._approval_channel, "config", None)
        if config:
            allowed = getattr(config, "allowed_users", [])
            if allowed:
                return str(allowed[0])

        return None

    def approve_for_session(self, tool_name: str) -> None:
        """Approve a tool for the rest of this session.

        Args:
            tool_name: The tool to approve.
        """
        self._session_approvals.add(tool_name)
