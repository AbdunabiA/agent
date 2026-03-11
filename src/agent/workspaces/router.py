"""Routes incoming messages from channels to the correct workspace.

Routing rules (in priority order):
1. Explicit mapping: Telegram user 12345 -> "work" workspace
2. Channel default: all Telegram messages -> "personal" workspace
3. Global default: unmatched -> "default" workspace

Configuration in agent.yaml::

    workspaces:
      routing:
        rules:
          - channel: telegram
            user_id: "12345"
            workspace: work
          - channel: webchat
            workspace: default
        default: default
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from agent.workspaces.config import ResolvedWorkspace, RoutingConfig

if TYPE_CHECKING:
    from agent.workspaces.manager import WorkspaceManager

logger = structlog.get_logger(__name__)


@dataclass
class RoutingRule:
    """A single routing rule."""

    channel: str
    workspace: str
    user_id: str | None = None
    pattern: str | None = None


class WorkspaceRouter:
    """Routes incoming messages to the correct workspace."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        rules: list[RoutingRule],
        default: str = "default",
    ) -> None:
        self.workspace_manager = workspace_manager
        self.rules = rules
        self.default = default
        self._workspace_cache: dict[str, ResolvedWorkspace] = {}

    def route(
        self,
        channel: str,
        user_id: str,
        message: str = "",
    ) -> ResolvedWorkspace:
        """Determine which workspace should handle this message.

        Priority:
        1. Exact match: channel + user_id
        2. Channel default: channel only (no user_id in rule)
        3. Wildcard channel
        4. Global default

        Args:
            channel: Channel name (e.g. "telegram", "webchat").
            user_id: User identifier within the channel.
            message: Message text (reserved for future pattern matching).

        Returns:
            The resolved workspace for this message.
        """
        # 1. Exact match (channel + user_id)
        for rule in self.rules:
            if rule.channel == channel and rule.user_id == user_id:
                logger.debug(
                    "route_exact_match",
                    channel=channel,
                    user_id=user_id,
                    workspace=rule.workspace,
                )
                return self._resolve(rule.workspace)

        # 2. Channel default (no user_id)
        for rule in self.rules:
            if rule.channel == channel and rule.user_id is None:
                logger.debug(
                    "route_channel_default",
                    channel=channel,
                    workspace=rule.workspace,
                )
                return self._resolve(rule.workspace)

        # 3. Wildcard channel
        for rule in self.rules:
            if rule.channel == "*" and rule.user_id is None:
                logger.debug(
                    "route_wildcard",
                    channel=channel,
                    workspace=rule.workspace,
                )
                return self._resolve(rule.workspace)

        # 4. Global default
        logger.debug(
            "route_global_default",
            channel=channel,
            user_id=user_id,
            workspace=self.default,
        )
        return self._resolve(self.default)

    def _resolve(self, workspace_name: str) -> ResolvedWorkspace:
        """Resolve and cache a workspace by name."""
        if workspace_name not in self._workspace_cache:
            self._workspace_cache[workspace_name] = (
                self.workspace_manager.resolve(workspace_name)
            )
        return self._workspace_cache[workspace_name]

    def invalidate_cache(self, workspace_name: str | None = None) -> None:
        """Clear cached workspace resolutions.

        Args:
            workspace_name: If given, only invalidate that workspace.
                            If None, clear the entire cache.
        """
        if workspace_name:
            self._workspace_cache.pop(workspace_name, None)
        else:
            self._workspace_cache.clear()

    @classmethod
    def from_config(
        cls,
        workspace_manager: WorkspaceManager,
        routing_config: RoutingConfig,
    ) -> WorkspaceRouter:
        """Create a router from a RoutingConfig object.

        Args:
            workspace_manager: The workspace manager to resolve workspaces.
            routing_config: Parsed routing configuration.

        Returns:
            Configured WorkspaceRouter instance.
        """
        rules: list[RoutingRule] = []
        for rule_cfg in routing_config.rules:
            rules.append(
                RoutingRule(
                    channel=rule_cfg.channel,
                    workspace=rule_cfg.workspace,
                    user_id=rule_cfg.user_id,
                    pattern=rule_cfg.pattern,
                )
            )

        return cls(
            workspace_manager=workspace_manager,
            rules=rules,
            default=routing_config.default,
        )
