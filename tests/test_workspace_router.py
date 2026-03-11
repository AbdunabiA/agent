"""Tests for WorkspaceRouter."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import AgentConfig, WorkspacesSection
from agent.workspaces.config import RoutingConfig, RoutingRuleConfig
from agent.workspaces.manager import WorkspaceManager
from agent.workspaces.router import RoutingRule, WorkspaceRouter


@pytest.fixture
def ws_config(tmp_path: Path) -> AgentConfig:
    """AgentConfig with workspaces directory in tmp_path."""
    return AgentConfig(
        workspaces=WorkspacesSection(directory=str(tmp_path / "workspaces")),
    )


@pytest.fixture
def manager(ws_config: AgentConfig) -> WorkspaceManager:
    mgr = WorkspaceManager(ws_config)
    mgr.create("default", display_name="Default", description="Default workspace")
    mgr.create("work", display_name="Work", description="Work workspace")
    mgr.create("personal", display_name="Personal", description="Personal workspace")
    return mgr


class TestRoute:
    def test_exact_match(self, manager: WorkspaceManager) -> None:
        """Exact channel + user_id match returns correct workspace."""
        router = WorkspaceRouter(
            workspace_manager=manager,
            rules=[
                RoutingRule(channel="telegram", workspace="work", user_id="12345"),
                RoutingRule(channel="telegram", workspace="personal", user_id="67890"),
            ],
            default="default",
        )

        ws = router.route("telegram", "12345")
        assert ws.name == "work"

        ws = router.route("telegram", "67890")
        assert ws.name == "personal"

    def test_channel_default(self, manager: WorkspaceManager) -> None:
        """Channel-only rule (no user_id) acts as channel default."""
        router = WorkspaceRouter(
            workspace_manager=manager,
            rules=[
                RoutingRule(channel="telegram", workspace="work", user_id="12345"),
                RoutingRule(channel="telegram", workspace="personal"),
            ],
            default="default",
        )

        # Known user → exact match
        ws = router.route("telegram", "12345")
        assert ws.name == "work"

        # Unknown user → channel default
        ws = router.route("telegram", "99999")
        assert ws.name == "personal"

    def test_wildcard_channel(self, manager: WorkspaceManager) -> None:
        """Wildcard '*' channel matches any unmatched channel."""
        router = WorkspaceRouter(
            workspace_manager=manager,
            rules=[
                RoutingRule(channel="telegram", workspace="work"),
                RoutingRule(channel="*", workspace="personal"),
            ],
            default="default",
        )

        ws = router.route("telegram", "123")
        assert ws.name == "work"

        ws = router.route("webchat", "abc")
        assert ws.name == "personal"

    def test_global_default(self, manager: WorkspaceManager) -> None:
        """No matching rule falls through to global default."""
        router = WorkspaceRouter(
            workspace_manager=manager,
            rules=[
                RoutingRule(channel="telegram", workspace="work", user_id="12345"),
            ],
            default="default",
        )

        ws = router.route("webchat", "abc")
        assert ws.name == "default"

    def test_workspace_cached(self, manager: WorkspaceManager) -> None:
        """Resolved workspaces are cached."""
        router = WorkspaceRouter(
            workspace_manager=manager,
            rules=[RoutingRule(channel="telegram", workspace="work")],
            default="default",
        )

        ws1 = router.route("telegram", "123")
        ws2 = router.route("telegram", "456")
        assert ws1 is ws2  # Same cached object

    def test_invalidate_cache(self, manager: WorkspaceManager) -> None:
        """invalidate_cache clears cached workspaces."""
        router = WorkspaceRouter(
            workspace_manager=manager,
            rules=[RoutingRule(channel="telegram", workspace="work")],
            default="default",
        )

        ws1 = router.route("telegram", "123")
        router.invalidate_cache("work")
        ws2 = router.route("telegram", "123")
        assert ws1 is not ws2

    def test_invalidate_all_cache(self, manager: WorkspaceManager) -> None:
        """invalidate_cache with no args clears entire cache."""
        router = WorkspaceRouter(
            workspace_manager=manager,
            rules=[RoutingRule(channel="telegram", workspace="work")],
            default="default",
        )

        router.route("telegram", "123")
        assert len(router._workspace_cache) > 0
        router.invalidate_cache()
        assert len(router._workspace_cache) == 0


class TestFromConfig:
    def test_parses_rules(self, manager: WorkspaceManager) -> None:
        """from_config correctly parses RoutingConfig."""
        routing_config = RoutingConfig(
            default="default",
            rules=[
                RoutingRuleConfig(
                    channel="telegram", workspace="work", user_id="12345"
                ),
                RoutingRuleConfig(channel="webchat", workspace="default"),
            ],
        )

        router = WorkspaceRouter.from_config(manager, routing_config)

        assert len(router.rules) == 2
        assert router.rules[0].channel == "telegram"
        assert router.rules[0].user_id == "12345"
        assert router.rules[0].workspace == "work"
        assert router.rules[1].channel == "webchat"
        assert router.rules[1].user_id is None
        assert router.default == "default"

    def test_empty_config(self, manager: WorkspaceManager) -> None:
        """from_config handles empty routing config."""
        routing_config = RoutingConfig()
        router = WorkspaceRouter.from_config(manager, routing_config)
        assert router.rules == []
        assert router.default == "default"

    def test_routing_works_after_from_config(
        self, manager: WorkspaceManager,
    ) -> None:
        """Router created from config routes correctly."""
        routing_config = RoutingConfig(
            default="default",
            rules=[
                RoutingRuleConfig(
                    channel="telegram", workspace="work", user_id="12345"
                ),
                RoutingRuleConfig(channel="telegram", workspace="personal"),
            ],
        )

        router = WorkspaceRouter.from_config(manager, routing_config)

        assert router.route("telegram", "12345").name == "work"
        assert router.route("telegram", "99999").name == "personal"
        assert router.route("webchat", "abc").name == "default"
