"""Tests for the tiered permission system."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.config import ToolsConfig
from agent.core.permissions import PermissionManager
from agent.tools.registry import ToolDefinition, ToolTier


@pytest.fixture
def permissions() -> PermissionManager:
    return PermissionManager(ToolsConfig())


def _make_tool_def(name: str, tier: ToolTier) -> ToolDefinition:
    """Helper to create a ToolDefinition for testing."""

    async def dummy() -> str:
        return ""

    return ToolDefinition(
        name=name,
        description="Test tool",
        tier=tier,
        parameters={"type": "object", "properties": {}},
        function=dummy,
    )


class TestPermissionManager:
    """Tests for PermissionManager."""

    async def test_safe_tier_auto_approved(self, permissions: PermissionManager) -> None:
        """SAFE tier tools should always be auto-approved."""
        tool_def = _make_tool_def("safe_tool", ToolTier.SAFE)
        result = await permissions.check_permission(tool_def, {})

        assert result.approved is True
        assert result.method == "auto"

    async def test_moderate_tier_auto_approved(self, permissions: PermissionManager) -> None:
        """MODERATE tier tools should be auto-approved by default."""
        tool_def = _make_tool_def("moderate_tool", ToolTier.MODERATE)
        result = await permissions.check_permission(tool_def, {})

        assert result.approved is True
        assert result.method == "auto"

    async def test_dangerous_tier_requires_approval(
        self, permissions: PermissionManager
    ) -> None:
        """DANGEROUS tier tools should require user approval."""
        tool_def = _make_tool_def("dangerous_tool", ToolTier.DANGEROUS)

        # Simulate user approving
        with patch("builtins.input", return_value="a"):
            result = await permissions.check_permission(tool_def, {"cmd": "rm -rf /"})

        assert result.approved is True
        assert result.method == "user"

    async def test_dangerous_tier_denied(self, permissions: PermissionManager) -> None:
        """User can deny dangerous tool execution."""
        tool_def = _make_tool_def("dangerous_tool", ToolTier.DANGEROUS)

        with patch("builtins.input", return_value="d"):
            result = await permissions.check_permission(tool_def, {})

        assert result.approved is False
        assert result.method == "denied"

    async def test_session_approval(self, permissions: PermissionManager) -> None:
        """Session approval should auto-approve subsequent calls."""
        tool_def = _make_tool_def("dangerous_tool", ToolTier.DANGEROUS)

        # Simulate user choosing session approval
        with patch("builtins.input", return_value="s"):
            result1 = await permissions.check_permission(tool_def, {})

        assert result1.approved is True
        assert result1.method == "session_approved"

        # Second call should be auto-approved without prompt
        result2 = await permissions.check_permission(tool_def, {})
        assert result2.approved is True
        assert result2.method == "session_approved"

    def test_approve_for_session(self, permissions: PermissionManager) -> None:
        """approve_for_session should add tool to session approvals."""
        permissions.approve_for_session("my_tool")
        assert "my_tool" in permissions._session_approvals

    async def test_eof_error_denied(self, permissions: PermissionManager) -> None:
        """EOFError during approval should deny."""
        tool_def = _make_tool_def("dangerous_tool", ToolTier.DANGEROUS)

        with patch("builtins.input", side_effect=EOFError):
            result = await permissions.check_permission(tool_def, {})

        assert result.approved is False
