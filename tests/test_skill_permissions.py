"""Tests for SkillPermissionManager — tier validation."""

from __future__ import annotations

import pytest

from agent.skills.base import SkillMetadata
from agent.skills.permissions import (
    SkillPermissionError,
    SkillPermissionManager,
)


@pytest.fixture
def pm() -> SkillPermissionManager:
    return SkillPermissionManager()


def test_get_max_tier_safe(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["safe"])
    assert pm.get_max_tier(meta) == "safe"


def test_get_max_tier_moderate(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["safe", "moderate"])
    assert pm.get_max_tier(meta) == "moderate"


def test_get_max_tier_dangerous(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["safe", "moderate", "dangerous"])
    assert pm.get_max_tier(meta) == "dangerous"


def test_get_max_tier_single_dangerous(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["dangerous"])
    assert pm.get_max_tier(meta) == "dangerous"


def test_validate_safe_tool_on_safe_skill(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["safe"])
    # Should not raise
    pm.validate_tool_registration(meta, "my_tool", "safe")


def test_validate_moderate_tool_on_moderate_skill(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["safe", "moderate"])
    pm.validate_tool_registration(meta, "my_tool", "moderate")


def test_validate_safe_tool_on_moderate_skill(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["moderate"])
    pm.validate_tool_registration(meta, "my_tool", "safe")


def test_validate_escalation_blocked(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["safe"])
    with pytest.raises(SkillPermissionError, match="moderate"):
        pm.validate_tool_registration(meta, "my_tool", "moderate")


def test_validate_dangerous_escalation_blocked(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["safe", "moderate"])
    with pytest.raises(SkillPermissionError, match="dangerous"):
        pm.validate_tool_registration(meta, "my_tool", "dangerous")


def test_validate_dangerous_on_dangerous_skill(pm: SkillPermissionManager) -> None:
    meta = SkillMetadata(name="test", permissions=["dangerous"])
    # Should not raise
    pm.validate_tool_registration(meta, "my_tool", "dangerous")
