"""Tests for the health diagnostic system."""

from __future__ import annotations

import pytest

from agent.core.doctor import HealthCheck, check_core, check_resources, check_security


@pytest.fixture
def config():
    """Minimal config for doctor tests."""
    from agent.config import get_config

    return get_config()


def test_health_check_dataclass() -> None:
    check = HealthCheck(
        name="Test",
        category="Core",
        status="pass",
        message="All good",
    )
    assert check.name == "Test"
    assert check.status == "pass"
    assert check.details == ""


def test_health_check_with_details() -> None:
    check = HealthCheck(
        name="Test",
        category="Core",
        status="fail",
        message="Error",
        details="Stack trace here",
    )
    assert check.details == "Stack trace here"


def test_check_core_returns_checks(config) -> None:
    checks = check_core(config)
    assert len(checks) > 0
    names = [c.name for c in checks]
    assert "Python Version" in names
    assert "Agent Version" in names
    assert "Operating System" in names


def test_check_core_all_have_category(config) -> None:
    checks = check_core(config)
    for check in checks:
        assert check.category == "Core"


def test_check_resources(config) -> None:
    checks = check_resources(config)
    # Should at least have disk space
    names = [c.name for c in checks]
    assert "Disk Space" in names


def test_check_security(config) -> None:
    checks = check_security(config)
    assert len(checks) > 0
    categories = {c.category for c in checks}
    assert categories == {"Security"}


@pytest.mark.asyncio
async def test_run_all_checks(config) -> None:
    from agent.core.doctor import run_all_checks

    checks = await run_all_checks(config)
    assert len(checks) > 0
    # All checks should have valid status
    for check in checks:
        assert check.status in ("pass", "warn", "fail")
        assert check.name
        assert check.category
