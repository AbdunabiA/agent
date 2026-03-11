"""Tests for self-healing error recovery."""

from __future__ import annotations

import pytest

from agent.core.recovery import ErrorCategory, ErrorRecovery


@pytest.fixture
def recovery() -> ErrorRecovery:
    return ErrorRecovery()


class TestErrorClassification:
    """Tests for error classification."""

    def test_timeout_is_transient(self, recovery: ErrorRecovery) -> None:
        error = TimeoutError("connection timed out")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.TRANSIENT

    def test_asyncio_timeout_is_transient(self, recovery: ErrorRecovery) -> None:
        error = TimeoutError()
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.TRANSIENT

    def test_connection_error_is_transient(self, recovery: ErrorRecovery) -> None:
        error = ConnectionError("connection refused")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.TRANSIENT

    def test_rate_limit_is_transient(self, recovery: ErrorRecovery) -> None:
        error = Exception("rate limit exceeded")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.TRANSIENT

    def test_permission_error(self, recovery: ErrorRecovery) -> None:
        error = PermissionError("access denied")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.PERMISSION

    def test_file_not_found_is_logic(self, recovery: ErrorRecovery) -> None:
        error = FileNotFoundError("no such file")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.LOGIC

    def test_value_error_is_logic(self, recovery: ErrorRecovery) -> None:
        error = ValueError("invalid argument")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.LOGIC

    def test_memory_error_is_resource(self, recovery: ErrorRecovery) -> None:
        error = MemoryError("out of memory")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.RESOURCE

    def test_401_is_auth(self, recovery: ErrorRecovery) -> None:
        error = Exception("HTTP 401 Unauthorized")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.AUTH

    def test_unknown_error(self, recovery: ErrorRecovery) -> None:
        error = RuntimeError("something weird happened")
        category = recovery.classify_error(error, "test_tool")
        assert category == ErrorCategory.UNKNOWN


class TestRecoveryActions:
    """Tests for recovery action decisions."""

    def test_transient_retry_with_backoff(self, recovery: ErrorRecovery) -> None:
        error = TimeoutError("timeout")
        category = ErrorCategory.TRANSIENT
        action = recovery.get_recovery_action(error, "test_tool", category)

        assert action.action == "retry"
        assert action.delay_seconds == 1  # First retry: 2^0

    def test_transient_increasing_backoff(self, recovery: ErrorRecovery) -> None:
        error = TimeoutError("timeout")
        category = ErrorCategory.TRANSIENT

        action1 = recovery.get_recovery_action(error, "test_tool", category)
        assert action1.delay_seconds == 1  # 2^0

        action2 = recovery.get_recovery_action(error, "test_tool", category)
        assert action2.delay_seconds == 2  # 2^1

        action3 = recovery.get_recovery_action(error, "test_tool", category)
        assert action3.delay_seconds == 4  # 2^2

    def test_transient_max_retries_exceeded(self, recovery: ErrorRecovery) -> None:
        error = TimeoutError("timeout")
        category = ErrorCategory.TRANSIENT

        # Exhaust retries
        for _ in range(3):
            recovery.get_recovery_action(error, "test_tool", category)

        action = recovery.get_recovery_action(error, "test_tool", category)
        assert action.action == "notify_user"

    def test_auth_notifies_user(self, recovery: ErrorRecovery) -> None:
        error = Exception("401 Unauthorized")
        category = ErrorCategory.AUTH
        action = recovery.get_recovery_action(error, "test_tool", category)

        assert action.action == "notify_user"
        assert "authentication" in action.reason.lower() or "credentials" in action.reason.lower()

    def test_permission_notifies_user(self, recovery: ErrorRecovery) -> None:
        error = PermissionError("access denied")
        category = ErrorCategory.PERMISSION
        action = recovery.get_recovery_action(error, "test_tool", category)

        assert action.action == "notify_user"

    def test_resource_notifies_user(self, recovery: ErrorRecovery) -> None:
        error = MemoryError()
        category = ErrorCategory.RESOURCE
        action = recovery.get_recovery_action(error, "test_tool", category)

        assert action.action == "notify_user"

    def test_reset_retries(self, recovery: ErrorRecovery) -> None:
        """Reset retries should clear the count."""
        error = TimeoutError("timeout")
        category = ErrorCategory.TRANSIENT

        # Use some retries
        recovery.get_recovery_action(error, "test_tool", category)
        recovery.get_recovery_action(error, "test_tool", category)

        # Reset
        recovery.reset_retries("test_tool")

        # Should start from 0 again
        action = recovery.get_recovery_action(error, "test_tool", category)
        assert action.delay_seconds == 1  # 2^0 = first retry
