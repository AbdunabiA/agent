"""Tests for the audit log."""

from __future__ import annotations

import pytest

from agent.core.audit import AuditLog


@pytest.fixture
def audit() -> AuditLog:
    return AuditLog()


class TestAuditLog:
    """Tests for AuditLog (in-memory fallback mode)."""

    async def test_log_entry_created(self, audit: AuditLog) -> None:
        """Logging should create an entry with all fields."""
        entry = await audit.log(
            tool_name="shell_exec",
            tool_call_id="call_1",
            input_data={"command": "ls"},
            output="file1.txt\nfile2.txt",
            status="success",
            duration_ms=50,
            trigger="user_message",
            session_id="session_1",
            approved_by="auto",
        )

        assert entry.tool_name == "shell_exec"
        assert entry.tool_call_id == "call_1"
        assert entry.status == "success"
        assert entry.duration_ms == 50
        assert entry.trigger == "user_message"
        assert entry.session_id == "session_1"
        assert entry.id  # Should have a UUID

    async def test_output_truncated_to_10kb(self, audit: AuditLog) -> None:
        """Output should be truncated to 10KB in audit entries."""
        big_output = "x" * 20_000
        entry = await audit.log(
            tool_name="test",
            tool_call_id="call_2",
            input_data={},
            output=big_output,
            status="success",
            duration_ms=0,
            trigger="user_message",
            session_id="s1",
        )

        assert len(entry.output) == 10240  # 10KB

    async def test_query_by_tool_name(self, audit: AuditLog) -> None:
        """Should filter entries by tool name."""
        await audit.log(
            tool_name="tool_a", tool_call_id="1", input_data={},
            output="a", status="success", duration_ms=0,
            trigger="user_message", session_id="s1",
        )
        await audit.log(
            tool_name="tool_b", tool_call_id="2", input_data={},
            output="b", status="success", duration_ms=0,
            trigger="user_message", session_id="s1",
        )

        entries = await audit.get_entries(tool_name="tool_a")
        assert len(entries) == 1
        assert entries[0].tool_name == "tool_a"

    async def test_query_by_status(self, audit: AuditLog) -> None:
        """Should filter entries by status."""
        await audit.log(
            tool_name="t1", tool_call_id="1", input_data={},
            output="ok", status="success", duration_ms=10,
            trigger="user_message", session_id="s1",
        )
        await audit.log(
            tool_name="t2", tool_call_id="2", input_data={},
            output="fail", status="error", duration_ms=5,
            trigger="user_message", session_id="s1",
        )

        entries = await audit.get_entries(status="error")
        assert len(entries) == 1
        assert entries[0].status == "error"

    async def test_get_entries_limit(self, audit: AuditLog) -> None:
        """Should respect the limit parameter."""
        for i in range(10):
            await audit.log(
                tool_name="t", tool_call_id=str(i), input_data={},
                output="", status="success", duration_ms=0,
                trigger="user_message", session_id="s1",
            )

        entries = await audit.get_entries(limit=5)
        assert len(entries) == 5

    async def test_entries_newest_first(self, audit: AuditLog) -> None:
        """Entries should be returned newest first."""
        await audit.log(
            tool_name="first", tool_call_id="1", input_data={},
            output="", status="success", duration_ms=0,
            trigger="user_message", session_id="s1",
        )
        await audit.log(
            tool_name="second", tool_call_id="2", input_data={},
            output="", status="success", duration_ms=0,
            trigger="user_message", session_id="s1",
        )

        entries = await audit.get_entries()
        assert entries[0].tool_name == "second"
        assert entries[1].tool_name == "first"

    async def test_stats_empty(self, audit: AuditLog) -> None:
        """Stats on empty log should return zeros."""
        stats = await audit.get_stats()
        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 0.0

    async def test_stats_calculation(self, audit: AuditLog) -> None:
        """Stats should calculate correctly."""
        await audit.log(
            tool_name="t1", tool_call_id="1", input_data={},
            output="ok", status="success", duration_ms=100,
            trigger="user_message", session_id="s1",
        )
        await audit.log(
            tool_name="t1", tool_call_id="2", input_data={},
            output="ok", status="success", duration_ms=200,
            trigger="user_message", session_id="s1",
        )
        await audit.log(
            tool_name="t2", tool_call_id="3", input_data={},
            output="err", status="error", duration_ms=50,
            trigger="user_message", session_id="s1", error="failed",
        )

        stats = await audit.get_stats()
        assert stats["total_calls"] == 3
        assert stats["success_count"] == 2
        assert stats["error_count"] == 1
        assert abs(stats["success_rate"] - 2 / 3) < 0.01
        assert stats["avg_duration_ms"] == 116  # (100+200+50)/3
        assert stats["tools_used"]["t1"] == 2
        assert stats["tools_used"]["t2"] == 1
