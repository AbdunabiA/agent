"""Tests for the SQLite-backed audit log."""

from __future__ import annotations

import pytest

from agent.core.audit import AuditLog
from agent.memory.database import Database


@pytest.fixture
async def db(tmp_path: object) -> Database:
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "audit_test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def audit(db: Database) -> AuditLog:
    """AuditLog with SQLite backend."""
    return AuditLog(db=db)


async def _log_entry(audit: AuditLog, **kwargs: object) -> object:
    """Helper to log an entry with defaults."""
    defaults = {
        "tool_name": "test_tool",
        "tool_call_id": "call_1",
        "input_data": {},
        "output": "ok",
        "status": "success",
        "duration_ms": 50,
        "trigger": "user_message",
        "session_id": "session_1",
        "approved_by": "auto",
    }
    defaults.update(kwargs)
    return await audit.log(**defaults)


class TestAuditSQLite:
    """Tests for SQLite-backed AuditLog."""

    async def test_entry_persisted(self, audit: AuditLog) -> None:
        """Entry should be persisted to SQLite."""
        entry = await _log_entry(audit, tool_name="shell_exec")

        entries = await audit.get_entries()
        assert len(entries) == 1
        assert entries[0].tool_name == "shell_exec"
        assert entries[0].id == entry.id

    async def test_entries_survive_reconnect(self, tmp_path: object) -> None:
        """Entries should survive closing and reopening the database."""
        db_path = str(tmp_path / "persist_test.db")

        # Write entry
        db1 = Database(db_path)
        await db1.connect()
        audit1 = AuditLog(db=db1)
        await _log_entry(audit1, tool_name="survived")
        await db1.close()

        # Reopen and verify
        db2 = Database(db_path)
        await db2.connect()
        audit2 = AuditLog(db=db2)
        entries = await audit2.get_entries()

        assert len(entries) == 1
        assert entries[0].tool_name == "survived"
        await db2.close()

    async def test_filter_by_tool_name(self, audit: AuditLog) -> None:
        """Should filter entries by tool_name."""
        await _log_entry(audit, tool_name="tool_a")
        await _log_entry(audit, tool_name="tool_b")

        entries = await audit.get_entries(tool_name="tool_a")
        assert len(entries) == 1
        assert entries[0].tool_name == "tool_a"

    async def test_filter_by_status(self, audit: AuditLog) -> None:
        """Should filter entries by status."""
        await _log_entry(audit, status="success")
        await _log_entry(audit, status="error", error="failed")

        entries = await audit.get_entries(status="error")
        assert len(entries) == 1
        assert entries[0].status == "error"

    async def test_combined_filters(self, audit: AuditLog) -> None:
        """Should support multiple filters combined."""
        await _log_entry(audit, tool_name="a", status="success")
        await _log_entry(audit, tool_name="a", status="error")
        await _log_entry(audit, tool_name="b", status="success")

        entries = await audit.get_entries(tool_name="a", status="error")
        assert len(entries) == 1

    async def test_entries_newest_first(self, audit: AuditLog) -> None:
        """Entries should be returned newest first."""
        await _log_entry(audit, tool_name="first")
        await _log_entry(audit, tool_name="second")

        entries = await audit.get_entries()
        assert entries[0].tool_name == "second"
        assert entries[1].tool_name == "first"

    async def test_entries_limit(self, audit: AuditLog) -> None:
        """Should respect the limit parameter."""
        for i in range(10):
            await _log_entry(audit, tool_call_id=str(i))

        entries = await audit.get_entries(limit=5)
        assert len(entries) == 5

    async def test_stats_from_sqlite(self, audit: AuditLog) -> None:
        """Stats should be calculated from SQLite data."""
        await _log_entry(audit, tool_name="t1", status="success", duration_ms=100)
        await _log_entry(audit, tool_name="t1", status="success", duration_ms=200)
        await _log_entry(
            audit, tool_name="t2", status="error", duration_ms=50, error="fail"
        )

        stats = await audit.get_stats()
        assert stats["total_calls"] == 3
        assert stats["success_count"] == 2
        assert stats["error_count"] == 1
        assert abs(stats["success_rate"] - 2 / 3) < 0.01
        assert stats["avg_duration_ms"] == 116  # (100+200+50)/3
        assert stats["tools_used"]["t1"] == 2
        assert stats["tools_used"]["t2"] == 1

    async def test_stats_empty(self, audit: AuditLog) -> None:
        """Stats on empty log should return zeros."""
        stats = await audit.get_stats()
        assert stats["total_calls"] == 0
        assert stats["success_rate"] == 0.0

    async def test_input_data_persisted_as_json(self, audit: AuditLog) -> None:
        """Input data should be serialized as JSON and round-trip correctly."""
        await _log_entry(audit, input_data={"command": "ls", "args": ["-la"]})

        entries = await audit.get_entries()
        assert entries[0].input_data == {"command": "ls", "args": ["-la"]}


class TestAuditFallback:
    """Tests for in-memory fallback when no database."""

    async def test_fallback_to_memory(self) -> None:
        """Should work without database (in-memory fallback)."""
        audit = AuditLog()  # No db
        await _log_entry(audit, tool_name="memory_tool")

        entries = await audit.get_entries()
        assert len(entries) == 1
        assert entries[0].tool_name == "memory_tool"

    async def test_fallback_stats(self) -> None:
        """Stats should work in memory-only mode."""
        audit = AuditLog()
        await _log_entry(audit, status="success", duration_ms=100)

        stats = await audit.get_stats()
        assert stats["total_calls"] == 1
        assert stats["success_count"] == 1
