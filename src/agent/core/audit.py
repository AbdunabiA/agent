"""Action audit log for tool executions.

Logs all tool executions for accountability and debugging.
Phase 2: In-memory storage.
Phase 4: Upgraded to SQLite persistence with in-memory fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

if TYPE_CHECKING:
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)


@dataclass
class AuditEntry:
    """A single audit log entry."""

    id: str
    timestamp: datetime
    tool_name: str
    tool_call_id: str
    input_data: dict
    output: str
    status: str  # "success", "error", "timeout", "denied", "blocked"
    duration_ms: int
    trigger: str  # "user_message", "heartbeat", "cron"
    session_id: str
    approved_by: str
    error: str | None = None


class AuditLog:
    """Logs all tool executions for accountability and debugging.

    When a Database is provided, entries are persisted to SQLite.
    Falls back to in-memory storage when no database is available.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._entries: list[AuditEntry] = []

    async def log(
        self,
        tool_name: str,
        tool_call_id: str,
        input_data: dict,
        output: str,
        status: str,
        duration_ms: int,
        trigger: str,
        session_id: str,
        approved_by: str = "auto",
        error: str | None = None,
    ) -> AuditEntry:
        """Log a tool execution.

        Args:
            tool_name: Name of the executed tool.
            tool_call_id: ID of the tool call from the LLM.
            input_data: Input arguments dict.
            output: Tool output (truncated to 10KB).
            status: Execution status.
            duration_ms: Execution duration in milliseconds.
            trigger: What triggered the execution.
            session_id: Session identifier.
            approved_by: How the execution was approved.
            error: Error message if failed.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            id=str(uuid4()),
            timestamp=datetime.now(),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            input_data=input_data,
            output=output[:10240],  # Truncate output to 10KB in audit
            status=status,
            duration_ms=duration_ms,
            trigger=trigger,
            session_id=session_id,
            approved_by=approved_by,
            error=error,
        )

        if self._db:
            await self._db.db.execute(
                """INSERT INTO audit_log
                   (id, timestamp, tool_name, tool_call_id, input_data, output,
                    status, duration_ms, trigger, session_id, approved_by, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.id,
                    entry.timestamp.isoformat(),
                    entry.tool_name,
                    entry.tool_call_id,
                    json.dumps(entry.input_data),
                    entry.output,
                    entry.status,
                    entry.duration_ms,
                    entry.trigger,
                    entry.session_id,
                    entry.approved_by,
                    entry.error,
                ),
            )
            await self._db.db.commit()
        else:
            self._entries.append(entry)

        logger.debug(
            "audit_logged",
            tool=tool_name,
            status=status,
            duration_ms=duration_ms,
        )

        return entry

    async def get_entries(
        self,
        limit: int = 50,
        tool_name: str | None = None,
        status: str | None = None,
    ) -> list[AuditEntry]:
        """Query audit log entries with optional filters.

        Args:
            limit: Maximum number of entries to return.
            tool_name: Filter by tool name.
            status: Filter by status.

        Returns:
            List of matching AuditEntry objects (newest first).
        """
        if self._db:
            return await self._get_entries_from_db(limit, tool_name, status)

        entries = self._entries
        if tool_name:
            entries = [e for e in entries if e.tool_name == tool_name]
        if status:
            entries = [e for e in entries if e.status == status]
        return list(reversed(entries[-limit:]))

    async def _get_entries_from_db(
        self,
        limit: int,
        tool_name: str | None,
        status: str | None,
    ) -> list[AuditEntry]:
        """Query entries from SQLite."""
        query = "SELECT * FROM audit_log"
        conditions: list[str] = []
        params: list[Any] = []

        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)
        if status:
            conditions.append("status = ?")
            params.append(status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self._db.db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            AuditEntry(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                tool_name=row["tool_name"],
                tool_call_id=row["tool_call_id"],
                input_data=json.loads(row["input_data"]),
                output=row["output"],
                status=row["status"],
                duration_ms=row["duration_ms"],
                trigger=row["trigger"],
                session_id=row["session_id"],
                approved_by=row["approved_by"],
                error=row["error"],
            )
            for row in rows
        ]

    async def get_stats(self) -> dict[str, Any]:
        """Get audit statistics.

        Returns:
            Dict with total_calls, success_count, error_count,
            success_rate, avg_duration_ms, tools_used.
        """
        empty_stats: dict[str, Any] = {
            "total_calls": 0,
            "success_count": 0,
            "error_count": 0,
            "success_rate": 0.0,
            "avg_duration_ms": 0,
            "tools_used": {},
        }

        if self._db:
            return await self._get_stats_from_db(empty_stats)

        if not self._entries:
            return empty_stats

        total = len(self._entries)
        success_count = sum(1 for e in self._entries if e.status == "success")
        error_count = sum(1 for e in self._entries if e.status == "error")
        avg_duration = sum(e.duration_ms for e in self._entries) / total

        tools_used: dict[str, int] = {}
        for entry in self._entries:
            tools_used[entry.tool_name] = tools_used.get(entry.tool_name, 0) + 1

        return {
            "total_calls": total,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / total if total > 0 else 0.0,
            "avg_duration_ms": int(avg_duration),
            "tools_used": tools_used,
        }

    async def _get_stats_from_db(self, empty_stats: dict[str, Any]) -> dict[str, Any]:
        """Calculate stats from SQLite."""
        async with self._db.db.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                AVG(duration_ms) as avg_duration
               FROM audit_log"""
        ) as cursor:
            row = await cursor.fetchone()

        if not row or row["total"] == 0:
            return empty_stats

        total = row["total"]
        success_count = row["success_count"]

        # Get per-tool counts
        async with self._db.db.execute(
            "SELECT tool_name, COUNT(*) as cnt FROM audit_log GROUP BY tool_name"
        ) as cursor:
            tool_rows = await cursor.fetchall()

        tools_used = {r["tool_name"]: r["cnt"] for r in tool_rows}

        return {
            "total_calls": total,
            "success_count": success_count,
            "error_count": row["error_count"],
            "success_rate": success_count / total if total > 0 else 0.0,
            "avg_duration_ms": int(row["avg_duration"]),
            "tools_used": tools_used,
        }
