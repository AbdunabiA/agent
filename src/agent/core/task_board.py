"""Task board — shared inter-agent task posting and tracking.

Agents can post tasks to each other on a shared board (e.g. QA finds a bug
and posts a ticket to backend_developer). Tickets are scoped to a parent
task_id so each orchestration run gets its own board view.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog

if TYPE_CHECKING:
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)


class TaskBoard:
    """Shared task board for inter-agent collaboration.

    Agents post tickets to each other, pick up work, and mark it done.
    All tickets are persisted to SQLite via the ``task_tickets`` table.

    Args:
        database: The project's async SQLite database handle.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------
    # Schema bootstrap (called from Database._migrate)
    # ------------------------------------------------------------------

    @staticmethod
    def migration_sql() -> str:
        """Return the CREATE TABLE statements for the task_tickets table."""
        return """
CREATE TABLE IF NOT EXISTS task_tickets (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    from_role TEXT NOT NULL,
    to_role TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'pending',
    context_json TEXT NOT NULL DEFAULT '{}',
    result TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_task_role_status
    ON task_tickets(task_id, to_role, status);
CREATE INDEX IF NOT EXISTS idx_tickets_task_status
    ON task_tickets(task_id, status);
"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def post_task(
        self,
        from_role: str,
        to_role: str,
        task_id: str,
        title: str,
        description: str,
        priority: str = "normal",
        context: dict | None = None,
    ) -> str:
        """Post a new ticket to the board.

        Args:
            from_role: Role posting the ticket.
            to_role: Role the ticket is assigned to.
            task_id: Parent orchestration task ID.
            title: Short ticket title.
            description: Detailed description.
            priority: 'blocker', 'normal', or 'low'.
            context: Optional structured context (files, snippets, etc.).

        Returns:
            The generated ticket ID.
        """
        ticket_id = f"tkt-{uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()
        context_json = json.dumps(context or {})

        await self._db.db.execute(
            """
            INSERT INTO task_tickets
                (id, task_id, from_role, to_role, title, description,
                 priority, status, context_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                ticket_id,
                task_id,
                from_role,
                to_role,
                title,
                description,
                priority,
                context_json,
                now,
            ),
        )
        await self._db.db.commit()

        logger.info(
            "task_board_ticket_posted",
            ticket_id=ticket_id,
            task_id=task_id,
            from_role=from_role,
            to_role=to_role,
            priority=priority,
        )
        return ticket_id

    async def get_my_tasks(self, role: str, task_id: str) -> list[dict]:
        """Get pending tickets assigned to a role.

        Args:
            role: The role to fetch tickets for.
            task_id: Parent orchestration task ID.

        Returns:
            List of ticket dicts ordered by priority DESC, created_at ASC.
        """
        priority_order = (
            "CASE priority"
            " WHEN 'blocker' THEN 0"
            " WHEN 'normal' THEN 1"
            " WHEN 'low' THEN 2"
            " ELSE 1 END"
        )
        async with self._db.db.execute(
            f"""
            SELECT id, task_id, from_role, to_role, title, description,
                   priority, status, context_json, result, created_at, completed_at
            FROM task_tickets
            WHERE task_id = ? AND to_role = ? AND status = 'pending'
            ORDER BY {priority_order} ASC, created_at ASC
            """,
            (task_id, role),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "task_id": row[1],
                "from_role": row[2],
                "to_role": row[3],
                "title": row[4],
                "description": row[5],
                "priority": row[6],
                "status": row[7],
                "context": json.loads(row[8] or "{}"),
                "result": row[9],
                "created_at": row[10],
                "completed_at": row[11],
            }
            for row in rows
        ]

    async def get_ticket(self, ticket_id: str) -> dict | None:
        """Look up a single ticket by ID.

        Args:
            ticket_id: The ticket ID.

        Returns:
            Ticket dict, or None if not found.
        """
        async with self._db.db.execute(
            """
            SELECT id, task_id, from_role, to_role, title, description,
                   priority, status, context_json, result, created_at, completed_at
            FROM task_tickets WHERE id = ?
            """,
            (ticket_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None
        return {
            "id": row[0],
            "task_id": row[1],
            "from_role": row[2],
            "to_role": row[3],
            "title": row[4],
            "description": row[5],
            "priority": row[6],
            "status": row[7],
            "context": json.loads(row[8] or "{}"),
            "result": row[9],
            "created_at": row[10],
            "completed_at": row[11],
        }

    async def start_ticket(self, ticket_id: str) -> None:
        """Mark a ticket as in_progress.

        Args:
            ticket_id: The ticket to start.
        """
        await self._db.db.execute(
            "UPDATE task_tickets SET status = 'in_progress' WHERE id = ?",
            (ticket_id,),
        )
        await self._db.db.commit()

    async def complete_ticket(self, ticket_id: str, result: str) -> None:
        """Mark a ticket as done with a result summary.

        Args:
            ticket_id: The ticket to complete.
            result: Summary of what was done.
        """
        now = datetime.now(UTC).isoformat()
        await self._db.db.execute(
            """
            UPDATE task_tickets
            SET status = 'done', result = ?, completed_at = ?
            WHERE id = ?
            """,
            (result, now, ticket_id),
        )
        await self._db.db.commit()

    async def get_board_summary(self, task_id: str) -> str:
        """Get a formatted summary of all tickets for a task.

        Args:
            task_id: Parent orchestration task ID.

        Returns:
            Human-readable board summary string.
        """
        async with self._db.db.execute(
            """
            SELECT id, from_role, to_role, title, status, priority, result
            FROM task_tickets
            WHERE task_id = ?
            ORDER BY created_at ASC
            """,
            (task_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return "📋 Task Board: No tickets."

        lines = ["📋 Task Board:"]
        for row in rows:
            ticket_id, from_role, to_role, title, status, priority, result = row
            if status == "done":
                icon = "✅"
                suffix = f" — {result[:80]}" if result else ""
            elif status == "in_progress":
                icon = "🔄"
                suffix = ""
            else:
                icon = "⏳"
                suffix = ""
            prio = " [BLOCKER]" if priority == "blocker" else ""
            lines.append(f"  {icon}{prio} [{from_role}→{to_role}] {title}{suffix}")

        return "\n".join(lines)

    async def get_pending_count(self, task_id: str) -> int:
        """Count pending tickets for a task.

        Args:
            task_id: Parent orchestration task ID.

        Returns:
            Number of pending tickets.
        """
        async with self._db.db.execute(
            "SELECT COUNT(*) FROM task_tickets WHERE task_id = ? AND status = 'pending'",
            (task_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def has_pending(self, task_id: str) -> bool:
        """Check if any pending tickets exist for a task.

        Args:
            task_id: Parent orchestration task ID.

        Returns:
            True if there are pending tickets.
        """
        return (await self.get_pending_count(task_id)) > 0

    async def get_roles_with_pending(self, task_id: str) -> set[str]:
        """Get roles that have pending tickets.

        Args:
            task_id: Parent orchestration task ID.

        Returns:
            Set of role names with pending work.
        """
        async with self._db.db.execute(
            """
            SELECT DISTINCT to_role FROM task_tickets
            WHERE task_id = ? AND status = 'pending'
            """,
            (task_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return {row[0] for row in rows}
