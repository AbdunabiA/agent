"""Working memory — shared persistent context store for task workers.

When multiple sub-agents collaborate on a task, they need to see what
other workers have already done.  WorkingMemory persists findings and
artifacts to SQLite so every worker spawned for the same ``task_id``
can receive accumulated team context in its system prompt.

Artifacts support full-text search via SQLite FTS5 when available,
falling back to LIKE-based search otherwise.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.core.events import EventBus
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)


class WorkingMemory:
    """Shared persistent context store scoped to orchestration tasks.

    Each finding is a key-value pair tagged with the worker role that
    produced it.  Artifacts are longer-form content (file diffs, plans,
    analysis) stored alongside metadata and searchable via FTS.

    Args:
        database: The project's async SQLite database handle.
    """

    def __init__(self, database: Database) -> None:
        self._db = database
        self._fts_available: bool | None = None
        self._validation_cols_ensured: bool = False

    # ------------------------------------------------------------------
    # Schema bootstrap (called from Database._migrate)
    # ------------------------------------------------------------------

    @staticmethod
    def migration_sql() -> str:
        """Return the CREATE TABLE statements for the task_memory table."""
        return """
CREATE TABLE IF NOT EXISTS task_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'finding',
    key TEXT,
    value_json TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_memory_task_role
    ON task_memory(task_id, role);
CREATE INDEX IF NOT EXISTS idx_task_memory_task_kind
    ON task_memory(task_id, kind);
"""

    @staticmethod
    def fts_sql() -> str:
        """Return FTS5 virtual table DDL (best-effort, may fail)."""
        return """
CREATE VIRTUAL TABLE IF NOT EXISTS task_memory_fts USING fts5(
    value_text,
    content='task_memory',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS task_memory_ai AFTER INSERT ON task_memory BEGIN
    INSERT INTO task_memory_fts(rowid, value_text)
    VALUES (new.id, json_extract(new.value_json, '$'));
END;
CREATE TRIGGER IF NOT EXISTS task_memory_ad AFTER DELETE ON task_memory BEGIN
    INSERT INTO task_memory_fts(task_memory_fts, rowid, value_text)
    VALUES ('delete', old.id, json_extract(old.value_json, '$'));
END;
"""

    async def _ensure_validation_columns(self) -> None:
        """Add validated_by/validated_at/status columns if missing.

        The database migration v8 targets the ``working_memory`` table,
        but the WorkingMemory class stores data in ``task_memory``.  This
        helper bridges the gap by adding the columns on first use.
        """
        if self._validation_cols_ensured:
            return
        try:
            async with self._db.db.execute(
                "PRAGMA table_info(task_memory)",
            ) as cursor:
                columns = {row[1] for row in await cursor.fetchall()}

            if "validated_by" not in columns:
                await self._db.db.execute(
                    "ALTER TABLE task_memory ADD COLUMN validated_by TEXT DEFAULT ''"
                )
                await self._db.db.execute(
                    "ALTER TABLE task_memory ADD COLUMN validated_at TEXT DEFAULT ''"
                )
                await self._db.db.execute(
                    "ALTER TABLE task_memory ADD COLUMN status TEXT DEFAULT 'pending'"
                )
                await self._db.db.commit()
        except Exception:
            pass
        self._validation_cols_ensured = True

    async def _ensure_fts(self) -> bool:
        """Check (once) whether FTS5 is usable and cache the answer."""
        if self._fts_available is not None:
            return self._fts_available
        try:
            async with self._db.db.execute("SELECT 1 FROM task_memory_fts LIMIT 0"):
                pass
            self._fts_available = True
        except Exception:
            self._fts_available = False
        return self._fts_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_finding(
        self,
        task_id: str,
        role: str,
        key: str,
        value: str,
    ) -> None:
        """Persist a key-value finding for a task.

        If a finding with the same ``(task_id, role, key)`` already exists
        it is updated in place; otherwise a new row is inserted.

        Args:
            task_id: Orchestration task identifier.
            role: Name of the worker role that produced this finding.
            key: Short identifier (e.g. ``"files_created"``).
            value: The finding content.
        """
        now = datetime.now(UTC).isoformat()
        value_json = json.dumps(value)

        # Upsert: update if exists, insert otherwise
        async with self._db.db.execute(
            "SELECT id FROM task_memory "
            "WHERE task_id = ? AND role = ? AND kind = 'finding' AND key = ?",
            (task_id, role, key),
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            await self._db.db.execute(
                "UPDATE task_memory SET value_json = ?, updated_at = ? WHERE id = ?",
                (value_json, now, row[0]),
            )
        else:
            await self._db.db.execute(
                "INSERT INTO task_memory "
                "(task_id, role, kind, key, value_json, created_at, updated_at) "
                "VALUES (?, ?, 'finding', ?, ?, ?, ?)",
                (task_id, role, key, value_json, now, now),
            )
        await self._db.db.commit()

        logger.debug(
            "working_memory_finding_saved",
            task_id=task_id,
            role=role,
            key=key,
        )

    async def save_artifact(
        self,
        task_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist a longer-form artifact (plan, diff, analysis).

        Args:
            task_id: Orchestration task identifier.
            role: Name of the worker role.
            content: The artifact text.
            metadata: Optional dict of extra info (serialised as JSON).

        Returns:
            The row id of the newly inserted artifact.
        """
        now = datetime.now(UTC).isoformat()
        value_json = json.dumps(content)
        meta_json = json.dumps(metadata) if metadata else None

        async with self._db.db.execute(
            "INSERT INTO task_memory "
            "(task_id, role, kind, key, value_json, metadata_json, created_at, updated_at) "
            "VALUES (?, ?, 'artifact', NULL, ?, ?, ?, ?)",
            (task_id, role, value_json, meta_json, now, now),
        ) as cursor:
            row_id = cursor.lastrowid or 0

        await self._db.db.commit()

        logger.debug(
            "working_memory_artifact_saved",
            task_id=task_id,
            role=role,
            row_id=row_id,
        )
        return row_id

    async def get_context_for_role(self, task_id: str, role: str) -> str:
        """Build a human-readable context block for a worker.

        Returns findings and recent artifacts produced by *other* roles
        for the same task, formatted as a prompt-friendly section.

        Args:
            task_id: Orchestration task identifier.
            role: The requesting worker's role name (excluded from own
                  output so it doesn't see stale copies of its own work).

        Returns:
            Formatted context string, or ``""`` if nothing exists yet.
        """
        sections: list[str] = []

        # 1. Findings from other roles
        async with self._db.db.execute(
            "SELECT role, key, value_json FROM task_memory "
            "WHERE task_id = ? AND role != ? AND kind = 'finding' "
            "ORDER BY created_at",
            (task_id, role),
        ) as cursor:
            findings = await cursor.fetchall()

        if findings:
            by_role: dict[str, list[str]] = {}
            for r in findings:
                r_role, r_key, r_val = r[0], r[1], json.loads(r[2])
                by_role.setdefault(r_role, []).append(f"- **{r_key}**: {r_val}")
            for r_name, items in by_role.items():
                sections.append(f"### {r_name}\n" + "\n".join(items))

        # 2. Recent artifacts from other roles (last 10)
        async with self._db.db.execute(
            "SELECT role, value_json, metadata_json FROM task_memory "
            "WHERE task_id = ? AND role != ? AND kind = 'artifact' "
            "ORDER BY created_at DESC LIMIT 10",
            (task_id, role),
        ) as cursor:
            artifacts = await cursor.fetchall()

        if artifacts:
            art_lines: list[str] = []
            for a in reversed(artifacts):  # chronological order
                a_role = a[0]
                a_val = json.loads(a[1])
                a_meta = json.loads(a[2]) if a[2] else {}
                label = a_meta.get("label", "artifact")
                # Truncate very long artifacts to keep prompt manageable
                preview = a_val[:2000] + ("…" if len(a_val) > 2000 else "")
                art_lines.append(f"**[{a_role} — {label}]**\n{preview}")
            sections.append("### Artifacts\n" + "\n\n".join(art_lines))

        if not sections:
            return ""

        return "## Team Context\n\n" + "\n\n".join(sections)

    async def search_artifacts(
        self,
        task_id: str,
        query: str,
        limit: int = 5,
    ) -> list[str]:
        """Search artifacts for a task by keyword.

        Uses FTS5 when available, falls back to ``LIKE`` otherwise.

        Args:
            task_id: Orchestration task identifier.
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching artifact texts.
        """
        results: list[str] = []

        if await self._ensure_fts():
            # FTS5 path — escape special characters to prevent parse errors
            safe_query = self._escape_fts_query(query)
            async with self._db.db.execute(
                "SELECT tm.value_json FROM task_memory tm "
                "JOIN task_memory_fts fts ON fts.rowid = tm.id "
                "WHERE tm.task_id = ? AND tm.kind = 'artifact' "
                "AND task_memory_fts MATCH ? "
                "LIMIT ?",
                (task_id, safe_query, limit),
            ) as cursor:
                rows = await cursor.fetchall()
            results = [json.loads(r[0]) for r in rows]
        else:
            # LIKE fallback
            like_pattern = f"%{query}%"
            async with self._db.db.execute(
                "SELECT value_json FROM task_memory "
                "WHERE task_id = ? AND kind = 'artifact' "
                "AND value_json LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (task_id, like_pattern, limit),
            ) as cursor:
                rows = await cursor.fetchall()
            results = [json.loads(r[0]) for r in rows]

        return results

    @staticmethod
    def _escape_fts_query(query: str) -> str:
        """Escape FTS5 special characters to prevent parse errors.

        Wraps each token in double quotes so characters like ``*``, ``:``,
        ``(``, ``)`` are treated as literals.
        """
        # Remove existing quotes to avoid nesting, then quote each token
        tokens = query.replace('"', "").split()
        if not tokens:
            return '""'
        return " ".join(f'"{t}"' for t in tokens)

    async def save_finding_with_notify(
        self,
        key: str,
        value: Any,
        role: str,
        task_id: str,
        event_bus: EventBus | None = None,
    ) -> None:
        """Save a finding and emit a FINDING_SAVED event.

        Delegates persistence to :meth:`save_finding`, then notifies
        other agents via the event bus so they can react immediately.

        Args:
            key: Short identifier for the finding.
            value: The finding content.
            role: Name of the worker role that produced this finding.
            task_id: Orchestration task identifier.
            event_bus: Optional event bus for broadcasting the save event.
        """
        from agent.core.events import Events

        await self.save_finding(task_id=task_id, role=role, key=key, value=value)
        if event_bus:
            await event_bus.emit(
                Events.FINDING_SAVED,
                {
                    "key": key,
                    "role": role,
                    "task_id": task_id,
                },
            )

    async def get_context_since(
        self,
        role: str,
        task_id: str,
        since_timestamp: float,
    ) -> str:
        """Get findings newer than a timestamp (incremental updates).

        Returns findings and artifacts produced by *other* roles after
        the given timestamp, formatted as a prompt-friendly markdown block.

        The ``created_at`` column (ISO-8601 text) is compared against
        the provided Unix timestamp converted to ISO format.

        Args:
            role: The requesting worker's role (excluded from results).
            task_id: Orchestration task identifier.
            since_timestamp: Unix timestamp; only rows created after this
                are returned.

        Returns:
            Formatted context string, or ``""`` if nothing new.
        """
        since_iso = datetime.fromtimestamp(since_timestamp, tz=UTC).isoformat()

        sections: list[str] = []

        # Findings from other roles since the given timestamp
        async with self._db.db.execute(
            "SELECT role, key, value_json FROM task_memory "
            "WHERE task_id = ? AND role != ? AND kind = 'finding' "
            "AND created_at > ? "
            "ORDER BY created_at",
            (task_id, role, since_iso),
        ) as cursor:
            findings = await cursor.fetchall()

        if findings:
            by_role: dict[str, list[str]] = {}
            for r in findings:
                r_role, r_key, r_val = r[0], r[1], json.loads(r[2])
                by_role.setdefault(r_role, []).append(f"- **{r_key}**: {r_val}")
            for r_name, items in by_role.items():
                sections.append(f"### {r_name}\n" + "\n".join(items))

        # Recent artifacts from other roles since the given timestamp
        async with self._db.db.execute(
            "SELECT role, value_json, metadata_json FROM task_memory "
            "WHERE task_id = ? AND role != ? AND kind = 'artifact' "
            "AND created_at > ? "
            "ORDER BY created_at DESC LIMIT 10",
            (task_id, role, since_iso),
        ) as cursor:
            artifacts = await cursor.fetchall()

        if artifacts:
            art_lines: list[str] = []
            for a in reversed(artifacts):
                a_role = a[0]
                a_val = json.loads(a[1])
                a_meta = json.loads(a[2]) if a[2] else {}
                label = a_meta.get("label", "artifact")
                preview = a_val[:2000] + ("\u2026" if len(a_val) > 2000 else "")
                art_lines.append(f"**[{a_role} \u2014 {label}]**\n{preview}")
            sections.append("### Artifacts\n" + "\n\n".join(art_lines))

        if not sections:
            return ""

        return "## New Updates\n\n" + "\n\n".join(sections)

    async def validate_finding(
        self,
        key: str,
        task_id: str,
        validator_role: str,
        approved: bool,
    ) -> bool:
        """Validate (approve or reject) a finding saved by another role.

        Updates the ``validated_by``, ``validated_at``, and ``status``
        columns added by database migration v8.

        Args:
            key: The finding key to validate.
            task_id: Orchestration task identifier.
            validator_role: Role of the agent performing the validation.
            approved: ``True`` to approve, ``False`` to reject.

        Returns:
            ``True`` if a matching row was found and updated, ``False``
            otherwise.
        """
        await self._ensure_validation_columns()
        status = "approved" if approved else "rejected"

        async with self._db.db.execute(
            "UPDATE task_memory "
            "SET validated_by = ?, validated_at = datetime('now'), status = ? "
            "WHERE key = ? AND task_id = ? AND kind = 'finding'",
            (validator_role, status, key, task_id),
        ) as cursor:
            updated = cursor.rowcount > 0

        await self._db.db.commit()

        if updated:
            logger.info(
                "working_memory_finding_validated",
                key=key,
                task_id=task_id,
                validator_role=validator_role,
                status=status,
            )

        return updated

    async def clear_task(self, task_id: str) -> int:
        """Delete all working memory for a task.

        Args:
            task_id: Orchestration task identifier.

        Returns:
            Number of rows deleted.
        """
        async with self._db.db.execute(
            "DELETE FROM task_memory WHERE task_id = ?",
            (task_id,),
        ) as cursor:
            deleted = cursor.rowcount

        await self._db.db.commit()

        logger.info("working_memory_cleared", task_id=task_id, deleted=deleted)
        return deleted
