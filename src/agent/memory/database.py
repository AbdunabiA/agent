"""SQLite database manager with schema migrations.

Provides async SQLite access via aiosqlite with WAL mode,
foreign key enforcement, and versioned schema migrations.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from datetime import datetime
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = 9

SCHEMA_SQL = """
-- Facts table: structured key-value memory
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'user',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence);

-- Conversations table: session metadata
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    channel_user_id TEXT NOT NULL,
    title TEXT,
    message_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_channel ON conversations(channel);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(channel_user_id);

-- Messages table: conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT,
    tool_call_id TEXT,
    model TEXT,
    token_usage TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);

-- Audit table: tool execution log
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    input_data TEXT NOT NULL,
    output TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    trigger TEXT NOT NULL,
    session_id TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_log(tool_name);
CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(timestamp);

-- Scheduled tasks table
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    type TEXT NOT NULL,
    schedule TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    channel TEXT,
    created_at TEXT NOT NULL,
    last_run TEXT
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


class Database:
    """Async SQLite database manager.

    Handles connection lifecycle, WAL mode, foreign keys,
    and versioned schema migrations.

    Usage:
        db = Database("data/agent.db")
        await db.connect()
        # ... use db.db for queries ...
        await db.close()
    """

    def __init__(self, db_path: str = "data/agent.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Connect to database and run migrations."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            await asyncio.to_thread(os.makedirs, db_dir, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute("PRAGMA cache_size = -10000")
        await self._db.execute("PRAGMA synchronous = NORMAL")
        await self._db.execute("PRAGMA temp_store = MEMORY")
        await self._db.execute("PRAGMA busy_timeout = 10000")

        # Corruption detection
        async with self._db.execute("PRAGMA integrity_check(1)") as cursor:
            row = await cursor.fetchone()
            result = row[0] if row else "ok"
        if result != "ok":
            logger.error(
                "database_corruption_detected",
                path=self.db_path,
                result=result,
            )
            await self._db.close()
            self._db = None
            corrupt_path = self.db_path + ".corrupt"
            await asyncio.to_thread(os.rename, self.db_path, corrupt_path)
            logger.info("database_renamed_corrupt", corrupt_path=corrupt_path)
            # Reconnect to create a fresh database
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            await self._db.execute("PRAGMA cache_size = -10000")
            await self._db.execute("PRAGMA synchronous = NORMAL")
            await self._db.execute("PRAGMA temp_store = MEMORY")
            await self._db.execute("PRAGMA busy_timeout = 10000")

        await self._migrate()
        logger.info("database_connected", path=self.db_path)

    async def _migrate(self) -> None:
        """Run schema migrations if needed."""
        current = 0
        try:
            async with self._db.execute("SELECT MAX(version) FROM schema_version") as cursor:
                row = await cursor.fetchone()
                if row and row[0] is not None:
                    current = row[0]
        except aiosqlite.OperationalError:
            current = 0

        starting_version = current

        if current < 1:
            await self._db.executescript(SCHEMA_SQL)
            current = 1

        if current < 2:
            await self._migrate_v2()
            current = 2

        if current < 3:
            await self._migrate_v3()
            current = 3

        if current < 4:
            await self._migrate_v4()
            current = 4

        if current < 5:
            await self._migrate_v5()
            current = 5

        if current < 6:
            await self._migrate_v6()
            current = 6

        if current < 7:
            await self._migrate_v7()
            current = 7

        if current < 8:
            await self._migrate_v8()
            current = 8

        if current < 9:
            await self._migrate_v9()
            current = 9

        if starting_version < SCHEMA_VERSION:
            await self._db.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            await self._db.commit()
            logger.info(
                "database_migrated",
                from_version=starting_version,
                to_version=SCHEMA_VERSION,
            )

    async def _migrate_v2(self) -> None:
        """Migration v1 -> v2: Add user_id and next_run to scheduled_tasks."""
        # Check existing columns
        async with self._db.execute("PRAGMA table_info(scheduled_tasks)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}

        if "user_id" not in columns:
            await self._db.execute("ALTER TABLE scheduled_tasks ADD COLUMN user_id TEXT")
        if "next_run" not in columns:
            await self._db.execute("ALTER TABLE scheduled_tasks ADD COLUMN next_run TEXT")
        await self._db.commit()
        logger.info("database_migrated_v2")

    async def _migrate_v3(self) -> None:
        """Migration v2 -> v3: Add task_memory table for WorkingMemory."""
        from agent.core.working_memory import WorkingMemory

        await self._db.executescript(WorkingMemory.migration_sql())
        # Best-effort FTS5 — silently skip if not compiled in
        try:
            await self._db.executescript(WorkingMemory.fts_sql())
        except Exception:
            logger.info("database_fts5_unavailable", note="artifact search will use LIKE")
        await self._db.commit()
        logger.info("database_migrated_v3")

    async def _migrate_v4(self) -> None:
        """Migration v3 -> v4: Add agent_spans table for AgentTracer."""
        from agent.observability.tracer import AgentTracer

        await self._db.executescript(AgentTracer.migration_sql())
        await self._db.commit()
        logger.info("database_migrated_v4")

    async def _migrate_v5(self) -> None:
        """Migration v4 -> v5: Add task_tickets table for TaskBoard."""
        from agent.core.task_board import TaskBoard

        await self._db.executescript(TaskBoard.migration_sql())
        await self._db.commit()
        logger.info("database_migrated_v5")

    async def _migrate_v6(self) -> None:
        """Migration v5 -> v6: Add cost_entries table for CostTracker."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS cost_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost REAL NOT NULL,
                channel TEXT NOT NULL DEFAULT 'cli',
                session_id TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cost_entries_model ON cost_entries(model);
            CREATE INDEX IF NOT EXISTS idx_cost_entries_timestamp ON cost_entries(timestamp);
        """)
        await self._db.commit()
        logger.info("database_migrated_v6")

    async def _migrate_v7(self) -> None:
        """Migration v6 -> v7: Add request_id column to audit_log."""
        async with self._db.execute("PRAGMA table_info(audit_log)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}

        if "request_id" not in columns:
            await self._db.execute(
                "ALTER TABLE audit_log ADD COLUMN request_id TEXT NOT NULL DEFAULT ''"
            )
        await self._db.commit()
        logger.info("database_migrated_v7")

    async def _migrate_v8(self) -> None:
        """Migration v7 -> v8: Agent messages table + working memory columns."""
        # Inter-agent message bus table
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS agent_messages (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                from_role TEXT NOT NULL,
                to_role TEXT,
                to_team TEXT,
                thread_id TEXT NOT NULL,
                content TEXT NOT NULL,
                msg_type TEXT NOT NULL DEFAULT 'question',
                reply_to TEXT,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_task
                ON agent_messages(task_id);
            CREATE INDEX IF NOT EXISTS idx_messages_thread
                ON agent_messages(thread_id);
            CREATE INDEX IF NOT EXISTS idx_messages_to_role
                ON agent_messages(task_id, to_role);
        """)

        # Working memory validation columns (if table exists)
        try:
            async with self._db.execute(
                "PRAGMA table_info(working_memory)",
            ) as cursor:
                columns = {row[1] for row in await cursor.fetchall()}

            if columns and "validated_by" not in columns:
                await self._db.execute(
                    "ALTER TABLE working_memory " "ADD COLUMN validated_by TEXT DEFAULT ''"
                )
                await self._db.execute(
                    "ALTER TABLE working_memory " "ADD COLUMN validated_at TEXT DEFAULT ''"
                )
                await self._db.execute(
                    "ALTER TABLE working_memory " "ADD COLUMN status TEXT DEFAULT 'pending'"
                )
        except Exception:
            pass  # working_memory may not exist yet

        await self._db.commit()
        logger.info("database_migrated_v8")

    async def _migrate_v9(self) -> None:
        """Migration v8 -> v9: Emotional/contextual metadata on facts."""
        async with self._db.execute("PRAGMA table_info(facts)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}

        new_columns = {
            "tone": "TEXT DEFAULT ''",
            "emotion": "TEXT DEFAULT ''",
            "priority": "TEXT DEFAULT 'normal'",
            "topic": "TEXT DEFAULT ''",
            "context_snippet": "TEXT DEFAULT ''",
            "temporal_reference": "TEXT",
            "next_action_date": "TEXT",
        }
        for col, col_type in new_columns.items():
            if col not in columns:
                await self._db.execute(f"ALTER TABLE facts ADD COLUMN {col} {col_type}")

        await self._db.executescript("""
            CREATE INDEX IF NOT EXISTS idx_facts_priority
                ON facts(priority);
            CREATE INDEX IF NOT EXISTS idx_facts_topic
                ON facts(topic);
            CREATE INDEX IF NOT EXISTS idx_facts_temporal
                ON facts(temporal_reference);
        """)
        await self._db.commit()
        logger.info("database_migrated_v9")

    async def checkpoint(self) -> None:
        """Run WAL checkpoint to keep WAL file size manageable."""
        if self._db:
            await self._db.execute("PRAGMA wal_checkpoint(PASSIVE)")

    async def backup(self, backup_dir: str = "./data/backups") -> str:
        """Create a timestamped backup of the database file.

        Checkpoints the WAL and copies the database file to the backup directory.

        Args:
            backup_dir: Directory to store backups. Created if it does not exist.

        Returns:
            Path to the backup file.
        """
        backup_path = Path(backup_dir)
        await asyncio.to_thread(backup_path.mkdir, parents=True, exist_ok=True)

        await self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = Path(self.db_path).stem
        dest = backup_path / f"{db_name}_{timestamp}.db"
        await asyncio.to_thread(shutil.copy2, self.db_path, str(dest))

        logger.info("database_backup_created", path=str(dest))
        return str(dest)

    def get_size_mb(self) -> float:
        """Return the database file size in megabytes.

        Returns:
            File size in MB, or 0.0 if the file does not exist.
        """
        try:
            size_bytes = Path(self.db_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except OSError:
            return 0.0

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("database_closed")

    @property
    def db(self) -> aiosqlite.Connection:
        """Get the active database connection.

        Raises:
            RuntimeError: If the database is not connected.
        """
        if not self._db:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db
