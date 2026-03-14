"""SQLite database manager with schema migrations.

Provides async SQLite access via aiosqlite with WAL mode,
foreign key enforcement, and versioned schema migrations.
"""

from __future__ import annotations

import asyncio
import os

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = 2

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
        await self._migrate()
        logger.info("database_connected", path=self.db_path)

    async def _migrate(self) -> None:
        """Run schema migrations if needed."""
        current = 0
        try:
            async with self._db.execute(
                "SELECT MAX(version) FROM schema_version"
            ) as cursor:
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
            await self._db.execute(
                "ALTER TABLE scheduled_tasks ADD COLUMN user_id TEXT"
            )
        if "next_run" not in columns:
            await self._db.execute(
                "ALTER TABLE scheduled_tasks ADD COLUMN next_run TEXT"
            )
        await self._db.commit()
        logger.info("database_migrated_v2")

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
