"""Tests for the SQLite database manager."""

from __future__ import annotations

import os

import pytest

from agent.memory.database import Database


@pytest.fixture
async def db(tmp_path: object) -> Database:
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


class TestDatabase:
    """Tests for Database manager."""

    async def test_connect_creates_file(self, tmp_path: object) -> None:
        """Connecting should create the database file."""
        db_path = str(tmp_path / "new.db")
        database = Database(db_path)
        await database.connect()

        assert os.path.exists(db_path)  # noqa: ASYNC240
        await database.close()

    async def test_schema_tables_created(self, db: Database) -> None:
        """All schema tables should exist after connect."""
        expected_tables = {
            "facts", "conversations", "messages",
            "audit_log", "scheduled_tasks", "schema_version",
        }

        async with db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            rows = await cursor.fetchall()

        table_names = {row[0] for row in rows}
        assert expected_tables.issubset(table_names)

    async def test_migration_idempotent(self, tmp_path: object) -> None:
        """Connecting twice should not error (idempotent migration)."""
        db_path = str(tmp_path / "idempotent.db")

        db1 = Database(db_path)
        await db1.connect()
        await db1.close()

        db2 = Database(db_path)
        await db2.connect()

        # Verify tables still exist
        async with db2.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            rows = await cursor.fetchall()

        table_names = {row[0] for row in rows}
        assert "facts" in table_names
        await db2.close()

    async def test_wal_mode_enabled(self, db: Database) -> None:
        """WAL journal mode should be enabled."""
        async with db.db.execute("PRAGMA journal_mode") as cursor:
            row = await cursor.fetchone()

        assert row[0] == "wal"

    async def test_foreign_keys_enabled(self, db: Database) -> None:
        """Foreign key constraints should be enabled."""
        async with db.db.execute("PRAGMA foreign_keys") as cursor:
            row = await cursor.fetchone()

        assert row[0] == 1

    async def test_schema_version_tracked(self, db: Database) -> None:
        """Schema version should be stored after migration."""
        async with db.db.execute(
            "SELECT MAX(version) FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0] >= 1

    async def test_db_property_raises_when_not_connected(self) -> None:
        """Accessing db property before connect should raise RuntimeError."""
        database = Database("unused.db")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = database.db

    async def test_close_sets_none(self, tmp_path: object) -> None:
        """After close, db property should raise."""
        db_path = str(tmp_path / "close_test.db")
        database = Database(db_path)
        await database.connect()
        await database.close()

        with pytest.raises(RuntimeError):
            _ = database.db

    async def test_creates_parent_directories(self, tmp_path: object) -> None:
        """Should create parent directories if they don't exist."""
        db_path = str(tmp_path / "nested" / "dir" / "test.db")
        database = Database(db_path)
        await database.connect()

        assert os.path.exists(db_path)  # noqa: ASYNC240
        await database.close()
