"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest

from agent.core.session import SessionStore
from agent.memory.database import Database
from agent.memory.store import FactStore


@pytest.fixture
async def test_database(tmp_path: object) -> Database:
    """Create a temporary Database instance using tmp_path."""
    db_path = str(tmp_path / "integration_test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def fact_store(test_database: Database) -> FactStore:
    """FactStore connected to test_database."""
    return FactStore(test_database)


@pytest.fixture
def session_manager(test_database: Database) -> SessionStore:
    """SessionManager (SessionStore) with test database."""
    return SessionStore(db=test_database)
