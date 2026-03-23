"""Tests for AgentTracer — hierarchical span tracing."""

from __future__ import annotations

import pytest

from agent.memory.database import Database
from agent.observability.tracer import AgentTracer


@pytest.fixture
async def db(tmp_path):
    """Provide a fresh database for each test."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def tracer(db):
    """Provide an AgentTracer backed by the test database."""
    return AgentTracer(db)


@pytest.mark.asyncio
async def test_span_creates_and_closes(tracer):
    async with tracer.span("task-1", "coder", "spawn") as s:
        s.tokens_input = 100
        s.tokens_output = 50
        s.metadata["files"] = ["main.py"]

    assert s.status == "ok"
    assert s.ended_at is not None
    assert s.duration_s is not None
    assert s.total_tokens == 150


@pytest.mark.asyncio
async def test_span_marks_error_on_exception(tracer):
    with pytest.raises(ValueError, match="test error"):
        async with tracer.span("task-1", "worker", "spawn") as s:
            raise ValueError("test error")

    assert s.status == "error"
    assert "test error" in s.error
    assert s.ended_at is not None


@pytest.mark.asyncio
async def test_span_persists_to_db(tracer, db):
    async with tracer.span("task-1", "researcher", "spawn") as s:
        s.tokens_input = 200

    async with db.db.execute(
        "SELECT span_id, role, status, tokens_input FROM agent_spans WHERE task_id = ?",
        ("task-1",),
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == s.span_id
    assert row[1] == "researcher"
    assert row[2] == "ok"
    assert row[3] == 200


@pytest.mark.asyncio
async def test_get_task_tree(tracer):
    async with tracer.span("task-1", "controller", "spawn") as parent:
        pass
    async with tracer.span("task-1", "coder", "spawn", parent_id=parent.span_id) as child:
        child.tokens_input = 1500

    tree = await tracer.get_task_tree("task-1")

    assert "task-1" in tree
    assert "controller" in tree
    assert "coder" in tree
    assert "1.5k tokens" in tree


@pytest.mark.asyncio
async def test_get_task_tree_empty(tracer):
    tree = await tracer.get_task_tree("nonexistent")
    assert "No spans" in tree


@pytest.mark.asyncio
async def test_get_stats(tracer):
    async with tracer.span("task-1", "coder", "spawn") as s:
        s.tokens_input = 100
        s.tokens_output = 50
    async with tracer.span("task-1", "reviewer", "spawn"):
        pass

    with pytest.raises(RuntimeError):
        async with tracer.span("task-1", "failing", "spawn"):
            raise RuntimeError("boom")

    stats = await tracer.get_stats("task-1")

    assert stats["total_tokens"] == 150
    assert stats["total_duration_s"] >= 0
    assert "coder" in stats["roles"]
    assert "failing" in stats["roles"]
    assert stats["roles"]["failing"]["error"] == 1
    assert stats["roles"]["coder"]["ok"] == 1


@pytest.mark.asyncio
async def test_get_stats_empty(tracer):
    stats = await tracer.get_stats("nonexistent")
    assert stats["total_tokens"] == 0
    assert stats["roles"] == {}


@pytest.mark.asyncio
async def test_get_recent_tasks(tracer):
    for i in range(7):
        async with tracer.span(f"task-{i}", "worker", "spawn"):
            pass

    recent = await tracer.get_recent_tasks(limit=5)
    assert len(recent) == 5
    # Most recent first
    assert recent[0]["task_id"] == "task-6"


@pytest.mark.asyncio
async def test_noop_tracer():
    """AgentTracer with no database is a no-op."""
    tracer = AgentTracer(database=None)

    async with tracer.span("task-1", "worker", "spawn") as s:
        s.tokens_input = 100

    assert s.status == "ok"
    tree = await tracer.get_task_tree("task-1")
    assert "No spans" in tree


@pytest.mark.asyncio
async def test_span_status_can_be_set_to_retry(tracer):
    async with tracer.span("task-1", "worker", "retry") as s:
        s.status = "retry"

    assert s.status == "retry"


@pytest.mark.asyncio
async def test_tree_shows_error_icon(tracer):
    with pytest.raises(ValueError):
        async with tracer.span("task-1", "buggy", "spawn"):
            raise ValueError("oops")

    tree = await tracer.get_task_tree("task-1")
    assert "\U0001f534" in tree  # 🔴
    assert "oops" in tree


@pytest.mark.asyncio
async def test_migration_creates_table(db):
    """Verify migration v4 created the agent_spans table."""
    async with db.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_spans'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_migration_creates_indexes(db):
    """Verify the expected indexes exist."""
    async with db.db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_spans_%'"
    ) as cursor:
        rows = await cursor.fetchall()
    index_names = {r[0] for r in rows}
    assert "idx_spans_task" in index_names
    assert "idx_spans_parent" in index_names
    assert "idx_spans_status" in index_names


@pytest.mark.asyncio
async def test_nested_spans_tree_structure(tracer):
    """Verify a 3-level tree renders correctly."""
    async with tracer.span("task-1", "controller", "spawn") as root:
        root.tokens_input = 50
    async with tracer.span("task-1", "architect", "spawn", parent_id=root.span_id) as arch:
        arch.tokens_input = 300
    async with tracer.span("task-1", "coder", "spawn", parent_id=arch.span_id) as coder:
        coder.tokens_input = 2000

    tree = await tracer.get_task_tree("task-1")
    lines = tree.split("\n")

    # Root line should have task info
    assert "task-1" in lines[0]
    # Should have at least 4 lines (header + 3 spans)
    assert len(lines) >= 4
    # All roles should appear
    assert any("controller" in line for line in lines)
    assert any("architect" in line for line in lines)
    assert any("coder" in line for line in lines)


@pytest.mark.asyncio
async def test_project_level_tree_structure(tracer):
    """Verify a project → stage → worker 3-level tree renders correctly."""
    # Project root span
    async with tracer.span("proj-abc", "code_review", "project") as proj:
        proj.metadata["instruction"] = "Review the codebase"
        proj.metadata["stages"] = 2

    # Stage 1 span
    async with tracer.span(
        "proj-abc",
        "analysis",
        "stage",
        parent_id=proj.span_id,
    ) as stage1:
        stage1.metadata["agents"] = 2

    # Worker spans under stage 1
    async with tracer.span(
        "proj-abc",
        "security_reviewer",
        "spawn_attempt_1",
        parent_id=stage1.span_id,
    ) as w1:
        w1.tokens_input = 500
        w1.metadata["output_preview"] = "Found 3 critical issues: hardcoded SECRET_KEY"

    async with tracer.span(
        "proj-abc",
        "qa_engineer",
        "spawn_attempt_1",
        parent_id=stage1.span_id,
    ) as w2:
        w2.tokens_input = 400
        w2.metadata["output_preview"] = "N+1 queries in favorites"

    # Stage 2 span
    async with tracer.span(
        "proj-abc",
        "fixes",
        "stage",
        parent_id=proj.span_id,
    ) as stage2:
        stage2.metadata["agents"] = 1

    tree = await tracer.get_task_tree("proj-abc")
    lines = tree.split("\n")

    # Header should show project name, not raw task_id
    assert "Project: code_review" in lines[0]
    # Should have project → stage → worker hierarchy
    assert any("analysis" in line and "2 agents" in line for line in lines)
    assert any("security_reviewer" in line for line in lines)
    assert any("hardcoded SECRET_KEY" in line for line in lines)
    assert any("qa_engineer" in line for line in lines)
    assert any("fixes" in line and "1 agents" in line for line in lines)


@pytest.mark.asyncio
async def test_project_in_recent_tasks(tracer):
    """get_recent_tasks should show project name and stage count."""
    # Create a project with stages
    async with tracer.span("proj-xyz", "my_project", "project") as proj:
        pass
    async with tracer.span(
        "proj-xyz",
        "stage1",
        "stage",
        parent_id=proj.span_id,
    ):
        pass
    async with tracer.span(
        "proj-xyz",
        "stage2",
        "stage",
        parent_id=proj.span_id,
    ):
        pass

    recent = await tracer.get_recent_tasks(limit=5)
    assert len(recent) >= 1
    proj_entry = next(t for t in recent if t["task_id"] == "proj-xyz")
    assert proj_entry["project_name"] == "my_project"
    assert proj_entry["stage_count"] == 2


@pytest.mark.asyncio
async def test_output_preview_in_tree(tracer):
    """Span metadata output_preview should appear in tree rendering."""
    async with tracer.span("task-prev", "worker", "spawn_attempt_1") as s:
        s.metadata["output_preview"] = "Found SQL injection vulnerability"

    tree = await tracer.get_task_tree("task-prev")
    assert "SQL injection" in tree
