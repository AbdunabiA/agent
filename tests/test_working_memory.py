"""Tests for WorkingMemory — shared persistent context store."""

from __future__ import annotations

import time

import pytest

from agent.core.events import EventBus, Events
from agent.core.working_memory import WorkingMemory
from agent.memory.database import Database


@pytest.fixture
async def db(tmp_path):
    """Provide a fresh in-memory-like database for each test."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    await database.connect()  # runs migrations including v3
    yield database
    await database.close()


@pytest.fixture
async def wm(db):
    """Provide a WorkingMemory instance backed by the test database."""
    return WorkingMemory(db)


@pytest.mark.asyncio
async def test_save_and_get_finding(wm):
    await wm.save_finding("task-1", "researcher", "files_found", "main.py, utils.py")

    ctx = await wm.get_context_for_role("task-1", "coder")
    assert "researcher" in ctx
    assert "files_found" in ctx
    assert "main.py, utils.py" in ctx


@pytest.mark.asyncio
async def test_findings_exclude_own_role(wm):
    await wm.save_finding("task-1", "coder", "status", "done")
    await wm.save_finding("task-1", "reviewer", "bugs", "none")

    ctx = await wm.get_context_for_role("task-1", "coder")
    assert "reviewer" in ctx
    assert "coder" not in ctx


@pytest.mark.asyncio
async def test_upsert_finding(wm):
    await wm.save_finding("task-1", "worker", "status", "in progress")
    await wm.save_finding("task-1", "worker", "status", "done")

    # Should have only one row, not two
    async with wm._db.db.execute(
        "SELECT COUNT(*) FROM task_memory WHERE task_id = ? AND key = 'status'",
        ("task-1",),
    ) as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1

    # Context for another role should show the updated value
    ctx = await wm.get_context_for_role("task-1", "other")
    assert "done" in ctx


@pytest.mark.asyncio
async def test_save_artifact_and_search(wm):
    await wm.save_artifact(
        "task-1",
        "coder",
        "def hello(): print('hi')",
        {"label": "code"},
    )
    await wm.save_artifact(
        "task-1",
        "coder",
        "Fixed the login bug in auth.py",
        {"label": "fix"},
    )

    results = await wm.search_artifacts("task-1", "login")
    assert len(results) >= 1
    assert any("login" in r for r in results)


@pytest.mark.asyncio
async def test_artifacts_in_context(wm):
    await wm.save_artifact(
        "task-1",
        "planner",
        "Step 1: Read code\nStep 2: Fix bug",
        {"label": "plan"},
    )

    ctx = await wm.get_context_for_role("task-1", "coder")
    assert "planner" in ctx
    assert "plan" in ctx
    assert "Step 1" in ctx


@pytest.mark.asyncio
async def test_clear_task(wm):
    await wm.save_finding("task-1", "a", "k", "v")
    await wm.save_artifact("task-1", "a", "content")
    await wm.save_finding("task-2", "a", "k", "v")

    deleted = await wm.clear_task("task-1")
    assert deleted == 2

    # task-1 context should be empty
    ctx = await wm.get_context_for_role("task-1", "other")
    assert ctx == ""

    # task-2 should be unaffected
    ctx2 = await wm.get_context_for_role("task-2", "other")
    assert "k" in ctx2


@pytest.mark.asyncio
async def test_empty_context_returns_empty_string(wm):
    ctx = await wm.get_context_for_role("nonexistent", "role")
    assert ctx == ""


@pytest.mark.asyncio
async def test_get_context_header(wm):
    await wm.save_finding("task-1", "scout", "target", "main.py")
    ctx = await wm.get_context_for_role("task-1", "coder")
    assert ctx.startswith("## Team Context")


@pytest.mark.asyncio
async def test_save_artifact_returns_row_id(wm):
    row_id = await wm.save_artifact("task-1", "worker", "some content")
    assert isinstance(row_id, int)
    assert row_id > 0


@pytest.mark.asyncio
async def test_search_artifacts_no_match(wm):
    await wm.save_artifact("task-1", "worker", "hello world")
    results = await wm.search_artifacts("task-1", "zzzznotfound")
    assert results == []


@pytest.mark.asyncio
async def test_migration_creates_table(db):
    """Verify the migration created the task_memory table."""
    async with db.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='task_memory'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_migration_creates_index(db):
    """Verify the migration created the expected index."""
    async with db.db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' " "AND name='idx_task_memory_task_role'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None


# ------------------------------------------------------------------
# Integration tests: WorkingMemory across project stages
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_stage_visibility(wm):
    """Workers in later stages see findings from earlier stages via shared task_id."""
    shared_project_id = "proj-integration-1"

    # Stage 1: security_reviewer saves findings
    await wm.save_finding(
        shared_project_id,
        "security_reviewer",
        "output_summary",
        "Found 3 critical issues: hardcoded SECRET_KEY in settings.py",
    )
    await wm.save_artifact(
        shared_project_id,
        "security_reviewer",
        "1. Hardcoded SECRET_KEY\n2. SQL injection in /login\n3. No CSRF tokens",
        {"label": "worker_output"},
    )

    # Stage 1: qa_engineer saves findings
    await wm.save_finding(
        shared_project_id,
        "qa_engineer",
        "output_summary",
        "N+1 queries in favorites endpoint, missing input validation",
    )

    # Stage 2: backend_developer should see BOTH reviewers' work
    ctx = await wm.get_context_for_role(shared_project_id, "backend_developer")

    assert "security_reviewer" in ctx
    assert "hardcoded SECRET_KEY" in ctx.lower() or "SECRET_KEY" in ctx
    assert "qa_engineer" in ctx
    assert "N+1 queries" in ctx
    # Artifact should be present
    assert "SQL injection" in ctx


@pytest.mark.asyncio
async def test_same_role_excluded_in_later_stage(wm):
    """A role re-run in a later stage doesn't see its own prior output."""
    shared_id = "proj-integration-2"

    # Stage 1: qa_engineer runs
    await wm.save_finding(shared_id, "qa_engineer", "output_summary", "bugs found")
    # Stage 1: developer runs
    await wm.save_finding(shared_id, "backend_developer", "output_summary", "fixed bugs")

    # Stage 2: qa_engineer re-runs — should see developer's work but NOT own prior output
    ctx = await wm.get_context_for_role(shared_id, "qa_engineer")
    assert "backend_developer" in ctx
    assert "fixed bugs" in ctx
    # Should NOT contain qa_engineer's own findings
    assert "qa_engineer" not in ctx


@pytest.mark.asyncio
async def test_multiple_findings_per_role(wm):
    """Multiple findings from one role are all visible to others."""
    shared_id = "proj-integration-3"

    await wm.save_finding(shared_id, "researcher", "finding_0", "Auth uses JWT")
    await wm.save_finding(shared_id, "researcher", "finding_1", "DB is PostgreSQL")
    await wm.save_finding(shared_id, "researcher", "output_summary", "Analyzed full stack")

    ctx = await wm.get_context_for_role(shared_id, "developer")
    assert "JWT" in ctx
    assert "PostgreSQL" in ctx
    assert "Analyzed full stack" in ctx


@pytest.mark.asyncio
async def test_artifact_truncation_in_context(wm):
    """Very long artifacts are truncated to 2000 chars in context."""
    shared_id = "proj-integration-4"

    long_content = "x" * 5000
    await wm.save_artifact(shared_id, "analyzer", long_content, {"label": "huge_output"})

    ctx = await wm.get_context_for_role(shared_id, "developer")
    # Should contain truncated content with ellipsis
    assert "…" in ctx
    # Should NOT contain the full 5000 chars
    assert "x" * 5000 not in ctx
    assert "x" * 2000 in ctx


@pytest.mark.asyncio
async def test_empty_working_memory_no_crash(wm):
    """Workers get empty string when no prior context exists — no crash."""
    ctx = await wm.get_context_for_role("proj-empty", "first_worker")
    assert ctx == ""


@pytest.mark.asyncio
async def test_search_across_project(wm):
    """Artifact search works across all roles in a project."""
    shared_id = "proj-search-1"

    await wm.save_artifact(
        shared_id,
        "security_reviewer",
        "Found SQL injection vulnerability in auth.py line 42",
        {"label": "security_finding"},
    )
    await wm.save_artifact(
        shared_id,
        "performance_engineer",
        "50 sequential HTTP requests to HN API, should be parallelized",
        {"label": "perf_finding"},
    )

    # Search should find security findings
    results = await wm.search_artifacts(shared_id, "SQL injection")
    assert len(results) >= 1
    assert any("SQL injection" in r for r in results)

    # Search should find performance findings
    results = await wm.search_artifacts(shared_id, "sequential")
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_findings_upsert_across_stages(wm):
    """Re-running a role updates (not duplicates) its findings."""
    shared_id = "proj-upsert-1"

    # Stage 1: developer saves output
    await wm.save_finding(shared_id, "developer", "output_summary", "First pass: 3 fixes")

    # Stage 2 (feedback loop): developer runs again, updates summary
    await wm.save_finding(shared_id, "developer", "output_summary", "Second pass: 5 fixes total")

    # Should have only one row for output_summary
    async with wm._db.db.execute(
        "SELECT COUNT(*) FROM task_memory "
        "WHERE task_id = ? AND role = 'developer' AND key = 'output_summary'",
        (shared_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row[0] == 1

    # Other roles should see the updated value
    ctx = await wm.get_context_for_role(shared_id, "qa_engineer")
    assert "5 fixes total" in ctx
    assert "3 fixes" not in ctx


# ------------------------------------------------------------------
# Tests: save_finding_with_notify
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_finding_with_notify_emits_event(wm):
    """save_finding_with_notify should persist the finding AND emit FINDING_SAVED."""
    bus = EventBus()
    received: list[dict] = []

    async def handler(data: dict) -> None:
        received.append(data)

    bus.on(Events.FINDING_SAVED, handler)

    await wm.save_finding_with_notify(
        key="test_key",
        value="test_value",
        role="qa",
        task_id="task-notify-1",
        event_bus=bus,
    )

    assert len(received) == 1
    assert received[0]["key"] == "test_key"
    assert received[0]["role"] == "qa"
    assert received[0]["task_id"] == "task-notify-1"

    # Also verify the finding was actually persisted
    ctx = await wm.get_context_for_role("task-notify-1", "coder")
    assert "test_key" in ctx
    assert "test_value" in ctx


@pytest.mark.asyncio
async def test_save_finding_with_notify_no_bus(wm):
    """save_finding_with_notify works without an event bus (no error)."""
    await wm.save_finding_with_notify(
        key="k",
        value="v",
        role="dev",
        task_id="task-notify-2",
        event_bus=None,
    )

    ctx = await wm.get_context_for_role("task-notify-2", "other")
    assert "k" in ctx


# ------------------------------------------------------------------
# Tests: get_context_since
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_context_since_returns_only_new(wm):
    """get_context_since should return only findings created after the timestamp."""
    # Save a finding
    await wm.save_finding("task-since-1", "dev", "old", "old data")

    # Record a midpoint timestamp after the first save
    midpoint = time.time()

    # Backdate the first finding by updating created_at directly
    await wm._db.db.execute(
        "UPDATE task_memory SET created_at = '2020-01-01T00:00:00+00:00' "
        "WHERE key = 'old' AND task_id = 'task-since-1'"
    )
    await wm._db.db.commit()

    # Save a newer finding (gets current timestamp)
    await wm.save_finding("task-since-1", "dev", "new", "new data")

    # Query as a different role, since a time between old and new
    result = await wm.get_context_since(
        role="qa",
        task_id="task-since-1",
        since_timestamp=midpoint - 1,
    )

    assert "new" in result
    assert "old" not in result


@pytest.mark.asyncio
async def test_get_context_since_empty(wm):
    """get_context_since returns '' when no new findings exist."""
    result = await wm.get_context_since(
        role="qa",
        task_id="task-since-empty",
        since_timestamp=time.time(),
    )
    assert result == ""


@pytest.mark.asyncio
async def test_get_context_since_excludes_own_role(wm):
    """get_context_since does not include findings from the requesting role."""
    await wm.save_finding("t-since-own", "qa", "mine", "my data")

    result = await wm.get_context_since(
        role="qa",
        task_id="t-since-own",
        since_timestamp=0.0,
    )

    assert result == ""


@pytest.mark.asyncio
async def test_get_context_since_header(wm):
    """get_context_since uses '## New Updates' header."""
    await wm.save_finding("t-since-hdr", "dev", "f1", "data")

    result = await wm.get_context_since(
        role="qa",
        task_id="t-since-hdr",
        since_timestamp=0.0,
    )

    assert result.startswith("## New Updates")


# ------------------------------------------------------------------
# Tests: validate_finding
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_finding_approves(wm):
    """validate_finding sets status to 'approved'."""
    await wm.save_finding("t-val-1", "dev", "design", "schema v2")

    result = await wm.validate_finding(
        key="design",
        task_id="t-val-1",
        validator_role="architect",
        approved=True,
    )

    assert result is True

    async with wm._db.db.execute(
        "SELECT validated_by, status FROM task_memory "
        "WHERE key = 'design' AND task_id = 't-val-1'",
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "architect"
    assert row[1] == "approved"


@pytest.mark.asyncio
async def test_validate_finding_rejects(wm):
    """validate_finding sets status to 'rejected'."""
    await wm.save_finding("t-val-2", "dev", "idea", "bad idea")

    result = await wm.validate_finding(
        key="idea",
        task_id="t-val-2",
        validator_role="qa",
        approved=False,
    )

    assert result is True

    async with wm._db.db.execute(
        "SELECT status FROM task_memory WHERE key = 'idea' AND task_id = 't-val-2'",
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "rejected"


@pytest.mark.asyncio
async def test_validate_finding_no_match(wm):
    """validate_finding returns False when no matching row exists."""
    result = await wm.validate_finding(
        key="nonexistent",
        task_id="t-val-3",
        validator_role="qa",
        approved=True,
    )

    assert result is False


# ------------------------------------------------------------------
# Tests: check_updates tool
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_updates_returns_incremental(db):
    """check_updates returns only findings since the last check."""
    from unittest.mock import MagicMock

    from agent.tools.builtins import collaboration

    # Reset module state
    collaboration._last_check_timestamps.clear()
    collaboration._current_role_var.set("qa")
    collaboration._current_task_id_var.set("task-upd-1")

    # Set up a mock board that exposes _db
    mock_board = MagicMock()
    mock_board._db = db
    original_board = collaboration._global_task_board
    original_bus = collaboration._global_message_bus
    collaboration._global_task_board = mock_board
    collaboration._global_message_bus = None

    try:
        wm = WorkingMemory(db)

        # Save a finding from another role
        await wm.save_finding("task-upd-1", "dev", "finding1", "value1")

        # First check — should see the finding
        result1 = await collaboration.check_updates_tool()
        assert "finding1" in result1

        # Backdate that finding so it's before the second check's window
        await db.db.execute(
            "UPDATE task_memory SET created_at = '2020-01-01T00:00:00+00:00' "
            "WHERE key = 'finding1' AND task_id = 'task-upd-1'"
        )
        await db.db.commit()

        # Second check — nothing new
        result2 = await collaboration.check_updates_tool()
        assert "No new updates" in result2
    finally:
        collaboration._global_task_board = original_board
        collaboration._global_message_bus = original_bus
        collaboration._last_check_timestamps.clear()


@pytest.mark.asyncio
async def test_check_updates_no_context():
    """check_updates fails gracefully without task context."""
    from agent.tools.builtins import collaboration

    collaboration._current_task_id_var.set("")

    result = await collaboration.check_updates_tool()
    assert "No active task context" in result


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_already_validated_finding(wm):
    """Re-validating an already validated finding should update it."""
    await wm.save_finding("t-reval", "dev", "design", "schema v2")

    # First validation: approve
    result1 = await wm.validate_finding(
        key="design",
        task_id="t-reval",
        validator_role="architect",
        approved=True,
    )
    assert result1 is True

    # Re-validate: reject
    result2 = await wm.validate_finding(
        key="design",
        task_id="t-reval",
        validator_role="qa",
        approved=False,
    )
    assert result2 is True

    async with wm._db.db.execute(
        "SELECT validated_by, status FROM task_memory "
        "WHERE key = 'design' AND task_id = 't-reval'",
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "qa"
    assert row[1] == "rejected"


@pytest.mark.asyncio
async def test_get_context_since_future_timestamp(wm):
    """Future timestamp should return empty (no findings in the future)."""
    await wm.save_finding("t-future", "dev", "finding", "value")

    import time

    future_ts = time.time() + 86400  # 24 hours in the future

    result = await wm.get_context_since(
        role="qa",
        task_id="t-future",
        since_timestamp=future_ts,
    )
    assert result == ""


@pytest.mark.asyncio
async def test_check_updates_combines_messages_and_findings(db):
    """check_updates should return both new findings AND unread messages."""
    from unittest.mock import MagicMock

    from agent.core.events import EventBus
    from agent.core.message_bus import AgentMessage, MessageBus
    from agent.tools.builtins import collaboration

    # Reset module state
    collaboration._last_check_timestamps.clear()
    collaboration._current_role_var.set("qa")
    collaboration._current_task_id_var.set("task-combo")

    # Set up mock board that exposes _db for working memory
    mock_board = MagicMock()
    mock_board._db = db

    # Set up a real message bus
    event_bus = EventBus()
    bus = MessageBus(event_bus=event_bus)

    original_board = collaboration._global_task_board
    original_bus = collaboration._global_message_bus
    collaboration._global_task_board = mock_board
    collaboration._global_message_bus = bus

    try:
        wm = WorkingMemory(db)

        # Save a finding from another role
        await wm.save_finding("task-combo", "dev", "status", "code complete")

        # Send a message to qa
        await bus.send(
            AgentMessage(
                task_id="task-combo",
                from_role="dev",
                to_role="qa",
                content="Please review the code",
            )
        )

        # check_updates should return both
        result = await collaboration.check_updates_tool()
        assert "status" in result
        assert "code complete" in result
        assert "Please review the code" in result
    finally:
        collaboration._global_task_board = original_board
        collaboration._global_message_bus = original_bus
        collaboration._last_check_timestamps.clear()


@pytest.mark.asyncio
async def test_save_finding_with_notify_persists_correctly(wm):
    """Finding should exist in DB after save_finding_with_notify."""
    from agent.core.events import EventBus

    bus = EventBus()
    await wm.save_finding_with_notify(
        key="arch_decision",
        value="Use microservices",
        role="architect",
        task_id="task-persist",
        event_bus=bus,
    )

    # Verify it's persisted and visible to other roles
    ctx = await wm.get_context_for_role("task-persist", "developer")
    assert "arch_decision" in ctx
    assert "Use microservices" in ctx


@pytest.mark.asyncio
async def test_incremental_updates_across_multiple_calls(wm):
    """Multiple get_context_since calls should return only truly new items each time."""
    import time

    # Save first finding
    await wm.save_finding("t-incr", "dev", "finding1", "first")

    ts1 = 0.0  # Get everything
    result1 = await wm.get_context_since(role="qa", task_id="t-incr", since_timestamp=ts1)
    assert "finding1" in result1
    assert "first" in result1

    # Record timestamp between saves
    ts2 = time.time()

    # Backdate the first finding so it's before ts2
    await wm._db.db.execute(
        "UPDATE task_memory SET created_at = '2020-01-01T00:00:00+00:00' "
        "WHERE key = 'finding1' AND task_id = 't-incr'"
    )
    await wm._db.db.commit()

    # Save second finding (gets current timestamp)
    await wm.save_finding("t-incr", "dev", "finding2", "second")

    # Second call with ts2 should only return finding2
    result2 = await wm.get_context_since(
        role="qa",
        task_id="t-incr",
        since_timestamp=ts2 - 1,
    )
    assert "finding2" in result2
    assert "second" in result2
    assert "finding1" not in result2
