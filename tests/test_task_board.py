"""Tests for TaskBoard and collaboration tools."""

from __future__ import annotations

import pytest

from agent.core.task_board import TaskBoard
from agent.memory.database import Database


@pytest.fixture
async def db(tmp_path):
    """Create a temporary database with task_tickets table."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def board(db):
    """Create a TaskBoard backed by the test database."""
    return TaskBoard(db)


class TestTaskBoard:
    """Tests for TaskBoard CRUD operations."""

    async def test_post_and_get_task(self, board: TaskBoard) -> None:
        ticket_id = await board.post_task(
            from_role="qa_engineer",
            to_role="backend_developer",
            task_id="task-001",
            title="Fix auth bug",
            description="Auth middleware returns 500 on expired tokens",
        )
        assert ticket_id.startswith("tkt-")

        tasks = await board.get_my_tasks(role="backend_developer", task_id="task-001")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Fix auth bug"
        assert tasks[0]["from_role"] == "qa_engineer"
        assert tasks[0]["status"] == "pending"

    async def test_get_my_tasks_empty(self, board: TaskBoard) -> None:
        tasks = await board.get_my_tasks(role="backend_developer", task_id="task-999")
        assert tasks == []

    async def test_complete_ticket(self, board: TaskBoard) -> None:
        ticket_id = await board.post_task(
            from_role="architect",
            to_role="backend_developer",
            task_id="task-002",
            title="Add rate limiting",
            description="Add rate limiting to API endpoints",
        )

        await board.complete_ticket(ticket_id, result="Added rate limiter middleware")

        # Should no longer appear in pending tasks
        tasks = await board.get_my_tasks(role="backend_developer", task_id="task-002")
        assert len(tasks) == 0

    async def test_start_ticket(self, board: TaskBoard) -> None:
        ticket_id = await board.post_task(
            from_role="qa_engineer",
            to_role="backend_developer",
            task_id="task-003",
            title="Fix login",
            description="Login page broken",
        )

        await board.start_ticket(ticket_id)

        # In-progress tickets should not appear in get_my_tasks (pending only)
        tasks = await board.get_my_tasks(role="backend_developer", task_id="task-003")
        assert len(tasks) == 0

    async def test_blocker_priority_sorts_first(self, board: TaskBoard) -> None:
        await board.post_task(
            from_role="qa_engineer",
            to_role="backend_developer",
            task_id="task-004",
            title="Normal task",
            description="Normal priority task",
            priority="normal",
        )
        await board.post_task(
            from_role="qa_engineer",
            to_role="backend_developer",
            task_id="task-004",
            title="Blocker task",
            description="Critical bug",
            priority="blocker",
        )

        tasks = await board.get_my_tasks(role="backend_developer", task_id="task-004")
        assert len(tasks) == 2
        assert tasks[0]["priority"] == "blocker"
        assert tasks[1]["priority"] == "normal"

    async def test_board_summary(self, board: TaskBoard) -> None:
        t1 = await board.post_task(
            from_role="qa_engineer",
            to_role="backend_developer",
            task_id="task-005",
            title="Fix auth middleware",
            description="Auth broken",
        )
        await board.complete_ticket(t1, result="fixed in commit abc123")

        await board.post_task(
            from_role="backend_developer",
            to_role="qa_engineer",
            task_id="task-005",
            title="Re-test after fix",
            description="Please re-test",
        )

        summary = await board.get_board_summary("task-005")
        assert "📋 Task Board:" in summary
        assert "✅" in summary
        assert "⏳" in summary
        assert "Fix auth middleware" in summary
        assert "Re-test after fix" in summary

    async def test_board_summary_empty(self, board: TaskBoard) -> None:
        summary = await board.get_board_summary("task-empty")
        assert "No tickets" in summary

    async def test_pending_count(self, board: TaskBoard) -> None:
        assert await board.get_pending_count("task-006") == 0
        assert not await board.has_pending("task-006")

        await board.post_task(
            from_role="qa",
            to_role="dev",
            task_id="task-006",
            title="T1",
            description="",
        )
        await board.post_task(
            from_role="qa",
            to_role="dev",
            task_id="task-006",
            title="T2",
            description="",
        )

        assert await board.get_pending_count("task-006") == 2
        assert await board.has_pending("task-006")

    async def test_roles_with_pending(self, board: TaskBoard) -> None:
        await board.post_task(
            from_role="qa",
            to_role="backend_developer",
            task_id="task-007",
            title="T1",
            description="",
        )
        await board.post_task(
            from_role="architect",
            to_role="qa_engineer",
            task_id="task-007",
            title="T2",
            description="",
        )

        roles = await board.get_roles_with_pending("task-007")
        assert roles == {"backend_developer", "qa_engineer"}

    async def test_report_bug_critical_is_blocker(self, board: TaskBoard) -> None:
        """Verify severity='critical' maps to priority='blocker'."""
        await board.post_task(
            from_role="qa_engineer",
            to_role="backend_developer",
            task_id="task-008",
            title="Critical auth bug",
            description="Auth completely broken",
            priority="blocker",
        )

        tasks = await board.get_my_tasks(role="backend_developer", task_id="task-008")
        assert len(tasks) == 1
        assert tasks[0]["priority"] == "blocker"

    async def test_context_preserved(self, board: TaskBoard) -> None:
        """Verify context dict is stored and retrieved correctly."""
        ctx = {"file_path": "src/auth.py", "line_number": 42}
        await board.post_task(
            from_role="qa",
            to_role="dev",
            task_id="task-009",
            title="Bug with context",
            description="test",
            context=ctx,
        )

        tasks = await board.get_my_tasks(role="dev", task_id="task-009")
        assert tasks[0]["context"] == ctx


class TestCollaborationTools:
    """Tests for the collaboration tool functions."""

    async def test_report_bug_tool(self, board: TaskBoard) -> None:
        from agent.tools.builtins.collaboration import (
            report_bug,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(board)
        set_collaboration_context("qa_engineer", "task-t01")

        result = await report_bug(
            title="Auth returns 500",
            description="Expired tokens cause 500",
            file_path="src/auth.py",
            line_number=42,
            severity="critical",
        )

        assert "🐛 Bug #" in result
        assert "Backend developer assigned" in result

        # Verify it landed on the board as a blocker
        tasks = await board.get_my_tasks(role="backend_developer", task_id="task-t01")
        assert len(tasks) == 1
        assert tasks[0]["priority"] == "blocker"

    async def test_request_review_tool(self, board: TaskBoard) -> None:
        from agent.tools.builtins.collaboration import (
            request_review,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(board)
        set_collaboration_context("backend_developer", "task-t02")

        result = await request_review(
            what_to_review="Auth middleware refactor",
            files_changed="src/auth.py, src/middleware.py",
            notes="Focus on error handling",
        )

        assert "👀 Review requested #" in result

        tasks = await board.get_my_tasks(role="qa_engineer", task_id="task-t02")
        assert len(tasks) == 1

    async def test_assign_task_tool(self, board: TaskBoard) -> None:
        from agent.tools.builtins.collaboration import (
            assign_task,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(board)
        set_collaboration_context("architect", "task-t03")

        result = await assign_task(
            to_role="backend_developer",
            title="Add rate limiting",
            description="Implement token bucket rate limiter",
        )

        assert "📋 Task #" in result
        assert "backend_developer" in result

    async def test_get_my_tasks_tool(self, board: TaskBoard) -> None:
        from agent.tools.builtins.collaboration import (
            get_my_tasks,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(board)

        # Post a task then check from the receiver's perspective
        await board.post_task(
            from_role="qa",
            to_role="dev",
            task_id="task-t04",
            title="Fix login",
            description="Login broken",
            priority="blocker",
        )

        set_collaboration_context("dev", "task-t04")
        result = await get_my_tasks()

        assert "1 pending task" in result
        assert "[BLOCKER]" in result
        assert "Fix login" in result

    async def test_get_my_tasks_tool_empty(self, board: TaskBoard) -> None:
        from agent.tools.builtins.collaboration import (
            get_my_tasks,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(board)
        set_collaboration_context("dev", "task-t05")

        result = await get_my_tasks()
        assert "No pending tasks" in result

    async def test_complete_my_task_tool(self, board: TaskBoard) -> None:
        from agent.tools.builtins.collaboration import (
            complete_my_task,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(board)
        set_collaboration_context("dev", "task-t06")

        ticket_id = await board.post_task(
            from_role="qa",
            to_role="dev",
            task_id="task-t06",
            title="Fix it",
            description="broken",
        )

        result = await complete_my_task(ticket_id=ticket_id, result_summary="Fixed it")
        assert "✅ Task #" in result

        # Verify it's done
        tasks = await board.get_my_tasks(role="dev", task_id="task-t06")
        assert len(tasks) == 0

    async def test_tools_without_board_return_error(self) -> None:
        from agent.tools.builtins.collaboration import (
            report_bug,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(None)  # type: ignore[arg-type]
        set_collaboration_context("qa", "task-x")

        result = await report_bug(title="x", description="y")
        assert "not available" in result

    async def test_tools_without_task_id_return_error(self, board: TaskBoard) -> None:
        from agent.tools.builtins.collaboration import (
            report_bug,
            set_collaboration_context,
            set_task_board,
        )

        set_task_board(board)
        set_collaboration_context("qa", "")  # no task_id

        result = await report_bug(title="x", description="y")
        assert "No active task context" in result


class TestMigration:
    """Test that migration v5 creates the table correctly."""

    async def test_task_tickets_table_exists(self, db: Database) -> None:
        async with db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='task_tickets'"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "task_tickets"
