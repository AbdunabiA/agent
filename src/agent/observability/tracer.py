"""Agent span tracer — full visibility into multi-agent execution.

Records hierarchical spans for every agent operation (spawns, consults,
delegations, retries) with timing, token usage, and error tracking.
Persists to SQLite for post-hoc analysis and the Telegram /trace command.

All database writes are wrapped in try/except — the tracer never crashes
the main agent flow.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import structlog

if TYPE_CHECKING:
    from agent.memory.database import Database

logger = structlog.get_logger(__name__)

SpanStatus = Literal["running", "ok", "error", "retry"]


@dataclass
class AgentSpan:
    """A single traced operation within the agent system.

    Spans form a tree: a task-level span parents controller spans,
    which parent worker spawns, which parent tool calls.

    Attributes:
        span_id: Unique identifier for this span.
        parent_id: Parent span id (None for root spans).
        task_id: The orchestration task this span belongs to.
        role: Agent role name (e.g. ``"controller"``, ``"qa_engineer"``).
        operation: What kind of work (``"spawn"``, ``"tool_call"``,
            ``"consult"``, ``"delegate"``, ``"retry"``).
        started_at: When the operation began.
        ended_at: When it finished (None while running).
        status: Current status.
        tokens_input: Input tokens consumed.
        tokens_output: Output tokens consumed.
        error: Error message if status is ``"error"``.
        metadata: Flexible extra data attached by callers.
    """

    span_id: str
    parent_id: str | None
    task_id: str
    role: str
    operation: str
    started_at: datetime
    ended_at: datetime | None = None
    status: SpanStatus = "running"
    tokens_input: int = 0
    tokens_output: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        """Duration in seconds, or None if still running."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def total_tokens(self) -> int:
        """Combined input + output tokens."""
        return self.tokens_input + self.tokens_output


# ------------------------------------------------------------------
# Status icons for tree rendering
# ------------------------------------------------------------------

_STATUS_ICON: dict[str, str] = {
    "ok": "\U0001f7e2",  # 🟢
    "error": "\U0001f534",  # 🔴
    "running": "\U0001f504",  # 🔄
    "retry": "\U0001f501",  # 🔁
}

_WAITING_ICON = "\u23f3"  # ⏳


class AgentTracer:
    """Hierarchical span tracer with SQLite persistence.

    Usage::

        tracer = AgentTracer(database)
        async with tracer.span("task-1", "coder", "spawn") as s:
            s.tokens_input = 500
            s.metadata["files"] = ["main.py"]

    If ``database`` is None the tracer operates as a no-op — all public
    methods return immediately without error.

    Args:
        database: The project's async SQLite database handle.
    """

    def __init__(self, database: Database | None = None) -> None:
        self._db = database

    # ------------------------------------------------------------------
    # Schema (called from Database._migrate)
    # ------------------------------------------------------------------

    @staticmethod
    def migration_sql() -> str:
        """Return the DDL for the ``agent_spans`` table."""
        return """
CREATE TABLE IF NOT EXISTS agent_spans (
    span_id TEXT PRIMARY KEY,
    parent_id TEXT,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    operation TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    tokens_input INTEGER NOT NULL DEFAULT 0,
    tokens_output INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_task ON agent_spans(task_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent ON agent_spans(parent_id);
CREATE INDEX IF NOT EXISTS idx_spans_status ON agent_spans(status);
CREATE INDEX IF NOT EXISTS idx_spans_role_status ON agent_spans(role, status);
"""

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def span(
        self,
        task_id: str,
        role: str,
        operation: str,
        parent_id: str | None = None,
    ) -> AsyncGenerator[AgentSpan, None]:
        """Create a traced span around an operation.

        On entry the span is persisted with status ``"running"``.
        On normal exit it is marked ``"ok"``.  If an exception is
        raised it is marked ``"error"`` and **re-raised** so the
        caller's error handling is unaffected.

        Args:
            task_id: Orchestration task identifier.
            role: Agent role name.
            operation: Operation type.
            parent_id: Optional parent span id for nesting.

        Yields:
            The mutable :class:`AgentSpan` — callers can set
            ``tokens_input``, ``tokens_output``, and ``metadata``.
        """
        s = AgentSpan(
            span_id=uuid4().hex[:12],
            parent_id=parent_id,
            task_id=task_id,
            role=role,
            operation=operation,
            started_at=datetime.now(UTC),
        )

        await self._persist_span(s)

        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.error = str(exc)[:1000]
            raise
        else:
            if s.status == "running":
                s.status = "ok"
        finally:
            s.ended_at = datetime.now(UTC)
            await self._persist_span(s)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_task_tree(self, task_id: str) -> str:
        """Build a formatted tree view of all spans for a task.

        Returns a unicode-art tree suitable for display in Telegram
        or a terminal.

        Args:
            task_id: The task to visualise.

        Returns:
            Formatted tree string, or a short message if no spans exist.
        """
        spans = await self._load_spans(task_id)
        if not spans:
            return f"No spans recorded for task {task_id}"

        # Build parent → children mapping
        by_parent: dict[str | None, list[AgentSpan]] = defaultdict(list)
        for s in spans:
            by_parent[s.parent_id].append(s)

        # Sort children by started_at
        for children in by_parent.values():
            children.sort(key=lambda x: x.started_at)

        # Calculate total duration
        earliest = min(s.started_at for s in spans)
        latest_end = max(
            (s.ended_at for s in spans if s.ended_at is not None),
            default=None,
        )
        if latest_end:
            total_s = (latest_end - earliest).total_seconds()
            total_str = f"{total_s:.1f}s"
        else:
            total_str = "running..."

        # Check for project-level root span to show project name
        roots = by_parent.get(None, [])
        project_root = next(
            (s for s in roots if s.operation == "project"),
            None,
        )
        if project_root:
            header = f"\U0001f4ca Project: {project_root.role} " f"({total_str} total)"
        else:
            header = f"\U0001f4ca Task: {task_id} ({total_str} total)"
        lines = [header]

        # Render tree from roots
        roots = by_parent.get(None, [])
        for i, root in enumerate(roots):
            is_last = i == len(roots) - 1
            self._render_node(root, by_parent, lines, "", is_last)

        return "\n".join(lines)

    async def get_stats(self, task_id: str) -> dict[str, Any]:
        """Aggregate statistics for a task.

        Args:
            task_id: The task to summarise.

        Returns:
            Dict with ``total_tokens``, ``total_duration_s``, and
            per-role status counts in ``roles``.
        """
        spans = await self._load_spans(task_id)
        if not spans:
            return {
                "total_tokens": 0,
                "total_duration_s": 0.0,
                "roles": {},
            }

        total_tokens = sum(s.total_tokens for s in spans)

        earliest = min(s.started_at for s in spans)
        latest_end = max(
            (s.ended_at for s in spans if s.ended_at is not None),
            default=None,
        )
        total_duration = (latest_end - earliest).total_seconds() if latest_end else 0.0

        roles: dict[str, dict[str, int]] = {}
        for s in spans:
            counters = roles.setdefault(s.role, {"ok": 0, "error": 0, "retry": 0})
            if s.status in counters:
                counters[s.status] += 1

        return {
            "total_tokens": total_tokens,
            "total_duration_s": round(total_duration, 2),
            "roles": roles,
        }

    async def get_recent_tasks(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return summary info for the most recent tasks.

        Args:
            limit: Maximum number of tasks to return.

        Returns:
            List of dicts with ``task_id``, ``status``, ``duration_s``,
            ``total_tokens``, and ``started_at``.
        """
        if not self._db:
            return []

        try:
            async with self._db.db.execute(
                "SELECT DISTINCT task_id, MIN(started_at) as first_start "
                "FROM agent_spans GROUP BY task_id "
                "ORDER BY first_start DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()

            results: list[dict[str, Any]] = []
            for row in rows:
                tid = row[0]
                spans = await self._load_spans(tid)
                if not spans:
                    continue

                statuses = {s.status for s in spans}
                if "error" in statuses:
                    overall = "error"
                elif "running" in statuses:
                    overall = "running"
                else:
                    overall = "ok"

                earliest = min(s.started_at for s in spans)
                latest_end = max(
                    (s.ended_at for s in spans if s.ended_at is not None),
                    default=None,
                )
                duration = (latest_end - earliest).total_seconds() if latest_end else None

                # Extract project name from root span if available
                project_name = None
                stage_count = 0
                for s in spans:
                    if s.parent_id is None and s.operation == "project":
                        project_name = s.role
                    if s.operation == "stage":
                        stage_count += 1

                results.append(
                    {
                        "task_id": tid,
                        "status": overall,
                        "duration_s": round(duration, 1) if duration else None,
                        "total_tokens": sum(s.total_tokens for s in spans),
                        "started_at": earliest.isoformat(),
                        "project_name": project_name,
                        "stage_count": stage_count,
                    }
                )

            return results
        except Exception as e:
            logger.warning("tracer_get_recent_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_node(
        self,
        span: AgentSpan,
        by_parent: dict[str | None, list[AgentSpan]],
        lines: list[str],
        prefix: str,
        is_last: bool,
    ) -> None:
        """Recursively render a span node into the tree lines."""
        connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "

        # Duration string
        if span.duration_s is not None:
            dur = f"{span.duration_s:.1f}s"
        elif span.status == "running":
            dur = "running..."
        else:
            dur = ""

        # Token string
        tok = ""
        if span.total_tokens > 0:
            if span.total_tokens >= 1000:
                tok = f"  ({span.total_tokens / 1000:.1f}k tokens)"
            else:
                tok = f"  ({span.total_tokens} tokens)"

        # Render based on operation type
        if span.operation == "project":
            icon = "\U0001f4ca"  # 📊
            instr = span.metadata.get("instruction", "")[:60]
            detail = f'  "{instr}"' if instr else ""
            line = f"{prefix}{connector}{icon} Project: {span.role}  {dur}{tok}{detail}"
        elif span.operation == "stage":
            icon = "\U0001f4cb"  # 📋
            agents = span.metadata.get("agents", "")
            agent_str = f"  ({agents} agents)" if agents else ""
            line = f"{prefix}{connector}{icon} {span.role}{agent_str}  {dur}{tok}"
        else:
            icon = _STATUS_ICON.get(span.status, _WAITING_ICON)
            # Operation suffix for non-spawn ops
            op_suffix = ""
            if span.operation not in ("spawn", ""):
                op_suffix = f" [{span.operation}]"
            # Output preview for worker spans
            preview = ""
            output_preview = span.metadata.get("output_preview", "")
            if output_preview:
                preview = f'  "{output_preview[:80]}"'
            line = f"{prefix}{connector}{icon} {span.role}{op_suffix}  {dur}{tok}{preview}"

        if span.error:
            line += f"  \u274c {span.error[:80]}"
        lines.append(line)

        # Render children
        children = by_parent.get(span.span_id, [])
        child_prefix = prefix + ("   " if is_last else "\u2502  ")
        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            self._render_node(child, by_parent, lines, child_prefix, child_is_last)

    async def _persist_span(self, span: AgentSpan) -> None:
        """Write or update a span row in SQLite.

        Silently logs and swallows any database error.
        """
        if not self._db:
            return

        try:
            await self._db.db.execute(
                "INSERT OR REPLACE INTO agent_spans "
                "(span_id, parent_id, task_id, role, operation, "
                " started_at, ended_at, status, tokens_input, tokens_output, "
                " error, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    span.span_id,
                    span.parent_id,
                    span.task_id,
                    span.role,
                    span.operation,
                    span.started_at.isoformat(),
                    span.ended_at.isoformat() if span.ended_at else None,
                    span.status,
                    span.tokens_input,
                    span.tokens_output,
                    span.error,
                    json.dumps(span.metadata) if span.metadata else None,
                ),
            )
            await self._db.db.commit()
        except Exception as e:
            logger.warning(
                "tracer_persist_failed",
                span_id=span.span_id,
                error=str(e),
            )

    async def _load_spans(self, task_id: str) -> list[AgentSpan]:
        """Load all spans for a task from the database."""
        if not self._db:
            return []

        try:
            async with self._db.db.execute(
                "SELECT span_id, parent_id, task_id, role, operation, "
                "started_at, ended_at, status, tokens_input, tokens_output, "
                "error, metadata_json "
                "FROM agent_spans WHERE task_id = ? ORDER BY started_at",
                (task_id,),
            ) as cursor:
                rows = await cursor.fetchall()

            spans: list[AgentSpan] = []
            for r in rows:
                spans.append(
                    AgentSpan(
                        span_id=r[0],
                        parent_id=r[1],
                        task_id=r[2],
                        role=r[3],
                        operation=r[4],
                        started_at=datetime.fromisoformat(r[5]),
                        ended_at=datetime.fromisoformat(r[6]) if r[6] else None,
                        status=r[7],
                        tokens_input=r[8] or 0,
                        tokens_output=r[9] or 0,
                        error=r[10],
                        metadata=json.loads(r[11]) if r[11] else {},
                    )
                )
            return spans
        except Exception as e:
            logger.warning("tracer_load_failed", task_id=task_id, error=str(e))
            return []
