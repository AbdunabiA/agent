"""Collaboration tools — let agents post tasks to each other on a shared board."""

from __future__ import annotations

import contextvars
import time
from typing import TYPE_CHECKING

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.core.message_bus import MessageBus
    from agent.core.task_board import TaskBoard

_global_task_board: TaskBoard | None = None
_global_message_bus: MessageBus | None = None

# Tracks the last check_updates timestamp per role.
_last_check_timestamps: dict[str, float] = {}

# Per-task context vars — set by the orchestrator before running each worker.
_current_role_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "collaboration_current_role",
    default="unknown",
)
_current_task_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "collaboration_current_task_id",
    default="",
)


def set_task_board(board: TaskBoard) -> None:
    """Set the global TaskBoard instance (called during startup).

    Args:
        board: The initialized TaskBoard.
    """
    global _global_task_board
    _global_task_board = board


def get_task_board() -> TaskBoard | None:
    """Get the global TaskBoard instance.

    Returns:
        The shared TaskBoard, or None if not initialized.
    """
    return _global_task_board


def set_message_bus(bus: MessageBus) -> None:
    """Set the global MessageBus instance (called during startup).

    Args:
        bus: The initialized MessageBus.
    """
    global _global_message_bus
    _global_message_bus = bus


def get_message_bus() -> MessageBus | None:
    """Get the global MessageBus instance.

    Returns:
        The shared MessageBus, or None if not initialized.
    """
    return _global_message_bus


def set_collaboration_context(role: str, task_id: str) -> None:
    """Set current role and task_id context for collaboration tools.

    Called by the orchestrator before running each worker.

    Args:
        role: The worker's role name.
        task_id: The parent orchestration task ID.
    """
    _current_role_var.set(role)
    _current_task_id_var.set(task_id)


@tool(
    name="report_bug",
    description=(
        "Report a bug found during testing or review. Posts a ticket "
        "to the backend developer. Use severity='critical' for blockers."
    ),
    tier=ToolTier.SAFE,
)
async def report_bug(
    title: str,
    description: str,
    file_path: str = "",
    line_number: int = 0,
    severity: str = "normal",
    suggested_fix: str = "",
) -> str:
    """Report a bug to the backend developer.

    Args:
        title: Short bug title.
        description: Detailed bug description.
        file_path: File where the bug was found.
        line_number: Line number in the file.
        severity: 'critical' or 'normal'.
        suggested_fix: Optional suggested fix.

    Returns:
        Confirmation with ticket ID.
    """
    board = get_task_board()
    if board is None:
        return "Task board not available. Cannot report bug."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context. Cannot report bug."

    priority = "blocker" if severity == "critical" else "normal"

    context: dict = {}
    if file_path:
        context["file_path"] = file_path
    if line_number:
        context["line_number"] = line_number
    if suggested_fix:
        context["suggested_fix"] = suggested_fix

    ticket_id = await board.post_task(
        from_role=current_role,
        to_role="backend_developer",
        task_id=task_id,
        title=title,
        description=description,
        priority=priority,
        context=context,
    )

    return f"🐛 Bug #{ticket_id} reported. Backend developer assigned."


@tool(
    name="request_review",
    description=(
        "Request a code review or QA check. Posts a ticket to the "
        "QA engineer with the files that changed."
    ),
    tier=ToolTier.SAFE,
)
async def request_review(
    what_to_review: str,
    files_changed: str = "",
    notes: str = "",
) -> str:
    """Request a review from QA.

    Args:
        what_to_review: Description of what needs reviewing.
        files_changed: Comma-separated list of changed files.
        notes: Additional notes for the reviewer.

    Returns:
        Confirmation with ticket ID.
    """
    board = get_task_board()
    if board is None:
        return "Task board not available. Cannot request review."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context. Cannot request review."

    context: dict = {}
    if files_changed:
        context["files_changed"] = [f.strip() for f in files_changed.split(",") if f.strip()]
    if notes:
        context["notes"] = notes

    ticket_id = await board.post_task(
        from_role=current_role,
        to_role="qa_engineer",
        task_id=task_id,
        title=f"Review: {what_to_review[:80]}",
        description=what_to_review,
        context=context,
    )

    return f"👀 Review requested #{ticket_id}."


@tool(
    name="assign_task",
    description=(
        "Assign a task to any role on the team. Use this for generic "
        "task assignments like 'architect, design the API'."
    ),
    tier=ToolTier.SAFE,
)
async def assign_task(
    to_role: str,
    title: str,
    description: str = "",
    context: str = "",
) -> str:
    """Assign a task to a specific role.

    Args:
        to_role: Target role name (e.g. 'backend_developer', 'qa_engineer').
        title: Short task title.
        description: Detailed task description.
        context: Optional JSON context string.

    Returns:
        Confirmation with ticket ID.
    """
    board = get_task_board()
    if board is None:
        return "Task board not available. Cannot assign task."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context. Cannot assign task."

    ctx: dict = {}
    if context:
        try:
            import json

            ctx = json.loads(context)
        except (json.JSONDecodeError, TypeError):
            ctx = {"raw": context}

    ticket_id = await board.post_task(
        from_role=current_role,
        to_role=to_role,
        task_id=task_id,
        title=title,
        description=description,
        context=ctx,
    )

    return f"📋 Task #{ticket_id} assigned to {to_role}."


@tool(
    name="get_my_tasks",
    description=(
        "Get your pending tasks from the task board. Shows tickets "
        "assigned to your role, ordered by priority."
    ),
    tier=ToolTier.SAFE,
)
async def get_my_tasks() -> str:
    """Get pending tasks for the current role.

    Returns:
        Formatted list of pending tickets, or a message if none.
    """
    board = get_task_board()
    if board is None:
        return "Task board not available."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context."

    tickets = await board.get_my_tasks(role=current_role, task_id=task_id)

    if not tickets:
        return "No pending tasks."

    lines = [f"You have {len(tickets)} pending task(s):"]
    for i, t in enumerate(tickets, 1):
        prio = "[BLOCKER] " if t["priority"] == "blocker" else f"[{t['priority']}] "
        lines.append(f"{i}. {prio}{t['title']} (from {t['from_role']}, #{t['id']})")
        if t["description"]:
            lines.append(f"   {t['description'][:200]}")

    return "\n".join(lines)


@tool(
    name="complete_my_task",
    description=(
        "Mark a task as complete. Provide the ticket ID and a summary " "of what you did."
    ),
    tier=ToolTier.SAFE,
)
async def complete_my_task(
    ticket_id: str,
    result_summary: str,
) -> str:
    """Mark a ticket as done.

    Args:
        ticket_id: The ticket ID to complete.
        result_summary: Summary of what was done.

    Returns:
        Confirmation message.
    """
    board = get_task_board()
    if board is None:
        return "Task board not available."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()

    # Ownership check: only the assigned role can complete a ticket
    if task_id:
        ticket = await board.get_ticket(ticket_id)
        if ticket and ticket.get("to_role") and ticket["to_role"] != current_role:
            return (
                f"Cannot complete ticket #{ticket_id} — it is assigned to "
                f"'{ticket['to_role']}', not '{current_role}'."
            )

    await board.complete_ticket(ticket_id=ticket_id, result=result_summary)

    return f"✅ Task #{ticket_id} marked complete."


@tool(
    name="save_finding",
    description=(
        "Save a finding or observation to the team's shared working memory. "
        "Other agents on the team can see your findings. Use this to record "
        "bugs found, test results, review observations, or any insight the team should know."
    ),
    tier=ToolTier.SAFE,
)
async def save_finding(
    key: str,
    value: str,
) -> str:
    """Save a finding to working memory.

    Args:
        key: Short identifier for the finding (e.g. 'bugs_found', 'test_results').
        value: The finding content.

    Returns:
        Confirmation message.
    """
    from agent.tools.builtins.collaboration import get_task_board

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context. Cannot save finding."

    # Try to save via WorkingMemory if available
    try:
        from agent.core.working_memory import WorkingMemory

        board = get_task_board()
        if board is not None and hasattr(board, "_db"):
            wm = WorkingMemory(board._db)
            await wm.save_finding(
                task_id=task_id,
                role=current_role,
                key=key,
                value=value,
            )
            return f"Finding '{key}' saved to working memory."
    except Exception:
        pass

    # Fallback: store as a task board ticket to self
    board = get_task_board()
    if board is not None:
        await board.post_task(
            from_role=current_role,
            to_role=current_role,
            task_id=task_id,
            title=f"Finding: {key}",
            description=value,
            priority="normal",
            context={"kind": "finding"},
        )
        return f"Finding '{key}' saved."

    return "Cannot save finding — no working memory or task board available."


@tool(
    name="ask_team",
    description=(
        "Ask a question to a specialist on the team. Routes through "
        "the consult_agent tool if available, otherwise returns a placeholder."
    ),
    tier=ToolTier.MODERATE,
)
async def ask_team(
    role: str,
    question: str,
) -> str:
    """Ask a specialist a question.

    Args:
        role: Role to consult (e.g. 'architect', 'qa_engineer').
        question: The question to ask.

    Returns:
        The specialist's answer.
    """
    # Try to route through consult_agent if the orchestrator is available
    try:
        from agent.core.subagent import ConsultRequest, SubAgentStatus
        from agent.tools.builtins.orchestration import get_orchestrator

        orchestrator = get_orchestrator()

        # Find a team that has this role
        target_team = None
        for team_name, team in orchestrator.teams.items():
            for r in team.roles:
                if r.name == role:
                    target_team = team_name
                    break
            if target_team:
                break

        if target_team is None:
            return f"No team found with role '{role}'. Available roles: check list_agent_teams."

        import uuid as _uuid

        from agent.tools.builtins.orchestration import get_nesting_depth

        request = ConsultRequest(
            requesting_agent_id=f"collab-{_uuid.uuid4().hex[:8]}",
            requesting_role=_current_role_var.get(),
            target_team=target_team,
            target_role=role,
            question=question,
        )

        response = await orchestrator.handle_consult(request, get_nesting_depth())

        if response.status == SubAgentStatus.COMPLETED:
            return f"Answer from {role}:\n{response.answer}"
        return f"Consultation with {role} failed: {response.error}"

    except (RuntimeError, ImportError):
        return f"Cannot consult {role} — orchestrator not available. " f"Question was: {question}"


# ---------------------------------------------------------------------------
# Inter-agent messaging tools
# ---------------------------------------------------------------------------


@tool(
    name="send_message",
    description="Send a message to another agent role",
    tier=ToolTier.SAFE,
)
async def send_message_tool(
    to_role: str,
    content: str,
    msg_type: str = "question",
) -> str:
    """Send a message to another agent role.

    Args:
        to_role: Target role name (e.g. 'backend_developer', 'qa_engineer').
        content: The message content.
        msg_type: Message type — 'question', 'answer', 'status', or 'alert'.

    Returns:
        Confirmation with message ID.
    """
    bus = get_message_bus()
    if bus is None:
        return "Message bus not available. Cannot send message."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context. Cannot send message."

    from agent.core.message_bus import AgentMessage

    msg = AgentMessage(
        task_id=task_id,
        from_role=current_role,
        to_role=to_role,
        content=content,
        msg_type=msg_type,
    )
    msg_id = await bus.send(msg)

    return f"Message {msg_id} sent to {to_role}."


@tool(
    name="read_messages",
    description="Read messages sent to you by other agents",
    tier=ToolTier.SAFE,
)
async def read_messages_tool(unread_only: bool = True) -> str:
    """Read messages sent to the current role.

    Args:
        unread_only: If True, only return unread messages.

    Returns:
        Formatted list of messages, or a notice if none.
    """
    bus = get_message_bus()
    if bus is None:
        return "Message bus not available."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context."

    messages = await bus.get_messages(
        task_id=task_id,
        role=current_role,
        unread_only=unread_only,
    )

    if not messages:
        label = "unread " if unread_only else ""
        return f"No {label}messages."

    lines = [f"You have {len(messages)} message(s):"]
    for i, m in enumerate(messages, 1):
        prefix = f"[{m.msg_type}]"
        lines.append(f"{i}. {prefix} From {m.from_role} (id={m.id}): {m.content[:300]}")
    return "\n".join(lines)


@tool(
    name="reply_message",
    description="Reply to a specific message",
    tier=ToolTier.SAFE,
)
async def reply_message_tool(message_id: str, content: str) -> str:
    """Reply to a specific message, keeping the same thread.

    Args:
        message_id: The ID of the message to reply to.
        content: The reply content.

    Returns:
        Confirmation with the reply message ID.
    """
    bus = get_message_bus()
    if bus is None:
        return "Message bus not available."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context."

    # Find the original message to get thread_id and sender
    original: object | None = None
    for msgs in bus._messages.values():
        for m in msgs:
            if m.id == message_id:
                original = m
                break
        if original is not None:
            break

    if original is None:
        return f"Message {message_id} not found."

    from agent.core.message_bus import AgentMessage

    reply = AgentMessage(
        task_id=task_id,
        from_role=current_role,
        to_role=original.from_role,  # type: ignore[union-attr]
        thread_id=original.thread_id,  # type: ignore[union-attr]
        content=content,
        msg_type="answer",
        reply_to=message_id,
    )
    reply_id = await bus.send(reply)

    return f"Reply {reply_id} sent to {original.from_role}."  # type: ignore[union-attr]


@tool(
    name="broadcast_team",
    description="Send status update to all agents",
    tier=ToolTier.SAFE,
)
async def broadcast_team_tool(
    content: str,
    msg_type: str = "status",
) -> str:
    """Broadcast a message to all agents on the team.

    Args:
        content: The message content.
        msg_type: Message type — typically 'status' or 'alert'.

    Returns:
        Confirmation with message ID.
    """
    bus = get_message_bus()
    if bus is None:
        return "Message bus not available. Cannot broadcast."

    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context."

    from agent.core.message_bus import AgentMessage

    msg = AgentMessage(
        task_id=task_id,
        from_role=current_role,
        to_role=None,
        content=content,
        msg_type=msg_type,
    )
    msg_id = await bus.broadcast(msg)

    return f"Broadcast {msg_id} sent to all agents."


@tool(
    name="check_updates",
    description=(
        "Check for new findings and messages since you last checked. "
        "Returns incremental updates from other agents."
    ),
    tier=ToolTier.SAFE,
)
async def check_updates_tool() -> str:
    """Check for new findings and messages since last check.

    Uses a per-role timestamp to return only data that arrived
    after the previous invocation.  On the first call for a role
    all existing findings are returned.

    Returns:
        Combined findings and unread messages, or a notice if
        nothing new.
    """
    current_role = _current_role_var.get()
    task_id = _current_task_id_var.get()
    if not task_id:
        return "No active task context. Cannot check updates."

    last_ts = _last_check_timestamps.get(current_role, 0.0)
    now = time.time()

    parts: list[str] = []

    # 1. New findings from working memory
    try:
        from agent.core.working_memory import WorkingMemory

        board = get_task_board()
        if board is not None and hasattr(board, "_db"):
            wm = WorkingMemory(board._db)
            context = await wm.get_context_since(
                role=current_role,
                task_id=task_id,
                since_timestamp=last_ts,
            )
            if context:
                parts.append(context)
    except Exception:
        pass

    # 2. Unread messages from the message bus
    bus = get_message_bus()
    if bus is not None:
        try:
            messages = await bus.get_messages(
                task_id=task_id,
                role=current_role,
                unread_only=True,
            )
            if messages:
                msg_lines = [f"## Unread Messages ({len(messages)})"]
                for m in messages:
                    msg_lines.append(
                        f"- **[{m.msg_type}]** from {m.from_role}: " f"{m.content[:300]}"
                    )
                parts.append("\n".join(msg_lines))
        except Exception:
            pass

    _last_check_timestamps[current_role] = now

    if not parts:
        return "No new updates since last check."

    return "\n\n".join(parts)
