"""Monitoring tools — let the LLM set up file/endpoint/repo watchers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.core.monitors import MonitorManager

_global_monitor_manager: MonitorManager | None = None


def set_monitor_manager(manager: MonitorManager) -> None:
    """Set the global MonitorManager instance (called during startup).

    Args:
        manager: The initialized MonitorManager.
    """
    global _global_monitor_manager
    _global_monitor_manager = manager


def get_monitor_manager() -> MonitorManager:
    """Get the global MonitorManager instance.

    Returns:
        The shared MonitorManager.

    Raises:
        RuntimeError: If set_monitor_manager() hasn't been called yet.
    """
    if _global_monitor_manager is None:
        raise RuntimeError(
            "MonitorManager not initialized. Restart the agent."
        )
    return _global_monitor_manager


@tool(
    name="set_monitor",
    description=(
        "Set up a monitor to watch for changes. Supported types: "
        "'file_change' (watch a file path), "
        "'http_endpoint' (watch a URL for status changes), "
        "'git_repo' (watch for new commits), "
        "'shell_command' (watch command output for changes). "
        "You'll be notified when a change is detected."
    ),
    tier=ToolTier.MODERATE,
)
async def set_monitor(
    type: str,
    target: str,
    interval: int = 60,
    description: str = "",
    channel: str = "",
    user_id: str = "",
) -> str:
    """Create a new monitor.

    Args:
        type: Monitor type: file_change, http_endpoint, git_repo, shell_command.
        target: What to monitor (file path, URL, git repo path, shell command).
        interval: Check interval in seconds (minimum 10).
        description: Human description of what's being monitored.
        channel: Channel to notify on (optional).
        user_id: User to notify (optional).

    Returns:
        Confirmation with monitor ID.
    """
    from agent.core.monitors import MonitorType

    manager = get_monitor_manager()

    try:
        monitor_type = MonitorType(type)
    except ValueError:
        valid = ", ".join(t.value for t in MonitorType)
        return f"Invalid monitor type: '{type}'. Valid types: {valid}"

    monitor = await manager.add_monitor(
        type=monitor_type,
        target=target,
        interval=interval,
        channel=channel or None,
        user_id=user_id or None,
        description=description,
    )

    return (
        f"Monitor created (ID: {monitor.id})\n"
        f"Type: {monitor.type}\n"
        f"Target: {monitor.target}\n"
        f"Check interval: {monitor.interval}s\n"
        f"Description: {monitor.description}"
    )


@tool(
    name="list_monitors",
    description="List all active monitors that are watching for changes.",
    tier=ToolTier.SAFE,
)
async def list_monitors_tool() -> str:
    """List active monitors.

    Returns:
        Formatted list of monitors.
    """
    manager = get_monitor_manager()
    monitors = manager.list_monitors()

    if not monitors:
        return "No active monitors."

    lines = [f"Active Monitors ({len(monitors)}):"]
    for m in monitors:
        last = m.get("last_check", "never")
        lines.append(
            f"  [{m['id']}] {m['type']}: {m['target']} "
            f"(every {m['interval']}s, last: {last})"
        )
        if m.get("description"):
            lines.append(f"    {m['description']}")

    return "\n".join(lines)


@tool(
    name="remove_monitor",
    description="Remove an active monitor by its ID.",
    tier=ToolTier.SAFE,
)
async def remove_monitor_tool(monitor_id: str) -> str:
    """Remove a monitor.

    Args:
        monitor_id: The monitor ID to remove.

    Returns:
        Status message.
    """
    manager = get_monitor_manager()
    removed = manager.remove_monitor(monitor_id)

    if removed:
        return f"Monitor {monitor_id} removed."
    return f"Monitor {monitor_id} not found."
