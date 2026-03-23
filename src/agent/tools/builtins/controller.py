"""Controller tools — let the main agent communicate with the Controller agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.core.controller import ControllerAgent

_global_controller: ControllerAgent | None = None


def set_controller(controller: ControllerAgent) -> None:
    """Set the global ControllerAgent instance (called during startup)."""
    global _global_controller
    _global_controller = controller


def get_controller() -> ControllerAgent:
    """Get the global ControllerAgent instance.

    Returns:
        The shared ControllerAgent.

    Raises:
        RuntimeError: If set_controller() hasn't been called yet.
    """
    if _global_controller is None:
        raise RuntimeError(
            "Controller not initialized. "
            "Enable use_controller in orchestration config and restart."
        )
    return _global_controller


@tool(
    name="assign_work",
    description=(
        "Send a work order to the Controller agent. The controller will "
        "plan, decompose, and delegate the work to specialist sub-agents. "
        "Returns immediately with an order_id for tracking."
    ),
    tier=ToolTier.MODERATE,
)
async def assign_work_tool(
    instruction: str,
    context: str = "",
    priority: int = 0,
) -> str:
    """Send a work order to the controller.

    Args:
        instruction: What needs to be done.
        context: Additional context for the work.
        priority: Priority level (0 = normal).

    Returns:
        Confirmation with order_id.
    """
    from agent.core.subagent import ControllerWorkOrder

    controller = get_controller()

    # Get user_id from context var
    user_id = ""
    try:
        from agent.tools.builtins.scheduler import _user_id_var

        user_id = _user_id_var.get("") or ""
    except Exception:
        pass

    order = ControllerWorkOrder(
        instruction=instruction,
        context=context,
        priority=priority,
        user_id=user_id,
    )
    try:
        await controller.submit_order(order)
    except Exception as e:
        return f"Failed to submit work order: {e}"

    # Emit task accepted event for channel notifications (best-effort)
    from agent.core.events import Events

    try:
        if controller.event_bus is not None:
            await controller.event_bus.emit(
                Events.TASK_ACCEPTED,
                {
                    "task_id": order.order_id,
                    "user_id": user_id,
                    "summary": instruction[:2000],
                },
            )
    except Exception:
        pass  # notification is non-critical

    return f"\u26a1 Task accepted (ID: {order.order_id}). " f"I'll notify you when done."


@tool(
    name="check_work_status",
    description=(
        "Check the status of controller work orders. "
        "Pass an order_id for a specific task, or leave empty for all active tasks."
    ),
    tier=ToolTier.SAFE,
)
async def check_work_status_tool(order_id: str = "") -> str:
    """Check status of work orders.

    Args:
        order_id: Specific order to check (empty = all active).

    Returns:
        Status information.
    """
    controller = get_controller()

    if order_id:
        return controller.get_task_summary(order_id)
    return controller.get_all_tasks_summary()


@tool(
    name="direct_controller",
    description=("Send a directive to the controller: stop, pause, or redirect a work order."),
    tier=ToolTier.MODERATE,
)
async def direct_controller_tool(
    order_id: str,
    command: str,
    details: str = "",
) -> str:
    """Send a directive to the controller.

    Args:
        order_id: The work order to target.
        command: Directive command (stop, pause, redirect).
        details: Additional details for the directive.

    Returns:
        Confirmation message.
    """
    from agent.core.subagent import ControllerDirective

    controller = get_controller()

    directive = ControllerDirective(
        order_id=order_id,
        command=command,
        details=details,
    )
    await controller.submit_directive(directive)
    return f"Directive '{command}' sent to controller for order '{order_id}'."
