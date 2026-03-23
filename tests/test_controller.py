"""Tests for the Controller agent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.controller import ControllerAgent
from agent.core.events import EventBus, Events
from agent.core.subagent import (
    ControllerDirective,
    ControllerTaskState,
    ControllerWorkOrder,
)


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.use_controller = True
    config.controller_model = None
    config.controller_max_turns = 30
    config.default_max_iterations = 5
    return config


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock()
    orch.spawn_subagent = AsyncMock()
    orch.cancel = AsyncMock(return_value=True)
    orch.tool_registry = MagicMock()
    orch.tool_registry.list_tools.return_value = []
    return orch


@pytest.fixture
def controller(mock_orchestrator, event_bus, mock_config):
    return ControllerAgent(
        orchestrator=mock_orchestrator,
        sdk_service=None,
        event_bus=event_bus,
        config=mock_config,
    )


class TestControllerWorkOrder:
    """Test ControllerWorkOrder dataclass."""

    def test_defaults(self):
        order = ControllerWorkOrder()
        assert order.order_id.startswith("wo-")
        assert order.instruction == ""
        assert order.priority == 0

    def test_custom_values(self):
        order = ControllerWorkOrder(
            instruction="Build a website",
            context="Use React",
            priority=1,
            user_id="123",
        )
        assert order.instruction == "Build a website"
        assert order.context == "Use React"
        assert order.priority == 1
        assert order.user_id == "123"


class TestControllerDirective:
    """Test ControllerDirective dataclass."""

    def test_defaults(self):
        directive = ControllerDirective()
        assert directive.order_id == ""
        assert directive.command == ""

    def test_stop_command(self):
        directive = ControllerDirective(
            order_id="wo-abc123",
            command="stop",
        )
        assert directive.command == "stop"


class TestControllerTaskState:
    """Test ControllerTaskState dataclass."""

    def test_defaults(self):
        state = ControllerTaskState()
        assert state.status == "pending"
        assert state.worker_task_ids == []
        assert state.summary == ""
        assert state.error == ""


class TestControllerAgent:
    """Test ControllerAgent lifecycle and operations."""

    async def test_start_stop(self, controller):
        await controller.start()
        assert controller._running is True
        assert controller._loop_task is not None

        await controller.stop()
        assert controller._running is False

    async def test_submit_order(self, controller):
        order = ControllerWorkOrder(instruction="Do something")
        await controller.submit_order(order)
        assert not controller._task_queue.empty()

    async def test_submit_directive(self, controller):
        directive = ControllerDirective(order_id="wo-123", command="stop")
        await controller.submit_directive(directive)
        assert not controller._task_queue.empty()

    async def test_get_task_summary_not_found(self, controller):
        result = controller.get_task_summary("nonexistent")
        assert "No task found" in result

    async def test_get_task_summary_found(self, controller):
        state = ControllerTaskState(
            order_id="wo-123",
            status="executing",
            summary="Working on it",
        )
        controller._active_tasks["wo-123"] = state
        result = controller.get_task_summary("wo-123")
        assert "wo-123" in result
        assert "executing" in result
        assert "Working on it" in result

    async def test_get_all_tasks_summary_empty(self, controller):
        result = controller.get_all_tasks_summary()
        assert "No active tasks" in result

    async def test_get_all_tasks_summary(self, controller):
        controller._active_tasks["wo-1"] = ControllerTaskState(
            order_id="wo-1",
            status="executing",
        )
        controller._active_tasks["wo-2"] = ControllerTaskState(
            order_id="wo-2",
            status="completed",
            summary="Done",
        )
        result = controller.get_all_tasks_summary()
        assert "wo-1" in result
        assert "wo-2" in result
        assert "Active Tasks (2)" in result

    async def test_handle_order_fallback(self, controller, mock_orchestrator):
        """Test order handling via fallback (no SDK)."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-123",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Task completed successfully",
        )

        events_received = []

        async def capture_event(data):
            events_received.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_STARTED, capture_event)
        controller.event_bus.on(Events.CONTROLLER_TASK_COMPLETED, capture_event)

        order = ControllerWorkOrder(
            instruction="Build something",
            user_id="user-1",
        )
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "completed"
        assert "Task completed" in state.summary
        assert len(events_received) == 2  # started + completed

    async def test_handle_order_failure(self, controller, mock_orchestrator):
        """Test order handling when worker fails."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-456",
            role_name="controller-worker",
            status=SubAgentStatus.FAILED,
            error="Something went wrong",
        )

        events_received = []

        async def capture_event(data):
            events_received.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_FAILED, capture_event)

        order = ControllerWorkOrder(instruction="Fail task")
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "failed"
        assert "Something went wrong" in state.error

    async def test_handle_directive_stop(self, controller, mock_orchestrator):
        """Test stop directive cancels worker tasks."""
        # Add an active task with a worker
        state = ControllerTaskState(
            order_id="wo-123",
            status="executing",
            worker_task_ids=["sa-worker"],
        )
        controller._active_tasks["wo-123"] = state

        # Create a mock worker task (use MagicMock, not AsyncMock,
        # since asyncio.Task.done() is a regular method)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        controller._worker_tasks["wo-123"] = mock_task

        directive = ControllerDirective(
            order_id="wo-123",
            command="stop",
        )
        await controller._handle_directive(directive)

        # The worker task should have been cancelled
        mock_task.cancel.assert_called_once()


class TestControllerTools:
    """Test controller tool functions."""

    async def test_assign_work_tool(self):
        from agent.tools.builtins.controller import (
            assign_work_tool,
            set_controller,
        )

        mock_controller = MagicMock()
        mock_controller.submit_order = AsyncMock()
        set_controller(mock_controller)

        result = await assign_work_tool(
            instruction="Build a website",
            context="Use React",
        )

        assert "Task accepted" in result
        mock_controller.submit_order.assert_called_once()

        # Cleanup
        set_controller(None)  # type: ignore[arg-type]

    async def test_check_work_status_tool(self):
        from agent.tools.builtins.controller import (
            check_work_status_tool,
            set_controller,
        )

        mock_controller = MagicMock()
        mock_controller.get_all_tasks_summary.return_value = "No active tasks."
        mock_controller.get_task_summary.return_value = "Order wo-123: executing"
        set_controller(mock_controller)

        # All tasks
        result = await check_work_status_tool()
        assert "No active tasks" in result

        # Specific task
        result = await check_work_status_tool(order_id="wo-123")
        assert "wo-123" in result

        set_controller(None)  # type: ignore[arg-type]

    async def test_direct_controller_tool(self):
        from agent.tools.builtins.controller import (
            direct_controller_tool,
            set_controller,
        )

        mock_controller = MagicMock()
        mock_controller.submit_directive = AsyncMock()
        set_controller(mock_controller)

        result = await direct_controller_tool(
            order_id="wo-123",
            command="stop",
        )

        assert "Directive 'stop' sent" in result
        mock_controller.submit_directive.assert_called_once()

        set_controller(None)  # type: ignore[arg-type]

    async def test_get_controller_not_initialized(self):
        from agent.tools.builtins.controller import get_controller, set_controller

        set_controller(None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="Controller not initialized"):
            get_controller()


class TestControllerEdgeCases:
    """Edge case tests for controller behavior."""

    async def test_directive_on_nonexistent_task(self, controller):
        """Directive targeting an unknown order_id should be silently ignored."""
        directive = ControllerDirective(order_id="wo-ghost", command="stop")
        # Should not raise
        await controller._handle_directive(directive)
        # No task should be created
        assert "wo-ghost" not in controller._active_tasks

    async def test_directive_unknown_command(self, controller):
        """Unknown directive command should log warning but not crash."""
        state = ControllerTaskState(order_id="wo-123", status="executing")
        controller._active_tasks["wo-123"] = state

        directive = ControllerDirective(order_id="wo-123", command="explode")
        await controller._handle_directive(directive)
        # Status should remain unchanged
        assert state.status == "executing"

    async def test_stop_directive_already_done_worker(self, controller):
        """Stop directive when worker task already finished should still update state."""
        state = ControllerTaskState(order_id="wo-123", status="executing")
        controller._active_tasks["wo-123"] = state

        mock_task = MagicMock()
        mock_task.done.return_value = True  # Already finished
        controller._worker_tasks["wo-123"] = mock_task

        directive = ControllerDirective(order_id="wo-123", command="stop")
        await controller._handle_directive(directive)

        # cancel should NOT be called on a finished task
        mock_task.cancel.assert_not_called()
        # But state should still be updated
        assert state.status == "cancelled"

    async def test_stop_directive_no_worker_task(self, controller):
        """Stop directive when no worker task exists (already cleaned up)."""
        state = ControllerTaskState(order_id="wo-123", status="executing")
        controller._active_tasks["wo-123"] = state
        # No entry in _worker_tasks

        directive = ControllerDirective(order_id="wo-123", command="stop")
        await controller._handle_directive(directive)
        assert state.status == "cancelled"

    async def test_order_cancelled_via_cancellation(self, controller):
        """Order that gets cancelled via CancelledError."""
        events_received = []

        async def capture_event(data):
            events_received.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_CANCELLED, capture_event)

        # Make _execute_order raise CancelledError
        async def mock_execute(*args):
            raise asyncio.CancelledError()

        controller._execute_order = mock_execute  # type: ignore[assignment]

        order = ControllerWorkOrder(instruction="Cancel me", user_id="u1")
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "cancelled"
        assert len(events_received) == 1
        assert events_received[0]["order_id"] == order.order_id

    async def test_order_unexpected_exception(self, controller):
        """Order that fails with unexpected exception."""
        events_received = []

        async def capture_event(data):
            events_received.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_FAILED, capture_event)

        async def mock_execute(*args):
            raise ValueError("Unexpected boom")

        controller._execute_order = mock_execute  # type: ignore[assignment]

        order = ControllerWorkOrder(instruction="Boom", user_id="u1")
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "failed"
        assert "Unexpected boom" in state.error
        assert len(events_received) == 1

    async def test_execute_order_with_context(self, controller, mock_orchestrator):
        """Verify context is passed through to the worker."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-ctx",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Done with context",
        )

        order = ControllerWorkOrder(
            instruction="Do work",
            context="Important context here",
        )
        await controller._handle_order(order)

        # Verify spawn_subagent was called with the context
        call_args = mock_orchestrator.spawn_subagent.call_args
        task = call_args[0][0]  # First positional arg
        assert task.context == "Important context here"

    async def test_execute_via_sdk_path(self, event_bus, mock_config):
        """Test the SDK execution path."""
        mock_orch = MagicMock()
        mock_orch.tool_registry = MagicMock()
        mock_orch.tool_registry.list_tools.return_value = []

        mock_sdk = MagicMock()
        mock_sdk.run_subagent = AsyncMock(return_value="SDK result text")

        ctrl = ControllerAgent(
            orchestrator=mock_orch,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            config=mock_config,
        )

        order = ControllerWorkOrder(instruction="SDK task", user_id="u1")
        await ctrl._handle_order(order)

        state = ctrl._active_tasks[order.order_id]
        assert state.status == "completed"
        assert "SDK result" in state.summary
        mock_sdk.run_subagent.assert_called_once()

    async def test_execute_via_sdk_failure(self, event_bus, mock_config):
        """Test SDK execution path when SDK raises."""
        mock_orch = MagicMock()
        mock_orch.tool_registry = MagicMock()
        mock_orch.tool_registry.list_tools.return_value = []

        mock_sdk = MagicMock()
        mock_sdk.run_subagent = AsyncMock(side_effect=RuntimeError("SDK broke"))

        ctrl = ControllerAgent(
            orchestrator=mock_orch,
            sdk_service=mock_sdk,
            event_bus=event_bus,
            config=mock_config,
        )

        order = ControllerWorkOrder(instruction="SDK fail", user_id="u1")
        await ctrl._handle_order(order)

        state = ctrl._active_tasks[order.order_id]
        assert state.status == "failed"
        assert "SDK broke" in state.error

    async def test_concurrent_orders(self, controller, mock_orchestrator):
        """Multiple concurrent orders should each get independent state."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        call_count = 0

        async def delayed_spawn(task):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return SubAgentResult(
                task_id=f"sa-{call_count}",
                role_name="controller-worker",
                status=SubAgentStatus.COMPLETED,
                output=f"Result {call_count}",
            )

        mock_orchestrator.spawn_subagent.side_effect = delayed_spawn

        order1 = ControllerWorkOrder(instruction="Task 1")
        order2 = ControllerWorkOrder(instruction="Task 2")

        await asyncio.gather(
            controller._handle_order(order1),
            controller._handle_order(order2),
        )

        assert controller._active_tasks[order1.order_id].status == "completed"
        assert controller._active_tasks[order2.order_id].status == "completed"
        assert call_count == 2

    async def test_run_loop_processes_mixed_items(self, controller, mock_orchestrator):
        """Run loop should process both orders and directives from queue."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-loop",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Loop result",
        )

        await controller.start()

        # Submit an order
        order = ControllerWorkOrder(instruction="Loop task")
        await controller.submit_order(order)

        # Give the loop time to process
        await asyncio.sleep(0.2)

        assert order.order_id in controller._active_tasks
        assert controller._active_tasks[order.order_id].status == "completed"

        await controller.stop()

    async def test_stop_with_running_workers(self, controller):
        """Stop should cancel all active worker tasks."""
        task = asyncio.create_task(asyncio.sleep(100))
        controller._worker_tasks["wo-long"] = task

        await controller.stop()

        assert task.cancelled() or task.done()

    async def test_stop_idempotent(self, controller):
        """Calling stop() multiple times should not raise."""
        await controller.start()
        await controller.stop()
        # Second stop should be safe
        await controller.stop()

    async def test_task_summary_with_error_and_workers(self, controller):
        """Task summary should include all fields when populated."""
        state = ControllerTaskState(
            order_id="wo-full",
            status="failed",
            worker_task_ids=["sa-1", "sa-2"],
            summary="Attempted work",
            error="Worker crashed",
        )
        controller._active_tasks["wo-full"] = state
        result = controller.get_task_summary("wo-full")

        assert "wo-full" in result
        assert "failed" in result
        assert "sa-1, sa-2" in result
        assert "Attempted work" in result
        assert "Worker crashed" in result

    async def test_all_tasks_summary_status_icons(self, controller):
        """Verify all status types get appropriate icons."""
        statuses = ["pending", "planning", "executing", "completed", "failed", "cancelled"]
        for i, status in enumerate(statuses):
            controller._active_tasks[f"wo-{i}"] = ControllerTaskState(
                order_id=f"wo-{i}",
                status=status,
            )

        result = controller.get_all_tasks_summary()
        assert f"Active Tasks ({len(statuses)})" in result
        for i in range(len(statuses)):
            assert f"wo-{i}" in result

    async def test_order_empty_result_text(self, controller, mock_orchestrator):
        """Handle order where worker returns empty output."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-empty",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="",  # empty output
        )

        order = ControllerWorkOrder(instruction="Empty result task")
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "completed"
        assert state.summary == "Completed"  # fallback when empty

    async def test_order_long_result_preserved(self, controller, mock_orchestrator):
        """Verify full result text is preserved in summary (no truncation)."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        long_output = "x" * 1000
        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-long",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output=long_output,
        )

        order = ControllerWorkOrder(instruction="Long result task")
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert len(state.summary) == 1000

    async def test_unique_order_ids(self):
        """Each work order should get a unique ID."""
        orders = [ControllerWorkOrder() for _ in range(100)]
        ids = {o.order_id for o in orders}
        assert len(ids) == 100


class TestControllerBugFixes:
    """Tests for specific bug fixes."""

    async def test_fix1_prune_finished_tasks(self, controller, mock_orchestrator):
        """FIX #1: _active_tasks should be pruned after exceeding threshold."""
        from agent.core.controller import _MAX_FINISHED_TASKS
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-prune",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Done",
        )

        # Create more than _MAX_FINISHED_TASKS completed orders
        for i in range(_MAX_FINISHED_TASKS + 50):
            order = ControllerWorkOrder(instruction=f"Task {i}")
            await controller._handle_order(order)

        # Should have pruned oldest entries
        finished_count = sum(
            1
            for s in controller._active_tasks.values()
            if s.status in {"completed", "failed", "cancelled"}
        )
        assert finished_count <= _MAX_FINISHED_TASKS

    async def test_fix2_cancelled_error_emit_protected(self, controller):
        """FIX #2: CancelledError in emit during shutdown should not propagate."""
        # Make event_bus.emit raise CancelledError
        original_emit = controller.event_bus.emit

        call_count = 0

        async def exploding_emit(event, data=None):
            nonlocal call_count
            call_count += 1
            if event == Events.CONTROLLER_TASK_CANCELLED:
                raise asyncio.CancelledError()
            return await original_emit(event, data)

        controller.event_bus.emit = exploding_emit  # type: ignore[assignment]

        async def mock_execute(*args):
            raise asyncio.CancelledError()

        controller._execute_order = mock_execute  # type: ignore[assignment]

        order = ControllerWorkOrder(instruction="Shutdown cancel", user_id="u1")
        # Should NOT raise — the CancelledError from emit should be suppressed
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "cancelled"

    async def test_fix3_no_double_cancelled_emission(self, controller):
        """FIX #3: Directive stop + CancelledError should emit CANCELLED only once.

        Simulates the real race: _handle_order creates the state, then
        a directive sets it to "cancelled" before the CancelledError fires.
        """
        cancelled_events = []

        async def capture_cancelled(data):
            cancelled_events.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_CANCELLED, capture_cancelled)

        order = ControllerWorkOrder(instruction="Race condition", user_id="u1")

        async def mock_execute(ord, state):
            # Simulate directive arriving mid-execution and marking cancelled
            state.status = "cancelled"
            raise asyncio.CancelledError()

        controller._execute_order = mock_execute  # type: ignore[assignment]
        await controller._handle_order(order)

        # The CancelledError handler should see status is already terminal
        # and skip the emit — only the directive's emit should fire (0 here
        # since we set status directly without emitting).
        assert len(cancelled_events) == 0

    async def test_fix3_directive_skip_already_terminal(self, controller):
        """FIX #3: Directive on already-completed task should not re-cancel."""
        cancelled_events = []

        async def capture_cancelled(data):
            cancelled_events.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_CANCELLED, capture_cancelled)

        state = ControllerTaskState(order_id="wo-done", status="completed")
        controller._active_tasks["wo-done"] = state

        directive = ControllerDirective(order_id="wo-done", command="stop")
        await controller._handle_directive(directive)

        # Status should remain completed, not be overwritten to cancelled
        assert state.status == "completed"
        assert len(cancelled_events) == 0

    async def test_fix4_directive_exception_doesnt_kill_loop(self, controller):
        """FIX #4: Exception in _handle_directive should not kill _run_loop."""

        # Make _handle_directive raise an exception

        async def exploding_directive(directive):
            raise ValueError("Bad directive processing")

        controller._handle_directive = exploding_directive  # type: ignore[assignment]

        await controller.start()

        # Submit a directive that will explode
        directive = ControllerDirective(order_id="wo-bad", command="stop")
        await controller.submit_directive(directive)

        # Give the loop time to process the error
        await asyncio.sleep(0.1)

        # The loop should still be running
        assert controller._running is True
        assert controller._loop_task is not None
        assert not controller._loop_task.done()

        await controller.stop()

    async def test_fix5_shutdown_stops_loop_before_workers(self, controller):
        """FIX #5: Shutdown should stop loop before cancelling workers."""
        await controller.start()

        # Create a real worker task
        worker = asyncio.create_task(asyncio.sleep(100))
        controller._worker_tasks["wo-order"] = worker

        await controller.stop()

        # Loop should have been cancelled first
        assert controller._loop_task is not None
        assert controller._loop_task.done()
        # Then workers
        assert worker.cancelled() or worker.done()

    async def test_fix6_progress_event_includes_user_id(
        self,
        controller,
        mock_orchestrator,
    ):
        """FIX #6: CONTROLLER_TASK_PROGRESS events must include user_id."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-prog",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Progress done",
        )

        progress_events = []

        async def capture_progress(data):
            progress_events.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_PROGRESS, capture_progress)

        order = ControllerWorkOrder(
            instruction="Track progress",
            user_id="user-42",
        )
        await controller._handle_order(order)

        assert len(progress_events) == 1
        assert progress_events[0]["user_id"] == "user-42"

    async def test_fix7_strenum_comparison_no_dot_value(
        self,
        controller,
        mock_orchestrator,
    ):
        """FIX #7: StrEnum comparison works without .value."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        # Verify that completed status works with direct comparison
        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-enum",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Enum test",
        )

        order = ControllerWorkOrder(instruction="Enum check")
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "completed"
        assert "Enum test" in state.summary

    async def test_fix8_cancelled_emit_doesnt_swallow_keyboard_interrupt(
        self,
        controller,
    ):
        """FIX #8: suppress(CancelledError) should NOT swallow KeyboardInterrupt."""
        original_emit = controller.event_bus.emit

        async def keyboard_interrupt_emit(event, data=None):
            if event == Events.CONTROLLER_TASK_CANCELLED:
                raise KeyboardInterrupt()
            return await original_emit(event, data)

        controller.event_bus.emit = keyboard_interrupt_emit  # type: ignore[assignment]

        async def mock_execute(*args):
            raise asyncio.CancelledError()

        controller._execute_order = mock_execute  # type: ignore[assignment]

        order = ControllerWorkOrder(instruction="KBI test", user_id="u1")

        # KeyboardInterrupt should NOT be suppressed
        with pytest.raises(KeyboardInterrupt):
            await controller._handle_order(order)

    async def test_fix9_task_state_stores_user_id(
        self,
        controller,
        mock_orchestrator,
    ):
        """FIX #9: ControllerTaskState should store user_id from the work order."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-uid",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="User ID test",
        )

        order = ControllerWorkOrder(instruction="UID check", user_id="user-99")
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.user_id == "user-99"

    async def test_fix9_directive_cancel_uses_state_user_id(
        self,
        controller,
    ):
        """FIX #9: Directive handler should use state.user_id, not getattr."""
        cancelled_events = []

        async def capture(data):
            cancelled_events.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_CANCELLED, capture)

        state = ControllerTaskState(
            order_id="wo-uid",
            status="executing",
            user_id="user-77",
        )
        controller._active_tasks["wo-uid"] = state

        directive = ControllerDirective(order_id="wo-uid", command="stop")
        await controller._handle_directive(directive)

        assert len(cancelled_events) == 1
        assert cancelled_events[0]["user_id"] == "user-77"

    async def test_fix10_start_idempotent(self, controller):
        """FIX #10: Calling start() twice should not create orphaned loop tasks."""
        await controller.start()
        first_loop = controller._loop_task

        await controller.start()  # Second call — should be a no-op
        second_loop = controller._loop_task

        assert first_loop is second_loop
        await controller.stop()

    async def test_fix11_completion_race_with_directive(
        self,
        controller,
        mock_orchestrator,
    ):
        """FIX #11: If directive cancels during _execute_order, completion should not overwrite."""

        completed_events = []
        cancelled_events = []

        async def capture_completed(data):
            completed_events.append(data)

        async def capture_cancelled(data):
            cancelled_events.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_COMPLETED, capture_completed)
        controller.event_bus.on(Events.CONTROLLER_TASK_CANCELLED, capture_cancelled)

        order = ControllerWorkOrder(instruction="Race test", user_id="u1")

        async def mock_execute(ord, state):
            # Simulate a directive arriving and cancelling during execution
            state.status = "cancelled"
            return "I finished anyway"

        controller._execute_order = mock_execute  # type: ignore[assignment]
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        # Status should remain "cancelled", not be overwritten to "completed"
        assert state.status == "cancelled"
        assert len(completed_events) == 0

    async def test_fix12_failed_emit_protected_from_cancelled_error(self, controller):
        """FIX #12: CONTROLLER_TASK_FAILED emit should be protected from CancelledError."""
        original_emit = controller.event_bus.emit

        async def exploding_emit(event, data=None):
            if event == Events.CONTROLLER_TASK_FAILED:
                raise asyncio.CancelledError()
            return await original_emit(event, data)

        controller.event_bus.emit = exploding_emit  # type: ignore[assignment]

        async def mock_execute(*args):
            raise ValueError("Something broke")

        controller._execute_order = mock_execute  # type: ignore[assignment]

        order = ControllerWorkOrder(instruction="Fail emit test", user_id="u1")
        # Should NOT raise — the CancelledError from emit should be suppressed
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        assert state.status == "failed"

    async def test_fix13_exception_handler_respects_terminal_status(self, controller):
        """FIX #13: except Exception should not overwrite cancelled status to failed."""
        failed_events = []

        async def capture_failed(data):
            failed_events.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_FAILED, capture_failed)

        order = ControllerWorkOrder(instruction="Error after cancel", user_id="u1")

        async def mock_execute(ord, state):
            # Simulate directive setting cancelled, then an exception fires
            state.status = "cancelled"
            raise RuntimeError("Something broke after cancellation")

        controller._execute_order = mock_execute  # type: ignore[assignment]
        await controller._handle_order(order)

        state = controller._active_tasks[order.order_id]
        # Status should remain "cancelled", NOT be overwritten to "failed"
        assert state.status == "cancelled"
        assert len(failed_events) == 0

    async def test_fix14_scoped_registry_hides_orchestration_tools(self):
        """FIX #14: Main agent should use ScopedToolRegistry, not unregister tools."""
        from agent.core.orchestrator import ScopedToolRegistry
        from agent.tools.registry import ToolRegistry

        # Simulate the startup pattern: create registry with orchestration tools
        parent = ToolRegistry()

        @parent.tool(name="spawn_subagent", description="Spawn a sub-agent")
        async def _spawn(instruction: str) -> str:
            return ""

        @parent.tool(name="assign_work", description="Assign work to controller")
        async def _assign(instruction: str) -> str:
            return ""

        # Controller should see spawn_subagent but NOT assign_work
        controller_excluded = {"assign_work", "check_work_status", "direct_controller"}
        controller_scoped = ScopedToolRegistry(
            parent=parent,
            denied_tools=controller_excluded,
        )

        # Main agent should see assign_work but NOT spawn_subagent
        orchestration_excluded = {"spawn_subagent"}
        main_scoped = ScopedToolRegistry(
            parent=parent,
            denied_tools=orchestration_excluded,
        )

        controller_schemas = controller_scoped.get_tool_schemas(enabled_only=True)
        main_schemas = main_scoped.get_tool_schemas(enabled_only=True)

        controller_names = {s["function"]["name"] for s in controller_schemas}
        main_names = {s["function"]["name"] for s in main_schemas}

        assert "spawn_subagent" in controller_names
        assert "assign_work" not in controller_names
        assert "assign_work" in main_names
        assert "spawn_subagent" not in main_names

    async def test_fix15_scoped_registry_preserves_dangerous_tools(self):
        """FIX #15: ScopedToolRegistry for main agent must not hide dangerous tools."""
        from agent.core.orchestrator import ScopedToolRegistry
        from agent.tools.registry import ToolRegistry, ToolTier

        parent = ToolRegistry()

        @parent.tool(
            name="shell_exec",
            description="Run shell commands",
            tier=ToolTier.DANGEROUS,
        )
        async def _shell(command: str) -> str:
            return ""

        @parent.tool(name="spawn_subagent", description="Spawn a sub-agent")
        async def _spawn(instruction: str) -> str:
            return ""

        # This mirrors the startup.py fix: exclude_dangerous=False
        main_scoped = ScopedToolRegistry(
            parent=parent,
            denied_tools={"spawn_subagent"},
            exclude_dangerous=False,
        )

        schemas = main_scoped.get_tool_schemas(enabled_only=True)
        names = {s["function"]["name"] for s in schemas}

        # Dangerous tools should still be visible
        assert "shell_exec" in names
        # Orchestration tools should be hidden
        assert "spawn_subagent" not in names


class TestControllerPromptEdgeCases:
    """Edge cases for prompt generation."""

    def test_no_orchestration_no_controller(self):
        """Neither orchestration nor controller should produce no delegation section."""
        from agent.config import AgentPersonaConfig
        from agent.llm.prompts import build_system_prompt

        prompt = build_system_prompt(
            AgentPersonaConfig(name="Agent"),
            orchestration_enabled=False,
            use_controller=False,
        )
        assert "assign_work" not in prompt
        assert "spawn_subagent" not in prompt
        assert "Orchestration" not in prompt
        assert "Work Delegation" not in prompt

    def test_controller_without_orchestration(self):
        """use_controller=True but orchestration_enabled=False should produce no section."""
        from agent.config import AgentPersonaConfig
        from agent.llm.prompts import build_system_prompt

        prompt = build_system_prompt(
            AgentPersonaConfig(name="Agent"),
            orchestration_enabled=False,
            use_controller=True,
        )
        # Controller mode requires orchestration_enabled=True
        assert "assign_work" not in prompt
        assert "spawn_subagent" not in prompt


class TestControllerToolEdgeCases:
    """Edge cases for controller tools."""

    async def test_assign_work_with_priority(self):
        """Verify priority is passed through."""
        from agent.tools.builtins.controller import assign_work_tool, set_controller

        mock_ctrl = MagicMock()
        mock_ctrl.submit_order = AsyncMock()
        set_controller(mock_ctrl)

        await assign_work_tool(instruction="Urgent", priority=5)

        call_args = mock_ctrl.submit_order.call_args
        order = call_args[0][0]
        assert order.priority == 5

        set_controller(None)  # type: ignore[arg-type]

    async def test_direct_controller_with_details(self):
        """Verify details are passed through to directive."""
        from agent.tools.builtins.controller import (
            direct_controller_tool,
            set_controller,
        )

        mock_ctrl = MagicMock()
        mock_ctrl.submit_directive = AsyncMock()
        set_controller(mock_ctrl)

        await direct_controller_tool(
            order_id="wo-123",
            command="redirect",
            details="Change to use Python instead",
        )

        call_args = mock_ctrl.submit_directive.call_args
        directive = call_args[0][0]
        assert directive.command == "redirect"
        assert directive.details == "Change to use Python instead"

        set_controller(None)  # type: ignore[arg-type]


class TestControllerEvents:
    """Test controller events in Events class."""

    def test_controller_events_exist(self):
        assert Events.CONTROLLER_TASK_STARTED == "controller.task.started"
        assert Events.CONTROLLER_TASK_PROGRESS == "controller.task.progress"
        assert Events.CONTROLLER_TASK_COMPLETED == "controller.task.completed"
        assert Events.CONTROLLER_TASK_FAILED == "controller.task.failed"
        assert Events.CONTROLLER_TASK_CANCELLED == "controller.task.cancelled"


class TestTaskNotificationDeduplication:
    """Verify user gets exactly ONE notification per completed/failed task.

    CONTROLLER_TASK_COMPLETED and TASK_COMPLETED_NOTIFY both fire from
    _handle_order, but only TASK_COMPLETED_NOTIFY should have a Telegram
    handler.  This test subscribes a mock "Telegram-like" listener to
    both old and new events and asserts the notification count.
    """

    async def test_completed_task_fires_one_user_notification(
        self,
        controller,
        mock_orchestrator,
    ):
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-dedup-1",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Done",
        )

        # Simulate the Telegram channel's subscription set:
        # - TASK_COMPLETED_NOTIFY is subscribed (new path)
        # - CONTROLLER_TASK_COMPLETED is NOT subscribed (removed)
        notifications: list[dict] = []

        async def on_task_completed_notify(data):
            notifications.append({"event": "task.completed.notify", **data})

        controller.event_bus.on(
            Events.TASK_COMPLETED_NOTIFY,
            on_task_completed_notify,
        )
        # NOTE: no subscription for CONTROLLER_TASK_COMPLETED — that's the point

        order = ControllerWorkOrder(
            instruction="Deduplicate me",
            user_id="user-42",
        )
        await controller._handle_order(order)

        assert len(notifications) == 1
        assert notifications[0]["user_id"] == "user-42"
        assert notifications[0]["task_id"] == order.order_id

    async def test_failed_task_fires_one_user_notification(
        self,
        controller,
        mock_orchestrator,
    ):
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-dedup-2",
            role_name="controller-worker",
            status=SubAgentStatus.FAILED,
            error="boom",
        )

        notifications: list[dict] = []

        async def on_task_failed_notify(data):
            notifications.append({"event": "task.failed.notify", **data})

        controller.event_bus.on(
            Events.TASK_FAILED_NOTIFY,
            on_task_failed_notify,
        )

        order = ControllerWorkOrder(
            instruction="Fail once",
            user_id="user-42",
        )
        await controller._handle_order(order)

        assert len(notifications) == 1
        assert notifications[0]["user_id"] == "user-42"
        assert notifications[0]["task_id"] == order.order_id
        assert "boom" in notifications[0]["error"]

    async def test_old_completed_event_still_fires_for_internal_use(
        self,
        controller,
        mock_orchestrator,
    ):
        """CONTROLLER_TASK_COMPLETED still fires (other components may use it)."""
        from agent.core.subagent import SubAgentResult, SubAgentStatus

        mock_orchestrator.spawn_subagent.return_value = SubAgentResult(
            task_id="sa-dedup-3",
            role_name="controller-worker",
            status=SubAgentStatus.COMPLETED,
            output="Done",
        )

        old_events: list[dict] = []

        async def capture(data):
            old_events.append(data)

        controller.event_bus.on(Events.CONTROLLER_TASK_COMPLETED, capture)

        order = ControllerWorkOrder(instruction="Still fires")
        await controller._handle_order(order)

        assert len(old_events) == 1  # internal event still emitted


class TestControllerPrompts:
    """Test system prompt changes for controller mode."""

    def test_controller_prompt(self):
        from agent.config import AgentPersonaConfig
        from agent.llm.prompts import build_system_prompt

        config = AgentPersonaConfig(name="Agent")
        prompt = build_system_prompt(
            config,
            orchestration_enabled=True,
            use_controller=True,
        )
        assert "assign_work" in prompt
        assert "check_work_status" in prompt
        assert "direct_controller" in prompt
        assert "CEO" in prompt
        # Should NOT contain direct orchestration instructions
        assert "spawn_subagent" not in prompt or "NEVER use spawn_subagent" in prompt

    def test_non_controller_prompt(self):
        from agent.config import AgentPersonaConfig
        from agent.llm.prompts import build_system_prompt

        config = AgentPersonaConfig(name="Agent")
        prompt = build_system_prompt(
            config,
            orchestration_enabled=True,
            use_controller=False,
        )
        assert "spawn_subagent" in prompt
        assert "assign_work" not in prompt


class TestControllerConfig:
    """Test config changes for controller."""

    def test_default_config(self):
        from agent.config import OrchestrationConfig

        config = OrchestrationConfig()
        assert config.use_controller is False
        assert config.controller_model is None
        assert config.controller_max_turns == 200

    def test_custom_config(self):
        from agent.config import OrchestrationConfig

        config = OrchestrationConfig(
            use_controller=True,
            controller_model="claude-sonnet-4-6",
            controller_max_turns=50,
        )
        assert config.use_controller is True
        assert config.controller_model == "claude-sonnet-4-6"
        assert config.controller_max_turns == 50
