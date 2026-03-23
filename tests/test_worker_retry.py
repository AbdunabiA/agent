"""Tests for worker retry resilience in the orchestrator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.core.events import EventBus, Events
from agent.core.subagent import SubAgentResult, SubAgentRole, SubAgentStatus, SubAgentTask


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.max_concurrent_agents = 5
    config.subagent_timeout = 120
    config.default_max_iterations = 5
    config.controller_model = None
    config.controller_max_turns = 30
    return config


def _make_role(name: str = "test-worker") -> SubAgentRole:
    return SubAgentRole(name=name, persona="test", max_iterations=3)


def _make_task(
    role: SubAgentRole | None = None,
    max_attempts: int = 3,
    timeout_seconds: int = 120,
    critical: bool = False,
) -> SubAgentTask:
    return SubAgentTask(
        role=role or _make_role(),
        instruction="do something",
        max_attempts=max_attempts,
        timeout_seconds=timeout_seconds,
        critical=critical,
    )


def _ok_result(task: SubAgentTask) -> SubAgentResult:
    return SubAgentResult(
        task_id=task.task_id,
        role_name=task.role.name,
        status=SubAgentStatus.COMPLETED,
        output="done",
    )


def _fail_result(task: SubAgentTask, error: str = "boom") -> SubAgentResult:
    return SubAgentResult(
        task_id=task.task_id,
        role_name=task.role.name,
        status=SubAgentStatus.FAILED,
        error=error,
    )


@pytest.fixture
def orchestrator(event_bus, mock_config):
    """Build an orchestrator with mocked internals."""
    from agent.core.orchestrator import SubAgentOrchestrator

    agent_loop = MagicMock()
    agent_loop.llm = MagicMock()
    agent_loop.tool_executor = MagicMock()
    tool_registry = MagicMock()

    orch = SubAgentOrchestrator(
        agent_loop=agent_loop,
        config=mock_config,
        event_bus=event_bus,
        tool_registry=tool_registry,
        sdk_service=None,
    )
    return orch


class TestSubAgentTaskDefaults:
    """Test new fields on SubAgentTask."""

    def test_default_timeout(self):
        task = _make_task()
        assert task.timeout_seconds == 120

    def test_default_max_attempts(self):
        task = _make_task()
        assert task.max_attempts == 3

    def test_default_critical(self):
        task = _make_task()
        assert task.critical is False

    def test_custom_values(self):
        task = _make_task(max_attempts=5, timeout_seconds=60, critical=True)
        assert task.max_attempts == 5
        assert task.timeout_seconds == 60
        assert task.critical is True


class TestWorkerRetry:
    """Test _run_worker_with_retry logic."""

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self, orchestrator):
        """Worker that succeeds on first attempt returns immediately."""
        task = _make_task(max_attempts=3)
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=ok)

        result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert orchestrator._single_spawn_attempt.call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self, orchestrator):
        """Worker fails first, succeeds second — only 2 attempts used."""
        task = _make_task(max_attempts=3)
        fail = _fail_result(task, "transient error")
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(
            side_effect=[fail, ok],
        )

        result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert orchestrator._single_spawn_attempt.call_count == 2

    @pytest.mark.asyncio
    async def test_all_attempts_fail(self, orchestrator, event_bus):
        """All 3 attempts fail → WorkerFailed event emitted."""
        task = _make_task(max_attempts=3)
        fail = _fail_result(task, "persistent failure")

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        events_received: list[dict] = []

        async def capture_failed(data):
            events_received.append(data)

        event_bus.on(Events.WORKER_FAILED, capture_failed)

        result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.FAILED
        assert result.error == "persistent failure"
        assert orchestrator._single_spawn_attempt.call_count == 3
        assert len(events_received) == 1
        assert events_received[0]["attempts"] == 3

    @pytest.mark.asyncio
    async def test_retrying_events_emitted(self, orchestrator, event_bus):
        """WorkerRetrying events fire between attempts."""
        task = _make_task(max_attempts=3)
        fail = _fail_result(task, "fail")

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        retry_events: list[dict] = []

        async def capture_retry(data):
            retry_events.append(data)

        event_bus.on(Events.WORKER_RETRYING, capture_retry)

        await orchestrator._run_worker_with_retry(task)

        # 3 attempts → 2 retries (between 1→2 and 2→3)
        assert len(retry_events) == 2
        assert retry_events[0]["attempt"] == 1
        assert retry_events[0]["retry_in_seconds"] == 1  # 2^0
        assert retry_events[1]["attempt"] == 2
        assert retry_events[1]["retry_in_seconds"] == 2  # 2^1

    @pytest.mark.asyncio
    async def test_succeeded_event_emitted(self, orchestrator, event_bus):
        """WorkerSucceeded event fires on success."""
        task = _make_task(max_attempts=3)
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=ok)

        succeeded_events: list[dict] = []

        async def capture_success(data):
            succeeded_events.append(data)

        event_bus.on(Events.WORKER_SUCCEEDED, capture_success)

        await orchestrator._run_worker_with_retry(task)

        assert len(succeeded_events) == 1
        assert succeeded_events[0]["role"] == "test-worker"

    @pytest.mark.asyncio
    async def test_cancelled_not_retried(self, orchestrator):
        """Cancelled workers are not retried."""
        task = _make_task(max_attempts=3)
        cancelled = SubAgentResult(
            task_id=task.task_id,
            role_name=task.role.name,
            status=SubAgentStatus.CANCELLED,
        )

        orchestrator._single_spawn_attempt = AsyncMock(return_value=cancelled)

        result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.CANCELLED
        assert orchestrator._single_spawn_attempt.call_count == 1

    @pytest.mark.asyncio
    async def test_single_attempt_no_retry(self, orchestrator, event_bus):
        """max_attempts=1 means no retry on failure."""
        task = _make_task(max_attempts=1)
        fail = _fail_result(task, "one-shot fail")

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        retry_events: list = []
        event_bus.on(Events.WORKER_RETRYING, lambda d: retry_events.append(d))

        result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.FAILED
        assert orchestrator._single_spawn_attempt.call_count == 1
        # No retry events for single-attempt tasks
        assert len(retry_events) == 0


class TestParallelIsolation:
    """Test that parallel workers don't cancel each other."""

    @pytest.mark.asyncio
    async def test_parallel_one_fails_other_succeeds(self, orchestrator):
        """In parallel execution, one failure doesn't cancel the other."""
        role_a = _make_role("worker-a")
        role_b = _make_role("worker-b")
        task_a = _make_task(role=role_a, max_attempts=1)
        task_b = _make_task(role=role_b, max_attempts=1)

        async def mock_spawn(task):
            if task.role.name == "worker-a":
                await asyncio.sleep(0.01)
                return _fail_result(task, "worker-a failed")
            else:
                await asyncio.sleep(0.01)
                return _ok_result(task)

        orchestrator.spawn_subagent = mock_spawn

        results = await orchestrator.spawn_parallel([task_a, task_b])

        assert len(results) == 2
        # Order matches input
        assert results[0].status == SubAgentStatus.FAILED
        assert results[0].role_name == "worker-a"
        assert results[1].status == SubAgentStatus.COMPLETED
        assert results[1].role_name == "worker-b"

    @pytest.mark.asyncio
    async def test_parallel_both_succeed(self, orchestrator):
        """Both parallel workers succeed."""
        task_a = _make_task(role=_make_role("a"), max_attempts=1)
        task_b = _make_task(role=_make_role("b"), max_attempts=1)

        async def mock_spawn(task):
            return _ok_result(task)

        orchestrator.spawn_subagent = mock_spawn

        results = await orchestrator.spawn_parallel([task_a, task_b])

        assert all(r.status == SubAgentStatus.COMPLETED for r in results)


class TestWorkerRetryEdgeCases:
    """Comprehensive edge-case tests for _run_worker_with_retry."""

    @pytest.mark.asyncio
    async def test_backoff_increases_exponentially(self, orchestrator):
        """Verify backoff delays: 1s, 2s, 4s between attempts."""
        task = _make_task(max_attempts=4)
        fail = _fail_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)
        sleep_calls: list[float] = []

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = lambda d: sleep_calls.append(d)
            await orchestrator._run_worker_with_retry(task)

        assert sleep_calls == [1, 2, 4]  # 2^0, 2^1, 2^2

    @pytest.mark.asyncio
    async def test_no_sleep_after_last_attempt(self, orchestrator):
        """No backoff delay after the final failed attempt."""
        task = _make_task(max_attempts=2)
        fail = _fail_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await orchestrator._run_worker_with_retry(task)

        # 2 attempts → 1 sleep (between attempt 1 and 2), not after attempt 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_completed_after_multiple_failures(self, orchestrator):
        """Succeeds on the Nth attempt after N-1 failures."""
        task = _make_task(max_attempts=5)
        fail = _fail_result(task)
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(
            side_effect=[fail, fail, fail, ok],
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert orchestrator._single_spawn_attempt.call_count == 4

    @pytest.mark.asyncio
    async def test_cancelled_on_second_attempt_stops_retry(self, orchestrator):
        """FAILED then CANCELLED → returns CANCELLED, no further retry."""
        task = _make_task(max_attempts=5)
        fail = _fail_result(task)
        cancelled = SubAgentResult(
            task_id=task.task_id,
            role_name=task.role.name,
            status=SubAgentStatus.CANCELLED,
        )

        orchestrator._single_spawn_attempt = AsyncMock(
            side_effect=[fail, cancelled],
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.CANCELLED
        assert orchestrator._single_spawn_attempt.call_count == 2

    @pytest.mark.asyncio
    async def test_mixed_statuses_correct_retry_events(
        self,
        orchestrator,
        event_bus,
    ):
        """FAILED → FAILED → COMPLETED emits exactly 2 RETRYING events."""
        task = _make_task(max_attempts=5)
        fail = _fail_result(task)
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(
            side_effect=[fail, fail, ok],
        )

        retry_events: list[dict] = []
        event_bus.on(Events.WORKER_RETRYING, lambda d: retry_events.append(d))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_worker_with_retry(task)

        assert len(retry_events) == 2

    @pytest.mark.asyncio
    async def test_max_attempts_one_no_retry_events(
        self,
        orchestrator,
        event_bus,
    ):
        """max_attempts=1 means single attempt, no retry events."""
        task = _make_task(max_attempts=1)
        fail = _fail_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        retry_events: list = []
        event_bus.on(Events.WORKER_RETRYING, lambda d: retry_events.append(d))

        result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.FAILED
        assert orchestrator._single_spawn_attempt.call_count == 1
        assert len(retry_events) == 0

    @pytest.mark.asyncio
    async def test_no_succeeded_event_on_failure(self, orchestrator, event_bus):
        """WORKER_SUCCEEDED never emitted when all attempts fail."""
        task = _make_task(max_attempts=2)
        fail = _fail_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        succeeded: list = []
        event_bus.on(Events.WORKER_SUCCEEDED, lambda d: succeeded.append(d))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_worker_with_retry(task)

        assert len(succeeded) == 0

    @pytest.mark.asyncio
    async def test_no_failed_event_on_success(self, orchestrator, event_bus):
        """WORKER_FAILED never emitted when first attempt succeeds."""
        task = _make_task(max_attempts=3)
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=ok)

        failed: list = []
        event_bus.on(Events.WORKER_FAILED, lambda d: failed.append(d))

        result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert len(failed) == 0

    @pytest.mark.asyncio
    async def test_retrying_event_has_correct_delay_values(
        self,
        orchestrator,
        event_bus,
    ):
        """Verify retry_in_seconds values in event payloads."""
        task = _make_task(max_attempts=4)
        fail = _fail_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        retry_events: list[dict] = []
        event_bus.on(Events.WORKER_RETRYING, lambda d: retry_events.append(d))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_worker_with_retry(task)

        assert [e["retry_in_seconds"] for e in retry_events] == [1, 2, 4]
        assert [e["attempt"] for e in retry_events] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_failed_event_includes_critical_flag(
        self,
        orchestrator,
        event_bus,
    ):
        """WORKER_FAILED payload includes critical flag."""
        task = _make_task(max_attempts=1, critical=True)
        fail = _fail_result(task, "critical failure")

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        failed_events: list[dict] = []
        event_bus.on(Events.WORKER_FAILED, lambda d: failed_events.append(d))

        await orchestrator._run_worker_with_retry(task)

        assert len(failed_events) == 1
        assert failed_events[0]["critical"] is True
        assert failed_events[0]["attempts"] == 1

    @pytest.mark.asyncio
    async def test_succeeded_event_includes_attempt_number(
        self,
        orchestrator,
        event_bus,
    ):
        """WORKER_SUCCEEDED payload includes which attempt succeeded."""
        task = _make_task(max_attempts=3)
        fail = _fail_result(task)
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(
            side_effect=[fail, ok],
        )

        succeeded: list[dict] = []
        event_bus.on(Events.WORKER_SUCCEEDED, lambda d: succeeded.append(d))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await orchestrator._run_worker_with_retry(task)

        assert len(succeeded) == 1
        assert succeeded[0]["attempt"] == 2
        assert succeeded[0]["role"] == "test-worker"

    @pytest.mark.asyncio
    async def test_error_message_from_last_attempt_preserved(
        self,
        orchestrator,
    ):
        """Final result contains the last attempt's error message."""
        task = _make_task(max_attempts=3)

        orchestrator._single_spawn_attempt = AsyncMock(
            side_effect=[
                _fail_result(task, "error 1"),
                _fail_result(task, "error 2"),
                _fail_result(task, "error 3"),
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await orchestrator._run_worker_with_retry(task)

        assert result.error == "error 3"

    @pytest.mark.asyncio
    async def test_spawn_subagent_backward_compatible(self, orchestrator):
        """spawn_subagent still works as before (delegates to _single_spawn_attempt)."""
        task = _make_task()
        ok = _ok_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=ok)

        result = await orchestrator.spawn_subagent(task)

        assert result.status == SubAgentStatus.COMPLETED
        assert orchestrator._single_spawn_attempt.call_count == 1

    @pytest.mark.asyncio
    async def test_large_max_attempts(self, orchestrator):
        """Handles large max_attempts correctly."""
        task = _make_task(max_attempts=10)
        fail = _fail_result(task)

        orchestrator._single_spawn_attempt = AsyncMock(return_value=fail)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await orchestrator._run_worker_with_retry(task)

        assert result.status == SubAgentStatus.FAILED
        assert orchestrator._single_spawn_attempt.call_count == 10

    @pytest.mark.asyncio
    async def test_retry_with_concurrency_limit_hit(self, orchestrator):
        """If concurrency limit hit on retry, result is FAILED."""
        task = _make_task(max_attempts=3)

        # First attempt succeeds normally but returns FAILED
        # Second attempt hits concurrency limit (also FAILED)
        concurrency_fail = SubAgentResult(
            task_id=task.task_id,
            role_name=task.role.name,
            status=SubAgentStatus.FAILED,
            error="Concurrency limit reached (5)",
        )
        orchestrator._single_spawn_attempt = AsyncMock(
            side_effect=[
                _fail_result(task, "transient"),
                concurrency_fail,
                _ok_result(task),
            ],
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await orchestrator._run_worker_with_retry(task)

        # Should keep retrying through concurrency failures
        assert result.status == SubAgentStatus.COMPLETED
        assert orchestrator._single_spawn_attempt.call_count == 3


class TestWorkerEvents:
    """Test event constants exist."""

    def test_worker_retrying_event_exists(self):
        assert Events.WORKER_RETRYING == "worker.retrying"

    def test_worker_failed_event_exists(self):
        assert Events.WORKER_FAILED == "worker.failed"

    def test_worker_succeeded_event_exists(self):
        assert Events.WORKER_SUCCEEDED == "worker.succeeded"


class TestControllerWorkerFailedHandler:
    """Test controller's reaction to WorkerFailed events."""

    @pytest.fixture
    def controller(self, event_bus, mock_config):
        from agent.core.controller import ControllerAgent

        orch = MagicMock()
        orch.spawn_subagent = AsyncMock()
        orch.tool_registry = MagicMock()
        orch.tool_registry.list_tools.return_value = []

        ctrl = ControllerAgent(
            orchestrator=orch,
            sdk_service=None,
            event_bus=event_bus,
            config=mock_config,
        )
        return ctrl

    @pytest.mark.asyncio
    async def test_non_critical_worker_failure_logged(self, controller, event_bus):
        """Non-critical worker failure is logged but doesn't fail the order."""
        from agent.core.subagent import ControllerTaskState

        state = ControllerTaskState(
            order_id="test-order",
            status="executing",
            worker_task_ids=["worker-123"],
        )
        controller._active_tasks["test-order"] = state

        await controller.start()
        try:
            await event_bus.emit(
                Events.WORKER_FAILED,
                {
                    "task_id": "worker-123",
                    "role": "coder",
                    "error": "timeout",
                    "attempts": 3,
                },
            )

            # Status should NOT change to failed for non-critical
            assert state.status == "executing"
        finally:
            await controller.stop()

    @pytest.mark.asyncio
    async def test_critical_worker_failure_marks_order_failed(self, controller, event_bus):
        """Critical worker failure marks the order as failed."""
        from agent.core.subagent import ControllerTaskState

        state = ControllerTaskState(
            order_id="test-order",
            status="executing",
            user_id="user-1",
            worker_task_ids=["critical-worker-123"],
        )
        controller._active_tasks["test-order"] = state

        failed_events: list[dict] = []

        async def capture_failed(data):
            failed_events.append(data)

        event_bus.on(Events.CONTROLLER_TASK_FAILED, capture_failed)

        await controller.start()
        try:
            await event_bus.emit(
                Events.WORKER_FAILED,
                {
                    "task_id": "critical-worker-123",
                    "role": "coder",
                    "error": "fatal",
                    "attempts": 3,
                },
            )

            assert state.status == "failed"
            assert "Critical worker" in state.error
            assert len(failed_events) == 1
        finally:
            await controller.stop()
