"""Tests for persistent SDK client features: idle timeout, resume, tool drift, prompt drift.

Covers the 4 issues implemented in the persistent client architecture:
- Issue 1: System prompt fingerprint drift detection
- Issue 2: MCP tool generation tracking and reconnect
- Issue 3: Idle timeout reaper
- Issue 4: Resume on crash via session_id storage
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.llm.claude_sdk import ClaudeSDKService, SDKTaskStatus
from agent.tools.registry import ToolTier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> ClaudeSDKService:
    """Create a minimal ClaudeSDKService for testing."""
    return ClaudeSDKService(working_dir="/tmp", max_turns=10)


@pytest.fixture
def registry():
    """Create a fresh ToolRegistry."""
    from agent.tools.registry import ToolRegistry

    return ToolRegistry()


@pytest.fixture
def service_with_registry(registry):
    """ClaudeSDKService wired to a ToolRegistry."""
    return ClaudeSDKService(
        working_dir="/tmp",
        max_turns=10,
        tool_registry=registry,
    )


@pytest.fixture
def service_with_soul():
    """ClaudeSDKService wired to a mock SoulLoader."""
    soul = MagicMock()
    soul.content = "You are a helpful assistant."
    return ClaudeSDKService(
        working_dir="/tmp",
        max_turns=10,
        soul_loader=soul,
    )


# ===================================================================
# Issue 3: Idle Timeout
# ===================================================================


class TestIdleTimeoutInit:
    """Test idle timeout state initialization."""

    def test_last_activity_starts_empty(self, service: ClaudeSDKService):
        assert service._last_activity == {}

    def test_idle_timeout_default(self, service: ClaudeSDKService):
        assert service._idle_timeout == 1800

    def test_idle_timeout_configurable(self):
        svc = ClaudeSDKService(working_dir="/tmp")
        svc._idle_timeout = 300
        assert svc._idle_timeout == 300

    def test_reaper_task_starts_none(self, service: ClaudeSDKService):
        assert service._reaper_task is None


class TestIdleTimeoutConfig:
    """Test idle_timeout in ClaudeSDKConfig."""

    def test_default_value(self):
        from agent.config import ClaudeSDKConfig

        cfg = ClaudeSDKConfig()
        assert cfg.idle_timeout == 1800

    def test_custom_value(self):
        from agent.config import ClaudeSDKConfig

        cfg = ClaudeSDKConfig(idle_timeout=600)
        assert cfg.idle_timeout == 600


class TestReaperLifecycle:
    """Test start_reaper / stop_reaper lifecycle."""

    @pytest.mark.asyncio
    async def test_start_reaper_creates_task(self, service: ClaudeSDKService):
        await service.start_reaper()
        assert service._reaper_task is not None
        assert not service._reaper_task.done()
        # Cleanup
        await service.stop_reaper()

    @pytest.mark.asyncio
    async def test_stop_reaper_cancels_task(self, service: ClaudeSDKService):
        await service.start_reaper()
        task = service._reaper_task
        await service.stop_reaper()
        assert service._reaper_task is None
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, service: ClaudeSDKService):
        await service.start_reaper()
        first_task = service._reaper_task
        await service.start_reaper()
        assert service._reaper_task is first_task
        await service.stop_reaper()

    @pytest.mark.asyncio
    async def test_double_stop_is_noop(self, service: ClaudeSDKService):
        await service.stop_reaper()  # No task to stop — should not raise
        assert service._reaper_task is None

    @pytest.mark.asyncio
    async def test_stop_after_stop_is_safe(self, service: ClaudeSDKService):
        await service.start_reaper()
        await service.stop_reaper()
        await service.stop_reaper()  # Should not raise


class TestReapIdleClients:
    """Test the reaper logic for disconnecting idle clients."""

    @pytest.mark.asyncio
    async def test_reaps_idle_client(self, service: ClaudeSDKService):
        """Client idle beyond timeout should be disconnected."""
        # Simulate a connected client that went idle 2000s ago
        mock_client = AsyncMock()
        service._clients["user1"] = mock_client
        service._last_activity["user1"] = time.monotonic() - 2000
        service._idle_timeout = 100  # 100s timeout

        # Manually call the reap logic (not the loop)
        now = time.monotonic()
        idle_ids = [
            tid for tid, last in service._last_activity.items()
            if (now - last) > service._idle_timeout and tid in service._clients
        ]
        for tid in idle_ids:
            await service.disconnect_client(tid)

        assert "user1" not in service._clients
        assert "user1" not in service._last_activity

    @pytest.mark.asyncio
    async def test_does_not_reap_active_client(self, service: ClaudeSDKService):
        """Client that was recently active should not be reaped."""
        mock_client = AsyncMock()
        service._clients["user1"] = mock_client
        service._last_activity["user1"] = time.monotonic()  # just now
        service._idle_timeout = 100

        now = time.monotonic()
        idle_ids = [
            tid for tid, last in service._last_activity.items()
            if (now - last) > service._idle_timeout and tid in service._clients
        ]
        assert len(idle_ids) == 0
        assert "user1" in service._clients

    @pytest.mark.asyncio
    async def test_reaps_only_idle_not_active(self, service: ClaudeSDKService):
        """Only idle clients should be reaped, active ones kept."""
        service._clients["idle_user"] = AsyncMock()
        service._clients["active_user"] = AsyncMock()
        service._last_activity["idle_user"] = time.monotonic() - 500
        service._last_activity["active_user"] = time.monotonic()
        service._idle_timeout = 100

        now = time.monotonic()
        idle_ids = [
            tid for tid, last in service._last_activity.items()
            if (now - last) > service._idle_timeout and tid in service._clients
        ]
        for tid in idle_ids:
            await service.disconnect_client(tid)

        assert "idle_user" not in service._clients
        assert "active_user" in service._clients


# ===================================================================
# Issue 4: Resume on Crash
# ===================================================================


class TestResumeSessionTracking:
    """Test session_id storage and resume behavior."""

    def test_last_session_ids_starts_empty(self, service: ClaudeSDKService):
        assert service._last_session_ids == {}

    def test_session_id_stored_manually(self, service: ClaudeSDKService):
        """Simulates what happens when ResultMessage arrives with session_id."""
        service._last_session_ids["user1"] = "sess-abc-123"
        assert service._last_session_ids["user1"] == "sess-abc-123"

    @pytest.mark.asyncio
    async def test_disconnect_preserves_session_id(self, service: ClaudeSDKService):
        """disconnect_client should keep _last_session_ids for resume."""
        service._clients["user1"] = AsyncMock()
        service._last_activity["user1"] = time.monotonic()
        service._last_session_ids["user1"] = "sess-abc-123"
        service._status["user1"] = SDKTaskStatus.COMPLETED

        await service.disconnect_client("user1")

        # Client gone, but session_id preserved
        assert "user1" not in service._clients
        assert "user1" not in service._last_activity
        assert service._last_session_ids["user1"] == "sess-abc-123"

    @pytest.mark.asyncio
    async def test_run_task_stream_retry_on_resume_failure(
        self, service: ClaudeSDKService
    ):
        """If resume fails, should retry without resume (fresh session)."""
        call_count = 0

        async def fake_impl(
            prompt, *, task_id, session_id, working_dir, on_permission, on_question
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and session_id is not None:
                from agent.llm.claude_sdk import SDKStreamEvent

                yield SDKStreamEvent(type="error", content="Resume failed")
            else:
                from agent.llm.claude_sdk import SDKStreamEvent

                yield SDKStreamEvent(type="result", content="Success")

        service._run_task_impl = fake_impl  # type: ignore[assignment]

        events = []
        async for event in service.run_task_stream(
            "hello", task_id="t1", session_id="old-session"
        ):
            events.append(event)

        # Should have retried: got "Success" from retry
        assert any(e.type == "result" and "Success" in e.content for e in events)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_run_task_stream_no_retry_without_session(
        self, service: ClaudeSDKService
    ):
        """If no session_id and error occurs, should NOT retry."""

        async def fake_impl(
            prompt, *, task_id, session_id, working_dir, on_permission, on_question
        ):
            from agent.llm.claude_sdk import SDKStreamEvent

            yield SDKStreamEvent(type="error", content="Some error")

        service._run_task_impl = fake_impl  # type: ignore[assignment]

        events = []
        async for event in service.run_task_stream("hello", task_id="t1"):
            events.append(event)

        # Error should pass through, no retry
        assert len(events) == 1
        assert events[0].type == "error"


# ===================================================================
# Issue 2: MCP Tool Generation Tracking
# ===================================================================


class TestToolGenerationCounter:
    """Test ToolRegistry generation counter."""

    def test_generation_starts_at_zero(self, registry):
        assert registry._generation == 0

    def test_register_increments(self, registry):
        @registry.tool(name="t1", description="test", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        assert registry._generation == 1

    def test_multiple_registers_increment(self, registry):
        @registry.tool(name="t1", description="test", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        @registry.tool(name="t2", description="test2", tier=ToolTier.SAFE)
        async def t2() -> str:
            return ""

        assert registry._generation == 2

    def test_enable_increments(self, registry):
        @registry.tool(name="t1", description="test", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        gen_before = registry._generation
        registry.enable_tool("t1")
        assert registry._generation == gen_before + 1

    def test_disable_increments(self, registry):
        @registry.tool(name="t1", description="test", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        gen_before = registry._generation
        registry.disable_tool("t1")
        assert registry._generation == gen_before + 1

    def test_unregister_increments(self, registry):
        @registry.tool(name="t1", description="test", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        gen_before = registry._generation
        registry.unregister_tool("t1")
        assert registry._generation == gen_before + 1

    def test_unregister_nonexistent_no_increment(self, registry):
        gen_before = registry._generation
        registry.unregister_tool("nonexistent")
        assert registry._generation == gen_before

    def test_enable_disable_cycle(self, registry):
        @registry.tool(name="t1", description="test", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        # register=1, disable=2, enable=3
        registry.disable_tool("t1")
        registry.enable_tool("t1")
        assert registry._generation == 3


class TestToolGenerationDriftDetection:
    """Test that SDK service detects tool generation changes."""

    def test_tool_generations_starts_empty(self, service_with_registry):
        assert service_with_registry._tool_generations == {}

    @pytest.mark.asyncio
    async def test_tool_change_triggers_reconnect(self, service_with_registry):
        """When tool generation changes, _ensure_client should disconnect old client."""
        svc = service_with_registry
        reg = svc.tool_registry

        # Simulate an existing connected client with gen=0
        mock_client = AsyncMock()
        svc._clients["user1"] = mock_client
        svc._tool_generations["user1"] = 0

        # Register a tool → gen becomes 1
        @reg.tool(name="new_tool", description="test", tier=ToolTier.SAFE)
        async def new_tool() -> str:
            return ""

        assert reg._generation == 1

        # _ensure_client should detect the drift and disconnect, then reconnect
        disconnect_called = False
        original_disconnect = svc.disconnect_client

        async def tracking_disconnect(task_id):
            nonlocal disconnect_called
            disconnect_called = True
            await original_disconnect(task_id)

        with patch.object(svc, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch.object(svc, "disconnect_client", side_effect=tracking_disconnect):
                with patch("claude_code_sdk.ClaudeCodeOptions") as MockOpts:
                    MockOpts.return_value = MagicMock()
                    with patch("claude_code_sdk.ClaudeSDKClient") as MockClient:
                        MockClient.return_value = AsyncMock()
                        await svc._ensure_client("user1")

        assert disconnect_called


# ===================================================================
# Issue 1: System Prompt Fingerprint
# ===================================================================


class TestPromptFingerprint:
    """Test system prompt fingerprint computation and drift detection."""

    def test_fingerprints_start_empty(self, service: ClaudeSDKService):
        assert service._prompt_fingerprints == {}

    def test_fingerprint_deterministic(self, service_with_soul):
        """Same inputs produce same fingerprint."""
        fp1 = service_with_soul._compute_prompt_fingerprint()
        fp2 = service_with_soul._compute_prompt_fingerprint()
        assert fp1 == fp2

    def test_fingerprint_changes_with_soul(self):
        """Different soul content produces different fingerprint."""
        soul1 = MagicMock()
        soul1.content = "You are helpful."
        svc1 = ClaudeSDKService(working_dir="/tmp", soul_loader=soul1)

        soul2 = MagicMock()
        soul2.content = "You are a pirate."
        svc2 = ClaudeSDKService(working_dir="/tmp", soul_loader=soul2)

        fp1 = svc1._compute_prompt_fingerprint()
        fp2 = svc2._compute_prompt_fingerprint()
        assert fp1 != fp2

    def test_fingerprint_changes_with_facts(self, service_with_soul):
        """Different facts produce different fingerprint."""
        fact_a = SimpleNamespace(key="user.name", value="Alice")
        fact_b = SimpleNamespace(key="user.name", value="Bob")

        fp1 = service_with_soul._compute_prompt_fingerprint(facts=[fact_a])
        fp2 = service_with_soul._compute_prompt_fingerprint(facts=[fact_b])
        assert fp1 != fp2

    def test_fingerprint_same_facts_same_hash(self, service_with_soul):
        """Same facts produce same fingerprint."""
        facts = [SimpleNamespace(key="user.name", value="Alice")]
        fp1 = service_with_soul._compute_prompt_fingerprint(facts=facts)
        fp2 = service_with_soul._compute_prompt_fingerprint(facts=facts)
        assert fp1 == fp2

    def test_fingerprint_no_soul_no_facts(self, service: ClaudeSDKService):
        """Service with no soul or facts should still produce a fingerprint."""
        fp = service._compute_prompt_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16  # sha256[:16]

    def test_fingerprint_length(self, service_with_soul):
        """Fingerprint should be a 16-char hex string."""
        fp = service_with_soul._compute_prompt_fingerprint()
        assert len(fp) == 16
        # Should be valid hex
        int(fp, 16)

    @pytest.mark.asyncio
    async def test_prompt_drift_triggers_reconnect(self, service_with_soul):
        """When prompt fingerprint changes, _ensure_client should reconnect."""
        svc = service_with_soul

        # Simulate an existing connected client with old fingerprint
        mock_client = AsyncMock()
        svc._clients["user1"] = mock_client
        svc._prompt_fingerprints["user1"] = "old_fingerprint_00"

        disconnect_called = False
        original_disconnect = svc.disconnect_client

        async def tracking_disconnect(task_id):
            nonlocal disconnect_called
            disconnect_called = True
            await original_disconnect(task_id)

        with patch.object(svc, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch.object(svc, "disconnect_client", side_effect=tracking_disconnect):
                with patch("claude_code_sdk.ClaudeCodeOptions") as MockOpts:
                    MockOpts.return_value = MagicMock()
                    with patch("claude_code_sdk.ClaudeSDKClient") as MockClient:
                        MockClient.return_value = AsyncMock()
                        await svc._ensure_client("user1")

        assert disconnect_called

    @pytest.mark.asyncio
    async def test_no_reconnect_when_fingerprint_unchanged(self, service_with_soul):
        """When prompt fingerprint matches, should reuse existing client."""
        svc = service_with_soul

        # Compute the real fingerprint
        real_fp = svc._compute_prompt_fingerprint()

        mock_client = AsyncMock()
        svc._clients["user1"] = mock_client
        svc._prompt_fingerprints["user1"] = real_fp

        with patch.object(svc, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            result = await svc._ensure_client("user1")

        # Should return existing client without disconnect
        assert result is mock_client


# ===================================================================
# disconnect_client cleanup
# ===================================================================


class TestDisconnectClientCleanup:
    """Test that disconnect_client cleans up all per-client state."""

    @pytest.mark.asyncio
    async def test_cleans_up_all_state(self, service: ClaudeSDKService):
        """disconnect_client should remove all per-client state except session_ids."""
        mock_client = AsyncMock()
        task_id = "user1"

        service._clients[task_id] = mock_client
        service._cancel_events[task_id] = asyncio.Event()
        service._status[task_id] = SDKTaskStatus.COMPLETED
        service._last_activity[task_id] = time.monotonic()
        service._tool_generations[task_id] = 5
        service._prompt_fingerprints[task_id] = "abc123"
        service._last_session_ids[task_id] = "sess-xyz"

        await service.disconnect_client(task_id)

        assert task_id not in service._clients
        assert task_id not in service._cancel_events
        assert task_id not in service._status
        assert task_id not in service._last_activity
        assert task_id not in service._tool_generations
        assert task_id not in service._prompt_fingerprints
        # Session ID preserved for resume
        assert service._last_session_ids[task_id] == "sess-xyz"

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_is_safe(self, service: ClaudeSDKService):
        """Disconnecting a non-existent client should not raise."""
        await service.disconnect_client("nonexistent")

    @pytest.mark.asyncio
    async def test_disconnect_calls_client_disconnect(self, service: ClaudeSDKService):
        """Should call client.disconnect() on the SDK client."""
        mock_client = AsyncMock()
        service._clients["user1"] = mock_client

        await service.disconnect_client("user1")

        mock_client.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_handles_client_error(self, service: ClaudeSDKService):
        """Should suppress errors from client.disconnect()."""
        mock_client = AsyncMock()
        mock_client.disconnect.side_effect = RuntimeError("connection lost")
        service._clients["user1"] = mock_client

        # Should not raise
        await service.disconnect_client("user1")
        assert "user1" not in service._clients


# ===================================================================
# Integration: _ensure_client with resume
# ===================================================================


class TestEnsureClientResume:
    """Test _ensure_client session_id / resume integration."""

    @pytest.mark.asyncio
    async def test_uses_explicit_session_id(self, service: ClaudeSDKService):
        """Explicit session_id should be passed as resume to options."""
        captured_options = {}

        class FakeClient:
            def __init__(self, options):
                captured_options["resume"] = getattr(options, "resume", None)

            async def connect(self):
                pass

        with patch.object(service, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch("claude_code_sdk.ClaudeSDKClient", FakeClient):
                with patch("claude_code_sdk.ClaudeCodeOptions") as MockOptions:
                    mock_opts = MagicMock()
                    MockOptions.return_value = mock_opts
                    await service._ensure_client("user1", session_id="explicit-sess")

                    # Check that resume was passed to ClaudeCodeOptions
                    call_kwargs = MockOptions.call_args[1]
                    assert call_kwargs.get("resume") == "explicit-sess"

        # Cleanup
        await service.disconnect_client("user1")

    @pytest.mark.asyncio
    async def test_falls_back_to_stored_session_id(self, service: ClaudeSDKService):
        """Without explicit session_id, should use _last_session_ids."""
        service._last_session_ids["user1"] = "stored-sess-456"

        with patch.object(service, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch("claude_code_sdk.ClaudeCodeOptions") as MockOptions:
                mock_opts = MagicMock()
                MockOptions.return_value = mock_opts
                with patch("claude_code_sdk.ClaudeSDKClient") as MockClient:
                    mock_client = AsyncMock()
                    MockClient.return_value = mock_client
                    await service._ensure_client("user1")

                    call_kwargs = MockOptions.call_args[1]
                    assert call_kwargs.get("resume") == "stored-sess-456"

        await service.disconnect_client("user1")

    @pytest.mark.asyncio
    async def test_no_resume_when_no_session(self, service: ClaudeSDKService):
        """When no session_id exists, resume should not be in options."""
        with patch.object(service, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch("claude_code_sdk.ClaudeCodeOptions") as MockOptions:
                mock_opts = MagicMock()
                MockOptions.return_value = mock_opts
                with patch("claude_code_sdk.ClaudeSDKClient") as MockClient:
                    mock_client = AsyncMock()
                    MockClient.return_value = mock_client
                    await service._ensure_client("user1")

                    call_kwargs = MockOptions.call_args[1]
                    assert "resume" not in call_kwargs

        await service.disconnect_client("user1")


# ===================================================================
# Integration: _ensure_client stores fingerprint and generation
# ===================================================================


class TestEnsureClientStoresState:
    """Test that _ensure_client records fingerprint and generation on connect."""

    @pytest.mark.asyncio
    async def test_stores_prompt_fingerprint(self, service_with_soul):
        svc = service_with_soul

        with patch.object(svc, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch("claude_code_sdk.ClaudeCodeOptions") as MockOptions:
                MockOptions.return_value = MagicMock()
                with patch("claude_code_sdk.ClaudeSDKClient") as MockClient:
                    MockClient.return_value = AsyncMock()
                    await svc._ensure_client("user1")

        assert "user1" in svc._prompt_fingerprints
        assert len(svc._prompt_fingerprints["user1"]) == 16
        await svc.disconnect_client("user1")

    @pytest.mark.asyncio
    async def test_stores_tool_generation(self, service_with_registry):
        svc = service_with_registry
        reg = svc.tool_registry

        @reg.tool(name="t1", description="test", tier=ToolTier.SAFE)
        async def t1() -> str:
            return ""

        with patch.object(svc, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch("claude_code_sdk.ClaudeCodeOptions") as MockOptions:
                MockOptions.return_value = MagicMock()
                with patch("claude_code_sdk.ClaudeSDKClient") as MockClient:
                    MockClient.return_value = AsyncMock()
                    await svc._ensure_client("user1")

        assert svc._tool_generations["user1"] == reg._generation
        await svc.disconnect_client("user1")

    @pytest.mark.asyncio
    async def test_stores_last_activity(self, service: ClaudeSDKService):
        with patch.object(service, "_query_memory", new_callable=AsyncMock, return_value=(None, None)):
            with patch("claude_code_sdk.ClaudeCodeOptions") as MockOptions:
                MockOptions.return_value = MagicMock()
                with patch("claude_code_sdk.ClaudeSDKClient") as MockClient:
                    MockClient.return_value = AsyncMock()
                    await service._ensure_client("user1")

        assert "user1" in service._last_activity
        assert service._last_activity["user1"] > 0
        await service.disconnect_client("user1")


# ===================================================================
# Concurrent query serialization (per-task lock)
# ===================================================================


class TestQueryLockSerialization:
    """Test that concurrent queries on the same task are serialized."""

    def test_query_locks_start_empty(self, service: ClaudeSDKService):
        assert service._query_locks == {}

    def test_get_query_lock_creates_lock(self, service: ClaudeSDKService):
        lock = service._get_query_lock("user1")
        assert isinstance(lock, asyncio.Lock)
        assert "user1" in service._query_locks

    def test_get_query_lock_reuses_same_lock(self, service: ClaudeSDKService):
        lock1 = service._get_query_lock("user1")
        lock2 = service._get_query_lock("user1")
        assert lock1 is lock2

    def test_different_tasks_get_different_locks(self, service: ClaudeSDKService):
        lock1 = service._get_query_lock("user1")
        lock2 = service._get_query_lock("user2")
        assert lock1 is not lock2

    @pytest.mark.asyncio
    async def test_disconnect_removes_lock(self, service: ClaudeSDKService):
        service._get_query_lock("user1")
        service._clients["user1"] = AsyncMock()
        await service.disconnect_client("user1")
        assert "user1" not in service._query_locks

    @pytest.mark.asyncio
    async def test_concurrent_queries_serialized(self, service: ClaudeSDKService):
        """Two concurrent messages for the same task should run sequentially."""
        execution_order: list[str] = []

        async def fake_impl(
            prompt, *, task_id, session_id, working_dir, on_permission, on_question
        ):
            from agent.llm.claude_sdk import SDKStreamEvent

            execution_order.append(f"start:{prompt}")
            await asyncio.sleep(0.05)  # Simulate processing time
            execution_order.append(f"end:{prompt}")
            yield SDKStreamEvent(type="result", content=prompt)

        service._run_task_locked = fake_impl  # type: ignore[assignment]

        # Launch two concurrent queries for the same task
        async def collect(prompt):
            events = []
            async for event in service._run_task_impl(
                prompt, task_id="user1"
            ):
                events.append(event)
            return events

        results = await asyncio.gather(collect("msg1"), collect("msg2"))

        # Both should succeed
        assert len(results[0]) == 1
        assert len(results[1]) == 1

        # They should NOT interleave — must be fully sequential
        assert execution_order == [
            "start:msg1", "end:msg1", "start:msg2", "end:msg2",
        ]

    @pytest.mark.asyncio
    async def test_different_tasks_run_concurrently(self, service: ClaudeSDKService):
        """Queries for different tasks should NOT block each other."""
        execution_order: list[str] = []

        async def fake_impl(
            prompt, *, task_id, session_id, working_dir, on_permission, on_question
        ):
            from agent.llm.claude_sdk import SDKStreamEvent

            execution_order.append(f"start:{task_id}")
            await asyncio.sleep(0.05)
            execution_order.append(f"end:{task_id}")
            yield SDKStreamEvent(type="result", content=prompt)

        service._run_task_locked = fake_impl  # type: ignore[assignment]

        async def collect(task_id, prompt):
            events = []
            async for event in service._run_task_impl(
                prompt, task_id=task_id
            ):
                events.append(event)
            return events

        await asyncio.gather(
            collect("user1", "hello"),
            collect("user2", "world"),
        )

        # Both should start before either ends (concurrent)
        starts = [e for e in execution_order if e.startswith("start:")]
        assert len(starts) == 2
        # Both starts should appear before both ends
        first_end = next(i for i, e in enumerate(execution_order) if e.startswith("end:"))
        assert first_end >= 2  # Both starts happened before first end


# ===================================================================
# Stale ResultMessage detection
# ===================================================================


class TestStaleResultSkipping:
    """Test that stale ResultMessages from previous queries are skipped.

    _safe_receive() works with already-parsed Message objects returned by
    client.receive_messages(), NOT raw dicts.
    """

    @staticmethod
    def _make_result(**kwargs: Any) -> Any:
        """Create a ResultMessage with sensible defaults."""
        from claude_code_sdk import ResultMessage

        defaults = {
            "subtype": "success",
            "duration_ms": 100,
            "duration_api_ms": 50,
            "is_error": False,
            "num_turns": 1,
            "session_id": "sess-000",
            "total_cost_usd": 0.001,
            "usage": {},
            "result": "",
        }
        defaults.update(kwargs)
        return ResultMessage(**defaults)

    @staticmethod
    def _make_assistant(text: str = "Hello!") -> Any:
        """Create an AssistantMessage."""
        from claude_code_sdk import AssistantMessage, TextBlock

        return AssistantMessage(
            content=[TextBlock(text=text)],
            model="claude-sonnet-4-6",
        )

    @pytest.mark.asyncio
    async def test_stale_result_skipped(self):
        """A ResultMessage before any AssistantMessage should be skipped."""
        from claude_code_sdk import AssistantMessage, ResultMessage

        stale_result = self._make_result(session_id="old-session")
        real_assistant = self._make_assistant("Hello!")
        real_result = self._make_result(
            session_id="new-session", duration_ms=2000, total_cost_usd=0.01,
        )

        parsed_messages = [stale_result, real_assistant, real_result]

        async def mock_receive_messages():
            for msg in parsed_messages:
                yield msg

        mock_client = MagicMock()
        mock_client.receive_messages = mock_receive_messages

        messages = []
        async for msg in ClaudeSDKService._safe_receive(mock_client):
            messages.append(msg)

        # Should have skipped the stale result, yielded assistant + real result
        assert len(messages) == 2
        assert isinstance(messages[0], AssistantMessage)
        assert isinstance(messages[1], ResultMessage)
        assert messages[1].session_id == "new-session"

    @pytest.mark.asyncio
    async def test_normal_flow_not_affected(self):
        """Normal flow (AssistantMessage before ResultMessage) should work."""
        from claude_code_sdk import AssistantMessage, ResultMessage

        parsed_messages = [
            self._make_assistant("Hi!"),
            self._make_result(session_id="sess-123"),
        ]

        async def mock_receive_messages():
            for msg in parsed_messages:
                yield msg

        mock_client = MagicMock()
        mock_client.receive_messages = mock_receive_messages

        messages = []
        async for msg in ClaudeSDKService._safe_receive(mock_client):
            messages.append(msg)

        assert len(messages) == 2
        assert isinstance(messages[0], AssistantMessage)
        assert isinstance(messages[1], ResultMessage)

    @pytest.mark.asyncio
    async def test_unknown_message_type_logged(self):
        """Non-Assistant/non-Result message types should be yielded as-is."""
        from claude_code_sdk import AssistantMessage, ResultMessage, SystemMessage

        # SystemMessage is a known parsed type that is neither Assistant nor Result
        system_msg = SystemMessage(
            subtype="init",
            data={"session_id": "sess-123"},
        )
        parsed_messages = [
            system_msg,
            self._make_assistant("Hi!"),
            self._make_result(session_id="sess-123"),
        ]

        async def mock_receive_messages():
            for msg in parsed_messages:
                yield msg

        mock_client = MagicMock()
        mock_client.receive_messages = mock_receive_messages

        messages = []
        async for msg in ClaudeSDKService._safe_receive(mock_client):
            messages.append(msg)

        # SystemMessage + AssistantMessage + ResultMessage = 3 messages
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], AssistantMessage)
        assert isinstance(messages[2], ResultMessage)

    @pytest.mark.asyncio
    async def test_multiple_stale_results_skipped(self):
        """Multiple stale ResultMessages should all be skipped."""
        from claude_code_sdk import AssistantMessage, ResultMessage

        parsed_messages = [
            self._make_result(session_id="stale-1", num_turns=0),
            self._make_result(session_id="stale-2", num_turns=0),
            self._make_assistant("Real response"),
            self._make_result(session_id="real-session", duration_ms=2000),
        ]

        async def mock_receive_messages():
            for msg in parsed_messages:
                yield msg

        mock_client = MagicMock()
        mock_client.receive_messages = mock_receive_messages

        messages = []
        async for msg in ClaudeSDKService._safe_receive(mock_client):
            messages.append(msg)

        assert len(messages) == 2
        assert isinstance(messages[0], AssistantMessage)
        assert isinstance(messages[1], ResultMessage)
        assert messages[1].session_id == "real-session"

    @pytest.mark.asyncio
    async def test_only_stale_results_no_assistant(self):
        """If stream has only stale results and no assistant, yield nothing."""
        parsed_messages = [
            self._make_result(session_id="stale-only"),
        ]

        async def mock_receive_messages():
            for msg in parsed_messages:
                yield msg

        mock_client = MagicMock()
        mock_client.receive_messages = mock_receive_messages

        messages = []
        async for msg in ClaudeSDKService._safe_receive(mock_client):
            messages.append(msg)

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_receive_messages_exception_unknown_type(self):
        """If receive_messages raises 'Unknown message type', should be caught."""
        async def mock_receive_messages():
            raise Exception("Unknown message type: foo_bar")
            yield  # make it a generator  # noqa: E501

        mock_client = MagicMock()
        mock_client.receive_messages = mock_receive_messages

        messages = []
        async for msg in ClaudeSDKService._safe_receive(mock_client):
            messages.append(msg)

        assert len(messages) == 0  # No crash, no messages

    @pytest.mark.asyncio
    async def test_receive_messages_exception_other_reraises(self):
        """Non 'Unknown message type' exceptions should propagate."""
        async def mock_receive_messages():
            raise RuntimeError("Connection lost")
            yield  # make it a generator  # noqa: E501

        mock_client = MagicMock()
        mock_client.receive_messages = mock_receive_messages

        with pytest.raises(RuntimeError, match="Connection lost"):
            async for _ in ClaudeSDKService._safe_receive(mock_client):
                pass


# ===================================================================
# Permission protocol patch (SDK v0.0.25 <-> CLI v2.1+ compat)
# ===================================================================


class TestPermissionProtocolPatch:
    """Test that the SDK permission protocol is patched to use new format."""

    @pytest.fixture(autouse=True)
    def _ensure_patch(self):
        """Ensure SDK patch is applied before each test."""
        from agent.llm.claude_sdk import sdk_available
        sdk_available()

    def test_patch_applied(self):
        """Patch should be applied when sdk_available() is called."""
        from agent.llm.claude_sdk import _SDK_PATCHED, sdk_available

        # Ensure the patch is triggered
        sdk_available()
        # Re-import to get updated value
        from agent.llm import claude_sdk
        assert claude_sdk._SDK_PATCHED is True

    def test_patched_method_exists(self):
        """Query._handle_control_request should be patched."""
        from agent.llm.claude_sdk import sdk_available

        sdk_available()  # Ensure patch is applied

        from claude_code_sdk._internal.query import Query as QuerySession

        method = QuerySession._handle_control_request
        assert method is not None

    @pytest.mark.asyncio
    async def test_allow_response_format(self):
        """Patched handler should return {behavior: 'allow', updatedInput: ...}."""
        import json

        from claude_code_sdk._internal.query import Query as QuerySession
        from claude_code_sdk.types import PermissionResultAllow

        written: list[str] = []

        # Create a minimal QuerySession-like object
        session = object.__new__(QuerySession)
        session.transport = MagicMock()
        session.transport.write = AsyncMock(side_effect=lambda s: written.append(s))

        tool_input = {"command": "ls -la"}

        async def fake_can_use_tool(name: str, inp: dict, ctx: Any) -> Any:
            return PermissionResultAllow(updated_input=inp)

        session.can_use_tool = fake_can_use_tool

        request = {
            "request_id": "req_1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": tool_input,
            },
        }

        await session._handle_control_request(request)

        assert len(written) == 1
        response = json.loads(written[0])
        inner = response["response"]["response"]
        assert inner["behavior"] == "allow"
        assert inner["updatedInput"] == tool_input

    @pytest.mark.asyncio
    async def test_deny_response_format(self):
        """Patched handler should return {behavior: 'deny', message: '...'}."""
        import json

        from claude_code_sdk._internal.query import Query as QuerySession
        from claude_code_sdk.types import PermissionResultDeny

        written: list[str] = []

        session = object.__new__(QuerySession)
        session.transport = MagicMock()
        session.transport.write = AsyncMock(side_effect=lambda s: written.append(s))

        async def fake_can_use_tool(name: str, inp: dict, ctx: Any) -> Any:
            return PermissionResultDeny(message="Not allowed")

        session.can_use_tool = fake_can_use_tool

        request = {
            "request_id": "req_2",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": {"command": "rm -rf /"},
            },
        }

        await session._handle_control_request(request)

        assert len(written) == 1
        response = json.loads(written[0])
        inner = response["response"]["response"]
        assert inner["behavior"] == "deny"
        assert inner["message"] == "Not allowed"

    @pytest.mark.asyncio
    async def test_allow_without_updated_input(self):
        """Allow without updated_input should not include updatedInput key."""
        import json

        from claude_code_sdk._internal.query import Query as QuerySession
        from claude_code_sdk.types import PermissionResultAllow

        written: list[str] = []

        session = object.__new__(QuerySession)
        session.transport = MagicMock()
        session.transport.write = AsyncMock(side_effect=lambda s: written.append(s))

        async def fake_can_use_tool(name: str, inp: dict, ctx: Any) -> Any:
            return PermissionResultAllow()

        session.can_use_tool = fake_can_use_tool

        request = {
            "request_id": "req_3",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Read",
                "input": {"file_path": "/tmp/test"},
            },
        }

        await session._handle_control_request(request)

        response = json.loads(written[0])
        inner = response["response"]["response"]
        assert inner["behavior"] == "allow"
        assert "updatedInput" not in inner
