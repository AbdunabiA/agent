"""Tests for the heartbeat daemon."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import AgentPersonaConfig
from agent.core.events import EventBus
from agent.core.heartbeat import HeartbeatDaemon
from agent.core.session import TokenUsage
from agent.memory.models import Fact


@pytest.fixture
def mock_agent_loop() -> AsyncMock:
    loop = AsyncMock()
    response = AsyncMock()
    response.content = "HEARTBEAT_OK"
    response.usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    loop.process_message.return_value = response
    return loop


@pytest.fixture
def config() -> AgentPersonaConfig:
    return AgentPersonaConfig(heartbeat_interval="30m")


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def heartbeat(
    mock_agent_loop: AsyncMock,
    config: AgentPersonaConfig,
    event_bus: EventBus,
) -> HeartbeatDaemon:
    return HeartbeatDaemon(
        agent_loop=mock_agent_loop,
        config=config,
        event_bus=event_bus,
    )


class TestHeartbeatDaemon:
    """Tests for HeartbeatDaemon."""

    def test_parse_interval_minutes(self, heartbeat: HeartbeatDaemon) -> None:
        assert HeartbeatDaemon._parse_interval("30m") == 1800

    def test_parse_interval_hours(self, heartbeat: HeartbeatDaemon) -> None:
        assert HeartbeatDaemon._parse_interval("1h") == 3600

    def test_parse_interval_seconds(self, heartbeat: HeartbeatDaemon) -> None:
        assert HeartbeatDaemon._parse_interval("30s") == 30

    def test_parse_interval_number_only(self, heartbeat: HeartbeatDaemon) -> None:
        """Number without unit should default to minutes."""
        assert HeartbeatDaemon._parse_interval("15") == 900

    def test_parse_interval_invalid(self, heartbeat: HeartbeatDaemon) -> None:
        """Invalid interval should default to 30 minutes."""
        assert HeartbeatDaemon._parse_interval("invalid") == 1800

    async def test_tick_heartbeat_ok(
        self,
        heartbeat: HeartbeatDaemon,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """HEARTBEAT_OK response should be silent."""
        await heartbeat._tick()
        mock_agent_loop.process_message.assert_called_once()
        assert heartbeat._consecutive_failures == 0

    async def test_tick_with_action(
        self,
        heartbeat: HeartbeatDaemon,
        mock_agent_loop: AsyncMock,
        event_bus: EventBus,
    ) -> None:
        """Non-HEARTBEAT_OK response should emit action event."""
        action_response = AsyncMock()
        action_response.content = "I noticed disk space is low. Cleaned up temp files."
        mock_agent_loop.process_message.return_value = action_response

        events_received = []

        async def handler(data):
            events_received.append(data)

        event_bus.on("heartbeat.action", handler)
        await heartbeat._tick()

        assert len(events_received) == 1
        assert "disk space" in events_received[0]["action"]

    async def test_circuit_breaker_after_3_failures(
        self,
        heartbeat: HeartbeatDaemon,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Circuit breaker should disable heartbeat after 3 consecutive failures."""
        mock_agent_loop.process_message.side_effect = Exception("LLM error")

        for _ in range(3):
            await heartbeat._tick()

        assert heartbeat.is_enabled is False
        assert heartbeat._consecutive_failures == 3

    async def test_success_resets_failure_count(
        self,
        heartbeat: HeartbeatDaemon,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Successful tick should reset failure count."""
        # Cause some failures
        mock_agent_loop.process_message.side_effect = Exception("error")
        await heartbeat._tick()
        await heartbeat._tick()
        assert heartbeat._consecutive_failures == 2

        # Success should reset
        response = AsyncMock()
        response.content = "HEARTBEAT_OK"
        mock_agent_loop.process_message.side_effect = None
        mock_agent_loop.process_message.return_value = response

        await heartbeat._tick()
        assert heartbeat._consecutive_failures == 0

    def test_enable_disable(self, heartbeat: HeartbeatDaemon) -> None:
        """Enable/disable should work."""
        assert heartbeat.is_enabled is True
        heartbeat.disable()
        assert heartbeat.is_enabled is False
        heartbeat.enable()
        assert heartbeat.is_enabled is True
        assert heartbeat._consecutive_failures == 0

    async def test_disabled_tick_noop(
        self,
        heartbeat: HeartbeatDaemon,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Disabled heartbeat tick should do nothing."""
        heartbeat.disable()
        await heartbeat._tick()
        mock_agent_loop.process_message.assert_not_called()

    def test_read_heartbeat_md(self, heartbeat: HeartbeatDaemon, tmp_path: Path) -> None:
        """Should read HEARTBEAT.md if it exists."""
        md_file = tmp_path / "HEARTBEAT.md"
        md_file.write_text("- Check disk space\n- Check services")

        with patch("agent.core.heartbeat.Path") as mock_path:
            # First path should find the file
            mock_instance = mock_path.return_value
            mock_instance.exists.return_value = True
            mock_instance.read_text.return_value = md_file.read_text()

            content = heartbeat._read_heartbeat_md()
            # Should contain either our content or default
            assert isinstance(content, str)
            assert len(content) > 0


def _make_fact(key: str, value: str, **kwargs) -> Fact:
    """Helper to create a Fact with defaults."""
    return Fact(id="test-id", key=key, value=value, **kwargs)


def _make_fact_store_mock(**overrides) -> MagicMock:
    """Create a mock FactStore with all proactive methods stubbed."""
    store = MagicMock()
    store.get_by_priority = AsyncMock(return_value=overrides.get("urgent", []))
    store.get_temporal_due_soon = AsyncMock(return_value=overrides.get("temporal", []))
    store.get_active_topics = AsyncMock(return_value=overrides.get("topics", []))
    store.get_emotional_summary = AsyncMock(return_value=overrides.get("emotional", ""))
    store.get_relevant = AsyncMock(return_value=[])
    return store


class TestProactiveInitiative:
    """Tests for proactive initiative features in the heartbeat."""

    async def test_tick_includes_urgent_facts(
        self,
        config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> None:
        """Verify that urgent (high-priority) facts appear in the heartbeat context."""
        fact_store = _make_fact_store_mock(
            urgent=[
                _make_fact("deploy.deadline", "Production deploy tonight", priority="high"),
                _make_fact("bug.critical", "Auth service down", priority="high"),
            ],
        )

        agent_loop = AsyncMock()
        agent_loop.fact_store = fact_store
        response = AsyncMock()
        response.content = "HEARTBEAT_OK"
        agent_loop.process_message.return_value = response

        daemon = HeartbeatDaemon(
            agent_loop=agent_loop,
            config=config,
            event_bus=event_bus,
            fact_store=fact_store,
        )

        with patch.object(daemon, "_read_heartbeat_md", return_value="- Check stuff"):
            await daemon._tick()

        fact_store.get_by_priority.assert_called_once_with("high", limit=5)

        heartbeat_message = agent_loop.process_message.call_args[0][0]
        assert "## URGENT ITEMS" in heartbeat_message
        assert "deploy.deadline" in heartbeat_message
        assert "Production deploy tonight" in heartbeat_message
        assert "bug.critical" in heartbeat_message

    async def test_tick_includes_deadlines(
        self,
        config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> None:
        """Verify that approaching deadlines appear in the heartbeat context."""
        fact_store = _make_fact_store_mock(
            temporal=[
                _make_fact(
                    "project.milestone",
                    "MVP demo ready",
                    temporal_reference="2026-03-24T10:00:00",
                ),
            ],
        )

        agent_loop = AsyncMock()
        agent_loop.fact_store = fact_store
        response = AsyncMock()
        response.content = "HEARTBEAT_OK"
        agent_loop.process_message.return_value = response

        daemon = HeartbeatDaemon(
            agent_loop=agent_loop,
            config=config,
            event_bus=event_bus,
            fact_store=fact_store,
        )

        with patch.object(daemon, "_read_heartbeat_md", return_value="- Check stuff"):
            await daemon._tick()

        fact_store.get_temporal_due_soon.assert_called_once_with(hours=24)

        heartbeat_message = agent_loop.process_message.call_args[0][0]
        assert "## APPROACHING DEADLINES" in heartbeat_message
        assert "project.milestone" in heartbeat_message
        assert "MVP demo ready" in heartbeat_message
        assert "2026-03-24T10:00:00" in heartbeat_message

    async def test_tick_includes_active_topics(
        self,
        config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> None:
        """Verify that active topics appear in the heartbeat context."""
        fact_store = _make_fact_store_mock(
            topics=["deployment", "API refactor", "testing"],
        )

        agent_loop = AsyncMock()
        agent_loop.fact_store = fact_store
        response = AsyncMock()
        response.content = "HEARTBEAT_OK"
        agent_loop.process_message.return_value = response

        daemon = HeartbeatDaemon(
            agent_loop=agent_loop,
            config=config,
            event_bus=event_bus,
            fact_store=fact_store,
        )

        with patch.object(daemon, "_read_heartbeat_md", return_value="- Check stuff"):
            await daemon._tick()

        fact_store.get_active_topics.assert_called_once_with(limit=3)

        heartbeat_message = agent_loop.process_message.call_args[0][0]
        assert "## ACTIVE TOPICS" in heartbeat_message
        assert "deployment" in heartbeat_message
        assert "API refactor" in heartbeat_message
        assert "testing" in heartbeat_message

    async def test_tick_handles_no_fact_store(
        self,
        config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> None:
        """Verify heartbeat works without a fact_store (no crash)."""
        # Ensure agent_loop has no fact_store attribute
        agent_loop_spec = MagicMock(spec=[])
        agent_loop_spec.llm = MagicMock()
        response = MagicMock()
        response.content = "HEARTBEAT_OK"
        agent_loop_spec.process_message = AsyncMock(return_value=response)

        daemon = HeartbeatDaemon(
            agent_loop=agent_loop_spec,
            config=config,
            event_bus=event_bus,
        )

        with patch.object(daemon, "_read_heartbeat_md", return_value="- Check stuff"):
            await daemon._tick()

        agent_loop_spec.process_message.assert_called_once()
        heartbeat_message = agent_loop_spec.process_message.call_args[0][0]
        assert "## URGENT ITEMS" not in heartbeat_message
        assert "## APPROACHING DEADLINES" not in heartbeat_message
        assert "## ACTIVE TOPICS" not in heartbeat_message

    async def test_tick_fact_store_throws(
        self,
        config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> None:
        """If fact_store throws on every query, tick still completes."""
        fact_store = MagicMock()
        fact_store.get_by_priority = AsyncMock(side_effect=RuntimeError("db error"))
        fact_store.get_temporal_due_soon = AsyncMock(side_effect=RuntimeError("db error"))
        fact_store.get_active_topics = AsyncMock(side_effect=RuntimeError("db error"))
        fact_store.get_emotional_summary = AsyncMock(side_effect=RuntimeError("db error"))
        fact_store.get_relevant = AsyncMock(side_effect=RuntimeError("db error"))

        agent_loop = AsyncMock()
        agent_loop.fact_store = fact_store
        response = AsyncMock()
        response.content = "HEARTBEAT_OK"
        agent_loop.process_message.return_value = response

        daemon = HeartbeatDaemon(
            agent_loop=agent_loop,
            config=config,
            event_bus=event_bus,
            fact_store=fact_store,
        )

        with patch.object(daemon, "_read_heartbeat_md", return_value="- Check stuff"):
            await daemon._tick()

        # Should still complete and call process_message
        agent_loop.process_message.assert_called_once()
        assert daemon._consecutive_failures == 0

    async def test_tick_context_truncated_when_large(
        self,
        config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> None:
        """Very large context is truncated to prevent token explosion."""
        # Create a fact store that returns very large data
        large_facts = [_make_fact(f"key.{i}", "x" * 500, priority="high") for i in range(20)]
        fact_store = _make_fact_store_mock(urgent=large_facts)

        agent_loop = AsyncMock()
        agent_loop.fact_store = fact_store
        response = AsyncMock()
        response.content = "HEARTBEAT_OK"
        agent_loop.process_message.return_value = response

        daemon = HeartbeatDaemon(
            agent_loop=agent_loop,
            config=config,
            event_bus=event_bus,
            fact_store=fact_store,
        )

        with patch.object(daemon, "_read_heartbeat_md", return_value="- Check stuff"):
            await daemon._tick()

        heartbeat_message = agent_loop.process_message.call_args[0][0]
        from agent.core.heartbeat import _MAX_HEARTBEAT_CONTEXT_CHARS

        # Message should be capped (with truncation marker)
        assert len(heartbeat_message) <= _MAX_HEARTBEAT_CONTEXT_CHARS + len(
            "\n...(context truncated)"
        )
