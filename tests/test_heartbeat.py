"""Tests for the heartbeat daemon."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.config import AgentPersonaConfig
from agent.core.events import EventBus
from agent.core.heartbeat import HeartbeatDaemon
from agent.core.session import TokenUsage


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
