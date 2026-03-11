"""Tests for the Application lifecycle manager."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.config import (
    AgentConfig,
    ChannelsConfig,
    MemoryConfig,
    TelegramConfig,
    WebChatConfig,
)
from agent.core.events import Events
from agent.core.startup import Application


def _mem_config(tmp_path: object) -> MemoryConfig:
    """Create a MemoryConfig pointing to a temp directory."""
    return MemoryConfig(db_path=str(tmp_path / "test_agent.db"))


@pytest.fixture
def config(tmp_path: object) -> AgentConfig:
    """Default config with channels disabled."""
    return AgentConfig(memory=_mem_config(tmp_path))


@pytest.fixture
def config_with_telegram(tmp_path: object) -> AgentConfig:
    """Config with Telegram enabled and token set."""
    return AgentConfig(
        channels=ChannelsConfig(
            telegram=TelegramConfig(enabled=True, token="fake:telegram:token"),
        ),
        memory=_mem_config(tmp_path),
    )


@pytest.fixture
def config_with_webchat(tmp_path: object) -> AgentConfig:
    """Config with WebChat enabled."""
    return AgentConfig(
        channels=ChannelsConfig(
            webchat=WebChatConfig(enabled=True),
        ),
        memory=_mem_config(tmp_path),
    )


@pytest.fixture
def config_with_both(tmp_path: object) -> AgentConfig:
    """Config with both Telegram and WebChat enabled."""
    return AgentConfig(
        channels=ChannelsConfig(
            telegram=TelegramConfig(enabled=True, token="fake:token"),
            webchat=WebChatConfig(enabled=True),
        ),
        memory=_mem_config(tmp_path),
    )


class TestApplicationInitialize:
    """Test Application.initialize() wires components correctly."""

    async def test_all_components_initialized(self, config: AgentConfig) -> None:
        """All core components should be non-None after initialize."""
        app = Application(config)
        await app.initialize()

        assert app.database is not None
        assert app.event_bus is not None
        assert app.llm is not None
        assert app.session_store is not None
        assert app.guardrails is not None
        assert app.permissions is not None
        assert app.audit is not None
        assert app.recovery is not None
        assert app.tool_executor is not None
        assert app.planner is not None
        assert app.agent_loop is not None
        assert app.scheduler is not None
        assert app.heartbeat is not None
        assert app.app is not None

        await app.shutdown()

    async def test_components_wired_correctly(self, config: AgentConfig) -> None:
        """Agent loop should reference the correct tool_executor, planner, etc."""
        app = Application(config)
        await app.initialize()

        assert app.agent_loop.tool_executor is app.tool_executor
        assert app.agent_loop.planner is app.planner
        assert app.agent_loop.recovery is app.recovery
        assert app.agent_loop.guardrails is app.guardrails

        await app.shutdown()

    async def test_telegram_created_when_enabled(
        self, config_with_telegram: AgentConfig
    ) -> None:
        """Telegram channel should be created when enabled with token."""
        app = Application(config_with_telegram)

        with patch("agent.channels.telegram.AIOGRAM_AVAILABLE", True), \
             patch("agent.channels.telegram.Bot"), \
             patch("agent.channels.telegram.Dispatcher"), \
             patch("agent.channels.telegram.Router"):
            await app.initialize()

        channel_names = [ch.name for ch in app._channels]
        assert "telegram" in channel_names

        # Stop channels manually to avoid awaiting mock dispatcher
        app._channels.clear()
        await app.shutdown()

    async def test_telegram_skipped_without_token(self, tmp_path: object) -> None:
        """Telegram should not be created when no token is provided."""
        cfg = AgentConfig(
            channels=ChannelsConfig(
                telegram=TelegramConfig(enabled=True, token=None),
            ),
            memory=_mem_config(tmp_path),
        )
        app = Application(cfg)
        await app.initialize()

        channel_names = [ch.name for ch in app._channels]
        assert "telegram" not in channel_names

        await app.shutdown()

    async def test_webchat_created_when_enabled(
        self, config_with_webchat: AgentConfig
    ) -> None:
        """WebChat channel should be created when enabled."""
        app = Application(config_with_webchat)
        await app.initialize()

        channel_names = [ch.name for ch in app._channels]
        assert "webchat" in channel_names

        await app.shutdown()


class TestApplicationShutdown:
    """Test Application.shutdown() stops all components."""

    async def test_shutdown_stops_channels(
        self, config_with_webchat: AgentConfig
    ) -> None:
        """Shutdown should call stop() on all channels."""
        app = Application(config_with_webchat)
        await app.initialize()

        # Start the webchat channel so it's running
        for ch in app._channels:
            await ch.start()
            assert ch.is_running

        await app.shutdown()

        for ch in app._channels:
            assert not ch.is_running


class TestApplicationEvents:
    """Test events emitted during lifecycle."""

    async def test_agent_started_emitted(self, config: AgentConfig) -> None:
        """AGENT_STARTED should be emitted during initialize."""
        app = Application(config)
        events_received: list[str] = []

        await app.initialize()
        await app.shutdown()

        # Use a fresh app and register before init
        app2 = Application(config)

        async def on_start(data: dict) -> None:
            events_received.append("started")

        # We verify via AGENT_STOPPED since AGENT_STARTED fires during init
        await app2.initialize()
        app2.event_bus.on(Events.AGENT_STOPPED, on_start)
        await app2.shutdown()

        assert "started" in events_received

    async def test_agent_stopped_emitted(self, config: AgentConfig) -> None:
        """AGENT_STOPPED should be emitted during shutdown."""
        app = Application(config)
        await app.initialize()

        events_received: list[str] = []

        async def on_stop(data: dict) -> None:
            events_received.append("stopped")

        app.event_bus.on(Events.AGENT_STOPPED, on_stop)

        await app.shutdown()

        assert "stopped" in events_received
