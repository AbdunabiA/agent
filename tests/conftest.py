"""Shared test fixtures."""

from __future__ import annotations

import pytest

from agent.config import (
    AgentConfig,
    AgentPersonaConfig,
    LoggingConfig,
    ModelsConfig,
)
from agent.core.events import EventBus


@pytest.fixture
def default_config() -> AgentConfig:
    """Create a default config for testing."""
    return AgentConfig()


@pytest.fixture
def event_bus() -> EventBus:
    """Create a fresh event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_config() -> AgentConfig:
    """Config with test values."""
    return AgentConfig(
        agent=AgentPersonaConfig(name="TestAgent"),
        models=ModelsConfig(default="gpt-4o-mini"),
        logging=LoggingConfig(level="DEBUG", format="console"),
    )
