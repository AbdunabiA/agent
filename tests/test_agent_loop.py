"""Tests for the main agent reasoning loop (mocked LLM)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.config import AgentPersonaConfig, ModelsConfig
from agent.core.agent_loop import AgentLoop
from agent.core.events import EventBus, Events
from agent.core.session import Session, TokenUsage
from agent.llm.provider import LLMProvider, LLMResponse


def _make_llm_response(content: str = "Hello!") -> LLMResponse:
    """Create a mock LLMResponse."""
    return LLMResponse(
        content=content,
        model="gpt-4o-mini",
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        finish_reason="stop",
    )


class TestAgentLoop:
    """Test AgentLoop functionality."""

    @pytest.fixture
    def persona_config(self) -> AgentPersonaConfig:
        return AgentPersonaConfig(name="TestAgent")

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def mock_llm(self) -> LLMProvider:
        config = ModelsConfig(default="gpt-4o-mini")
        provider = LLMProvider(config)
        return provider

    @pytest.fixture
    def agent_loop(
        self,
        mock_llm: LLMProvider,
        persona_config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> AgentLoop:
        return AgentLoop(llm=mock_llm, config=persona_config, event_bus=event_bus)

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_process_message_returns_response(
        self,
        mock_acompletion: AsyncMock,
        agent_loop: AgentLoop,
    ) -> None:
        mock_acompletion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hi there!", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )

        session = Session()
        response = await agent_loop.process_message("Hello", session)

        assert response.content == "Hi there!"
        assert response.model == "gpt-4o-mini"

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_session_history_maintained(
        self,
        mock_acompletion: AsyncMock,
        agent_loop: AgentLoop,
    ) -> None:
        mock_acompletion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Response 1", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        session = Session()
        await agent_loop.process_message("Message 1", session)

        # Session should have user + assistant messages
        assert session.message_count == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Message 1"
        assert session.messages[1].role == "assistant"
        assert session.messages[1].content == "Response 1"

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_system_prompt_included(
        self,
        mock_acompletion: AsyncMock,
        agent_loop: AgentLoop,
    ) -> None:
        mock_acompletion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="OK", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

        session = Session()
        await agent_loop.process_message("Hi", session)

        call_args = mock_acompletion.call_args
        messages = call_args[1]["messages"]

        # First message should be the system prompt
        assert messages[0]["role"] == "system"
        assert "TestAgent" in messages[0]["content"]

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_events_emitted(
        self,
        mock_acompletion: AsyncMock,
        agent_loop: AgentLoop,
        event_bus: EventBus,
    ) -> None:
        mock_acompletion.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hey", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        incoming_events: list[object] = []
        outgoing_events: list[object] = []

        async def on_incoming(data: object) -> None:
            incoming_events.append(data)

        async def on_outgoing(data: object) -> None:
            outgoing_events.append(data)

        event_bus.on(Events.MESSAGE_INCOMING, on_incoming)
        event_bus.on(Events.MESSAGE_OUTGOING, on_outgoing)

        session = Session()
        await agent_loop.process_message("Hello", session)

        assert len(incoming_events) == 1
        assert len(outgoing_events) == 1

    @patch("agent.llm.provider.litellm.acompletion", new_callable=AsyncMock)
    async def test_llm_failure_propagates(
        self,
        mock_acompletion: AsyncMock,
        agent_loop: AgentLoop,
    ) -> None:
        mock_acompletion.side_effect = Exception("LLM Error")

        session = Session()
        with pytest.raises(Exception, match="LLM Error"):
            await agent_loop.process_message("Hi", session)
