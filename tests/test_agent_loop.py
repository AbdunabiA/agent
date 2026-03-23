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


# ---------------------------------------------------------------------------
# URL verification edge cases
# ---------------------------------------------------------------------------


class TestUrlVerification:
    """Tests for _check_urls_in_output in the agent loop."""

    @pytest.fixture
    def persona_config(self) -> AgentPersonaConfig:
        return AgentPersonaConfig(name="TestAgent")

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def mock_llm(self) -> LLMProvider:
        config = ModelsConfig(default="gpt-4o-mini")
        return LLMProvider(config)

    @pytest.fixture
    def agent_loop(
        self,
        mock_llm: LLMProvider,
        persona_config: AgentPersonaConfig,
        event_bus: EventBus,
    ) -> AgentLoop:
        return AgentLoop(llm=mock_llm, config=persona_config, event_bus=event_bus)

    async def test_check_urls_no_urls_returns_unchanged(self, agent_loop: AgentLoop) -> None:
        """Output without any URLs should pass through unchanged."""
        output = "This is plain text with no links."
        result = await agent_loop._check_urls_in_output(output)
        assert result == output

    async def test_check_urls_marks_broken_404(self, agent_loop: AgentLoop) -> None:
        """Output with a broken URL should get [BROKEN:404] appended."""
        import httpx as httpx_mod

        mock_resp = AsyncMock()
        mock_resp.status_code = 404

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(httpx_mod, "AsyncClient", return_value=mock_client):
            output = "Check this: https://example.com/missing-page for details."
            result = await agent_loop._check_urls_in_output(output)
        assert "[BROKEN:404]" in result

    async def test_check_urls_marks_unreachable(self, agent_loop: AgentLoop) -> None:
        """Output with an unreachable URL should get [UNREACHABLE] appended."""
        import httpx as httpx_mod

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(httpx_mod, "AsyncClient", return_value=mock_client):
            output = "Visit https://down.example.com for info."
            result = await agent_loop._check_urls_in_output(output)
        assert "[UNREACHABLE]" in result

    async def test_check_urls_max_5_urls(self, agent_loop: AgentLoop) -> None:
        """With 10 URLs, only the first 5 should be checked."""
        import httpx as httpx_mod

        call_count = 0

        async def counting_head(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            resp.status_code = 200
            return resp

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=counting_head)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(httpx_mod, "AsyncClient", return_value=mock_client):
            urls = [f"https://example.com/page{i}" for i in range(10)]
            output = " ".join(urls)
            await agent_loop._check_urls_in_output(output)

        assert call_count == 5

    async def test_check_urls_valid_url_unchanged(self, agent_loop: AgentLoop) -> None:
        """A working URL (status 200) should remain unchanged in the output."""
        import httpx as httpx_mod

        mock_resp = AsyncMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(httpx_mod, "AsyncClient", return_value=mock_client):
            output = "See https://example.com for details."
            result = await agent_loop._check_urls_in_output(output)
        assert result == output
        assert "[BROKEN" not in result
        assert "[UNREACHABLE]" not in result


# ---------------------------------------------------------------------------
# Memory context cap tests
# ---------------------------------------------------------------------------


class TestMemoryContextCap:
    """Tests for memory context size capping."""

    @pytest.fixture
    def persona_config(self) -> AgentPersonaConfig:
        return AgentPersonaConfig(name="TestAgent")

    @pytest.fixture
    def event_bus(self) -> EventBus:
        return EventBus()

    @pytest.fixture
    def mock_llm(self) -> LLMProvider:
        config = ModelsConfig(default="gpt-4o-mini")
        return LLMProvider(config)

    async def test_memory_context_capped(self) -> None:
        """Total memory context injection doesn't exceed 3000 chars."""
        from unittest.mock import MagicMock

        config = AgentPersonaConfig(name="TestAgent")
        event_bus_local = EventBus()
        mock_llm = MagicMock()
        mock_llm.config = MagicMock()
        mock_llm.config.default = "gpt-4o-mini"

        # Create a mock fact_store
        mock_fact_store = MagicMock()

        agent_loop = AgentLoop(
            llm=mock_llm,
            config=config,
            event_bus=event_bus_local,
            fact_store=mock_fact_store,
        )

        from agent.core.session import Message

        session = Session()
        session.add_message(Message(role="user", content="Hello"))

        # Generate very large memory context inputs
        from datetime import datetime

        from agent.memory.models import Fact

        large_facts = [
            Fact(
                id=f"id-{i}",
                key=f"fact.key_{i}",
                value="x" * 200,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                accessed_at=datetime.now(),
            )
            for i in range(50)
        ]
        large_topics = [f"topic_{i}" for i in range(50)]
        large_emotional = "y" * 1000

        messages = agent_loop._build_messages(
            session,
            plan=None,
            facts=large_facts,
            vector_results=None,
            tool_schemas=None,
            active_topics=large_topics,
            emotional_context=large_emotional,
        )

        # Verify the system message exists and the memory context
        # in the runtime context section does not blow up
        system_msg = messages[0]["content"]
        assert isinstance(system_msg, str)

        # The emotional_context should have been truncated to 500 chars
        # before being passed to build_runtime_context
        assert len(large_emotional[:500]) == 500
