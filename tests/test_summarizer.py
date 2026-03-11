"""Tests for ConversationSummarizer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.session import Message, Session
from agent.memory.summarizer import ConversationSummarizer


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.completion = AsyncMock()
    return llm


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock VectorStore."""
    store = MagicMock()
    store.add = AsyncMock(return_value="doc-id")
    return store


class TestConversationSummarizer:
    """Tests for ConversationSummarizer class."""

    @pytest.mark.asyncio
    async def test_summarize_session(
        self, mock_llm: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """Summarize a session and store in vector store."""
        mock_llm.completion.return_value = MagicMock(
            content="User discussed deploying a Python project with Docker."
        )

        summarizer = ConversationSummarizer(mock_llm, mock_vector_store)
        session = Session()
        session.add_message(Message(role="user", content="How do I deploy with Docker?"))
        session.add_message(
            Message(role="assistant", content="You can create a Dockerfile...")
        )

        result = await summarizer.summarize_session(session)

        assert result == "User discussed deploying a Python project with Docker."
        mock_vector_store.add.assert_called_once()

        # Check metadata
        call_kwargs = mock_vector_store.add.call_args.kwargs
        assert call_kwargs["metadata"]["session_id"] == session.id
        assert call_kwargs["metadata"]["type"] == "summary"
        assert call_kwargs["metadata"]["message_count"] == 2

    @pytest.mark.asyncio
    async def test_summarize_empty_session(
        self, mock_llm: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """Empty session returns empty string."""
        summarizer = ConversationSummarizer(mock_llm, mock_vector_store)
        session = Session()

        result = await summarizer.summarize_session(session)

        assert result == ""
        mock_llm.completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_if_needed_below_threshold(
        self, mock_llm: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """Below threshold returns None."""
        summarizer = ConversationSummarizer(mock_llm, mock_vector_store)
        session = Session()
        for i in range(5):
            session.add_message(Message(role="user", content=f"msg {i}"))

        result = await summarizer.summarize_if_needed(session, threshold=20)

        assert result is None
        mock_llm.completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_if_needed_above_threshold(
        self, mock_llm: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """Above threshold triggers summarization."""
        mock_llm.completion.return_value = MagicMock(
            content="Long conversation summary."
        )

        summarizer = ConversationSummarizer(mock_llm, mock_vector_store)
        session = Session()
        for i in range(25):
            session.add_message(Message(role="user", content=f"msg {i}"))

        result = await summarizer.summarize_if_needed(session, threshold=20)

        assert result == "Long conversation summary."
        mock_llm.completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_propagates(
        self, mock_llm: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """LLM failure propagates as exception."""
        mock_llm.completion.side_effect = Exception("API error")

        summarizer = ConversationSummarizer(mock_llm, mock_vector_store)
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        with pytest.raises(Exception, match="API error"):
            await summarizer.summarize_session(session)

    @pytest.mark.asyncio
    async def test_only_user_assistant_messages(
        self, mock_llm: MagicMock, mock_vector_store: MagicMock
    ) -> None:
        """Only user and assistant messages are included in summary prompt."""
        mock_llm.completion.return_value = MagicMock(content="Summary")

        summarizer = ConversationSummarizer(mock_llm, mock_vector_store)
        session = Session()
        session.add_message(Message(role="system", content="System message"))
        session.add_message(Message(role="user", content="Hello"))
        session.add_message(Message(role="tool", content="tool output"))
        session.add_message(Message(role="assistant", content="Hi there"))

        await summarizer.summarize_session(session)

        prompt = mock_llm.completion.call_args.kwargs["messages"][0]["content"]
        assert "System message" not in prompt
        assert "tool output" not in prompt
        assert "Hello" in prompt
        assert "Hi there" in prompt
