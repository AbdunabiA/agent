"""Tests for FactExtractor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.session import Message, Session
from agent.memory.extraction import FactExtractor, _parse_facts_json


class TestParseFactsJson:
    """Tests for _parse_facts_json helper."""

    def test_valid_json_array(self) -> None:
        """Parse a valid JSON array."""
        text = '[{"key": "user.name", "value": "Alice", "category": "user"}]'
        result = _parse_facts_json(text)
        assert len(result) == 1
        assert result[0]["key"] == "user.name"

    def test_markdown_fenced_json(self) -> None:
        """Parse JSON wrapped in markdown code fences."""
        text = '```json\n[{"key": "user.name", "value": "Bob", "category": "user"}]\n```'
        result = _parse_facts_json(text)
        assert len(result) == 1
        assert result[0]["value"] == "Bob"

    def test_malformed_json_returns_empty(self) -> None:
        """Malformed JSON returns empty list."""
        result = _parse_facts_json("not json at all")
        assert result == []

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        assert _parse_facts_json("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns empty list."""
        assert _parse_facts_json("   ") == []

    def test_json_object_not_array(self) -> None:
        """A JSON object (not array) returns empty list."""
        result = _parse_facts_json('{"key": "value"}')
        assert result == []

    def test_json_array_in_surrounding_text(self) -> None:
        """Extracts JSON array from surrounding text."""
        text = (
            'Here are the facts:\n'
            '[{"key": "user.name", "value": "Eve", "category": "user"}]'
            '\nDone.'
        )
        result = _parse_facts_json(text)
        assert len(result) == 1

    def test_empty_array(self) -> None:
        """Parse an empty JSON array."""
        assert _parse_facts_json("[]") == []


class TestFactExtractor:
    """Tests for FactExtractor class."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.completion = AsyncMock()
        return llm

    @pytest.fixture
    def mock_fact_store(self) -> MagicMock:
        """Create a mock FactStore."""
        store = MagicMock()
        store.set = AsyncMock()
        return store

    @pytest.mark.asyncio
    async def test_extract_from_messages(
        self, mock_llm: MagicMock, mock_fact_store: MagicMock
    ) -> None:
        """Extract facts from messages successfully."""
        facts_json = json.dumps([
            {"key": "user.name", "value": "Alice", "category": "user"},
            {"key": "preference.language", "value": "Python", "category": "preference"},
        ])
        mock_llm.completion.return_value = MagicMock(content=facts_json)

        # Mock fact_store.set to return a Fact-like object
        mock_fact = MagicMock()
        mock_fact_store.set.return_value = mock_fact

        extractor = FactExtractor(mock_llm, mock_fact_store)
        messages = [
            {"role": "user", "content": "My name is Alice and I like Python"},
        ]

        result = await extractor.extract_from_messages(messages)

        assert len(result) == 2
        assert mock_fact_store.set.call_count == 2

        # Check that source="extracted" and confidence=0.8 were used
        call_kwargs = mock_fact_store.set.call_args_list[0].kwargs
        assert call_kwargs["source"] == "extracted"
        assert call_kwargs["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(
        self, mock_llm: MagicMock, mock_fact_store: MagicMock
    ) -> None:
        """Disabled extractor returns empty list."""
        extractor = FactExtractor(mock_llm, mock_fact_store, enabled=False)
        result = await extractor.extract_from_messages(
            [{"role": "user", "content": "hello"}]
        )
        assert result == []
        mock_llm.completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(
        self, mock_llm: MagicMock, mock_fact_store: MagicMock
    ) -> None:
        """LLM failure returns empty list."""
        mock_llm.completion.side_effect = Exception("API error")

        extractor = FactExtractor(mock_llm, mock_fact_store)
        result = await extractor.extract_from_messages(
            [{"role": "user", "content": "test"}]
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_from_session_last_pair(
        self, mock_llm: MagicMock, mock_fact_store: MagicMock
    ) -> None:
        """Only the last 2 messages are processed."""
        mock_llm.completion.return_value = MagicMock(content="[]")

        extractor = FactExtractor(mock_llm, mock_fact_store)
        session = Session()
        session.add_message(Message(role="user", content="old message"))
        session.add_message(Message(role="assistant", content="old reply"))
        session.add_message(Message(role="user", content="My name is Bob"))
        session.add_message(Message(role="assistant", content="Nice to meet you, Bob"))

        await extractor.extract_from_session(session)

        # LLM should have been called with only the last 2 messages
        call_args = mock_llm.completion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "My name is Bob" in prompt
        assert "old message" not in prompt

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty(
        self, mock_llm: MagicMock, mock_fact_store: MagicMock
    ) -> None:
        """Empty messages list returns empty."""
        extractor = FactExtractor(mock_llm, mock_fact_store)
        result = await extractor.extract_from_messages([])
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_empty_key_value(
        self, mock_llm: MagicMock, mock_fact_store: MagicMock
    ) -> None:
        """Facts with empty key or value are skipped."""
        facts_json = json.dumps([
            {"key": "", "value": "Alice", "category": "user"},
            {"key": "user.name", "value": "", "category": "user"},
            {"key": "valid.key", "value": "valid value", "category": "general"},
        ])
        mock_llm.completion.return_value = MagicMock(content=facts_json)
        mock_fact_store.set.return_value = MagicMock()

        extractor = FactExtractor(mock_llm, mock_fact_store)
        result = await extractor.extract_from_messages(
            [{"role": "user", "content": "test"}]
        )

        assert len(result) == 1
        assert mock_fact_store.set.call_count == 1
