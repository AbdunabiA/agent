"""Tests for the conversation session manager."""

from __future__ import annotations

from typing import Any

from agent.core.session import Message, Session, TokenUsage, content_as_text


class TestSession:
    """Test Session functionality."""

    def test_create_session_auto_id(self) -> None:
        session = Session()
        assert session.id is not None
        assert len(session.id) > 0

    def test_create_session_custom_id(self) -> None:
        session = Session(session_id="test-123")
        assert session.id == "test-123"

    def test_add_message(self) -> None:
        session = Session()
        msg = Message(role="user", content="Hello")
        session.add_message(msg)

        assert session.message_count == 1
        assert session.messages[0].content == "Hello"
        assert session.messages[0].role == "user"

    def test_get_history(self) -> None:
        session = Session()
        session.add_message(Message(role="user", content="Hi"))
        session.add_message(Message(role="assistant", content="Hello!"))

        history = session.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "Hello!"}

    def test_history_limit(self) -> None:
        session = Session()
        for i in range(10):
            session.add_message(Message(role="user", content=f"Message {i}"))

        history = session.get_history(max_messages=3)
        assert len(history) == 3
        assert history[0]["content"] == "Message 7"
        assert history[2]["content"] == "Message 9"

    def test_clear(self) -> None:
        session = Session()
        session.add_message(Message(role="user", content="Hi"))
        session.add_message(Message(role="assistant", content="Hello!"))

        session.clear()
        assert session.message_count == 0
        assert session.messages == []

    def test_total_tokens(self) -> None:
        session = Session()
        session.add_message(
            Message(
                role="user",
                content="Hi",
                usage=TokenUsage(input_tokens=5, output_tokens=0, total_tokens=5),
            )
        )
        session.add_message(
            Message(
                role="assistant",
                content="Hello!",
                usage=TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15),
            )
        )

        assert session.total_tokens == 20

    def test_total_tokens_no_usage(self) -> None:
        session = Session()
        session.add_message(Message(role="user", content="Hi"))
        assert session.total_tokens == 0

    def test_message_count(self) -> None:
        session = Session()
        assert session.message_count == 0

        session.add_message(Message(role="user", content="One"))
        assert session.message_count == 1

        session.add_message(Message(role="assistant", content="Two"))
        assert session.message_count == 2

    def test_history_format_matches_llm_api(self) -> None:
        session = Session()
        session.add_message(Message(role="user", content="What is 2+2?"))
        session.add_message(Message(role="assistant", content="4"))

        history = session.get_history()
        for entry in history:
            assert "role" in entry
            assert "content" in entry
            assert isinstance(entry["role"], str)
            assert isinstance(entry["content"], (str, list))

    def test_updated_at_changes(self) -> None:
        session = Session()
        created = session.updated_at

        session.add_message(Message(role="user", content="test"))
        assert session.updated_at >= created


class TestMultimodalContent:
    """Test multimodal (list) content in messages."""

    def test_message_accepts_list_content(self) -> None:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Screenshot captured"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        msg = Message(role="tool", content=content)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_get_history_passes_list_content_through(self) -> None:
        session = Session()
        multimodal: list[dict[str, Any]] = [
            {"type": "text", "text": "Screenshot info"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        session.add_message(Message(role="tool", content=multimodal, tool_call_id="tc1"))

        history = session.get_history()
        assert len(history) == 1
        assert isinstance(history[0]["content"], list)
        assert history[0]["content"][0]["type"] == "text"

    def test_content_as_text_with_string(self) -> None:
        assert content_as_text("hello world") == "hello world"

    def test_content_as_text_with_multimodal(self) -> None:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Screenshot captured"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "more text"},
        ]
        result = content_as_text(content)
        assert "Screenshot captured" in result
        assert "more text" in result
        assert "base64" not in result

    def test_content_as_text_empty_list(self) -> None:
        assert content_as_text([]) == ""


class TestTokenUsage:
    """Test TokenUsage functionality."""

    def test_estimated_cost(self) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = usage.estimated_cost
        assert cost > 0
        assert isinstance(cost, float)

    def test_zero_tokens_zero_cost(self) -> None:
        usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        assert usage.estimated_cost == 0.0


class TestMessage:
    """Test Message dataclass."""

    def test_default_timestamp(self) -> None:
        msg = Message(role="user", content="test")
        assert msg.timestamp is not None

    def test_optional_fields_default_none(self) -> None:
        msg = Message(role="user", content="test")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.model is None
        assert msg.usage is None
