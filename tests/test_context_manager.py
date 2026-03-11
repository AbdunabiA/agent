"""Tests for the context window manager."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.core.context import (
    _DEFAULT_LIMIT,
    build_messages,
    estimate_tokens,
    get_model_limit,
)
from agent.core.session import Message, Session


class TestGetModelLimit:
    """Test model limit detection."""

    def test_claude_models(self) -> None:
        assert get_model_limit("claude-3-5-sonnet-20241022") == 200_000
        assert get_model_limit("claude-sonnet-4-5-20250929") == 200_000
        assert get_model_limit("claude-opus-4-20250514") == 200_000
        assert get_model_limit("claude-haiku-4-5-20251001") == 200_000

    def test_gpt4o(self) -> None:
        assert get_model_limit("gpt-4o") == 128_000
        assert get_model_limit("gpt-4o-mini") == 128_000

    def test_gemini(self) -> None:
        assert get_model_limit("gemini-1.5-pro") == 1_000_000
        assert get_model_limit("gemini-pro") == 32_000

    def test_unknown_model_returns_default(self) -> None:
        assert get_model_limit("ollama/llama3") == _DEFAULT_LIMIT
        assert get_model_limit("some-custom-model") == _DEFAULT_LIMIT


class TestEstimateTokens:
    """Test token estimation."""

    def test_roughly_4_chars_per_token(self) -> None:
        text = "a" * 400
        tokens = estimate_tokens(text)
        assert tokens == 100

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        # "hello" = 5 chars -> 1 token
        assert estimate_tokens("hello") == 1


class TestBuildMessages:
    """Test message building with context window constraints."""

    def test_system_prompt_always_included(self) -> None:
        session = Session()
        result = build_messages(session, "You are helpful.")
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_plan_appended_to_system(self) -> None:
        session = Session()
        plan = MagicMock()
        plan.to_context_string.return_value = "Step 1: Do this"

        result = build_messages(session, "System prompt.", plan=plan)
        assert "ACTIVE PLAN" in result[0]["content"]
        assert "Step 1: Do this" in result[0]["content"]

    def test_history_included(self) -> None:
        session = Session()
        session.add_message(Message(role="user", content="Hello"))
        session.add_message(Message(role="assistant", content="Hi there!"))

        result = build_messages(session, "System.")
        assert len(result) == 3  # system + 2 history
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_history_trimmed_to_fit_budget(self) -> None:
        session = Session()
        # Add many long messages that exceed context window
        for i in range(100):
            session.add_message(
                Message(role="user", content=f"Message {i}: " + "x" * 1000)
            )
            session.add_message(
                Message(role="assistant", content=f"Reply {i}: " + "y" * 1000)
            )

        # Use a model with small context window
        result = build_messages(
            session, "System.", model="some-tiny-model"
        )

        # Should have fewer messages than session history
        assert len(result) < 201  # system + 200 history messages
        # System prompt is always first
        assert result[0]["role"] == "system"
        # Most recent messages should be kept (newest-first trimming)
        if len(result) > 1:
            last = result[-1]
            # Last message should be one of the most recent
            assert "Reply 99" in last["content"] or "Message 99" in last["content"]

    def test_response_reservation_applied(self) -> None:
        session = Session()
        # The budget should be limit - response_reservation - system_tokens
        # With default model (8000 tokens) and a system prompt of ~4 tokens,
        # budget = 8000 - 4096 - ~4 = ~3900 tokens = ~15600 chars of history
        for _ in range(20):
            session.add_message(
                Message(role="user", content="x" * 2000)
            )

        result = build_messages(session, "Hi.", model="unknown-model")
        # With 8000 limit, 4096 reserved, only ~3900 tokens of history fits
        # Each message is ~500 tokens, so roughly 7-8 messages should fit
        history_count = len(result) - 1  # minus system
        assert history_count < 20

    def test_tool_schemas_budget_accounted(self) -> None:
        session = Session()
        for _ in range(20):
            session.add_message(Message(role="user", content="x" * 2000))

        # Large tool schemas eat into budget
        schemas = [{"name": f"tool_{i}", "description": "d" * 500} for i in range(10)]

        result_with_tools = build_messages(
            session, "System.", tool_schemas=schemas, model="unknown-model"
        )
        result_without = build_messages(
            session, "System.", model="unknown-model"
        )

        # With tool schemas, fewer history messages should fit
        assert len(result_with_tools) <= len(result_without)
