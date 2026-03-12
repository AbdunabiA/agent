"""Tests for context assembly with memory injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.core.context import _IMAGE_TOKEN_ESTIMATE, build_messages, estimate_tokens
from agent.core.session import Message, Session


@dataclass
class FakeFact:
    """Minimal fact-like object for testing."""

    key: str
    value: str
    id: str = "test-id"
    category: str = "general"
    confidence: float = 1.0
    source: str = "test"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0


@dataclass
class FakeVectorResult:
    """Minimal vector result-like object for testing."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.2

    @property
    def score(self) -> float:
        """Similarity score."""
        return max(0.0, 1.0 - self.distance)


class TestContextAssemblyWithMemory:
    """Tests for build_messages with facts and vector_results."""

    def test_facts_in_system_prompt(self) -> None:
        """Facts are injected into the system prompt."""
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        facts = [
            FakeFact(key="user.name", value="Alice"),
            FakeFact(key="user.language", value="Python"),
        ]

        messages = build_messages(
            session=session,
            system_prompt="You are a helpful assistant.",
            facts=facts,
        )

        system_content = messages[0]["content"]
        assert "KNOWN FACTS ABOUT THE USER:" in system_content
        assert "user.name: Alice" in system_content
        assert "user.language: Python" in system_content

    def test_vector_results_in_system_prompt(self) -> None:
        """Vector results are injected into the system prompt."""
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        vectors = [
            FakeVectorResult(
                id="v1",
                text="User discussed deploying with Docker",
                distance=0.2,
            ),
            FakeVectorResult(
                id="v2",
                text="User prefers dark mode",
                distance=0.4,
            ),
        ]

        messages = build_messages(
            session=session,
            system_prompt="You are a helpful assistant.",
            vector_results=vectors,
        )

        system_content = messages[0]["content"]
        assert "RELATED PAST CONVERSATIONS:" in system_content
        assert "User discussed deploying with Docker" in system_content
        assert "Relevance: 80%" in system_content
        assert "Relevance: 60%" in system_content

    def test_both_facts_and_vectors(self) -> None:
        """Both facts and vectors are present in system prompt."""
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        facts = [FakeFact(key="user.name", value="Bob")]
        vectors = [
            FakeVectorResult(id="v1", text="Past discussion", distance=0.3)
        ]

        messages = build_messages(
            session=session,
            system_prompt="Base prompt.",
            facts=facts,
            vector_results=vectors,
        )

        system_content = messages[0]["content"]
        assert "KNOWN FACTS ABOUT THE USER:" in system_content
        assert "RELATED PAST CONVERSATIONS:" in system_content
        assert "user.name: Bob" in system_content
        assert "Past discussion" in system_content

    def test_no_facts_or_vectors_backward_compat(self) -> None:
        """Without facts/vectors, system prompt is unchanged (backward compat)."""
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        messages = build_messages(
            session=session,
            system_prompt="You are helpful.",
        )

        system_content = messages[0]["content"]
        assert system_content == "You are helpful."
        assert "KNOWN FACTS" not in system_content
        assert "RELATED PAST" not in system_content

    def test_empty_facts_list(self) -> None:
        """Empty facts list doesn't add section."""
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        messages = build_messages(
            session=session,
            system_prompt="Base prompt.",
            facts=[],
        )

        system_content = messages[0]["content"]
        assert "KNOWN FACTS" not in system_content

    def test_empty_vectors_list(self) -> None:
        """Empty vectors list doesn't add section."""
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        messages = build_messages(
            session=session,
            system_prompt="Base prompt.",
            vector_results=[],
        )

        system_content = messages[0]["content"]
        assert "RELATED PAST" not in system_content

    def test_token_budget_includes_memory(self) -> None:
        """Memory content is counted in the token budget.

        The system prompt grows when facts/vectors are added,
        which means less room for history. We verify that history
        is properly trimmed.
        """
        session = Session()
        # Add many messages to fill history
        for i in range(100):
            session.add_message(
                Message(role="user", content=f"Message {i} " + "x" * 200)
            )

        # Build with very small model limit to force trimming
        facts = [FakeFact(key=f"fact.{i}", value="v" * 100) for i in range(10)]

        messages_with_facts = build_messages(
            session=session,
            system_prompt="Base prompt.",
            facts=facts,
            model="gpt-4",  # 8192 token limit
        )
        messages_without_facts = build_messages(
            session=session,
            system_prompt="Base prompt.",
            model="gpt-4",
        )

        # With facts, fewer history messages should fit
        assert len(messages_with_facts) <= len(messages_without_facts)

    def test_multimodal_content_token_estimation(self) -> None:
        """Multimodal messages with images use ~1600 tokens per image."""
        session = Session()
        session.add_message(Message(role="user", content="take a screenshot"))
        # Simulate a tool result with image content
        multimodal_content: list[dict[str, Any]] = [
            {"type": "text", "text": "Screenshot captured: 1920x1080"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        session.add_message(
            Message(role="tool", content=multimodal_content, tool_call_id="tc1")
        )
        # Also add matching assistant message with tool_calls so sanitizer keeps it
        from agent.core.session import ToolCall

        session.add_message(
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc1", name="screen_capture", arguments={})],
            )
        )

        messages = build_messages(
            session=session,
            system_prompt="You are helpful.",
            model="claude-3-5-sonnet",  # 200k limit, won't trim
        )

        # The multimodal message should be included
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert isinstance(tool_msgs[0]["content"], list)

    def test_image_token_constant(self) -> None:
        """Image token estimate should be a reasonable number."""
        assert _IMAGE_TOKEN_ESTIMATE == 1600
        # A text-only message should use regular estimation
        assert estimate_tokens("hello world") == len("hello world") // 4

    def test_facts_with_plan(self) -> None:
        """Facts and plan both appear in system prompt."""
        session = Session()
        session.add_message(Message(role="user", content="hello"))

        class FakePlan:
            def to_context_string(self) -> str:
                return "Step 1: Do thing"

        facts = [FakeFact(key="user.name", value="Charlie")]

        messages = build_messages(
            session=session,
            system_prompt="Base prompt.",
            facts=facts,
            plan=FakePlan(),
        )

        system_content = messages[0]["content"]
        assert "user.name: Charlie" in system_content
        assert "ACTIVE PLAN:" in system_content
        assert "Step 1: Do thing" in system_content
