"""Tests for WebSocket endpoints (/ws/events and /ws/chat)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agent.config import AgentConfig, GatewayConfig
from agent.core.agent_loop import StreamEvent
from agent.core.audit import AuditLog
from agent.core.events import EventBus, Events
from agent.core.session import TokenUsage
from agent.gateway.app import create_app
from agent.gateway.routes.ws import ConnectionManager
from agent.llm.provider import LLMResponse
from agent.tools.registry import ToolRegistry

# --- Fixtures ---


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def config_no_auth() -> AgentConfig:
    """Config with auth disabled."""
    cfg = AgentConfig()
    cfg.gateway = GatewayConfig(auth_token=None)
    return cfg


@pytest.fixture
def config_with_auth() -> AgentConfig:
    """Config with auth enabled."""
    cfg = AgentConfig()
    cfg.gateway = GatewayConfig(auth_token="test-secret-token")
    return cfg


@pytest.fixture
def mock_agent_loop() -> AsyncMock:
    """Mock agent loop that returns a canned response."""
    from agent.core.session import Message

    loop = AsyncMock()

    async def fake_process_message(user_message: str, session, **kwargs):
        session.add_message(Message(role="user", content=user_message))
        response = LLMResponse(
            content="Hello from agent!",
            model="test-model",
            tool_calls=None,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        session.add_message(
            Message(
                role="assistant",
                content=response.content,
                model=response.model,
                usage=response.usage,
            )
        )
        return response

    loop.process_message.side_effect = fake_process_message

    async def fake_process_message_stream(user_message: str, session, **kwargs):
        session.add_message(Message(role="user", content=user_message))
        response = LLMResponse(
            content="Hello from agent!",
            model="test-model",
            tool_calls=None,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        session.add_message(
            Message(
                role="assistant",
                content=response.content,
                model=response.model,
                usage=response.usage,
            )
        )
        # Yield chunks then done event
        yield StreamEvent(type="chunk", content="Hello from agent!")
        yield StreamEvent(type="done", response=response)

    loop.process_message_stream = fake_process_message_stream
    return loop


@pytest.fixture
def mock_agent_loop_error() -> AsyncMock:
    """Mock agent loop that raises an error."""
    loop = AsyncMock()
    loop.process_message.side_effect = RuntimeError("LLM provider unavailable")

    async def fake_stream_error(user_message: str, session, **kwargs):
        raise RuntimeError("LLM provider unavailable")
        # Make it an async generator (unreachable yield makes Python treat it as one)
        yield  # noqa: unreachable  # pragma: no cover

    loop.process_message_stream = fake_stream_error
    return loop


def _create_app(
    config: AgentConfig,
    agent_loop: AsyncMock,
    event_bus: EventBus,
) -> TestClient:
    """Helper to create a TestClient."""
    app = create_app(
        config=config,
        agent_loop=agent_loop,
        event_bus=event_bus,
        audit=AuditLog(),
        tool_registry=ToolRegistry(),
        heartbeat=None,
    )
    return TestClient(app)


# --- ConnectionManager unit tests ---


class TestConnectionManager:
    """Unit tests for ConnectionManager."""

    def test_initial_counts(self) -> None:
        mgr = ConnectionManager()
        assert mgr.event_connection_count == 0
        assert mgr.chat_connection_count == 0

    @pytest.mark.asyncio
    async def test_connect_disconnect_events(self) -> None:
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.connect_events(mock_ws)
        assert mgr.event_connection_count == 1
        mock_ws.accept.assert_awaited_once()

        await mgr.disconnect_events(mock_ws)
        assert mgr.event_connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_events_not_connected(self) -> None:
        """Disconnecting a WS that isn't connected should not error."""
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.disconnect_events(mock_ws)
        assert mgr.event_connection_count == 0

    @pytest.mark.asyncio
    async def test_connect_disconnect_chat(self) -> None:
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.connect_chat(mock_ws, "sess-1")
        assert mgr.chat_connection_count == 1
        mock_ws.accept.assert_awaited_once()

        await mgr.disconnect_chat("sess-1")
        assert mgr.chat_connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_chat_not_connected(self) -> None:
        mgr = ConnectionManager()
        await mgr.disconnect_chat("nonexistent")
        assert mgr.chat_connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_event(self) -> None:
        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect_events(ws1)
        await mgr.connect_events(ws2)

        event = {"event": "test", "data": {"key": "value"}}
        await mgr.broadcast_event(event)

        ws1.send_json.assert_awaited_once_with(event)
        ws2.send_json.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self) -> None:
        mgr = ConnectionManager()
        live_ws = AsyncMock()
        dead_ws = AsyncMock()
        dead_ws.send_json.side_effect = RuntimeError("connection closed")

        await mgr.connect_events(live_ws)
        await mgr.connect_events(dead_ws)
        assert mgr.event_connection_count == 2

        await mgr.broadcast_event({"event": "test"})
        # Dead WS should be removed
        assert mgr.event_connection_count == 1

    @pytest.mark.asyncio
    async def test_send_to_chat(self) -> None:
        mgr = ConnectionManager()
        mock_ws = AsyncMock()
        await mgr.connect_chat(mock_ws, "sess-1")

        data = {"type": "response.end", "content": "hi"}
        await mgr.send_to_chat("sess-1", data)
        mock_ws.send_json.assert_awaited_once_with(data)

    @pytest.mark.asyncio
    async def test_send_to_chat_nonexistent(self) -> None:
        """Sending to a nonexistent session should silently do nothing."""
        mgr = ConnectionManager()
        await mgr.send_to_chat("nonexistent", {"type": "test"})

    @pytest.mark.asyncio
    async def test_send_to_chat_removes_dead(self) -> None:
        mgr = ConnectionManager()
        dead_ws = AsyncMock()
        dead_ws.send_json.side_effect = RuntimeError("closed")
        await mgr.connect_chat(dead_ws, "sess-1")

        await mgr.send_to_chat("sess-1", {"type": "test"})
        assert mgr.chat_connection_count == 0


# --- /ws/events endpoint tests ---


class TestWebSocketEvents:
    """Tests for /ws/events endpoint."""

    def test_connect_no_auth(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Connect to /ws/events without auth → accepted."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/events") as ws:
            ws.send_text("ping")
            data = ws.receive_text()
            assert data == "pong"

    def test_connect_with_valid_token(
        self, config_with_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Connect with valid token → accepted."""
        client = _create_app(config_with_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/events?token=test-secret-token") as ws:
            ws.send_text("ping")
            data = ws.receive_text()
            assert data == "pong"

    def test_connect_with_invalid_token(
        self, config_with_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Connect with invalid token → rejected with 4001."""
        client = _create_app(config_with_auth, mock_agent_loop, event_bus)
        with (
            pytest.raises(Exception),  # noqa: B017
            client.websocket_connect("/ws/events?token=wrong") as ws,
        ):
            ws.send_text("ping")

    def test_connect_without_token_when_required(
        self, config_with_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Connect without token when auth required → rejected."""
        client = _create_app(config_with_auth, mock_agent_loop, event_bus)
        with (
            pytest.raises(Exception),  # noqa: B017
            client.websocket_connect("/ws/events") as ws,
        ):
            ws.send_text("ping")

    def test_ping_pong(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Client sends 'ping', server replies 'pong'."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/events") as ws:
            ws.send_text("ping")
            assert ws.receive_text() == "pong"
            ws.send_text("ping")
            assert ws.receive_text() == "pong"


# --- /ws/chat endpoint tests ---


class TestWebSocketChat:
    """Tests for /ws/chat endpoint."""

    def test_connect_no_auth(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Connect to /ws/chat without auth → accepted."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_connect_with_valid_token(
        self, config_with_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Connect with valid token → accepted."""
        client = _create_app(config_with_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/chat?token=test-secret-token") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_connect_with_invalid_token(
        self, config_with_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Connect with invalid token → rejected."""
        client = _create_app(config_with_auth, mock_agent_loop, event_bus)
        with (
            pytest.raises(Exception),  # noqa: B017
            client.websocket_connect("/ws/chat?token=wrong") as ws,
        ):
            ws.send_json({"type": "ping"})

    def test_chat_message_flow(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Send message → receive response.start + typing + chunks + typing off + response.end."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "message", "content": "Hello"})

            # 1. response.start
            msg = ws.receive_json()
            assert msg["type"] == "response.start"
            assert "session_id" in msg

            # 2. typing on
            msg = ws.receive_json()
            assert msg["type"] == "typing"
            assert msg["status"] is True

            # 3. response.chunk (streamed content)
            msg = ws.receive_json()
            assert msg["type"] == "response.chunk"
            assert msg["content"] == "Hello from agent!"

            # 4. typing off
            msg = ws.receive_json()
            assert msg["type"] == "typing"
            assert msg["status"] is False

            # 5. response.end
            msg = ws.receive_json()
            assert msg["type"] == "response.end"
            assert msg["content"] == "Hello from agent!"
            assert msg["model"] == "test-model"
            assert msg["usage"]["input_tokens"] == 10
            assert msg["usage"]["output_tokens"] == 5

    def test_chat_with_session_id(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Providing session_id query param reuses or creates with that ID."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/chat?session_id=my-session") as ws:
            ws.send_json({"type": "message", "content": "Hi"})

            msg = ws.receive_json()
            assert msg["type"] == "response.start"
            assert msg["session_id"] == "my-session"

    def test_chat_error_handling(
        self,
        config_no_auth: AgentConfig,
        mock_agent_loop_error: AsyncMock,
        event_bus: EventBus,
    ) -> None:
        """Processing error → typing off + error message sent."""
        client = _create_app(config_no_auth, mock_agent_loop_error, event_bus)
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "message", "content": "Hello"})

            # response.start
            msg = ws.receive_json()
            assert msg["type"] == "response.start"

            # typing on
            msg = ws.receive_json()
            assert msg["type"] == "typing"
            assert msg["status"] is True

            # error (sent before typing off in _handle_litellm_message)
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "LLM provider unavailable" in msg["message"]

            # typing off
            msg = ws.receive_json()
            assert msg["type"] == "typing"
            assert msg["status"] is False

    def test_chat_empty_message_ignored(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Empty message content should be ignored, not processed."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/chat") as ws:
            # Send empty message
            ws.send_json({"type": "message", "content": ""})
            # Then send a valid one — if empty was ignored, this should work
            ws.send_json({"type": "message", "content": "Real message"})

            msg = ws.receive_json()
            assert msg["type"] == "response.start"

    def test_chat_ping_pong(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Chat ping/pong uses JSON protocol."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data == {"type": "pong"}

    def test_multiple_messages_same_session(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Multiple messages in one connection reuse the same session."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)
        with client.websocket_connect("/ws/chat") as ws:
            # First message
            ws.send_json({"type": "message", "content": "First"})
            msg1_start = ws.receive_json()
            session_id = msg1_start["session_id"]
            # Drain remaining responses
            ws.receive_json()  # typing on
            ws.receive_json()  # response.chunk
            ws.receive_json()  # typing off
            ws.receive_json()  # response.end

            # Second message
            ws.send_json({"type": "message", "content": "Second"})
            msg2_start = ws.receive_json()
            assert msg2_start["session_id"] == session_id


# --- Event forwarding tests ---


class TestEventForwarding:
    """Tests for EventBus → WebSocket event forwarding."""

    def test_events_received_on_ws(
        self, config_no_auth: AgentConfig, mock_agent_loop: AsyncMock, event_bus: EventBus
    ) -> None:
        """Events emitted on bus should arrive at /ws/events subscribers."""
        client = _create_app(config_no_auth, mock_agent_loop, event_bus)

        with client.websocket_connect("/ws/events") as ws:
            # Emit an event through the bus on a fresh event loop
            import asyncio

            asyncio.run(event_bus.emit(Events.AGENT_STARTED, {"status": "ready"}))

            msg = ws.receive_json()
            assert msg["event"] == "agent.started"
            assert "timestamp" in msg
            assert msg["data"]["status"] == "ready"

    def test_chat_triggers_tool_events(
        self, config_no_auth: AgentConfig, event_bus: EventBus
    ) -> None:
        """Tool events during chat processing are forwarded to the chat WS."""
        from agent.core.session import Message

        loop_mock = AsyncMock()

        async def stream_with_tool_events(user_message: str, session, **kwargs):
            session.add_message(Message(role="user", content=user_message))
            # Yield tool events
            yield StreamEvent(
                type="tool.execute",
                data={"tool": "shell_exec", "arguments": {"cmd": "ls"}, "iteration": 1},
            )
            yield StreamEvent(
                type="tool.result",
                data={"tool": "shell_exec", "success": True, "output": "file.txt"},
            )
            # Then final response
            response = LLMResponse(
                content="Done!",
                model="test-model",
                tool_calls=None,
                usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
            session.add_message(
                Message(role="assistant", content=response.content, model=response.model)
            )
            yield StreamEvent(type="chunk", content="Done!")
            yield StreamEvent(type="done", response=response)

        loop_mock.process_message_stream = stream_with_tool_events

        client = _create_app(config_no_auth, loop_mock, event_bus)

        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "message", "content": "Run ls"})

            # Collect all messages until response.end
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "response.end":
                    break

            types = [m["type"] for m in messages]
            assert "response.start" in types
            assert "response.end" in types
            # Tool events should be present
            assert "tool.execute" in types
            assert "tool.result" in types

            # Verify tool execute content
            tool_exec = next(m for m in messages if m["type"] == "tool.execute")
            assert tool_exec["tool"] == "shell_exec"

            # Verify tool result content
            tool_res = next(m for m in messages if m["type"] == "tool.result")
            assert tool_res["success"] is True
