"""Tests for FastAPI gateway endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from agent import __version__
from agent.config import AgentConfig, GatewayConfig
from agent.core.audit import AuditLog
from agent.core.events import EventBus
from agent.core.session import TokenUsage
from agent.gateway.app import create_app
from agent.llm.provider import LLMResponse
from agent.tools.registry import ToolRegistry, ToolTier


@pytest.fixture
def config() -> AgentConfig:
    """Config with auth disabled for easy testing."""
    cfg = AgentConfig()
    cfg.gateway = GatewayConfig(auth_token=None)
    return cfg


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Registry with a test tool."""
    reg = ToolRegistry()

    @reg.tool(name="test_tool", description="A test tool", tier=ToolTier.SAFE)
    async def test_tool(text: str) -> str:
        return f"result: {text}"

    return reg


@pytest.fixture
def mock_agent_loop() -> AsyncMock:
    """Mock agent loop that adds messages to session and returns response."""
    from agent.core.session import Message

    loop = AsyncMock()

    async def fake_process_message(message: str, session, **kwargs):
        # Mimic real agent loop: add user + assistant messages
        session.add_message(Message(role="user", content=message))
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
    return loop


@pytest.fixture
def audit_log() -> AuditLog:
    return AuditLog()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def app(
    config: AgentConfig,
    mock_agent_loop: AsyncMock,
    event_bus: EventBus,
    audit_log: AuditLog,
    tool_registry: ToolRegistry,
) -> TestClient:
    """Create test client with full gateway app."""
    application = create_app(
        config=config,
        agent_loop=mock_agent_loop,
        event_bus=event_bus,
        audit=audit_log,
        tool_registry=tool_registry,
        heartbeat=None,
    )
    return TestClient(application)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_ok(self, app: TestClient) -> None:
        resp = app.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__
        assert "uptime_seconds" in data
        assert "timestamp" in data


class TestStatusEndpoint:
    """Tests for GET /api/v1/status."""

    def test_status_returns_info(self, app: TestClient) -> None:
        resp = app.get("/api/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "active_sessions" in data
        assert "heartbeat" in data
        assert "tools" in data


class TestChatEndpoint:
    """Tests for POST /api/v1/chat."""

    def test_chat_sends_message(self, app: TestClient) -> None:
        resp = app.post("/api/v1/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hello from agent!"
        assert data["session_id"]
        assert data["model"] == "test-model"
        assert data["usage"]["total_tokens"] == 15

    def test_chat_with_session_id(self, app: TestClient) -> None:
        """Sending a session_id should reuse the session."""
        resp1 = app.post("/api/v1/chat", json={"message": "First"})
        session_id = resp1.json()["session_id"]

        resp2 = app.post(
            "/api/v1/chat",
            json={"message": "Second", "session_id": session_id},
        )
        assert resp2.json()["session_id"] == session_id

    def test_chat_empty_message_returns_422(self, app: TestClient) -> None:
        resp = app.post("/api/v1/chat", json={"message": ""})
        assert resp.status_code == 422


class TestConversationsEndpoint:
    """Tests for GET /api/v1/conversations."""

    def test_list_empty(self, app: TestClient) -> None:
        resp = app.get("/api/v1/conversations")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_chat(self, app: TestClient) -> None:
        app.post("/api/v1/chat", json={"message": "Hello"})
        resp = app.get("/api/v1/conversations")
        assert resp.status_code == 200
        sessions = resp.json()
        assert len(sessions) == 1
        assert sessions[0]["message_count"] >= 1


class TestConversationMessagesEndpoint:
    """Tests for GET /api/v1/conversations/{session_id}/messages."""

    def test_get_messages(self, app: TestClient) -> None:
        chat_resp = app.post("/api/v1/chat", json={"message": "Hello"})
        session_id = chat_resp.json()["session_id"]

        resp = app.get(f"/api/v1/conversations/{session_id}/messages")
        assert resp.status_code == 200
        messages = resp.json()
        assert len(messages) >= 1

    def test_get_messages_not_found(self, app: TestClient) -> None:
        resp = app.get("/api/v1/conversations/nonexistent/messages")
        assert resp.status_code == 404


class TestAuditEndpoints:
    """Tests for GET /api/v1/audit and /audit/stats."""

    def test_audit_empty(self, app: TestClient) -> None:
        resp = app.get("/api/v1/audit")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_audit_stats_empty(self, app: TestClient) -> None:
        resp = app.get("/api/v1/audit/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_calls"] == 0


class TestToolsEndpoint:
    """Tests for GET /api/v1/tools."""

    def test_list_tools(self, app: TestClient) -> None:
        resp = app.get("/api/v1/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["tier"] == "safe"


class TestControlEndpoint:
    """Tests for POST /api/v1/control."""

    def test_control_no_heartbeat(self, app: TestClient) -> None:
        """Control without heartbeat should return 503."""
        resp = app.post("/api/v1/control", json={"action": "pause"})
        assert resp.status_code == 503

    def test_control_invalid_action(self, app: TestClient) -> None:
        resp = app.post("/api/v1/control", json={"action": "invalid"})
        # Either 400 (invalid action) or 503 (no heartbeat) is acceptable
        assert resp.status_code in (400, 503)


class TestControlWithHeartbeat:
    """Tests for control endpoint with heartbeat daemon."""

    def test_pause_resume(self, config: AgentConfig) -> None:
        """Pause and resume should toggle heartbeat."""
        mock_heartbeat = MagicMock()
        mock_heartbeat.is_enabled = True
        mock_heartbeat.last_tick = None

        mock_loop = AsyncMock()
        application = create_app(
            config=config,
            agent_loop=mock_loop,
            event_bus=EventBus(),
            audit=AuditLog(),
            tool_registry=ToolRegistry(),
            heartbeat=mock_heartbeat,
        )
        client = TestClient(application)

        resp = client.post("/api/v1/control", json={"action": "pause"})
        assert resp.status_code == 200
        mock_heartbeat.disable.assert_called_once()

        resp = client.post("/api/v1/control", json={"action": "resume"})
        assert resp.status_code == 200
        mock_heartbeat.enable.assert_called_once()


class TestConfigEndpoint:
    """Tests for GET /api/v1/config."""

    def test_config_returns_masked(self, app: TestClient) -> None:
        resp = app.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent" in data
        assert "models" in data
        assert "gateway" in data
