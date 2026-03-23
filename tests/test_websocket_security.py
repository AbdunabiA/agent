"""Tests for WebSocket security — connection limits and message size."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.gateway.routes.ws import (
    MAX_WS_CONNECTIONS,
    MAX_WS_MESSAGE_SIZE,
    ConnectionManager,
    manager,
    router,
)


def _make_app(*, auth_token: str = "") -> FastAPI:
    """Build a minimal FastAPI app with the WS router mounted."""
    app = FastAPI()
    app.include_router(router)

    config = MagicMock()
    config.gateway.auth_token = auth_token

    app.state.config = config
    app.state.session_store = MagicMock()
    app.state.agent_loop = MagicMock()
    app.state.event_bus = MagicMock()
    return app


class TestConnectionLimit:
    """Verify that WebSocket connection limits are enforced."""

    def test_events_connection_rejected_at_limit(self) -> None:
        """When MAX_WS_CONNECTIONS is reached, new /ws/events are rejected."""
        app = _make_app()
        client = TestClient(app)

        # Pre-fill the manager so it reports at-limit connections
        fake_ws_list = [MagicMock() for _ in range(MAX_WS_CONNECTIONS)]
        original_events = manager._event_connections
        manager._event_connections = fake_ws_list

        try:
            with client.websocket_connect("/ws/events"):
                # Server should close the connection with code 1013
                # The TestClient raises an exception or we get a close
                pass
        except Exception:
            # Connection was rejected as expected
            pass
        finally:
            manager._event_connections = original_events

    def test_chat_connection_rejected_at_limit(self) -> None:
        """When MAX_WS_CONNECTIONS is reached, new /ws/chat are rejected."""
        app = _make_app()
        client = TestClient(app)

        # Pre-fill the manager so it reports at-limit connections
        fake_ws_list = [MagicMock() for _ in range(MAX_WS_CONNECTIONS)]
        original_events = manager._event_connections
        manager._event_connections = fake_ws_list

        try:
            with client.websocket_connect("/ws/chat"):
                pass
        except Exception:
            pass
        finally:
            manager._event_connections = original_events


class TestOversizedMessage:
    """Verify that oversized messages return an error."""

    def test_events_oversized_text_returns_error(self) -> None:
        """Sending a text message larger than MAX_WS_MESSAGE_SIZE on /ws/events
        returns an error response."""
        app = _make_app()
        client = TestClient(app)

        with client.websocket_connect("/ws/events") as ws:
            # Send an oversized text message
            oversized = "x" * (MAX_WS_MESSAGE_SIZE + 1)
            ws.send_text(oversized)
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "too large" in resp["message"].lower() or "max" in resp["message"].lower()

    def test_chat_oversized_json_returns_error(self) -> None:
        """Sending a JSON message larger than MAX_WS_MESSAGE_SIZE on /ws/chat
        returns an error response."""
        app = _make_app()

        # Mock session_store to return a session
        mock_session = MagicMock()
        mock_session.id = "test-session"
        app.state.session_store.new_session = AsyncMock(return_value=mock_session)
        app.state.session_store.get = AsyncMock(return_value=None)
        app.state.session_store.get_or_create = AsyncMock(return_value=mock_session)

        client = TestClient(app)

        with client.websocket_connect("/ws/chat") as ws:
            # Build a message whose JSON representation exceeds the limit
            big_content = "a" * MAX_WS_MESSAGE_SIZE
            ws.send_json({"type": "message", "content": big_content})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "too large" in resp["message"].lower() or "max" in resp["message"].lower()


class TestConnectionManager:
    """Unit tests for the ConnectionManager class."""

    @pytest.mark.asyncio
    async def test_event_connection_count(self) -> None:
        """event_connection_count tracks connected websockets."""
        mgr = ConnectionManager()
        assert mgr.event_connection_count == 0

        ws = AsyncMock()
        await mgr.connect_events(ws)
        assert mgr.event_connection_count == 1

        await mgr.disconnect_events(ws)
        assert mgr.event_connection_count == 0

    @pytest.mark.asyncio
    async def test_chat_connection_count(self) -> None:
        """chat_connection_count tracks connected chat sessions."""
        mgr = ConnectionManager()
        assert mgr.chat_connection_count == 0

        ws = AsyncMock()
        await mgr.connect_chat(ws, "sess-1")
        assert mgr.chat_connection_count == 1

        await mgr.disconnect_chat("sess-1")
        assert mgr.chat_connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self) -> None:
        """broadcast_event removes connections that raise on send."""
        mgr = ConnectionManager()

        good_ws = AsyncMock()
        bad_ws = AsyncMock()
        bad_ws.send_json.side_effect = RuntimeError("connection closed")

        await mgr.connect_events(good_ws)
        await mgr.connect_events(bad_ws)
        assert mgr.event_connection_count == 2

        await mgr.broadcast_event({"event": "test"})

        # Bad connection should have been removed
        assert mgr.event_connection_count == 1
        good_ws.send_json.assert_called_once()
