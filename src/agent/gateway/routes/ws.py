"""WebSocket endpoints for real-time communication.

Two WebSocket endpoints:
- /ws/events — Dashboard subscribes to real-time agent events (read-only stream)
- /ws/chat   — Interactive chat via WebSocket (bidirectional)

Auth is enforced via ?token= query parameter when gateway.auth_token is configured.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from agent.core.events import EventBus

logger = structlog.get_logger(__name__)

router = APIRouter()


async def _safe_send(ws: WebSocket, data: dict[str, Any]) -> bool:
    """Send JSON to a WebSocket, returning False if the connection is closed."""
    try:
        await ws.send_json(data)
    except (WebSocketDisconnect, RuntimeError):
        return False
    return True


class ConnectionManager:
    """Manages active WebSocket connections.

    Two connection types:
    - "events": Dashboard subscribes to real-time agent events (read-only)
    - "chat": Interactive chat via WebSocket (bidirectional)
    """

    def __init__(self) -> None:
        self._event_connections: list[WebSocket] = []
        self._chat_connections: dict[str, WebSocket] = {}  # session_id → ws

    async def connect_events(self, ws: WebSocket) -> None:
        """Accept and register an events subscriber."""
        await ws.accept()
        self._event_connections.append(ws)

    async def disconnect_events(self, ws: WebSocket) -> None:
        """Remove an events subscriber."""
        if ws in self._event_connections:
            self._event_connections.remove(ws)

    async def connect_chat(self, ws: WebSocket, session_id: str) -> None:
        """Accept and register a chat connection for a session."""
        await ws.accept()
        self._chat_connections[session_id] = ws

    async def disconnect_chat(self, session_id: str) -> None:
        """Remove a chat connection by session ID."""
        self._chat_connections.pop(session_id, None)

    async def broadcast_event(self, event: dict[str, Any]) -> None:
        """Send event to all events subscribers. Remove dead connections."""
        dead: list[WebSocket] = []
        for ws in self._event_connections:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._event_connections.remove(ws)

    async def send_to_chat(self, session_id: str, data: dict[str, Any]) -> None:
        """Send data to a specific chat session."""
        ws = self._chat_connections.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self._chat_connections.pop(session_id, None)

    @property
    def event_connection_count(self) -> int:
        """Number of active events subscribers."""
        return len(self._event_connections)

    @property
    def chat_connection_count(self) -> int:
        """Number of active chat connections."""
        return len(self._chat_connections)


# Global instance shared across the module
manager = ConnectionManager()


def _verify_token(websocket: WebSocket, token: str | None) -> bool:
    """Check WebSocket auth token against config.

    Returns True if authorized, False otherwise.
    """
    config = websocket.app.state.config
    auth_token = config.gateway.auth_token
    return not (auth_token and token != auth_token)


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, token: str | None = None) -> None:
    """Real-time event stream for dashboard.

    Auth: verify token query param if gateway auth_token configured.

    Events sent as JSON:
    {
        "event": "tool.execute",
        "timestamp": "2026-03-04T...",
        "data": { "tool": "shell_exec", ... }
    }

    Client can send pings; server responds with pong.
    Connection stays open indefinitely.
    """
    if not _verify_token(websocket, token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await manager.connect_events(websocket)
    logger.info("ws_events_connected", total=manager.event_connection_count)

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await manager.disconnect_events(websocket)
        logger.info("ws_events_disconnected", total=manager.event_connection_count)


async def _handle_litellm_message(
    websocket: WebSocket,
    agent_loop: Any,
    session: Any,
    content: str,
) -> None:
    """Process a chat message via the LiteLLM agent loop (streaming)."""
    final_response = None
    async for event in agent_loop.process_message_stream(
        user_message=content,
        session=session,
        trigger="user_message",
    ):
        if event.type == "chunk":
            await _safe_send(websocket, {
                "type": "response.chunk",
                "content": event.content,
            })
        elif event.type == "tool.execute":
            await _safe_send(websocket, {
                "type": "tool.execute",
                "tool": event.data.get("tool", ""),
                "arguments": event.data.get("arguments", {}),
            })
        elif event.type == "tool.result":
            await _safe_send(websocket, {
                "type": "tool.result",
                "tool": event.data.get("tool", ""),
                "success": event.data.get("success", False),
                "output": event.data.get("output", ""),
            })
        elif event.type == "done":
            final_response = event.response

    await _safe_send(websocket, {"type": "typing", "status": False})

    if final_response:
        await _safe_send(websocket, {
            "type": "response.end",
            "content": final_response.content,
            "model": final_response.model,
            "usage": {
                "input_tokens":
                    final_response.usage.input_tokens
                    if final_response.usage else 0,
                "output_tokens":
                    final_response.usage.output_tokens
                    if final_response.usage else 0,
            },
        })


async def _handle_sdk_message(
    websocket: WebSocket,
    sdk_service: Any,
    session: Any,
    content: str,
) -> None:
    """Process a chat message via Claude Agent SDK (streaming)."""
    accumulated_text = ""
    model = "claude-sdk"
    input_tokens = 0
    output_tokens = 0

    async for event in sdk_service.run_task_stream(
        prompt=content,
        task_id=session.id,
        session_id=getattr(session, "sdk_session_id", None),
    ):
        if event.type == "text":
            # Only accumulate main agent text, not subagent output
            if not (event.data and event.data.get("subagent")):
                accumulated_text += event.content
                await _safe_send(websocket, {
                    "type": "response.chunk",
                    "content": event.content,
                })
        elif event.type == "tool_use":
            await _safe_send(websocket, {
                "type": "tool.execute",
                "tool": event.data.get("tool", ""),
                "arguments": event.data.get("input", {}),
            })
        elif event.type == "result":
            # Capture final metadata
            input_tokens = event.data.get("input_tokens", 0)
            output_tokens = event.data.get("output_tokens", 0)
            sdk_sid = event.data.get("session_id")
            if sdk_sid:
                session.sdk_session_id = sdk_sid
        elif event.type == "error":
            await _safe_send(websocket, {"type": "typing", "status": False})
            await _safe_send(websocket, {
                "type": "error",
                "message": event.content,
            })
            return

    await _safe_send(websocket, {"type": "typing", "status": False})
    await _safe_send(websocket, {
        "type": "response.end",
        "content": accumulated_text,
        "model": model,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    })


@router.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    token: str | None = None,
    session_id: str | None = None,
) -> None:
    """Interactive chat via WebSocket.

    Auth: verify token query param.

    Client → Server protocol:
    {"type": "message", "content": "user text"}
    {"type": "voice.data", "audio": "<base64>", "mime_type": "audio/webm"}
    {"type": "ping"}

    Server → Client protocol:
    {"type": "response.start", "session_id": "uuid"}
    {"type": "response.end", "content": "...", "model": "...", "usage": {...}}
    {"type": "tool.execute", "tool": "...", "arguments": {...}}
    {"type": "tool.result", "tool": "...", "success": true, "output": "..."}
    {"type": "voice.transcription", "text": "...", "language": "en"}
    {"type": "voice.audio", "audio": "<base64>", "mime_type": "audio/ogg", "duration": 3.5}
    {"type": "error", "message": "error description"}
    {"type": "typing", "status": true/false}
    {"type": "pong"}
    """
    if not _verify_token(websocket, token):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Get or create session
    session_store = websocket.app.state.session_store
    agent_loop = websocket.app.state.agent_loop
    event_bus = websocket.app.state.event_bus

    if session_id:
        session = await session_store.get(session_id)
        if not session:
            session = await session_store.get_or_create(session_id=session_id, channel="webchat")
    else:
        session = await session_store.new_session(channel="webchat")

    await manager.connect_chat(websocket, session.id)
    logger.info("ws_chat_connected", session_id=session.id)

    # Tool event forwarding helpers (used only for non-streaming paths like voice)
    async def _forward_tool_execute(data: dict[str, Any]) -> None:
        await manager.send_to_chat(session.id, {
            "type": "tool.execute",
            "tool": data.get("tool", ""),
            "arguments": data.get("arguments", {}),
        })

    async def _forward_tool_result(data: dict[str, Any]) -> None:
        await manager.send_to_chat(session.id, {
            "type": "tool.result",
            "tool": data.get("tool", ""),
            "success": data.get("success", False),
            "output": data.get("output", ""),
        })

    from agent.core.events import Events

    try:
        while True:
            raw = await websocket.receive_json()
            msg_type = raw.get("type")

            if msg_type == "ping":
                await _safe_send(websocket, {"type": "pong"})
                continue

            if msg_type == "voice.data":
                # Voice input from client
                voice_pipeline = getattr(websocket.app.state, "voice_pipeline", None)
                audio_b64 = raw.get("audio", "")
                voice_mime = raw.get("mime_type", "audio/webm")

                if not audio_b64 or not voice_pipeline:
                    await _safe_send(websocket, {
                        "type": "error",
                        "message": "Voice pipeline not available or no audio data",
                    })
                    continue

                audio_data = base64.b64decode(audio_b64)

                if not await _safe_send(websocket, {
                    "type": "response.start",
                    "session_id": session.id,
                }):
                    continue
                await _safe_send(websocket, {"type": "typing", "status": True})

                # Register tool forwarding for voice (non-streaming path)
                event_bus.on(Events.TOOL_EXECUTE, _forward_tool_execute)
                event_bus.on(Events.TOOL_RESULT, _forward_tool_result)
                try:
                    # Transcribe
                    if not voice_pipeline.is_llm_native():
                        stt_result = await voice_pipeline.transcribe(audio_data, voice_mime)
                        await _safe_send(websocket, {
                            "type": "voice.transcription",
                            "text": stt_result.text,
                            "language": stt_result.language,
                        })
                        user_text = stt_result.text
                    else:
                        # LLM native — for now just send a placeholder
                        user_text = "[Voice message]"

                    # Process through agent loop (non-streaming for voice)
                    response = await agent_loop.process_message(
                        user_message=user_text,
                        session=session,
                        trigger="user_message",
                    )

                    await _safe_send(websocket, {"type": "typing", "status": False})
                    await _safe_send(websocket, {
                        "type": "response.end",
                        "content": response.content,
                        "model": response.model,
                        "usage": {
                            "input_tokens": response.usage.input_tokens
                            if response.usage else 0,
                            "output_tokens": response.usage.output_tokens
                            if response.usage else 0,
                        },
                    })

                    # Synthesize voice response
                    if voice_pipeline.should_voice_reply("webchat"):
                        tts_result = await voice_pipeline.synthesize(response.content)
                        if tts_result:
                            audio_out_b64 = base64.b64encode(
                                tts_result.audio_data
                            ).decode()
                            await _safe_send(websocket, {
                                "type": "voice.audio",
                                "audio": audio_out_b64,
                                "mime_type": tts_result.mime_type,
                                "duration": tts_result.duration_seconds,
                            })

                except Exception as e:
                    logger.error("ws_voice_error", session_id=session.id, error=str(e))
                    await _safe_send(websocket, {"type": "typing", "status": False})
                    await _safe_send(websocket, {"type": "error", "message": str(e)})
                finally:
                    event_bus.off(Events.TOOL_EXECUTE, _forward_tool_execute)
                    event_bus.off(Events.TOOL_RESULT, _forward_tool_result)

                continue

            if msg_type == "message":
                content = raw.get("content", "").strip()
                if not content:
                    continue

                # Signal response start
                if not await _safe_send(websocket, {
                    "type": "response.start",
                    "session_id": session.id,
                }):
                    continue

                # Typing indicator on
                await _safe_send(websocket, {"type": "typing", "status": True})

                sdk_service = getattr(websocket.app.state, "sdk_service", None)

                try:
                    if sdk_service is not None:
                        # --- Claude SDK backend ---
                        await _handle_sdk_message(
                            websocket, sdk_service, session, content,
                        )
                    else:
                        # --- LiteLLM backend ---
                        await _handle_litellm_message(
                            websocket, agent_loop, session, content,
                        )

                except Exception as e:
                    logger.error(
                        "ws_chat_error",
                        session_id=session.id,
                        error=str(e),
                    )
                    await _safe_send(websocket, {"type": "typing", "status": False})
                    await _safe_send(websocket, {
                        "type": "error",
                        "message": str(e),
                    })

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        await manager.disconnect_chat(session.id)
        logger.info("ws_chat_disconnected", session_id=session.id)


_forwarding_registered_bus: int | None = None


def setup_event_forwarding(event_bus: EventBus) -> None:
    """Bridge internal EventBus → WebSocket event subscribers.

    Call this during app startup to forward all relevant events
    to connected /ws/events clients. Safe to call multiple times
    with the same event_bus; handlers are only registered once per bus.
    """
    global _forwarding_registered_bus
    bus_id = id(event_bus)
    if _forwarding_registered_bus == bus_id:
        return
    _forwarding_registered_bus = bus_id

    from agent.core.events import Events

    events_to_forward = [
        Events.MESSAGE_INCOMING,
        Events.MESSAGE_OUTGOING,
        Events.TOOL_EXECUTE,
        Events.TOOL_RESULT,
        Events.HEARTBEAT_TICK,
        Events.HEARTBEAT_ACTION,
        Events.MEMORY_UPDATE,
        Events.AGENT_ERROR,
        Events.AGENT_STARTED,
        Events.AGENT_STOPPED,
    ]

    for event_name in events_to_forward:
        async def _forward(data: Any, evt: str = event_name) -> None:
            await manager.broadcast_event({
                "event": evt,
                "timestamp": datetime.now().isoformat(),
                "data": data,
            })

        event_bus.on(event_name, _forward)
