"""WebSocket chat channel adapter.

Thin wrapper around the gateway's ConnectionManager for sending
messages and typing indicators to connected WebSocket clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from agent.channels.base import BaseChannel, OutgoingMessage

if TYPE_CHECKING:
    from agent.core.agent_loop import AgentLoop
    from agent.core.events import EventBus
    from agent.core.session import SessionStore

logger = structlog.get_logger(__name__)


class WebChatChannel(BaseChannel):
    """WebSocket-based chat channel.

    Sends messages and typing indicators through the gateway's
    ConnectionManager to connected WebSocket clients.
    """

    def __init__(
        self,
        config: Any,
        event_bus: EventBus,
        session_store: SessionStore,
        agent_loop: AgentLoop,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, session_store=session_store)
        self.agent_loop = agent_loop

    @property
    def name(self) -> str:
        """Channel identifier."""
        return "webchat"

    async def start(self) -> None:
        """Start the webchat channel."""
        self._running = True
        logger.info("webchat_started")

    async def stop(self) -> None:
        """Stop the webchat channel."""
        self._running = False
        logger.info("webchat_stopped")

    async def send_message(self, message: OutgoingMessage) -> None:
        """Send a message to a WebSocket chat client.

        Args:
            message: The outgoing message to send.
        """
        from agent.gateway.routes.ws import manager

        await manager.send_to_chat(
            session_id=message.channel_user_id,
            data={
                "type": "response.end",
                "content": message.content,
            },
        )

    async def send_typing(self, channel_user_id: str) -> None:
        """Send a typing indicator to a WebSocket chat client.

        Args:
            channel_user_id: Session ID of the connected client.
        """
        from agent.gateway.routes.ws import manager

        await manager.send_to_chat(
            session_id=channel_user_id,
            data={"type": "typing", "status": True},
        )
