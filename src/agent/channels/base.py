"""Abstract base channel interface and message data classes.

Defines the contract all messaging channels (Telegram, WebChat, etc.) must implement,
plus the common data classes for incoming/outgoing messages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.core.events import EventBus
    from agent.core.session import SessionStore

logger = structlog.get_logger(__name__)


@dataclass
class Attachment:
    """A file or media attachment on a message."""

    type: str  # "photo", "document", "audio", "video", "voice", "sticker"
    file_id: str | None = None
    file_path: str | None = None
    mime_type: str | None = None
    file_name: str | None = None
    file_size: int | None = None
    data: bytes | None = None


@dataclass
class IncomingMessage:
    """A message received from a channel."""

    channel: str
    channel_user_id: str
    content: str
    message_id: str | None = None
    reply_to: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OutgoingMessage:
    """A message to send through a channel."""

    content: str
    channel_user_id: str
    attachments: list[Attachment] = field(default_factory=list)
    reply_to: str | None = None
    parse_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChannel(ABC):
    """Abstract base class for all messaging channels.

    Subclasses must implement:
        - name (property): Channel identifier string
        - start(): Begin receiving messages
        - stop(): Shut down the channel
        - send_message(msg): Send an outgoing message
        - send_typing(channel_user_id): Show typing indicator
    """

    def __init__(
        self,
        config: Any,
        event_bus: EventBus,
        session_store: SessionStore,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.session_store = session_store
        self._running: bool = False
        self._paused: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique channel identifier (e.g. 'telegram', 'webchat')."""
        ...

    @abstractmethod
    async def start(self) -> None:
        """Start receiving messages from this channel."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        ...

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> None:
        """Send a message through this channel.

        Args:
            message: The outgoing message to send.
        """
        ...

    @abstractmethod
    async def send_typing(self, channel_user_id: str) -> None:
        """Show a typing indicator to the user.

        Args:
            channel_user_id: The user to show typing to.
        """
        ...

    async def send_approval_request(
        self,
        channel_user_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        request_id: str,
    ) -> bool:
        """Ask the user to approve a tool execution.

        Default implementation auto-approves. Channels can override
        to show interactive approval UI.

        Args:
            channel_user_id: The user to ask.
            tool_name: Name of the tool requesting approval.
            arguments: Tool call arguments.
            request_id: Unique identifier for this approval request.

        Returns:
            True if approved, False if denied.
        """
        return True

    @property
    def is_running(self) -> bool:
        """Whether the channel is currently running."""
        return self._running

    def pause(self) -> None:
        """Pause message processing (messages are ignored)."""
        self._paused = True
        logger.info("channel_paused", channel=self.name)

    def resume(self) -> None:
        """Resume message processing."""
        self._paused = False
        logger.info("channel_resumed", channel=self.name)

    def _make_session_id(self, channel_user_id: str) -> str:
        """Create a deterministic session ID for a channel user.

        Args:
            channel_user_id: The user's ID within this channel.

        Returns:
            Session ID in the format 'channelname:userid'.
        """
        return f"{self.name}:{channel_user_id}"
