"""Proactive messenger — sends unsolicited messages to users.

Used by heartbeat, monitors, and scheduled tasks to proactively
notify users without waiting for them to send a message first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent.core.events import Events

if TYPE_CHECKING:
    from agent.channels.base import BaseChannel
    from agent.core.events import EventBus

logger = structlog.get_logger(__name__)


class ProactiveMessenger:
    """Sends proactive messages to users via available channels.

    Holds references to all messaging channels and provides
    methods for sending unsolicited notifications.
    """

    def __init__(
        self,
        event_bus: EventBus,
        channels: list[BaseChannel] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self._channels: list[BaseChannel] = channels or []
        self._channels_by_name: dict[str, BaseChannel] = {}
        self._update_channel_index()

    def add_channel(self, channel: BaseChannel) -> None:
        """Register a channel for proactive messaging.

        Args:
            channel: The channel to add.
        """
        self._channels.append(channel)
        self._update_channel_index()

    def _update_channel_index(self) -> None:
        """Rebuild the name -> channel lookup."""
        self._channels_by_name = {ch.name: ch for ch in self._channels}

    async def send_proactive(
        self,
        user_id: str,
        content: str,
        channel: str | None = None,
    ) -> bool:
        """Send an unsolicited message to a specific user.

        Args:
            user_id: Target user ID.
            content: Message content.
            channel: Optional channel name. Uses first available if not specified.

        Returns:
            True if sent successfully, False otherwise.
        """
        from agent.channels.base import OutgoingMessage

        target = None
        if channel and channel in self._channels_by_name:
            target = self._channels_by_name[channel]
        elif self._channels:
            target = self._channels[0]

        if not target:
            logger.warning(
                "proactive_no_channel",
                user_id=user_id,
                content=content[:100],
            )
            return False

        try:
            await target.send_message(
                OutgoingMessage(
                    content=content,
                    channel_user_id=user_id,
                    parse_mode="Markdown",
                )
            )

            await self.event_bus.emit(Events.PROACTIVE_MESSAGE, {
                "user_id": user_id,
                "channel": target.name,
                "content": content[:200],
            })

            logger.info(
                "proactive_sent",
                user_id=user_id,
                channel=target.name,
                length=len(content),
            )
            return True

        except Exception as e:
            logger.error(
                "proactive_send_failed",
                user_id=user_id,
                channel=target.name,
                error=str(e),
            )
            return False

    async def send_to_all_known_users(self, content: str) -> int:
        """Broadcast a message to all channels (e.g., morning briefing).

        Iterates over registered channels and calls their ``broadcast``
        method if available, falling back to ``send_message`` with each
        known user ID from ``get_known_user_ids``.  Channels that
        implement neither are skipped.

        Args:
            content: Message to broadcast.

        Returns:
            Number of channels notified successfully.
        """
        from agent.channels.base import OutgoingMessage

        sent = 0
        for channel in self._channels:
            try:
                # Prefer a channel-level broadcast if implemented
                if hasattr(channel, "broadcast"):
                    await channel.broadcast(content)
                    sent += 1
                    continue

                # Otherwise send individually to known users
                user_ids: list[str] = []
                if hasattr(channel, "get_known_user_ids"):
                    user_ids = await channel.get_known_user_ids()

                if not user_ids:
                    logger.debug(
                        "broadcast_no_users",
                        channel=channel.name,
                    )
                    continue

                for uid in user_ids:
                    try:
                        await channel.send_message(
                            OutgoingMessage(
                                content=content,
                                channel_user_id=uid,
                                parse_mode="Markdown",
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "broadcast_user_failed",
                            channel=channel.name,
                            user_id=uid,
                            error=str(e),
                        )
                sent += 1
            except Exception as e:
                logger.warning(
                    "broadcast_channel_failed",
                    channel=channel.name,
                    error=str(e),
                )

        return sent
