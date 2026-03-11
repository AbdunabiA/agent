"""Messaging channels — abstract interface and adapters.

TelegramChannel is not re-exported here because aiogram is an optional dependency.
Import it directly: ``from agent.channels.telegram import TelegramChannel``
"""

from agent.channels.base import Attachment, BaseChannel, IncomingMessage, OutgoingMessage

__all__ = [
    "Attachment",
    "BaseChannel",
    "IncomingMessage",
    "OutgoingMessage",
]
