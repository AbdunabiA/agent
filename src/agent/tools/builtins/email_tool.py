"""Email sending and reading tool."""

from __future__ import annotations

import asyncio
import email as email_lib
import email.mime.multipart
import email.mime.text
import imaplib
import smtplib
from typing import Any

import structlog

from agent.config import get_config
from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)


def _get_email_config() -> Any:
    """Return the email tool config, or None if credentials are missing."""
    cfg = get_config().tools.email
    if not cfg.email or not cfg.password:
        return None
    return cfg


def _build_message(
    from_addr: str,
    to: str,
    subject: str,
    body: str,
    extra_headers: dict[str, str] | None = None,
) -> email.mime.multipart.MIMEMultipart:
    """Build a MIMEMultipart email message."""
    msg = email.mime.multipart.MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to
    msg["Subject"] = subject

    if extra_headers:
        for key, value in extra_headers.items():
            msg[key] = value

    content_type = "html" if body.strip().lower().startswith("<html") else "plain"
    msg.attach(email.mime.text.MIMEText(body, content_type))
    return msg


async def _send_smtp(msg: email.mime.multipart.MIMEMultipart, cfg: Any) -> str:
    """Send an email via SMTP, preferring async aiosmtplib if available."""
    try:
        import aiosmtplib

        await aiosmtplib.send(
            msg,
            hostname=cfg.smtp_host,
            port=cfg.smtp_port,
            username=cfg.email,
            password=cfg.password,
            start_tls=cfg.use_tls,
        )
    except ImportError:
        logger.debug("aiosmtplib_not_available", fallback="smtplib")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send_smtp_sync, msg, cfg)
    return f"Email sent to {msg['To']}: {msg['Subject']}"


def _send_smtp_sync(msg: email.mime.multipart.MIMEMultipart, cfg: Any) -> None:
    """Synchronous SMTP send fallback."""
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as server:
        if cfg.use_tls:
            server.starttls()
        server.login(cfg.email, cfg.password)
        server.send_message(msg)


def _imap_connect(cfg: Any) -> imaplib.IMAP4_SSL:
    """Connect and login to the IMAP server."""
    mail = imaplib.IMAP4_SSL(cfg.imap_host, cfg.imap_port)
    mail.login(cfg.email, cfg.password)
    return mail


def _parse_email_summary(raw_email: bytes) -> dict[str, str]:
    """Parse a raw email into a summary dict."""
    msg = email_lib.message_from_bytes(raw_email)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="replace")
                break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(errors="replace")

    return {
        "from": msg.get("From", "Unknown"),
        "subject": msg.get("Subject", "(no subject)"),
        "date": msg.get("Date", "Unknown"),
        "message_id": msg.get("Message-ID", ""),
        "preview": body[:100].replace("\n", " ").strip(),
    }


def _read_messages(folder: str, limit: int) -> str:
    """Read the last N messages from an IMAP folder (blocking)."""
    cfg = _get_email_config()
    if cfg is None:
        return "[ERROR] Email not configured. Set email and password in config."

    mail = _imap_connect(cfg)
    try:
        mail.select(folder, readonly=True)
        _, data = mail.search(None, "ALL")
        msg_ids = data[0].split()
        if not msg_ids:
            return "No messages found."

        # Get the last N message IDs
        selected = msg_ids[-limit:]
        results: list[str] = []
        for msg_id in reversed(selected):
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            if msg_data and msg_data[0] is not None:
                raw = msg_data[0]
                if isinstance(raw, tuple) and len(raw) > 1:
                    summary = _parse_email_summary(raw[1])
                    results.append(
                        f"From: {summary['from']}, "
                        f"Subject: {summary['subject']}, "
                        f"Date: {summary['date']}, "
                        f"Preview: {summary['preview']}"
                    )
        return "\n\n".join(results) if results else "No messages found."
    finally:
        mail.logout()


def _search_messages(subject: str, from_addr: str, folder: str, limit: int) -> str:
    """Search messages by subject/from criteria (blocking)."""
    cfg = _get_email_config()
    if cfg is None:
        return "[ERROR] Email not configured. Set email and password in config."

    mail = _imap_connect(cfg)
    try:
        mail.select(folder, readonly=True)

        criteria: list[str] = []
        if subject:
            criteria.append(f'SUBJECT "{subject}"')
        if from_addr:
            criteria.append(f'FROM "{from_addr}"')
        search_str = " ".join(criteria) if criteria else "ALL"

        _, data = mail.search(None, search_str)
        msg_ids = data[0].split()
        if not msg_ids:
            return "No matching messages found."

        selected = msg_ids[-limit:]
        results: list[str] = []
        for msg_id in reversed(selected):
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            if msg_data and msg_data[0] is not None:
                raw = msg_data[0]
                if isinstance(raw, tuple) and len(raw) > 1:
                    summary = _parse_email_summary(raw[1])
                    results.append(
                        f"From: {summary['from']}, "
                        f"Subject: {summary['subject']}, "
                        f"Date: {summary['date']}, "
                        f"Preview: {summary['preview']}"
                    )
        return "\n\n".join(results) if results else "No matching messages found."
    finally:
        mail.logout()


def _fetch_original_message(message_id: str, folder: str) -> dict[str, str] | None:
    """Fetch an original message by Message-ID for replying (blocking)."""
    cfg = _get_email_config()
    if cfg is None:
        return None

    mail = _imap_connect(cfg)
    try:
        mail.select(folder, readonly=True)
        _, data = mail.search(None, f'HEADER Message-ID "{message_id}"')
        msg_ids = data[0].split()
        if not msg_ids:
            return None

        _, msg_data = mail.fetch(msg_ids[0], "(RFC822)")
        if msg_data and msg_data[0] is not None:
            raw = msg_data[0]
            if isinstance(raw, tuple) and len(raw) > 1:
                summary = _parse_email_summary(raw[1])
                summary["message_id"] = message_id
                return summary
        return None
    finally:
        mail.logout()


@tool(
    name="email",
    description=("Send and read emails via SMTP/IMAP. " "Actions: send, read, search, reply"),
    tier=ToolTier.MODERATE,
)
async def email_tool(
    action: str,
    to: str = "",
    subject: str = "",
    body: str = "",
    message_id: str = "",
    folder: str = "INBOX",
    limit: int = 10,
) -> str:
    """Send and read emails.

    Args:
        action: One of 'send', 'read', 'search', 'reply'.
        to: Recipient email address (for send/reply).
        subject: Email subject (for send/search).
        body: Email body text. If it starts with '<html', sent as HTML.
        message_id: Message-ID header for reply action.
        folder: IMAP folder to read/search (default: INBOX).
        limit: Maximum messages to return for read/search (default: 10).

    Returns:
        Result message or formatted email list.
    """
    action = action.lower().strip()

    if action == "send":
        cfg = _get_email_config()
        if cfg is None:
            return (
                "[ERROR] Email not configured. "
                "Set tools.email.email and tools.email.password in agent.yaml, "
                "or EMAIL_ADDRESS and EMAIL_PASSWORD environment variables."
            )
        if not to:
            return "[ERROR] 'to' is required for send action."
        if not subject:
            return "[ERROR] 'subject' is required for send action."

        msg = _build_message(cfg.email, to, subject, body)
        try:
            return await _send_smtp(msg, cfg)
        except Exception as e:
            logger.error("email_send_failed", error=str(e))
            return f"[ERROR] Failed to send email: {e}"

    elif action == "read":
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, _read_messages, folder, limit)
        except Exception as e:
            logger.error("email_read_failed", error=str(e))
            return f"[ERROR] Failed to read emails: {e}"

    elif action == "search":
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, _search_messages, subject, to, folder, limit)
        except Exception as e:
            logger.error("email_search_failed", error=str(e))
            return f"[ERROR] Failed to search emails: {e}"

    elif action == "reply":
        cfg = _get_email_config()
        if cfg is None:
            return (
                "[ERROR] Email not configured. "
                "Set tools.email.email and tools.email.password in agent.yaml, "
                "or EMAIL_ADDRESS and EMAIL_PASSWORD environment variables."
            )
        if not message_id:
            return "[ERROR] 'message_id' is required for reply action."
        if not body:
            return "[ERROR] 'body' is required for reply action."

        loop = asyncio.get_event_loop()
        try:
            original = await loop.run_in_executor(None, _fetch_original_message, message_id, folder)
        except Exception as e:
            logger.error("email_fetch_original_failed", error=str(e))
            return f"[ERROR] Failed to fetch original message: {e}"

        if original is None:
            return f"[ERROR] Could not find message with ID: {message_id}"

        reply_to = to or original["from"]
        reply_subject = subject or (
            original["subject"]
            if original["subject"].startswith("Re:")
            else f"Re: {original['subject']}"
        )

        msg = _build_message(
            cfg.email,
            reply_to,
            reply_subject,
            body,
            extra_headers={"In-Reply-To": message_id, "References": message_id},
        )
        try:
            return await _send_smtp(msg, cfg)
        except Exception as e:
            logger.error("email_reply_failed", error=str(e))
            return f"[ERROR] Failed to send reply: {e}"

    else:
        return f"[ERROR] Unknown action: {action}. Valid actions: send, read, search, reply"
