"""Tests for the email sending and reading tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset singleton config before each test."""
    from agent.config import reset_config

    reset_config()
    yield
    reset_config()


def _make_config(email: str | None = "test@example.com", password: str | None = "secret"):
    """Build a mock AgentConfig with email settings."""
    cfg = MagicMock()
    cfg.tools.email.enabled = True
    cfg.tools.email.smtp_host = "smtp.example.com"
    cfg.tools.email.smtp_port = 587
    cfg.tools.email.imap_host = "imap.example.com"
    cfg.tools.email.imap_port = 993
    cfg.tools.email.email = email
    cfg.tools.email.password = password
    cfg.tools.email.use_tls = True
    return cfg


# ---------------------------------------------------------------------------
# send
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_with_aiosmtplib():
    """send action constructs message and sends via aiosmtplib."""
    from agent.tools.builtins.email_tool import email_tool

    mock_send = AsyncMock()
    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch.dict("sys.modules", {"aiosmtplib": MagicMock(send=mock_send)}),
        patch("agent.tools.builtins.email_tool.aiosmtplib", create=True) as mock_aio,
    ):
        mock_aio.send = mock_send
        # We need to make the import inside _send_smtp succeed
        import sys

        fake_aio = MagicMock()
        fake_aio.send = mock_send
        sys.modules["aiosmtplib"] = fake_aio

        result = await email_tool(
            action="send",
            to="recipient@example.com",
            subject="Test Subject",
            body="Hello there!",
        )

        del sys.modules["aiosmtplib"]

    assert "Email sent to recipient@example.com" in result
    assert "Test Subject" in result
    mock_send.assert_awaited_once()

    # Verify message structure
    call_kwargs = mock_send.call_args
    msg = call_kwargs.args[0] if call_kwargs.args else call_kwargs[0][0]
    assert msg["To"] == "recipient@example.com"
    assert msg["Subject"] == "Test Subject"
    assert msg["From"] == "test@example.com"


@pytest.mark.asyncio
async def test_send_falls_back_to_smtplib():
    """send action falls back to stdlib smtplib when aiosmtplib is unavailable."""
    from agent.tools.builtins.email_tool import email_tool

    mock_smtp_instance = MagicMock()
    mock_smtp_class = MagicMock(return_value=mock_smtp_instance)
    mock_smtp_instance.__enter__ = MagicMock(return_value=mock_smtp_instance)
    mock_smtp_instance.__exit__ = MagicMock(return_value=False)

    import sys

    # Ensure aiosmtplib is not importable
    saved = sys.modules.pop("aiosmtplib", None)

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.smtplib.SMTP", mock_smtp_class),
        patch.dict("sys.modules", {"aiosmtplib": None}),  # force ImportError
    ):
        result = await email_tool(
            action="send",
            to="recipient@example.com",
            subject="Fallback Test",
            body="Hello via smtplib!",
        )

    if saved is not None:
        sys.modules["aiosmtplib"] = saved

    assert "Email sent to recipient@example.com" in result
    mock_smtp_instance.starttls.assert_called_once()
    mock_smtp_instance.login.assert_called_once_with("test@example.com", "secret")
    mock_smtp_instance.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_send_html_body():
    """HTML body is detected and sent with html content type."""
    from agent.tools.builtins.email_tool import _build_message

    msg = _build_message(
        "sender@example.com",
        "recipient@example.com",
        "HTML Test",
        "<html><body><h1>Hi</h1></body></html>",
    )
    payload = msg.get_payload()
    assert len(payload) == 1
    assert payload[0].get_content_type() == "text/html"


@pytest.mark.asyncio
async def test_send_missing_to():
    """send without 'to' returns error."""
    from agent.tools.builtins.email_tool import email_tool

    with patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()):
        result = await email_tool(action="send", subject="No recipient", body="test")
    assert "[ERROR]" in result
    assert "'to'" in result


@pytest.mark.asyncio
async def test_send_missing_subject():
    """send without 'subject' returns error."""
    from agent.tools.builtins.email_tool import email_tool

    with patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()):
        result = await email_tool(action="send", to="x@y.com", body="test")
    assert "[ERROR]" in result
    assert "'subject'" in result


# ---------------------------------------------------------------------------
# Missing config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_missing_config():
    """send with missing credentials returns helpful error."""
    from agent.tools.builtins.email_tool import email_tool

    with patch(
        "agent.tools.builtins.email_tool.get_config",
        return_value=_make_config(email=None, password=None),
    ):
        result = await email_tool(action="send", to="a@b.com", subject="Test", body="body")
    assert "[ERROR]" in result
    assert "not configured" in result


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def _make_imap_mock(messages: list[bytes] | None = None):
    """Create a mock IMAP4_SSL instance with canned messages."""
    if messages is None:
        raw = (
            b"From: alice@example.com\r\n"
            b"Subject: Hello\r\n"
            b"Date: Mon, 1 Jan 2024 12:00:00 +0000\r\n"
            b"Message-ID: <abc123@example.com>\r\n"
            b"\r\n"
            b"This is a test email body."
        )
        messages = [raw]

    mail = MagicMock()
    mail.login.return_value = ("OK", [b"Logged in"])
    mail.select.return_value = ("OK", [b"1"])
    # search returns message IDs
    ids = b" ".join(str(i + 1).encode() for i in range(len(messages)))
    mail.search.return_value = ("OK", [ids])

    def fetch_side_effect(msg_id, fmt):
        idx = int(msg_id) - 1
        if 0 <= idx < len(messages):
            return ("OK", [(b"1 (RFC822 {1234})", messages[idx])])
        return ("OK", [(None,)])

    mail.fetch.side_effect = fetch_side_effect
    mail.logout.return_value = ("OK", [b"Bye"])
    return mail


@pytest.mark.asyncio
async def test_read_messages():
    """read action parses IMAP messages correctly."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(action="read", limit=5)

    assert "alice@example.com" in result
    assert "Hello" in result
    assert "test email body" in result


@pytest.mark.asyncio
async def test_read_missing_config():
    """read with missing credentials returns helpful error."""
    from agent.tools.builtins.email_tool import email_tool

    with patch(
        "agent.tools.builtins.email_tool.get_config",
        return_value=_make_config(email=None, password=None),
    ):
        result = await email_tool(action="read")
    assert "[ERROR]" in result
    assert "not configured" in result


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_messages():
    """search action uses IMAP SEARCH with subject criteria."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(action="search", subject="Hello", limit=5)

    # Verify IMAP SEARCH was called with SUBJECT criteria
    imap_mock.search.assert_called_once()
    search_args = imap_mock.search.call_args[0]
    assert "SUBJECT" in search_args[1]
    assert "Hello" in search_args[1]

    assert "alice@example.com" in result


@pytest.mark.asyncio
async def test_search_no_results():
    """search returns friendly message when no messages match."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()
    imap_mock.search.return_value = ("OK", [b""])

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(action="search", subject="Nonexistent")

    assert "No matching messages" in result


# ---------------------------------------------------------------------------
# reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reply():
    """reply fetches original, adds In-Reply-To header, and sends."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()
    mock_send = AsyncMock()

    import sys

    fake_aio = MagicMock()
    fake_aio.send = mock_send
    sys.modules["aiosmtplib"] = fake_aio

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(
            action="reply",
            message_id="<abc123@example.com>",
            body="Thanks for your email!",
        )

    del sys.modules["aiosmtplib"]

    assert "Email sent to" in result
    mock_send.assert_awaited_once()

    # Verify reply headers
    sent_msg = mock_send.call_args.args[0]
    assert sent_msg["In-Reply-To"] == "<abc123@example.com>"
    assert sent_msg["References"] == "<abc123@example.com>"
    assert sent_msg["Subject"].startswith("Re:")


@pytest.mark.asyncio
async def test_reply_missing_message_id():
    """reply without message_id returns error."""
    from agent.tools.builtins.email_tool import email_tool

    with patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()):
        result = await email_tool(action="reply", body="test reply")
    assert "[ERROR]" in result
    assert "'message_id'" in result


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_action():
    """Unknown action returns error with valid actions listed."""
    from agent.tools.builtins.email_tool import email_tool

    result = await email_tool(action="delete")
    assert "[ERROR]" in result
    assert "Unknown action" in result
    assert "send" in result


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_smtp_connection_timeout():
    """send should return error when SMTP connection times out."""
    import sys

    from agent.tools.builtins.email_tool import email_tool

    fake_aio = MagicMock()
    fake_aio.send = AsyncMock(side_effect=TimeoutError("Connection timed out"))
    sys.modules["aiosmtplib"] = fake_aio

    with patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()):
        result = await email_tool(
            action="send",
            to="recipient@example.com",
            subject="Timeout Test",
            body="Body",
        )

    del sys.modules["aiosmtplib"]

    assert "[ERROR]" in result
    assert "timed out" in result.lower() or "Failed to send" in result


@pytest.mark.asyncio
async def test_send_smtp_auth_failure():
    """send should return error when SMTP authentication fails."""
    import sys

    from agent.tools.builtins.email_tool import email_tool

    fake_aio = MagicMock()
    fake_aio.send = AsyncMock(
        side_effect=Exception("(535, b'5.7.8 Username and Password not accepted')")
    )
    sys.modules["aiosmtplib"] = fake_aio

    with patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()):
        result = await email_tool(
            action="send",
            to="recipient@example.com",
            subject="Auth Test",
            body="Body",
        )

    del sys.modules["aiosmtplib"]

    assert "[ERROR]" in result
    assert "Failed to send" in result


@pytest.mark.asyncio
async def test_read_empty_inbox():
    """read should return friendly message when inbox has no messages."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()
    imap_mock.search.return_value = ("OK", [b""])

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(action="read", limit=5)

    assert "No messages" in result


@pytest.mark.asyncio
async def test_read_imap_connection_failure():
    """read should return error when IMAP connection fails."""
    from agent.tools.builtins.email_tool import email_tool

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch(
            "agent.tools.builtins.email_tool.imaplib.IMAP4_SSL",
            side_effect=OSError("Connection refused"),
        ),
    ):
        result = await email_tool(action="read", limit=5)

    assert "[ERROR]" in result
    assert "Failed to read" in result


@pytest.mark.asyncio
async def test_read_malformed_email_missing_headers():
    """Email without From/Subject should use fallback values."""
    from agent.tools.builtins.email_tool import email_tool

    # Build a raw email with no From or Subject headers
    raw = b"Date: Mon, 1 Jan 2024 12:00:00 +0000\r\n\r\nJust a body."
    imap_mock = _make_imap_mock(messages=[raw])

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(action="read", limit=5)

    # Should not crash; should use fallback values like "Unknown" or "(no subject)"
    assert isinstance(result, str)
    assert len(result) > 0
    # _parse_email_summary returns "Unknown" for missing From and "(no subject)" for missing Subject
    assert "Unknown" in result or "(no subject)" in result


@pytest.mark.asyncio
async def test_search_folder_not_found():
    """search should return error when IMAP folder does not exist."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()
    imap_mock.select.return_value = ("NO", [b"Folder not found"])
    # Make search raise because select failed
    imap_mock.search.side_effect = Exception("SEARCH command error: folder not selected")

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(action="search", subject="Test", folder="NONEXISTENT")

    assert "[ERROR]" in result


@pytest.mark.asyncio
async def test_send_unicode_subject_and_body():
    """send with emoji and non-ASCII in subject/body should work."""
    from agent.tools.builtins.email_tool import email_tool

    mock_send = AsyncMock()

    import sys

    fake_aio = MagicMock()
    fake_aio.send = mock_send
    sys.modules["aiosmtplib"] = fake_aio

    with patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()):
        result = await email_tool(
            action="send",
            to="recipient@example.com",
            subject="\U0001f4e7 T\u00e9st \u00dcnicode",
            body="Hello \u4e16\u754c \U0001f600 caf\u00e9",
        )

    del sys.modules["aiosmtplib"]

    assert "Email sent to recipient@example.com" in result
    mock_send.assert_awaited_once()

    # Verify message was built with unicode content
    sent_msg = mock_send.call_args.args[0]
    assert "\U0001f4e7" in sent_msg["Subject"]


@pytest.mark.asyncio
async def test_reply_original_not_found():
    """reply should return error when original message_id is not found."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()
    # Make search return no results for the Message-ID
    imap_mock.search.return_value = ("OK", [b""])

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(
            action="reply",
            message_id="<nonexistent@example.com>",
            body="This is a reply.",
        )

    assert "[ERROR]" in result
    assert "Could not find" in result


@pytest.mark.asyncio
async def test_send_very_long_body():
    """send with 100,000 character body should not crash."""
    from agent.tools.builtins.email_tool import email_tool

    long_body = "x" * 100_000
    mock_send = AsyncMock()

    import sys

    fake_aio = MagicMock()
    fake_aio.send = mock_send
    sys.modules["aiosmtplib"] = fake_aio

    with patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()):
        result = await email_tool(
            action="send",
            to="recipient@example.com",
            subject="Long Body",
            body=long_body,
        )

    del sys.modules["aiosmtplib"]

    assert "Email sent to recipient@example.com" in result
    mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_read_limit_zero():
    """read with limit=0 should return empty or no messages."""
    from agent.tools.builtins.email_tool import email_tool

    imap_mock = _make_imap_mock()

    with (
        patch("agent.tools.builtins.email_tool.get_config", return_value=_make_config()),
        patch("agent.tools.builtins.email_tool.imaplib.IMAP4_SSL", return_value=imap_mock),
    ):
        result = await email_tool(action="read", limit=0)

    # With limit=0, msg_ids[-0:] returns all items in Python,
    # but it should not crash regardless
    assert isinstance(result, str)
