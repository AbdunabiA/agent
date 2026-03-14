"""Tests for Telegram media handlers (voice, photo, document) and /soul command."""

from __future__ import annotations

import contextlib
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.channels.telegram import TelegramChannel
from agent.core.events import EventBus
from agent.core.session import SessionStore


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def session_store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def mock_agent_loop() -> MagicMock:
    loop = MagicMock()
    loop.system_prompt = "You are a helpful assistant."
    loop.llm = MagicMock()
    loop.llm.completion = AsyncMock()
    loop.process_message = AsyncMock()
    return loop


def _make_telegram_config(token: str = "fake:token") -> MagicMock:
    cfg = MagicMock()
    cfg.token = token
    cfg.allowed_users = []
    return cfg


def _make_message(
    user_id: int = 42,
    text: str | None = None,
    caption: str | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.text = text
    msg.caption = caption
    msg.answer = AsyncMock()
    return msg


@pytest.fixture
def channel(
    event_bus: EventBus,
    session_store: SessionStore,
    mock_agent_loop: MagicMock,
) -> TelegramChannel:
    """Create a TelegramChannel with mocked bot (no aiogram import needed for logic tests)."""
    with patch("agent.channels.telegram.AIOGRAM_AVAILABLE", True):
        cfg = _make_telegram_config()
        with patch("agent.channels.telegram.Bot"), \
             patch("agent.channels.telegram.Dispatcher"), \
             patch("agent.channels.telegram.Router"):
            ch = TelegramChannel(
                config=cfg,
                event_bus=event_bus,
                session_store=session_store,
                agent_loop=mock_agent_loop,
            )
            ch._bot = MagicMock()
            ch._bot.get_file = AsyncMock()
            ch._bot.download_file = AsyncMock()
            ch._bot.send_message = AsyncMock()
            ch._bot.send_chat_action = AsyncMock()
            return ch


async def _drain_background_tasks(channel: Any) -> None:
    """Await all background tasks spawned by the channel."""
    all_tasks = [
        task
        for tasks in list(channel._background_tasks.values())
        for task, _desc in list(tasks)
    ]
    for task in all_tasks:
        if not task.done():
            with contextlib.suppress(Exception):
                await task


class TestVoice:
    """Voice message handling tests."""

    async def test_voice_downloads_and_forwards_to_llm(
        self, channel: TelegramChannel, mock_agent_loop: MagicMock
    ) -> None:
        """Voice message should be downloaded, base64-encoded, and sent to LLM."""
        msg = _make_message()
        msg.voice = MagicMock()
        msg.voice.file_id = "voice123"

        # Mock file download
        file_mock = MagicMock()
        file_mock.file_path = "voice/file.ogg"
        channel._bot.get_file.return_value = file_mock

        async def fake_download(path: str, buf: BytesIO) -> None:
            buf.write(b"fake-ogg-data")

        channel._bot.download_file.side_effect = fake_download

        # Mock LLM response
        llm_response = MagicMock()
        llm_response.content = "I heard your voice message."
        mock_agent_loop.llm.completion.return_value = llm_response

        await channel._handle_voice(msg)
        await _drain_background_tasks(channel)

        mock_agent_loop.llm.completion.assert_awaited_once()
        call_args = mock_agent_loop.llm.completion.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else None)
        assert messages is not None
        # Last message should have multimodal content
        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)

    async def test_voice_fallback_on_error(
        self, channel: TelegramChannel, mock_agent_loop: MagicMock
    ) -> None:
        """Voice handler should send fallback message on LLM error."""
        msg = _make_message()
        msg.voice = MagicMock()
        msg.voice.file_id = "voice456"

        channel._bot.get_file.side_effect = Exception("Download failed")

        await channel._handle_voice(msg)

        msg.answer.assert_awaited_once()
        answer_text = msg.answer.call_args[0][0]
        assert "couldn't process" in answer_text.lower()


class TestPhoto:
    """Photo message handling tests."""

    async def test_photo_with_caption(
        self, channel: TelegramChannel, mock_agent_loop: MagicMock
    ) -> None:
        """Photo with caption should include both image and text in multimodal content."""
        msg = _make_message(caption="What's in this image?")
        photo_mock = MagicMock()
        photo_mock.file_id = "photo123"
        msg.photo = [MagicMock(), photo_mock]  # last is highest res

        file_mock = MagicMock()
        file_mock.file_path = "photos/file.jpg"
        channel._bot.get_file.return_value = file_mock

        async def fake_download(path: str, buf: BytesIO) -> None:
            buf.write(b"fake-jpeg-data")

        channel._bot.download_file.side_effect = fake_download

        llm_response = MagicMock()
        llm_response.content = "I see a cat."
        mock_agent_loop.llm.completion.return_value = llm_response

        await channel._handle_photo(msg)

        mock_agent_loop.llm.completion.assert_awaited_once()
        call_args = mock_agent_loop.llm.completion.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else None)
        last_msg = messages[-1]
        content = last_msg["content"]
        assert isinstance(content, list)
        # Should have image_url and text blocks
        types = [block["type"] for block in content]
        assert "image_url" in types
        assert "text" in types

    async def test_photo_without_caption(
        self, channel: TelegramChannel, mock_agent_loop: MagicMock
    ) -> None:
        """Photo without caption should have only image_url block."""
        msg = _make_message()
        photo_mock = MagicMock()
        photo_mock.file_id = "photo789"
        msg.photo = [photo_mock]

        file_mock = MagicMock()
        file_mock.file_path = "photos/file2.jpg"
        channel._bot.get_file.return_value = file_mock

        async def fake_download(path: str, buf: BytesIO) -> None:
            buf.write(b"fake-jpeg-data")

        channel._bot.download_file.side_effect = fake_download

        llm_response = MagicMock()
        llm_response.content = "Nice photo."
        mock_agent_loop.llm.completion.return_value = llm_response

        await channel._handle_photo(msg)

        call_args = mock_agent_loop.llm.completion.call_args
        messages = call_args.kwargs.get("messages", call_args.args[0] if call_args.args else None)
        last_msg = messages[-1]
        content = last_msg["content"]
        types = [block["type"] for block in content]
        assert "image_url" in types
        assert "text" not in types


class TestDocument:
    """Document upload handling tests."""

    async def test_document_saved_and_agent_informed(
        self, channel: TelegramChannel, mock_agent_loop: MagicMock, tmp_path: Path
    ) -> None:
        """Document should be saved to uploads dir and agent informed."""
        msg = _make_message(caption="Check this file")
        msg.document = MagicMock()
        msg.document.file_id = "doc123"
        msg.document.file_name = "report.pdf"
        msg.document.mime_type = "application/pdf"
        msg.document.file_size = 12345

        file_mock = MagicMock()
        file_mock.file_path = "documents/report.pdf"
        channel._bot.get_file.return_value = file_mock

        async def fake_download(path: str, buf: BytesIO) -> None:
            buf.write(b"fake-pdf-content")

        channel._bot.download_file.side_effect = fake_download

        llm_response = MagicMock()
        llm_response.content = "I received your PDF."
        mock_agent_loop.process_message.return_value = llm_response

        with patch("agent.channels.telegram._UPLOAD_DIR", tmp_path / "uploads"):
            await channel._handle_document(msg)
            await _drain_background_tasks(channel)

        mock_agent_loop.process_message.assert_awaited_once()
        call_text = mock_agent_loop.process_message.call_args[0][0]
        assert "report.pdf" in call_text
        assert "12345" in call_text
        assert "application/pdf" in call_text


class TestSoulCommand:
    """Tests for the /soul command."""

    async def test_soul_displays_content(
        self, channel: TelegramChannel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'/soul' should display the current soul.md content."""
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("I am a friendly assistant.", encoding="utf-8")

        msg = _make_message(text="/soul")

        # Monkeypatch Path("soul.md") to point to tmp_path
        original_path = Path

        def patched_path(p: str) -> Path:
            if p == "soul.md":
                return soul_file
            return original_path(p)

        monkeypatch.setattr("agent.channels.telegram.Path", patched_path)
        await channel._cmd_soul(msg)

        msg.answer.assert_awaited_once()
        answer_text = msg.answer.call_args[0][0]
        assert "I am a friendly assistant." in answer_text

    async def test_soul_edit_writes_file(
        self, channel: TelegramChannel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'/soul edit <text>' should write new content to soul.md."""
        soul_file = tmp_path / "soul.md"

        msg = _make_message(text="/soul edit You are a pirate assistant.")

        original_path = Path

        def patched_path(p: str) -> Path:
            if p == "soul.md":
                return soul_file
            return original_path(p)

        monkeypatch.setattr("agent.channels.telegram.Path", patched_path)
        await channel._cmd_soul(msg)

        msg.answer.assert_awaited_once()
        answer_text = msg.answer.call_args[0][0]
        assert "updated" in answer_text.lower()
        assert soul_file.read_text(encoding="utf-8") == "You are a pirate assistant."
