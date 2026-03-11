"""Tests for text-to-speech providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.voice.config import TTSConfig
from agent.voice.tts import (
    EdgeTTS,
    OpenAITTS,
    clean_for_speech,
    create_tts,
)


class TestCleanForSpeech:
    def test_removes_code_blocks(self) -> None:
        text = "Here is code:\n```python\nprint('hello')\n```\nDone."
        result = clean_for_speech(text)
        assert "print" not in result
        assert "code block omitted" in result

    def test_removes_inline_code(self) -> None:
        text = "Use the `pip install` command."
        result = clean_for_speech(text)
        assert "`" not in result

    def test_removes_markdown_bold(self) -> None:
        text = "This is **bold** text."
        result = clean_for_speech(text)
        assert "**" not in result
        assert "bold" in result

    def test_removes_markdown_italic(self) -> None:
        text = "This is *italic* text."
        result = clean_for_speech(text)
        assert result.count("*") == 0
        assert "italic" in result

    def test_removes_headers(self) -> None:
        text = "## My Header\nSome content."
        result = clean_for_speech(text)
        assert "##" not in result
        assert "My Header" in result

    def test_replaces_urls(self) -> None:
        text = "Visit https://example.com/page for details."
        result = clean_for_speech(text)
        assert "https://" not in result
        assert "link" in result

    def test_removes_markdown_links(self) -> None:
        text = "Click [here](https://example.com) for more."
        result = clean_for_speech(text)
        assert "[" not in result
        assert "here" in result

    def test_preserves_punctuation(self) -> None:
        text = "Hello, world! How are you? Fine."
        result = clean_for_speech(text)
        assert "," in result
        assert "!" in result
        assert "?" in result

    def test_normalizes_whitespace(self) -> None:
        text = "First paragraph.\n\n\nSecond paragraph."
        result = clean_for_speech(text)
        assert "\n\n\n" not in result


class TestEdgeTTS:
    async def test_synthesizes_text(self) -> None:
        config = TTSConfig(provider="edge_tts", output_format="mp3")

        mock_chunks = [
            {"type": "audio", "data": b"chunk1"},
            {"type": "audio", "data": b"chunk2"},
            {"type": "WordBoundary", "data": None},
        ]

        with patch("edge_tts.Communicate") as mock_comm_cls:
            mock_comm = MagicMock()

            async def mock_stream():
                for chunk in mock_chunks:
                    yield chunk

            mock_comm.stream = mock_stream
            mock_comm_cls.return_value = mock_comm

            tts = EdgeTTS(config)
            result = await tts.synthesize("Hello world")

            assert result.audio_data == b"chunk1chunk2"
            assert result.mime_type == "audio/mpeg"
            assert result.voice == "en-US-AriaNeural"

    async def test_truncates_long_text(self) -> None:
        config = TTSConfig(provider="edge_tts", max_text_length=10, output_format="mp3")

        with patch("edge_tts.Communicate") as mock_comm_cls:
            mock_comm = MagicMock()

            async def mock_stream():
                yield {"type": "audio", "data": b"audio"}

            mock_comm.stream = mock_stream
            mock_comm_cls.return_value = mock_comm

            tts = EdgeTTS(config)
            result = await tts.synthesize("A" * 100)
            assert result.audio_data == b"audio"

    async def test_raises_on_empty_text(self) -> None:
        config = TTSConfig(provider="edge_tts")
        tts = EdgeTTS(config)

        with pytest.raises(ValueError, match="No speakable text"):
            await tts.synthesize("   ")

    async def test_list_voices(self) -> None:
        config = TTSConfig(provider="edge_tts")

        mock_voices = [
            {
                "ShortName": "en-US-AriaNeural", "Gender": "Female",
                "Locale": "en-US", "FriendlyName": "Aria",
            },
            {
                "ShortName": "ru-RU-SvetlanaNeural", "Gender": "Female",
                "Locale": "ru-RU", "FriendlyName": "Svetlana",
            },
        ]

        with patch("edge_tts.list_voices", new_callable=AsyncMock, return_value=mock_voices):
            tts = EdgeTTS(config)
            voices = await tts.list_voices()
            assert len(voices) == 2
            assert voices[0]["name"] == "en-US-AriaNeural"

    async def test_list_voices_filtered(self) -> None:
        config = TTSConfig(provider="edge_tts")

        mock_voices = [
            {
                "ShortName": "en-US-AriaNeural", "Gender": "Female",
                "Locale": "en-US", "FriendlyName": "Aria",
            },
            {
                "ShortName": "ru-RU-SvetlanaNeural", "Gender": "Female",
                "Locale": "ru-RU", "FriendlyName": "Svetlana",
            },
        ]

        with patch("edge_tts.list_voices", new_callable=AsyncMock, return_value=mock_voices):
            tts = EdgeTTS(config)
            voices = await tts.list_voices("ru")
            assert len(voices) == 1
            assert voices[0]["name"] == "ru-RU-SvetlanaNeural"

    async def test_opus_conversion_fallback(self) -> None:
        """When ffmpeg is missing, falls back to mp3."""
        config = TTSConfig(provider="edge_tts", output_format="opus")

        with patch("edge_tts.Communicate") as mock_comm_cls:
            mock_comm = MagicMock()

            async def mock_stream():
                yield {"type": "audio", "data": b"mp3data"}

            mock_comm.stream = mock_stream
            mock_comm_cls.return_value = mock_comm

            # Mock _convert_to_opus to return None (ffmpeg not found)
            opus_patch = "agent.voice.tts._convert_to_opus"
            with patch(opus_patch, new_callable=AsyncMock, return_value=None):
                tts = EdgeTTS(config)
                result = await tts.synthesize("Hello")
                # Should fall back to mp3
                assert result.audio_data == b"mp3data"
                assert result.mime_type == "audio/mpeg"


class TestOpenAITTS:
    async def test_calls_openai_endpoint(self) -> None:
        config = TTSConfig(
            provider="openai", openai_model="tts-1",
            openai_voice="alloy", output_format="mp3",
        )

        mock_response = MagicMock()
        mock_response.content = b"audio_bytes"
        mock_response.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            tts = OpenAITTS(config)
            result = await tts.synthesize("Hello")

            assert result.audio_data == b"audio_bytes"
            assert result.voice == "alloy"

    async def test_list_voices(self) -> None:
        config = TTSConfig(provider="openai")
        tts = OpenAITTS(config)
        voices = await tts.list_voices()
        assert len(voices) == 6
        names = [v["name"] for v in voices]
        assert "alloy" in names
        assert "nova" in names


class TestCreateTTS:
    def test_creates_edge_tts(self) -> None:
        config = TTSConfig(provider="edge_tts")
        tts = create_tts(config)
        assert isinstance(tts, EdgeTTS)

    def test_creates_openai_tts(self) -> None:
        config = TTSConfig(provider="openai")
        tts = create_tts(config)
        assert isinstance(tts, OpenAITTS)

    def test_unknown_provider_raises(self) -> None:
        config = TTSConfig(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            create_tts(config)
