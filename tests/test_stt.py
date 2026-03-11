"""Tests for speech-to-text providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.voice.config import STTConfig
from agent.voice.stt import (
    DeepgramSTT,
    LLMNativeSTT,
    WhisperAPISTT,
    WhisperLocalSTT,
    create_stt,
)


class TestLLMNativeSTT:
    async def test_returns_sentinel_marker(self) -> None:
        stt = LLMNativeSTT()
        result = await stt.transcribe(b"fake audio", "audio/ogg")
        assert result.text == "__LLM_NATIVE__"
        assert result.confidence == 1.0


class TestWhisperAPISTT:
    async def test_calls_openai_endpoint(self) -> None:
        config = STTConfig(provider="whisper_api", whisper_model="whisper-1")

        mock_response = MagicMock()
        mock_response.json.return_value = {"text": "Hello world", "language": "en"}
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

            stt = WhisperAPISTT(config)
            result = await stt.transcribe(b"fake audio", "audio/ogg")

            assert result.text == "Hello world"
            assert result.language == "en"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "transcriptions" in call_args[0][0]

    async def test_raises_without_api_key(self) -> None:
        config = STTConfig(provider="whisper_api")
        stt = WhisperAPISTT(config)

        with patch.dict("os.environ", {}, clear=True), \
                pytest.raises(ValueError, match="OPENAI_API_KEY"):
            await stt.transcribe(b"audio", "audio/ogg")


class TestWhisperLocalSTT:
    async def test_transcribes_via_model(self) -> None:
        config = STTConfig(provider="whisper_local", whisper_local_model="base")

        mock_segment = MagicMock()
        mock_segment.text = "Hello from whisper"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 3.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        stt = WhisperLocalSTT(config)
        stt._model = mock_model

        result = await stt.transcribe(b"fake audio data", "audio/ogg")

        assert result.text == "Hello from whisper"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.duration_seconds == 3.5


class TestDeepgramSTT:
    async def test_calls_deepgram_endpoint(self) -> None:
        config = STTConfig(provider="deepgram", deepgram_model="nova-2")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {"transcript": "Hello deepgram", "confidence": 0.98}
                        ],
                        "detected_language": "en",
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"DEEPGRAM_API_KEY": "test-key"}),
            patch("httpx.AsyncClient") as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            stt = DeepgramSTT(config)
            result = await stt.transcribe(b"fake audio", "audio/ogg")

            assert result.text == "Hello deepgram"
            assert result.confidence == 0.98

    async def test_raises_without_api_key(self) -> None:
        config = STTConfig(provider="deepgram")
        stt = DeepgramSTT(config)

        with patch.dict("os.environ", {}, clear=True), \
                pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
            await stt.transcribe(b"audio", "audio/ogg")


class TestCreateSTT:
    def test_creates_llm_native(self) -> None:
        config = STTConfig(provider="llm_native")
        stt = create_stt(config)
        assert isinstance(stt, LLMNativeSTT)

    def test_creates_whisper_api(self) -> None:
        config = STTConfig(provider="whisper_api")
        stt = create_stt(config)
        assert isinstance(stt, WhisperAPISTT)

    def test_creates_whisper_local(self) -> None:
        config = STTConfig(provider="whisper_local")
        stt = create_stt(config)
        assert isinstance(stt, WhisperLocalSTT)

    def test_creates_deepgram(self) -> None:
        config = STTConfig(provider="deepgram")
        stt = create_stt(config)
        assert isinstance(stt, DeepgramSTT)

    def test_unknown_provider_raises(self) -> None:
        config = STTConfig(provider="nonexistent")
        with pytest.raises(ValueError, match="Unknown STT provider"):
            create_stt(config)
