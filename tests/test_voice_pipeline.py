"""Tests for voice processing pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from agent.core.events import EventBus, Events
from agent.voice.config import STTConfig, TTSConfig, VoiceConfig
from agent.voice.pipeline import VoicePipeline
from agent.voice.stt import STTResult
from agent.voice.tts import TTSResult


class TestVoicePipeline:
    def _make_pipeline(self, **overrides) -> tuple[VoicePipeline, EventBus]:
        defaults = {
            "stt": STTConfig(provider="whisper_api"),
            "tts": TTSConfig(enabled=True, provider="edge_tts"),
            "auto_voice_reply": True,
            "voice_reply_channels": ["telegram"],
        }
        defaults.update(overrides)
        config = VoiceConfig(**defaults)
        event_bus = EventBus()
        pipeline = VoicePipeline(config, event_bus)
        return pipeline, event_bus

    async def test_transcribe_calls_stt_and_emits_event(self) -> None:
        pipeline, event_bus = self._make_pipeline()

        mock_result = STTResult(text="Hello", language="en", confidence=0.99)
        pipeline.stt = MagicMock()
        pipeline.stt.transcribe = AsyncMock(return_value=mock_result)

        events_received: list[dict] = []
        event_bus.on(Events.VOICE_TRANSCRIBED, lambda data: events_received.append(data))

        result = await pipeline.transcribe(b"audio", "audio/ogg")

        assert result.text == "Hello"
        assert result.language == "en"
        pipeline.stt.transcribe.assert_called_once_with(b"audio", "audio/ogg")
        assert len(events_received) == 1
        assert events_received[0]["text"] == "Hello"

    async def test_synthesize_calls_tts_and_emits_event(self) -> None:
        pipeline, event_bus = self._make_pipeline()

        mock_result = TTSResult(
            audio_data=b"audio_bytes",
            mime_type="audio/ogg",
            duration_seconds=2.5,
            voice="en-US-AriaNeural",
        )
        pipeline.tts = MagicMock()
        pipeline.tts.synthesize = AsyncMock(return_value=mock_result)

        events_received: list[dict] = []
        event_bus.on(Events.VOICE_SYNTHESIZED, lambda data: events_received.append(data))

        result = await pipeline.synthesize("Hello world")

        assert result is not None
        assert result.audio_data == b"audio_bytes"
        pipeline.tts.synthesize.assert_called_once_with("Hello world")
        assert len(events_received) == 1
        assert events_received[0]["voice"] == "en-US-AriaNeural"

    async def test_synthesize_returns_none_when_tts_disabled(self) -> None:
        pipeline, _ = self._make_pipeline(tts=TTSConfig(enabled=False))

        result = await pipeline.synthesize("Hello world")
        assert result is None

    async def test_synthesize_returns_none_on_error(self) -> None:
        pipeline, _ = self._make_pipeline()
        pipeline.tts = MagicMock()
        pipeline.tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS failed"))

        result = await pipeline.synthesize("Hello")
        assert result is None

    def test_should_voice_reply_telegram(self) -> None:
        pipeline, _ = self._make_pipeline(
            auto_voice_reply=True,
            voice_reply_channels=["telegram"],
        )
        assert pipeline.should_voice_reply("telegram") is True
        assert pipeline.should_voice_reply("webchat") is False

    def test_should_voice_reply_disabled(self) -> None:
        pipeline, _ = self._make_pipeline(auto_voice_reply=False)
        assert pipeline.should_voice_reply("telegram") is False

    def test_should_voice_reply_tts_disabled(self) -> None:
        pipeline, _ = self._make_pipeline(tts=TTSConfig(enabled=False))
        assert pipeline.should_voice_reply("telegram") is False

    def test_is_llm_native(self) -> None:
        pipeline, _ = self._make_pipeline(stt=STTConfig(provider="llm_native"))
        assert pipeline.is_llm_native() is True

    def test_is_not_llm_native(self) -> None:
        pipeline, _ = self._make_pipeline(stt=STTConfig(provider="whisper_api"))
        assert pipeline.is_llm_native() is False

    async def test_list_voices_delegates_to_tts(self) -> None:
        pipeline, _ = self._make_pipeline()
        mock_voices = [{"name": "test", "gender": "female", "language": "en"}]
        pipeline.tts = MagicMock()
        pipeline.tts.list_voices = AsyncMock(return_value=mock_voices)

        voices = await pipeline.list_voices("en")
        assert voices == mock_voices
        pipeline.tts.list_voices.assert_called_once_with("en")

    async def test_list_voices_empty_when_tts_disabled(self) -> None:
        pipeline, _ = self._make_pipeline(tts=TTSConfig(enabled=False))
        voices = await pipeline.list_voices()
        assert voices == []
