"""Voice processing pipeline.

Orchestrates the full voice flow:
1. Receive audio -> STT -> text
2. Process text through agent loop -> response text
3. Response text -> TTS -> audio
4. Send audio back to channel
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent.core.events import Events
from agent.voice.config import VoiceConfig
from agent.voice.stt import BaseSTT, STTResult, create_stt
from agent.voice.tts import BaseTTS, TTSResult, create_tts

if TYPE_CHECKING:
    from agent.core.events import EventBus

logger = structlog.get_logger(__name__)


class VoicePipeline:
    """Manages the voice interaction flow."""

    def __init__(self, config: VoiceConfig, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self.stt: BaseSTT = create_stt(config.stt)
        self.tts: BaseTTS | None = create_tts(config.tts) if config.tts.enabled else None

    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/ogg") -> STTResult:
        """Transcribe audio to text.

        If provider is 'llm_native', returns a special marker.
        Otherwise, returns actual transcription.
        """
        result = await self.stt.transcribe(audio_data, mime_type)

        await self.event_bus.emit(
            Events.VOICE_TRANSCRIBED,
            {
                "text": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "provider": self.config.stt.provider,
            },
        )

        logger.info(
            "voice_transcribed",
            text_length=len(result.text),
            language=result.language,
            provider=self.config.stt.provider,
        )

        return result

    async def synthesize(self, text: str) -> TTSResult | None:
        """Convert response text to speech audio.

        Returns None if TTS is disabled.
        """
        if self.tts is None:
            return None

        try:
            result = await self.tts.synthesize(text)

            await self.event_bus.emit(
                Events.VOICE_SYNTHESIZED,
                {
                    "text_length": len(text),
                    "audio_size": len(result.audio_data),
                    "voice": result.voice,
                    "provider": self.config.tts.provider,
                },
            )

            logger.info(
                "voice_synthesized",
                text_length=len(text),
                audio_size=len(result.audio_data),
                voice=result.voice,
            )

            return result

        except Exception as e:
            logger.error("tts_failed", error=str(e))
            return None

    def should_voice_reply(self, channel: str) -> bool:
        """Check if voice reply is enabled for this channel."""
        return (
            self.config.auto_voice_reply
            and self.config.tts.enabled
            and channel in self.config.voice_reply_channels
        )

    def is_llm_native(self) -> bool:
        """Check if STT uses LLM native audio (no standalone transcription)."""
        return self.config.stt.provider == "llm_native"

    async def list_voices(self, language: str = "") -> list[dict[str, str]]:
        """List available TTS voices."""
        if self.tts is None:
            return []
        return await self.tts.list_voices(language)
