"""Voice configuration models."""

from __future__ import annotations

from pydantic import BaseModel


class STTConfig(BaseModel):
    """Speech-to-text configuration."""

    provider: str = "llm_native"
    # Providers:
    #   "llm_native"    — send audio to LLM directly (works with Gemini/GPT-4o)
    #   "whisper_api"   — OpenAI Whisper API
    #   "whisper_local" — Local faster-whisper
    #   "deepgram"      — Deepgram API

    # Whisper API settings
    whisper_model: str = "whisper-1"

    # Local whisper settings
    whisper_local_model: str = "base"  # tiny, base, small, medium, large
    whisper_local_device: str = "cpu"  # cpu or cuda

    # Deepgram settings
    deepgram_model: str = "nova-2"

    # Common settings
    language: str = ""  # Empty = auto-detect. ISO 639-1 code ("en", "uz", "ru")


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""

    enabled: bool = True
    provider: str = "edge_tts"
    # Providers:
    #   "edge_tts" — Microsoft Edge TTS (free, high quality, many languages)
    #   "openai"   — OpenAI TTS API

    # edge-tts settings
    edge_voice: str = "en-US-AriaNeural"
    edge_rate: str = "+0%"
    edge_pitch: str = "+0Hz"

    # OpenAI TTS settings
    openai_model: str = "tts-1"  # tts-1 or tts-1-hd
    openai_voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer

    # Common settings
    output_format: str = "opus"  # opus (small, good for Telegram), mp3, wav
    max_text_length: int = 4000


class VoiceConfig(BaseModel):
    """Top-level voice configuration."""

    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()

    # Voice mode behavior
    auto_voice_reply: bool = True
    voice_reply_channels: list[str] = ["telegram"]
    voice_transcription_prefix: bool = True  # Show transcription before responding
