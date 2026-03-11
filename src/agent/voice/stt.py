"""Speech-to-text providers.

Converts audio bytes to text. Supports multiple backends:
llm_native, whisper_api, whisper_local, deepgram.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog

from agent.voice.config import STTConfig

logger = structlog.get_logger(__name__)

_EXT_MAP: dict[str, str] = {
    "audio/ogg": "ogg",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/webm": "webm",
}


@dataclass
class STTResult:
    """Result from speech-to-text processing."""

    text: str
    language: str = ""
    confidence: float = 1.0
    duration_seconds: float = 0.0


class BaseSTT(ABC):
    """Abstract base class for STT providers."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/ogg") -> STTResult:
        """Transcribe audio to text."""


class LLMNativeSTT(BaseSTT):
    """Send audio directly to LLM for transcription.

    Works with Gemini 2.0 Flash, GPT-4o, etc.
    Returns a sentinel marker — the caller sends raw audio to the LLM.
    """

    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/ogg") -> STTResult:
        return STTResult(text="__LLM_NATIVE__")


class WhisperAPISTT(BaseSTT):
    """OpenAI Whisper API for speech-to-text."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config

    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/ogg") -> STTResult:
        """Send audio to OpenAI Whisper API."""
        import httpx

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for Whisper API STT")

        ext = _EXT_MAP.get(mime_type, "ogg")

        async with httpx.AsyncClient() as client:
            files = {"file": (f"audio.{ext}", audio_data, mime_type)}
            data: dict[str, str] = {"model": self.config.whisper_model}
            if self.config.language:
                data["language"] = self.config.language

            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

        return STTResult(
            text=result.get("text", ""),
            language=result.get("language", self.config.language),
        )


class WhisperLocalSTT(BaseSTT):
    """Local Whisper model using faster-whisper.

    Requires: pip install faster-whisper
    Downloads model on first use.
    """

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            self._model = WhisperModel(
                self.config.whisper_local_model,
                device=self.config.whisper_local_device,
                compute_type="int8" if self.config.whisper_local_device == "cpu" else "float16",
            )
        return self._model

    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/ogg") -> STTResult:
        """Transcribe with local Whisper model."""
        dot_ext_map = {"audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/wav": ".wav"}
        ext = dot_ext_map.get(mime_type, ".ogg")

        fd, temp_path = tempfile.mkstemp(suffix=ext)
        try:
            os.write(fd, audio_data)
            os.close(fd)

            model = self._get_model()
            loop = asyncio.get_event_loop()

            segments, info = await loop.run_in_executor(
                None,
                lambda: model.transcribe(  # type: ignore[union-attr]
                    temp_path,
                    language=self.config.language or None,
                ),
            )

            text_parts: list[str] = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            text = " ".join(text_parts)

            return STTResult(
                text=text,
                language=info.language,
                confidence=info.language_probability,
                duration_seconds=info.duration,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)


class DeepgramSTT(BaseSTT):
    """Deepgram API for speech-to-text."""

    def __init__(self, config: STTConfig) -> None:
        self.config = config

    async def transcribe(self, audio_data: bytes, mime_type: str = "audio/ogg") -> STTResult:
        """Send audio to Deepgram API."""
        import httpx

        api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY required for Deepgram STT")

        params: dict[str, str] = {"model": self.config.deepgram_model, "smart_format": "true"}
        if self.config.language:
            params["language"] = self.config.language
        else:
            params["detect_language"] = "true"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": mime_type,
                },
                params=params,
                content=audio_data,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

        channel = result["results"]["channels"][0]
        transcript = channel["alternatives"][0]

        return STTResult(
            text=transcript.get("transcript", ""),
            confidence=transcript.get("confidence", 0.0),
            language=channel.get("detected_language", ""),
        )


def create_stt(config: STTConfig) -> BaseSTT:
    """Factory function to create the configured STT provider."""
    providers: dict[str, callable] = {
        "llm_native": lambda: LLMNativeSTT(),
        "whisper_api": lambda: WhisperAPISTT(config),
        "whisper_local": lambda: WhisperLocalSTT(config),
        "deepgram": lambda: DeepgramSTT(config),
    }

    factory = providers.get(config.provider)
    if not factory:
        raise ValueError(
            f"Unknown STT provider: {config.provider}. "
            f"Available: {list(providers.keys())}"
        )

    return factory()
