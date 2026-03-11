"""Text-to-speech providers.

Converts text to audio. Primary provider: edge-tts (free, high quality).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from agent.voice.config import TTSConfig

logger = structlog.get_logger(__name__)


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""

    audio_data: bytes
    mime_type: str  # "audio/ogg", "audio/mpeg"
    duration_seconds: float
    voice: str


class BaseTTS(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """Convert text to speech audio."""

    @abstractmethod
    async def list_voices(self, language: str = "") -> list[dict[str, str]]:
        """List available voices, optionally filtered by language."""


def clean_for_speech(text: str) -> str:
    """Clean text for natural speech output.

    Removes markdown formatting, code blocks, URLs, and excessive whitespace.
    Keeps natural punctuation and numbers.
    """
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " (code block omitted) ", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown emphasis
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # italic

    # Remove markdown links, keep text (before URL replacement)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Replace URLs
    text = re.sub(r"https?://\S+", "link", text)

    # Remove bullet points
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


class EdgeTTS(BaseTTS):
    """Microsoft Edge TTS — free, high quality, 300+ voices, 75+ languages.

    Popular voices:
    - en-US-AriaNeural (female, natural)
    - en-US-GuyNeural (male, natural)
    - ru-RU-SvetlanaNeural (Russian female)
    - uz-UZ-MadinaNeural (Uzbek female)
    - uz-UZ-SardorNeural (Uzbek male)
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> TTSResult:
        """Convert text to speech using edge-tts."""
        import edge_tts  # type: ignore[import-untyped]

        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length] + "..."
            logger.warning("tts_text_truncated", max_length=self.config.max_text_length)

        text = clean_for_speech(text)

        if not text.strip():
            raise ValueError("No speakable text after cleaning")

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.config.edge_voice,
            rate=self.config.edge_rate,
            pitch=self.config.edge_pitch,
        )

        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_data = b"".join(audio_chunks)

        if not audio_data:
            raise ValueError("TTS produced no audio output")

        # edge-tts outputs mp3 by default
        mime_type = "audio/mpeg"

        # Convert to opus if requested
        if self.config.output_format == "opus":
            converted = await _convert_to_opus(audio_data)
            if converted is not None:
                audio_data = converted
                mime_type = "audio/ogg"

        # Estimate duration (~150 words per minute)
        word_count = len(text.split())
        estimated_duration = word_count / 150 * 60

        return TTSResult(
            audio_data=audio_data,
            mime_type=mime_type,
            duration_seconds=estimated_duration,
            voice=self.config.edge_voice,
        )

    async def list_voices(self, language: str = "") -> list[dict[str, str]]:
        """List available edge-tts voices."""
        import edge_tts  # type: ignore[import-untyped]

        voices: list[dict[str, Any]] = await edge_tts.list_voices()

        if language:
            voices = [v for v in voices if v.get("Locale", "").startswith(language)]

        return [
            {
                "name": v["ShortName"],
                "gender": v.get("Gender", ""),
                "language": v.get("Locale", ""),
                "friendly_name": v.get("FriendlyName", ""),
            }
            for v in voices
        ]


class OpenAITTS(BaseTTS):
    """OpenAI TTS API.

    Voices: alloy, echo, fable, onyx, nova, shimmer
    Models: tts-1 (fast), tts-1-hd (high quality)
    """

    def __init__(self, config: TTSConfig) -> None:
        self.config = config

    async def synthesize(self, text: str) -> TTSResult:
        """Convert text to speech using OpenAI TTS API."""
        import httpx

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI TTS")

        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]

        text = clean_for_speech(text)

        response_format = "opus" if self.config.output_format == "opus" else "mp3"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.config.openai_model,
                    "voice": self.config.openai_voice,
                    "input": text,
                    "response_format": response_format,
                },
                timeout=30,
            )
            resp.raise_for_status()

        mime_type = "audio/ogg" if self.config.output_format == "opus" else "audio/mpeg"

        return TTSResult(
            audio_data=resp.content,
            mime_type=mime_type,
            duration_seconds=len(text.split()) / 150 * 60,
            voice=self.config.openai_voice,
        )

    async def list_voices(self, language: str = "") -> list[dict[str, str]]:
        """List available OpenAI TTS voices."""
        return [
            {"name": "alloy", "gender": "neutral", "language": "multilingual"},
            {"name": "echo", "gender": "male", "language": "multilingual"},
            {"name": "fable", "gender": "male", "language": "multilingual"},
            {"name": "onyx", "gender": "male", "language": "multilingual"},
            {"name": "nova", "gender": "female", "language": "multilingual"},
            {"name": "shimmer", "gender": "female", "language": "multilingual"},
        ]


async def _convert_to_opus(mp3_data: bytes) -> bytes | None:
    """Convert MP3 to OGG/Opus using ffmpeg.

    Returns None if ffmpeg is not available (caller falls back to mp3).
    """
    fd_in, input_path = tempfile.mkstemp(suffix=".mp3")
    output_path = input_path.replace(".mp3", ".ogg")

    try:
        os.write(fd_in, mp3_data)
        os.close(fd_in)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            input_path,
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-y",
            output_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

        if proc.returncode == 0:
            data = await asyncio.to_thread(Path(output_path).read_bytes)
            return data
        else:
            logger.warning("opus_conversion_failed", returncode=proc.returncode)
            return None
    except FileNotFoundError:
        logger.debug("ffmpeg_not_found", msg="falling back to mp3")
        return None
    finally:
        for p in [input_path, output_path]:
            with contextlib.suppress(OSError):
                os.unlink(p)


def create_tts(config: TTSConfig) -> BaseTTS:
    """Factory function to create the configured TTS provider."""
    providers: dict[str, callable] = {
        "edge_tts": lambda: EdgeTTS(config),
        "openai": lambda: OpenAITTS(config),
    }

    factory = providers.get(config.provider)
    if not factory:
        raise ValueError(
            f"Unknown TTS provider: {config.provider}. "
            f"Available: {list(providers.keys())}"
        )

    return factory()
