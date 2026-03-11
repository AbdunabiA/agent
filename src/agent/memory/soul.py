"""Soul.md loader — agent personality system.

Loads, caches, watches for changes, and supports updating
the soul.md personality file that drives the agent's system prompt.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_SOUL = """\
# Agent Soul

You are **Agent**, a personal AI assistant running locally on the user's machine.

## Personality
- You are helpful, concise, and proactive
- You speak naturally, like a knowledgeable colleague
- You are honest about what you don't know
- You prefer action over discussion — when you can do something, you do it

## Behavior
- When asked to do something, plan your approach first for complex tasks
- If something fails, try a different approach before giving up
- Remember what the user tells you and use that context
- Be proactive: if you notice something relevant, mention it

## Communication Style
- Keep responses concise unless the user asks for detail
- Use code blocks for code, commands, and technical output
- Don't use excessive emojis or formatting
- Match the user's language (respond in the language they write in)
"""


class SoulLoader:
    """Loads and caches soul.md personality content.

    Searches for soul.md in multiple locations, caches the content,
    and supports live-reload when the file changes on disk.

    Usage::

        loader = SoulLoader()
        content = loader.load()
        if loader.reload_if_changed():
            content = loader.load()
        loader.update("New personality content")
    """

    def __init__(self, explicit_path: str | None = None) -> None:
        self._explicit_path = explicit_path
        self._cached_content: str | None = None
        self._cached_mtime: float | None = None
        self._resolved_path: Path | None = None

    def load(self) -> str:
        """Load soul.md content from the first available location.

        Search order:
        1. Explicit path (from config or constructor)
        2. ./soul.md (current working directory)
        3. ~/.config/agent/soul.md

        Returns:
            Soul content string. Falls back to built-in default if no file found.
        """
        if self._cached_content is not None:
            return self._cached_content

        path = self._find_path()
        if path is not None:
            self._resolved_path = path
            self._cached_content = path.read_text(encoding="utf-8")
            self._cached_mtime = path.stat().st_mtime
            logger.info("soul_loaded", path=str(path))
            return self._cached_content

        self._cached_content = _default_soul()
        logger.info("soul_using_default")
        return self._cached_content

    def reload_if_changed(self) -> bool:
        """Check if soul.md has changed on disk and reload if so.

        Returns:
            True if the content was reloaded, False otherwise.
        """
        if self._resolved_path is None or not self._resolved_path.exists():
            return False

        current_mtime = self._resolved_path.stat().st_mtime
        if self._cached_mtime is not None and current_mtime > self._cached_mtime:
            self._cached_content = self._resolved_path.read_text(encoding="utf-8")
            self._cached_mtime = current_mtime
            logger.info("soul_reloaded", path=str(self._resolved_path))
            return True

        return False

    def update(self, content: str) -> None:
        """Write new content to soul.md and update cache.

        Writes to the resolved path, or creates ./soul.md if no path was found.

        Args:
            content: New soul.md content.
        """
        path = self._resolved_path or Path("soul.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self._resolved_path = path
        self._cached_content = content
        self._cached_mtime = path.stat().st_mtime
        logger.info("soul_updated", path=str(path))

    async def async_update(self, content: str) -> None:
        """Async version of update() — avoids blocking the event loop."""
        path = self._resolved_path or Path("soul.md")
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        self._resolved_path = path
        self._cached_content = content
        self._cached_mtime = (await asyncio.to_thread(path.stat)).st_mtime
        logger.info("soul_updated", path=str(path))

    @property
    def content(self) -> str:
        """Get cached content, loading if needed."""
        return self.load()

    @property
    def path(self) -> Path | None:
        """Get the resolved path, if any."""
        return self._resolved_path

    def _find_path(self) -> Path | None:
        """Find the first available soul.md file."""
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(Path(self._explicit_path))

        candidates.append(Path("soul.md"))
        candidates.append(Path.home() / ".config" / "agent" / "soul.md")

        for path in candidates:
            if path.exists() and path.is_file():
                return path

        return None


def _default_soul() -> str:
    """Return the built-in default soul content."""
    return _DEFAULT_SOUL
