"""Tests for SoulLoader."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from agent.memory.soul import SoulLoader, _default_soul


class TestSoulLoader:
    """Tests for SoulLoader class."""

    def test_load_from_explicit_path(self, tmp_path: Path) -> None:
        """Load soul.md from an explicit path."""
        soul_file = tmp_path / "custom_soul.md"
        soul_file.write_text("Custom personality", encoding="utf-8")

        loader = SoulLoader(explicit_path=str(soul_file))
        content = loader.load()

        assert content == "Custom personality"
        assert loader.path == soul_file

    def test_load_from_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Load soul.md from current working directory."""
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("CWD personality", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        loader = SoulLoader()
        content = loader.load()

        assert content == "CWD personality"

    def test_default_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fall back to default when no soul.md found."""
        monkeypatch.chdir(tmp_path)

        loader = SoulLoader()
        content = loader.load()

        assert content == _default_soul()
        assert "Agent Soul" in content

    def test_caching(self, tmp_path: Path) -> None:
        """Content is cached after first load."""
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("Original", encoding="utf-8")

        loader = SoulLoader(explicit_path=str(soul_file))
        first = loader.load()
        second = loader.load()

        assert first == second == "Original"

    def test_reload_if_changed_detects_change(self, tmp_path: Path) -> None:
        """reload_if_changed detects file modification."""
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("Version 1", encoding="utf-8")

        loader = SoulLoader(explicit_path=str(soul_file))
        loader.load()

        # Ensure different mtime
        time.sleep(0.05)
        soul_file.write_text("Version 2", encoding="utf-8")
        # Force mtime to be different
        new_mtime = os.path.getmtime(str(soul_file)) + 1
        os.utime(str(soul_file), (new_mtime, new_mtime))

        assert loader.reload_if_changed() is True
        assert loader.content == "Version 2"

    def test_reload_if_changed_no_change(self, tmp_path: Path) -> None:
        """reload_if_changed returns False when file unchanged."""
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("Stable", encoding="utf-8")

        loader = SoulLoader(explicit_path=str(soul_file))
        loader.load()

        assert loader.reload_if_changed() is False

    def test_reload_no_resolved_path(self) -> None:
        """reload_if_changed returns False when no path resolved."""
        loader = SoulLoader()
        assert loader.reload_if_changed() is False

    def test_update_writes_file(self, tmp_path: Path) -> None:
        """update() writes content to disk and updates cache."""
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("Old content", encoding="utf-8")

        loader = SoulLoader(explicit_path=str(soul_file))
        loader.load()
        loader.update("New content")

        assert loader.content == "New content"
        assert soul_file.read_text(encoding="utf-8") == "New content"

    def test_update_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """update() creates soul.md if it doesn't exist."""
        monkeypatch.chdir(tmp_path)

        loader = SoulLoader()
        loader.update("Brand new soul")

        assert (tmp_path / "soul.md").read_text(encoding="utf-8") == "Brand new soul"
        assert loader.content == "Brand new soul"

    def test_content_property(self, tmp_path: Path) -> None:
        """content property triggers load on first access."""
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("Via property", encoding="utf-8")

        loader = SoulLoader(explicit_path=str(soul_file))
        assert loader.content == "Via property"

    def test_explicit_path_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit path takes priority over CWD soul.md."""
        cwd_soul = tmp_path / "soul.md"
        cwd_soul.write_text("CWD", encoding="utf-8")
        custom_soul = tmp_path / "custom" / "soul.md"
        custom_soul.parent.mkdir()
        custom_soul.write_text("Custom", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        loader = SoulLoader(explicit_path=str(custom_soul))
        assert loader.load() == "Custom"
