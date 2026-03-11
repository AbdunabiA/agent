"""Tests for filesystem access controls — read/write path separation and deny_paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tools.registry import ToolPermissionError


class TestReadWriteSeparation:
    """Tests for split read/write path validation."""

    def test_read_path_allows_broad_access(self) -> None:
        """Read validation with root='/' should allow any valid path."""
        from agent.tools.builtins.filesystem import _validate_read_path

        with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
            mock_config.return_value = ("/", "~", [])
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "test.txt")
                Path(filepath).write_text("test")
                result = _validate_read_path(filepath)
                assert result == Path(filepath).resolve()

    def test_write_path_blocks_outside_write_root(self) -> None:
        """Write validation should block paths outside write_root."""
        from agent.tools.builtins.filesystem import _validate_write_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set write_root to a subdirectory
            write_root = os.path.join(tmpdir, "allowed")
            os.makedirs(write_root)

            with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
                mock_config.return_value = ("/", write_root, [])

                # Writing inside write_root should work
                allowed_path = os.path.join(write_root, "ok.txt")
                result = _validate_write_path(allowed_path)
                assert result == Path(allowed_path).resolve()

                # Writing outside write_root should fail
                blocked_path = os.path.join(tmpdir, "outside.txt")
                with pytest.raises(ToolPermissionError, match="outside allowed write root"):
                    _validate_write_path(blocked_path)

    @pytest.mark.skipif(os.name == "nt", reason="Unix system files don't exist on Windows")
    def test_write_blocks_etc(self) -> None:
        """Write should block /etc/passwd even if write_root would allow it."""
        from agent.tools.builtins.filesystem import _validate_write_path

        with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
            mock_config.return_value = ("/", "/", [])
            with pytest.raises(ToolPermissionError, match="protected system file"):
                _validate_write_path("/etc/passwd")

    async def test_file_read_uses_read_path(self) -> None:
        """file_read should use _validate_read_path (broad access)."""
        from agent.tools.builtins.filesystem import file_read

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "readable.txt")
            Path(filepath).write_text("can read this")  # noqa: ASYNC240

            with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
                mock_config.return_value = ("/", "~", [])
                result = await file_read(path=filepath)
                assert "can read this" in result

    async def test_file_write_uses_write_path(self) -> None:
        """file_write should use _validate_write_path (restricted)."""
        from agent.tools.builtins.filesystem import file_write

        with tempfile.TemporaryDirectory() as tmpdir:
            write_root = os.path.join(tmpdir, "writable")
            os.makedirs(write_root)

            with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
                mock_config.return_value = ("/", write_root, [])

                # Can write inside write_root
                filepath = os.path.join(write_root, "test.txt")
                result = await file_write(path=filepath, content="hello")
                assert "Written" in result

                # Cannot write outside write_root
                blocked = os.path.join(tmpdir, "blocked.txt")
                with pytest.raises(ToolPermissionError):
                    await file_write(path=blocked, content="nope")

    async def test_file_list_uses_read_path(self) -> None:
        """file_list should use _validate_read_path (broad access)."""
        from agent.tools.builtins.filesystem import file_list

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file.txt").write_text("content")  # noqa: ASYNC240

            with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
                mock_config.return_value = ("/", "~", [])
                result = await file_list(path=tmpdir)
                assert "file.txt" in result


class TestDenyPaths:
    """Tests for deny_paths blocking."""

    def test_deny_path_blocks_read(self) -> None:
        """Paths in deny_paths should be blocked for reading."""
        from agent.tools.builtins.filesystem import _validate_read_path

        with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
            mock_config.return_value = ("/", "~", ["/proc/kcore"])
            with pytest.raises(ToolPermissionError, match="blocked path"):
                _validate_read_path("/proc/kcore")

    def test_deny_path_blocks_write(self) -> None:
        """Paths in deny_paths should be blocked for writing."""
        from agent.tools.builtins.filesystem import _validate_write_path

        with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
            mock_config.return_value = ("/", "/", ["/dev/sda"])
            with pytest.raises(ToolPermissionError, match="blocked path"):
                _validate_write_path("/dev/sda")


class TestPathTraversal:
    """Tests for path traversal attack prevention."""

    def test_traversal_blocked_read(self) -> None:
        """Path traversal should be blocked in read validation."""
        from agent.tools.builtins.filesystem import _validate_read_path

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config,
        ):
            # Use tmpdir as root so traversal out of it fails
            mock_config.return_value = (tmpdir, "~", [])
            with pytest.raises(ToolPermissionError, match="outside allowed read root"):
                _validate_read_path(os.path.join(tmpdir, "..", "..", "etc", "passwd"))

    def test_traversal_blocked_write(self) -> None:
        """Path traversal should be blocked in write validation."""
        from agent.tools.builtins.filesystem import _validate_write_path

        with tempfile.TemporaryDirectory() as tmpdir:
            write_root = os.path.join(tmpdir, "safe")
            os.makedirs(write_root)

            with patch("agent.tools.builtins.filesystem._get_fs_config") as mock_config:
                mock_config.return_value = ("/", write_root, [])
                with pytest.raises(ToolPermissionError, match="outside allowed write root"):
                    _validate_write_path(os.path.join(write_root, "..", "..", "etc", "passwd"))

    def test_legacy_validate_path_still_works(self) -> None:
        """The old _validate_path function should still work for backwards compat."""
        from agent.tools.builtins.filesystem import _validate_path

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            Path(filepath).write_text("test")
            result = _validate_path(filepath, tmpdir)
            assert result == Path(filepath).resolve()

            with pytest.raises(ToolPermissionError, match="outside allowed root"):
                _validate_path("../../etc/passwd", tmpdir)


class TestConfigDefaults:
    """Tests for config default values."""

    def test_filesystem_config_defaults(self) -> None:
        from agent.config import ToolsFilesystemConfig

        config = ToolsFilesystemConfig()
        assert config.root == "/"
        assert config.write_root == "~"
        assert config.max_file_size == 10 * 1024 * 1024
        assert "/proc/kcore" in config.deny_paths
        assert "/dev/sda" in config.deny_paths
