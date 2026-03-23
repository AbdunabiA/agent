"""Tests for file rollback system."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.core.rollback import RollbackManager


class TestRollbackManager:
    """Tests for the RollbackManager class."""

    def _make_manager(self, tmpdir: str, max_backups: int = 100) -> RollbackManager:
        """Create a RollbackManager with a temp backup directory."""
        backup_dir = os.path.join(tmpdir, "backups")
        return RollbackManager(backup_dir=backup_dir, max_backups=max_backups)

    def test_create_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            # Create a file to back up
            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("original content")

            entry = manager.create_backup(original)

            assert entry.id
            assert entry.original_path == str(Path(original).resolve())
            assert entry.size_bytes == len("original content")
            assert entry.operation == "overwrite"
            assert Path(entry.backup_path).exists()
            assert Path(entry.backup_path).read_text() == "original content"

    def test_create_backup_nonexistent_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            with pytest.raises(FileNotFoundError):
                manager.create_backup(os.path.join(tmpdir, "nonexistent.txt"))

    def test_rollback_restores_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("version 1")

            entry = manager.create_backup(original)

            # Overwrite the file
            Path(original).write_text("version 2")
            assert Path(original).read_text() == "version 2"

            # Rollback
            success = manager.rollback(entry.id)

            assert success
            assert Path(original).read_text() == "version 1"

    def test_rollback_nonexistent_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            success = manager.rollback("nonexistent_id")
            assert not success

    def test_rollback_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("old content")
            manager.create_backup(original)

            Path(original).write_text("new content")

            success = manager.rollback_by_path(original)
            assert success
            assert Path(original).read_text() == "old content"

    def test_rollback_by_path_uses_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            original = os.path.join(tmpdir, "test.txt")

            # Create first backup
            Path(original).write_text("version 1")
            manager.create_backup(original)

            # Create second backup
            Path(original).write_text("version 2")
            manager.create_backup(original)

            # Overwrite
            Path(original).write_text("version 3")

            # Rollback should restore version 2 (latest backup)
            success = manager.rollback_by_path(original)
            assert success
            assert Path(original).read_text() == "version 2"

    def test_list_backups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            file_a = os.path.join(tmpdir, "a.txt")
            file_b = os.path.join(tmpdir, "b.txt")
            Path(file_a).write_text("file a")
            Path(file_b).write_text("file b")

            manager.create_backup(file_a)
            manager.create_backup(file_b)

            # All backups
            all_backups = manager.list_backups()
            assert len(all_backups) == 2

            # Filtered by path
            a_backups = manager.list_backups(path=file_a)
            assert len(a_backups) == 1
            assert a_backups[0].original_path == str(Path(file_a).resolve())

    def test_list_backups_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("content")

            for _ in range(5):
                manager.create_backup(original)

            limited = manager.list_backups(limit=3)
            assert len(limited) == 3

    def test_prune_oldest_backups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir, max_backups=3)

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("content")

            entries = []
            for _ in range(5):
                entries.append(manager.create_backup(original))

            # Only 3 should remain after pruning
            assert len(manager._entries) == 3

            # The oldest entries should have been pruned
            remaining_ids = {e.id for e in manager._entries}
            assert entries[0].id not in remaining_ids
            assert entries[1].id not in remaining_ids

    def test_load_existing_backups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = os.path.join(tmpdir, "backups")

            # Create a manager and add a backup
            manager1 = RollbackManager(backup_dir=backup_dir)
            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("content")
            entry = manager1.create_backup(original)

            # Create a new manager that loads existing backups
            manager2 = RollbackManager(backup_dir=backup_dir)
            assert len(manager2._entries) == 1
            assert manager2._entries[0].id == entry.id

    def test_create_backup_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._make_manager(tmpdir)

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("content")

            entry = manager.create_backup(
                original,
                operation="delete",
                session_id="session-123",
                tool_call_id="call-456",
            )

            assert entry.operation == "delete"
            assert entry.session_id == "session-123"
            assert entry.tool_call_id == "call-456"


class TestRollbackTools:
    """Tests for rollback tool functions."""

    async def test_file_rollback_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=os.path.join(tmpdir, "backups"))

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("original")  # noqa: ASYNC240
            entry = manager.create_backup(original)
            Path(original).write_text("modified")  # noqa: ASYNC240

            with patch("agent.core.rollback.get_rollback_manager", return_value=manager):
                from agent.core.rollback import file_rollback

                result = await file_rollback(backup_id=entry.id)

            assert "Restored" in result
            assert Path(original).read_text() == "original"  # noqa: ASYNC240

    async def test_file_rollback_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=os.path.join(tmpdir, "backups"))

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("original")  # noqa: ASYNC240
            manager.create_backup(original)
            Path(original).write_text("modified")  # noqa: ASYNC240

            with patch("agent.core.rollback.get_rollback_manager", return_value=manager):
                from agent.core.rollback import file_rollback

                result = await file_rollback(file_path=original)

            assert "Restored" in result

    async def test_file_rollback_no_args(self) -> None:
        from agent.core.rollback import file_rollback

        with patch("agent.core.rollback.get_rollback_manager"):
            result = await file_rollback()

        assert "ERROR" in result

    async def test_file_backups_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=os.path.join(tmpdir, "backups"))

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("content")  # noqa: ASYNC240
            manager.create_backup(original)

            with patch("agent.core.rollback.get_rollback_manager", return_value=manager):
                from agent.core.rollback import file_backups

                result = await file_backups()

            assert "1 backup" in result

    async def test_file_backups_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=os.path.join(tmpdir, "backups"))

            with patch("agent.core.rollback.get_rollback_manager", return_value=manager):
                from agent.core.rollback import file_backups

                result = await file_backups()

            assert "No backups found" in result


class TestFileWriteIntegration:
    """Tests that file_write creates backups before overwriting."""

    async def test_overwrite_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=os.path.join(tmpdir, "backups"))

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("original content")  # noqa: ASYNC240

            patch_target = "agent.tools.builtins.filesystem.get_rollback_manager"
            fs_patch = "agent.tools.builtins.filesystem._get_fs_config"
            with (
                patch(patch_target, return_value=manager),
                patch(fs_patch, return_value=("/", tmpdir, [])),
            ):
                from agent.tools.builtins.filesystem import file_write

                await file_write(path=original, content="new content")

            # Backup should have been created
            backups = manager.list_backups()
            assert len(backups) == 1
            assert Path(backups[0].backup_path).read_text() == "original content"  # noqa: ASYNC240

            # File should have new content
            assert Path(original).read_text() == "new content"  # noqa: ASYNC240

    async def test_append_does_not_create_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=os.path.join(tmpdir, "backups"))

            original = os.path.join(tmpdir, "test.txt")
            Path(original).write_text("original")  # noqa: ASYNC240

            patch_target = "agent.tools.builtins.filesystem.get_rollback_manager"
            fs_patch = "agent.tools.builtins.filesystem._get_fs_config"
            with (
                patch(patch_target, return_value=manager),
                patch(fs_patch, return_value=("/", tmpdir, [])),
            ):
                from agent.tools.builtins.filesystem import file_write

                await file_write(path=original, content=" appended", append=True)

            # No backup for append operations
            assert len(manager.list_backups()) == 0

    async def test_new_file_does_not_create_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=os.path.join(tmpdir, "backups"))

            new_file = os.path.join(tmpdir, "new.txt")

            patch_target = "agent.tools.builtins.filesystem.get_rollback_manager"
            fs_patch = "agent.tools.builtins.filesystem._get_fs_config"
            with (
                patch(patch_target, return_value=manager),
                patch(fs_patch, return_value=("/", tmpdir, [])),
            ):
                from agent.tools.builtins.filesystem import file_write

                await file_write(path=new_file, content="new content")

            # No backup for new file creation
            assert len(manager.list_backups()) == 0
