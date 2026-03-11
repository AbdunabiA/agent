"""File rollback system.

Whenever file_write overwrites an existing file, a backup is created in data/backups/.
This module manages those backups and provides rollback functionality.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)


@dataclass
class BackupEntry:
    """A single file backup record."""

    id: str
    original_path: str
    backup_path: str
    created_at: str  # ISO format
    size_bytes: int
    operation: str  # "overwrite" or "delete"
    session_id: str = ""
    tool_call_id: str = ""
    rolled_back: bool = False


class RollbackManager:
    """Manages file backups and rollback operations.

    Backups are stored in data/backups/ with structure:
        data/backups/
        +-- <timestamp>_<short_id>/
        |   +-- metadata.json
        |   +-- file_copy
        +-- ...

    Max backups configurable (default 100). Oldest pruned first.
    """

    def __init__(self, backup_dir: str = "data/backups", max_backups: int = 100) -> None:
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self._entries: list[BackupEntry] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Scan backup directory and load existing entries."""
        if not self.backup_dir.exists():
            return

        for entry_dir in sorted(self.backup_dir.iterdir()):
            meta_path = entry_dir / "metadata.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    self._entries.append(BackupEntry(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("backup_metadata_corrupt", path=str(meta_path), error=str(e))

        logger.info("backups_loaded", count=len(self._entries))

    def create_backup(
        self,
        file_path: str,
        operation: str = "overwrite",
        session_id: str = "",
        tool_call_id: str = "",
    ) -> BackupEntry:
        """Create a backup of a file before modifying it.

        Args:
            file_path: Path to the file to back up.
            operation: Type of operation ("overwrite" or "delete").
            session_id: Current session ID.
            tool_call_id: ID of the tool call that triggered the backup.

        Returns:
            The created BackupEntry.
        """
        source = Path(file_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")

        backup_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        entry_dir = self.backup_dir / f"{timestamp}_{backup_id}"
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Copy the file
        backup_file = entry_dir / "file_copy"
        shutil.copy2(str(source), str(backup_file))

        entry = BackupEntry(
            id=backup_id,
            original_path=str(source),
            backup_path=str(backup_file),
            created_at=datetime.now(tz=UTC).isoformat(),
            size_bytes=source.stat().st_size,
            operation=operation,
            session_id=session_id,
            tool_call_id=tool_call_id,
        )

        # Write metadata
        meta_path = entry_dir / "metadata.json"
        meta_path.write_text(json.dumps(asdict(entry), indent=2), encoding="utf-8")

        self._entries.append(entry)
        logger.info(
            "backup_created",
            id=backup_id,
            original=str(source),
            size=entry.size_bytes,
        )

        # Prune if over limit
        self.prune()

        return entry

    def rollback(self, backup_id: str) -> bool:
        """Restore a file from backup.

        Args:
            backup_id: The backup ID to restore.

        Returns:
            True if restored successfully, False otherwise.
        """
        entry = self._find_entry(backup_id)
        if entry is None:
            logger.error("backup_not_found", id=backup_id)
            return False

        backup_file = Path(entry.backup_path)
        if not backup_file.exists():
            logger.error("backup_file_missing", id=backup_id, path=entry.backup_path)
            return False

        target = Path(entry.original_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(backup_file), str(target))

        entry.rolled_back = True
        self._save_entry_metadata(entry)

        logger.info("backup_restored", id=backup_id, target=str(target))
        return True

    def rollback_by_path(self, file_path: str) -> bool:
        """Rollback the latest backup for a given file path.

        Args:
            file_path: The original file path to restore.

        Returns:
            True if restored, False if no backup found.
        """
        resolved = str(Path(file_path).resolve())
        for entry in reversed(self._entries):
            if entry.original_path == resolved and not entry.rolled_back:
                return self.rollback(entry.id)
        return False

    def list_backups(
        self, path: str | None = None, limit: int = 20
    ) -> list[BackupEntry]:
        """List recent backups, optionally filtered by original path.

        Args:
            path: If provided, only show backups for this file.
            limit: Maximum number of entries to return.

        Returns:
            List of BackupEntry objects, newest first.
        """
        entries = self._entries
        if path:
            resolved = str(Path(path).resolve())
            entries = [e for e in entries if e.original_path == resolved]
        return list(reversed(entries[-limit:]))

    def prune(self) -> int:
        """Remove oldest backups exceeding max_backups.

        Returns:
            Number of backups pruned.
        """
        pruned = 0
        while len(self._entries) > self.max_backups:
            oldest = self._entries.pop(0)
            backup_file = Path(oldest.backup_path)
            entry_dir = backup_file.parent
            if entry_dir.exists():
                shutil.rmtree(str(entry_dir), ignore_errors=True)
            pruned += 1

        if pruned:
            logger.info("backups_pruned", count=pruned)
        return pruned

    def _find_entry(self, backup_id: str) -> BackupEntry | None:
        """Find a backup entry by ID."""
        for entry in self._entries:
            if entry.id == backup_id:
                return entry
        return None

    def _save_entry_metadata(self, entry: BackupEntry) -> None:
        """Save updated metadata for an entry."""
        backup_file = Path(entry.backup_path)
        meta_path = backup_file.parent / "metadata.json"
        if meta_path.exists():
            meta_path.write_text(json.dumps(asdict(entry), indent=2), encoding="utf-8")


# Global rollback manager instance
_rollback_manager: RollbackManager | None = None


def get_rollback_manager() -> RollbackManager:
    """Get or create the global RollbackManager instance."""
    global _rollback_manager
    if _rollback_manager is None:
        _rollback_manager = RollbackManager()
    return _rollback_manager


# --- Tool functions ---


@tool(
    name="file_rollback",
    description=(
        "Restore a file to a previous version from backup. "
        "Use when a file_write produced wrong results and you need to undo it."
    ),
    tier=ToolTier.MODERATE,
)
async def file_rollback(
    backup_id: str | None = None, file_path: str | None = None
) -> str:
    """Rollback a file.

    Args:
        backup_id: Specific backup ID to restore. If not provided, uses latest for file_path.
        file_path: The file path to rollback. Used to find latest backup if backup_id not given.
    """
    manager = get_rollback_manager()

    if backup_id:
        success = await asyncio.to_thread(manager.rollback, backup_id)
        if success:
            return f"Restored backup {backup_id} successfully."
        return f"[ERROR] Could not restore backup {backup_id}. Backup not found or file missing."

    if file_path:
        success = await asyncio.to_thread(manager.rollback_by_path, file_path)
        if success:
            return f"Restored latest backup for {file_path} successfully."
        return f"[ERROR] No backup found for {file_path}."

    return "[ERROR] Provide either backup_id or file_path."


@tool(
    name="file_backups",
    description="List available file backups that can be rolled back.",
    tier=ToolTier.SAFE,
)
async def file_backups(file_path: str | None = None, limit: int = 10) -> str:
    """List backups.

    Args:
        file_path: Filter backups for a specific file path. If None, show all recent backups.
        limit: Maximum number of backups to show.
    """
    manager = get_rollback_manager()
    entries = manager.list_backups(path=file_path, limit=limit)

    if not entries:
        return "[No backups found]"

    lines: list[str] = []
    for e in entries:
        status = " (rolled back)" if e.rolled_back else ""
        lines.append(
            f"- ID: {e.id} | {e.original_path} | {e.size_bytes} bytes | "
            f"{e.operation} | {e.created_at}{status}"
        )

    return f"Found {len(entries)} backup(s):\n" + "\n".join(lines)
