"""File system tools: read, write, and list files.

Read access uses config `tools.filesystem.root` (default: /, full filesystem).
Write access uses config `tools.filesystem.write_root` (default: ~, home only).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog

from agent.config import get_config
from agent.core.rollback import get_rollback_manager
from agent.tools.registry import ToolPermissionError, ToolTier, tool

logger = structlog.get_logger(__name__)


def _get_fs_config() -> tuple[str, str, list[str]]:
    """Get filesystem config: (root, write_root, deny_paths)."""
    try:
        config = get_config()
        fs = config.tools.filesystem
        return fs.root, fs.write_root, fs.deny_paths
    except Exception:
        return "/", "~", ["/proc/kcore", "/dev/sda", "/dev/nvme", "/boot/efi"]


def _check_deny_paths(resolved: Path, deny_paths: list[str]) -> None:
    """Check if a resolved path is in the deny list.

    Raises:
        ToolPermissionError: If path matches a denied path.
    """
    resolved_str = str(resolved)
    for denied in deny_paths:
        denied_resolved = str(Path(denied).resolve())
        if resolved_str == denied_resolved or resolved_str.startswith(denied_resolved + os.sep):
            raise ToolPermissionError(f"Access denied: {resolved} is a blocked path")


def _validate_read_path(path: str) -> Path:
    """Validate a path for reading. Uses filesystem.root (default: /).

    Rules:
    - Must be within filesystem.root
    - Must not be in deny_paths
    - Resolves symlinks to prevent escapes

    Returns:
        Resolved absolute Path.

    Raises:
        ToolPermissionError: If path is outside root or in deny_paths.
    """
    root, _, deny_paths = _get_fs_config()
    resolved = Path(os.path.expanduser(path)).resolve()
    root_resolved = Path(os.path.expanduser(root)).resolve()

    if not resolved.is_relative_to(root_resolved):
        raise ToolPermissionError(f"Access denied: {path} is outside allowed read root {root}")

    _check_deny_paths(resolved, deny_paths)
    return resolved


def _validate_write_path(path: str) -> Path:
    """Validate a path for writing. Uses filesystem.write_root (default: ~).

    Rules:
    - Must be within filesystem.write_root (stricter than read)
    - Must not be in deny_paths
    - Must not be a system file (/etc/passwd, /etc/shadow, etc.)
    - Resolves symlinks

    Returns:
        Resolved absolute Path.

    Raises:
        ToolPermissionError: If path is outside write_root or blocked.
    """
    _, write_root, deny_paths = _get_fs_config()
    resolved = Path(os.path.expanduser(path)).resolve()
    write_root_resolved = Path(os.path.expanduser(write_root)).resolve()

    if not resolved.is_relative_to(write_root_resolved):
        raise ToolPermissionError(
            f"Access denied: {path} is outside allowed write root {write_root}"
        )

    _check_deny_paths(resolved, deny_paths)

    # Block system-critical files even if inside write_root
    # Resolve the system paths too so symlinks are handled (macOS: /etc → /private/etc)
    system_files = {str(Path(p).resolve()) for p in ("/etc/passwd", "/etc/shadow", "/etc/sudoers")}
    if str(resolved) in system_files:
        raise ToolPermissionError(f"Access denied: {resolved} is a protected system file")

    return resolved


# Keep for backwards compatibility with existing tests
def _validate_path(path: str, root: str) -> Path:
    """Resolve and validate a path is within the allowed root.

    Args:
        path: The path to validate.
        root: The allowed filesystem root.

    Returns:
        Resolved absolute Path.

    Raises:
        ToolPermissionError: If path escapes the root.
    """
    resolved = Path(os.path.expanduser(path)).resolve()
    root_resolved = Path(os.path.expanduser(root)).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ToolPermissionError(f"Access denied: {path} is outside allowed root {root}")
    return resolved


def _format_size(size: int) -> str:
    """Format file size in human-readable form."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"


def _sync_read_file(resolved: Path, max_lines: int | None) -> str:
    """Sync file read logic — run via asyncio.to_thread()."""
    if not resolved.exists():
        return f"[ERROR] File not found: {resolved}"

    if not resolved.is_file():
        return f"[ERROR] Not a file: {resolved}"

    # Check if binary
    try:
        with open(resolved, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                size = resolved.stat().st_size
                return f"[Binary file: {_format_size(size)}]"
    except OSError as e:
        return f"[ERROR] Cannot read file: {e}"

    # Read text
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"[ERROR] Cannot read file: {e}"

    # Truncate large files
    max_size = 1024 * 1024  # 1MB
    if len(content) > max_size:
        content = content[:max_size]
        size_str = _format_size(resolved.stat().st_size)
        content += f"\n\n[File truncated: showing first 1MB of {size_str}]"

    # Apply max_lines limit
    if max_lines is not None:
        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            content += f"\n\n[Showing first {max_lines} of {len(lines)} lines]"

    return content


def _sync_write_file(resolved: Path, content: str, append: bool) -> str:
    """Sync file write logic — run via asyncio.to_thread()."""
    # Validate parent directory is within write root to prevent directory escape
    _, write_root, _ = _get_fs_config()
    write_root_resolved = Path(os.path.expanduser(write_root)).resolve()
    if not resolved.parent.is_relative_to(write_root_resolved):
        return f"[ERROR] Parent directory {resolved.parent} is outside write root {write_root}"

    resolved.parent.mkdir(parents=True, exist_ok=True)

    # Create backup before overwriting existing files
    if resolved.exists() and not append:
        try:
            manager = get_rollback_manager()
            manager.create_backup(str(resolved), operation="overwrite")
        except Exception as e:
            logger.warning("backup_failed", path=str(resolved), error=str(e))

    try:
        mode = "a" if append else "w"
        with open(resolved, mode, encoding="utf-8") as f:
            f.write(content)

        size = len(content.encode("utf-8"))
        action = "Appended" if append else "Written"
        return f"{action} {_format_size(size)} to {resolved}"
    except OSError as e:
        return f"[ERROR] Cannot write file: {e}"


def _sync_list_dir(resolved: Path, max_depth: int, show_hidden: bool) -> str:
    """Sync directory listing logic — run via asyncio.to_thread()."""
    if not resolved.exists():
        return f"[ERROR] Path not found: {resolved}"

    if not resolved.is_dir():
        return f"[ERROR] Not a directory: {resolved}"

    lines: list[str] = []
    entry_count = 0
    max_entries = 200

    def _walk(dir_path: Path, prefix: str, depth: int) -> None:
        nonlocal entry_count
        if depth > max_depth or entry_count >= max_entries:
            return

        try:
            entries = sorted(
                dir_path.iterdir(),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            lines.append(f"{prefix}[Permission denied]")
            return

        # Filter hidden files
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        for i, entry in enumerate(entries):
            if entry_count >= max_entries:
                remaining = len(entries) - i
                lines.append(f"{prefix}[... and {remaining} more entries]")
                break

            entry_count += 1
            is_last = i == len(entries) - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            extension = "    " if is_last else "\u2502   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                if depth < max_depth:
                    _walk(entry, prefix + extension, depth + 1)
            else:
                try:
                    size = _format_size(entry.stat().st_size)
                except OSError:
                    size = "?"
                lines.append(f"{prefix}{connector}{entry.name} ({size})")

    lines.append(f"{resolved.name}/")
    _walk(resolved, "", 1)

    if entry_count >= max_entries:
        lines.append(f"\n[Listing truncated at {max_entries} entries]")

    return "\n".join(lines)


@tool(
    name="file_read",
    description=(
        "Read the contents of a file. Returns the full file content as text. "
        "Can read files anywhere on the filesystem. "
        "Use this to inspect configuration files, source code, documents, logs, etc."
    ),
    tier=ToolTier.SAFE,
)
async def file_read(path: str, max_lines: int | None = None) -> str:
    """Read a file's contents.

    Args:
        path: Path to the file (absolute or relative). Supports ~ for home.
        max_lines: If set, only return the first N lines.

    Returns:
        File content as text.
    """
    resolved = _validate_read_path(path)
    return await asyncio.to_thread(_sync_read_file, resolved, max_lines)


@tool(
    name="file_write",
    description=(
        "Write content to a file. Creates the file if it doesn't exist. "
        "Creates parent directories if needed. "
        "Writes are restricted to the configured write root (home directory by default). "
        "Use this to create scripts, config files, documents, or modify existing files."
    ),
    tier=ToolTier.DANGEROUS,
)
async def file_write(path: str, content: str, append: bool = False) -> str:
    """Write content to a file.

    Args:
        path: Path to the file. Supports ~ for home.
        content: The content to write.
        append: If True, append to file instead of overwriting.

    Returns:
        Confirmation message.
    """
    resolved = _validate_write_path(path)
    return await asyncio.to_thread(_sync_write_file, resolved, content, append)


@tool(
    name="file_list",
    description=(
        "List files and directories at a given path. "
        "Returns a formatted listing showing names, sizes, and types. "
        "Can browse any directory on the filesystem. "
        "Use this to explore directory structures and find files."
    ),
    tier=ToolTier.SAFE,
)
async def file_list(path: str = ".", max_depth: int = 1, show_hidden: bool = False) -> str:
    """List directory contents.

    Args:
        path: Directory path to list. Default is current directory.
        max_depth: How deep to recurse. 1 = current dir only.
        show_hidden: Include hidden files (starting with .).

    Returns:
        Formatted directory listing.
    """
    resolved = _validate_read_path(path)
    return await asyncio.to_thread(_sync_list_dir, resolved, max_depth, show_hidden)
