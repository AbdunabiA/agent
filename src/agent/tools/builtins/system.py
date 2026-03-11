"""Cross-platform system information and control tools.

Works on Linux, macOS, and Windows without modification.
Uses Python stdlib + psutil for all OS interaction.
"""

from __future__ import annotations

import fnmatch
import os
import platform
from datetime import datetime
from pathlib import Path

from agent.tools.registry import ToolTier, tool


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _dir_item_count(path: Path) -> int:
    """Count items in a directory, returning 0 on permission error."""
    try:
        return len(list(path.iterdir()))
    except PermissionError:
        return 0


def _get_file_type(suffix: str) -> str:
    """Map file extension to a type name."""
    types = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".html": "html", ".css": "css", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".md": "markdown", ".txt": "text", ".csv": "csv",
        ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image", ".svg": "image",
        ".mp3": "audio", ".wav": "audio", ".ogg": "audio",
        ".mp4": "video", ".mkv": "video", ".avi": "video",
        ".pdf": "pdf", ".docx": "document", ".xlsx": "spreadsheet",
        ".zip": "archive", ".tar": "archive", ".gz": "archive",
        ".exe": "executable", ".sh": "script", ".bat": "script",
    }
    return types.get(suffix.lower(), "file")


def _get_file_icon(file_type: str) -> str:
    """Get an icon for a file type."""
    icons = {
        "python": "PY", "javascript": "JS", "typescript": "TS",
        "html": "WEB", "css": "CSS", "json": "CFG", "yaml": "CFG",
        "markdown": "MD", "text": "TXT", "csv": "CSV",
        "image": "IMG", "audio": "AUD", "video": "VID",
        "pdf": "PDF", "document": "DOC", "spreadsheet": "XLS",
        "archive": "ZIP", "executable": "EXE", "script": "SH",
    }
    return f"[{icons.get(file_type, 'FILE')}]"


@tool(
    name="system_info",
    description=(
        "Get comprehensive system information: OS, architecture, hostname, "
        "CPU, RAM, disk, Python version. Works on Linux, macOS, and Windows."
    ),
    tier=ToolTier.SAFE,
)
async def system_info() -> str:
    """Get system information."""
    import psutil

    mem = psutil.virtual_memory()
    disk_root = "/" if platform.system() != "Windows" else "C:\\"
    disk = psutil.disk_usage(disk_root)
    cpu_freq = psutil.cpu_freq()

    lines = [
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"Platform: {platform.platform()}",
        f"Hostname: {platform.node()}",
        f"Python: {platform.python_version()}",
        "",
        f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} logical)",
    ]

    if cpu_freq:
        lines.append(f"CPU Freq: {cpu_freq.current:.0f} MHz")

    lines.extend([
        f"CPU Usage: {psutil.cpu_percent(interval=0.1)}%",
        "",
        f"RAM: {mem.total / (1024**3):.1f} GB total, "
        f"{mem.available / (1024**3):.1f} GB available ({mem.percent}% used)",
        "",
        f"Disk ({disk_root}): {disk.total / (1024**3):.1f} GB total, "
        f"{disk.free / (1024**3):.1f} GB free ({disk.percent}% used)",
        "",
        f"User: {os.getenv('USER') or os.getenv('USERNAME', 'unknown')}",
        f"Home: {Path.home()}",
        f"CWD: {os.getcwd()}",
    ])

    return "\n".join(lines)


@tool(
    name="list_directory",
    description=(
        "List contents of any directory with details (size, type, modified date). "
        "Works on any path the user has read access to. Cross-platform."
    ),
    tier=ToolTier.SAFE,
)
async def list_directory(
    path: str = ".",
    show_hidden: bool = False,
    sort_by: str = "name",
    max_items: int = 100,
) -> str:
    """List directory contents with details.

    Args:
        path: Directory path to list. Supports ~ for home, . for current.
        show_hidden: Include hidden files (starting with .).
        sort_by: Sort by "name", "size", "modified", or "type".
        max_items: Maximum number of items to return.
    """
    target = Path(path).expanduser().resolve()  # noqa: ASYNC240

    if not target.exists():
        return f"[ERROR] Path does not exist: {target}"
    if not target.is_dir():
        return f"[ERROR] Not a directory: {target}"

    try:
        entries = list(target.iterdir())
    except PermissionError:
        return f"[ERROR] Permission denied: {target}"

    if not show_hidden:
        entries = [e for e in entries if not e.name.startswith(".")]

    # Build info for each entry
    items: list[dict[str, object]] = []
    for entry in entries:
        try:
            stat = entry.stat()
            is_dir = entry.is_dir()
            size = stat.st_size if not is_dir else _dir_item_count(entry)

            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            items.append({
                "name": entry.name + ("/" if is_dir else ""),
                "type": "dir" if is_dir else _get_file_type(entry.suffix),
                "size": size,
                "size_str": _format_size(stat.st_size) if not is_dir else f"{size} items",
                "modified": modified,
                "path": str(entry),
            })
        except (PermissionError, OSError):
            items.append({
                "name": entry.name,
                "type": "?",
                "size": 0,
                "size_str": "?",
                "modified": "?",
                "path": str(entry),
            })

    # Sort
    if sort_by == "size":
        items.sort(key=lambda x: x["size"], reverse=True)  # type: ignore[arg-type]
    elif sort_by == "modified":
        items.sort(key=lambda x: x["modified"], reverse=True)  # type: ignore[arg-type]
    elif sort_by == "type":
        items.sort(key=lambda x: (x["type"], x["name"]))  # type: ignore[arg-type]
    else:
        # Directories first, then by name
        name = ""
        items.sort(
            key=lambda x: (
                0 if str(x["name"]).endswith("/") else 1,
                str(x.get("name", name)).lower(),
            )
        )

    # Truncate
    total_count = len(items)
    items = items[:max_items]

    # Format output
    lines = [f"{target} ({total_count} items)", ""]

    for item in items:
        icon = "[DIR]" if str(item["name"]).endswith("/") else _get_file_icon(str(item["type"]))
        lines.append(
            f"  {icon:6s} {str(item['name']):40s} {str(item['size_str']):>10s}"
            f"  {item['modified']}"
        )

    if total_count > max_items:
        lines.append(f"\n  ... and {total_count - max_items} more items")

    return "\n".join(lines)


@tool(
    name="find_files",
    description=(
        "Search for files by name pattern, extension, or content. "
        "Recursively searches directories. Works cross-platform."
    ),
    tier=ToolTier.SAFE,
)
async def find_files(
    directory: str = ".",
    pattern: str = "*",
    extension: str = "",
    contains: str = "",
    max_depth: int = 5,
    max_results: int = 50,
) -> str:
    """Find files matching criteria.

    Args:
        directory: Starting directory to search.
        pattern: Glob pattern for file names (e.g., "*.py", "test_*").
        extension: Filter by extension (e.g., ".py", ".js"). Include the dot.
        contains: Search for files containing this text (searches file content).
        max_depth: Maximum directory depth to recurse.
        max_results: Maximum number of results to return.
    """
    target = Path(directory).expanduser().resolve()  # noqa: ASYNC240
    if not target.exists():
        return f"[ERROR] Directory does not exist: {target}"

    results: list[dict[str, str]] = []
    skip_dirs = {
        "node_modules", "__pycache__", ".git", ".venv", "venv",
        ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
    }

    def _search(dir_path: Path, depth: int) -> None:
        if depth > max_depth or len(results) >= max_results:
            return

        try:
            for entry in sorted(dir_path.iterdir()):
                if len(results) >= max_results:
                    return

                if entry.name.startswith("."):
                    continue

                if entry.is_dir():
                    if entry.name in skip_dirs:
                        continue
                    _search(entry, depth + 1)

                elif entry.is_file():
                    if not fnmatch.fnmatch(entry.name, pattern):
                        continue

                    if extension and entry.suffix != extension:
                        continue

                    if contains:
                        try:
                            text = entry.read_text(errors="ignore")[:100_000]
                            if contains.lower() not in text.lower():
                                continue
                        except (PermissionError, OSError):
                            continue

                    try:
                        stat = entry.stat()
                        results.append({
                            "path": str(entry),
                            "size": _format_size(stat.st_size),
                            "modified": datetime.fromtimestamp(
                                stat.st_mtime
                            ).strftime("%Y-%m-%d %H:%M"),
                        })
                    except (PermissionError, OSError):
                        pass
        except PermissionError:
            pass

    _search(target, 0)

    if not results:
        return f"No files found matching criteria in {target}"

    lines = [f"Found {len(results)} files:"]
    for r in results:
        lines.append(f"  {r['path']}  ({r['size']}, {r['modified']})")

    return "\n".join(lines)


@tool(
    name="disk_usage",
    description="Show disk usage for all mounted drives/partitions. Cross-platform.",
    tier=ToolTier.SAFE,
)
async def disk_usage() -> str:
    """Show disk usage for all partitions."""
    import psutil

    partitions = psutil.disk_partitions(all=False)
    lines = ["Disk Usage:"]

    for p in partitions:
        try:
            usage = psutil.disk_usage(p.mountpoint)
            lines.append(
                f"  {p.device} -> {p.mountpoint}\n"
                f"    Type: {p.fstype} | "
                f"Total: {usage.total / (1024**3):.1f} GB | "
                f"Used: {usage.used / (1024**3):.1f} GB ({usage.percent}%) | "
                f"Free: {usage.free / (1024**3):.1f} GB"
            )
        except (PermissionError, OSError):
            lines.append(f"  {p.device} -> {p.mountpoint} (access denied)")

    return "\n".join(lines)


@tool(
    name="running_processes",
    description="List running processes with CPU and memory usage. Cross-platform.",
    tier=ToolTier.SAFE,
)
async def running_processes(sort_by: str = "memory", limit: int = 20) -> str:
    """List running processes.

    Args:
        sort_by: Sort by "memory", "cpu", or "name".
        limit: Number of processes to show.
    """
    import psutil

    procs: list[dict[str, object]] = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
        try:
            info = p.info  # type: ignore[attr-defined]
            procs.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if sort_by == "cpu":
        procs.sort(key=lambda x: x.get("cpu_percent", 0) or 0, reverse=True)
    elif sort_by == "name":
        procs.sort(key=lambda x: str(x.get("name", "")).lower())
    else:
        procs.sort(key=lambda x: x.get("memory_percent", 0) or 0, reverse=True)

    procs = procs[:limit]

    lines = [f"{'PID':>7}  {'CPU%':>6}  {'MEM%':>6}  {'Status':10}  Name"]
    lines.append("-" * 60)

    for p in procs:
        lines.append(
            f"{p.get('pid', '?'):>7}  "
            f"{(p.get('cpu_percent') or 0):>5.1f}%  "
            f"{(p.get('memory_percent') or 0):>5.1f}%  "
            f"{str(p.get('status', '?')):10}  "
            f"{p.get('name', '?')}"
        )

    return "\n".join(lines)


@tool(
    name="environment_vars",
    description="List or get environment variables. Sensitive values are masked. Cross-platform.",
    tier=ToolTier.SAFE,
)
async def environment_vars(name: str = "", filter_prefix: str = "") -> str:
    """Get environment variables.

    Args:
        name: Get a specific variable by name. If empty, lists all.
        filter_prefix: Filter variables by prefix (e.g., "PATH", "PYTHON", "HOME").
    """
    sensitive_keywords = {"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL"}

    if name:
        value = os.environ.get(name)
        if value is None:
            return f"Environment variable '{name}' is not set."
        if any(s in name.upper() for s in sensitive_keywords):
            return f"{name} = ******* (masked for security)"
        return f"{name} = {value}"

    env = dict(os.environ)

    if filter_prefix:
        env = {k: v for k, v in env.items() if k.upper().startswith(filter_prefix.upper())}

    lines: list[str] = []
    for k in sorted(env.keys()):
        v = env[k]
        if any(s in k.upper() for s in sensitive_keywords):
            v = "*******"
        lines.append(f"  {k} = {v[:100]}{'...' if len(v) > 100 else ''}")

    return f"Environment Variables ({len(lines)}):\n" + "\n".join(lines)
