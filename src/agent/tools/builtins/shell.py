"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import os

from agent.tools.registry import ToolTier, tool


@tool(
    name="shell_exec",
    description=(
        "Execute a shell command on the user's machine and return the output. "
        "Use this for running CLI commands, scripts, git operations, package management, "
        "system administration, and any task that can be done via the terminal. "
        "The command runs in a shell. Both stdout and stderr are returned. "
        "IMPORTANT: Also use this to open programs, URLs, and files VISIBLY for the user. "
        "On Windows use 'start <url/program>', on macOS use 'open <url/program>', "
        "on Linux use 'xdg-open <url/program>'. Examples: "
        "'start https://youtube.com' opens YouTube in the user's browser, "
        "'start notepad' opens Notepad, 'start calc' opens Calculator."
    ),
    tier=ToolTier.MODERATE,
)
async def shell_exec(
    command: str,
    timeout: int = 30,  # noqa: ASYNC109
    working_dir: str | None = None,
) -> str:
    """Execute a shell command.

    Args:
        command: The bash command to execute. Can be multi-line.
        timeout: Maximum execution time in seconds. Default 30.
        working_dir: Working directory for the command. Default is current dir.

    Returns:
        Combined stdout and stderr output.
    """
    cwd = os.path.expanduser(working_dir) if working_dir else None  # noqa: ASYNC240

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        await process.communicate()
        return f"[ERROR] Command timed out after {timeout}s: {command}"

    output_parts = []
    if stdout:
        output_parts.append(stdout.decode(errors="replace"))
    if stderr:
        output_parts.append(f"[STDERR]\n{stderr.decode(errors='replace')}")

    result = "\n".join(output_parts) if output_parts else "[No output]"

    # Add exit code if non-zero
    if process.returncode != 0:
        result += f"\n[Exit code: {process.returncode}]"

    return result
