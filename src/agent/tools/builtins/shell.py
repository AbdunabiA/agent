"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import os
import pathlib

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
    tier=ToolTier.DANGEROUS,
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
    # Defense-in-depth: guardrails check before execution
    from agent.config import get_config
    from agent.core.guardrails import Guardrails

    config = get_config()
    guardrails = Guardrails(config.tools)
    check = guardrails.check_command(command)
    if not check.allowed:
        return f"[BLOCKED] {check.reason}"

    cwd = os.path.expanduser(working_dir) if working_dir else None  # noqa: ASYNC240

    # Validate working_dir is within filesystem root
    if cwd is not None:
        cwd_path = pathlib.Path(cwd).resolve()  # noqa: ASYNC240
        fs_root = pathlib.Path(  # noqa: ASYNC240
            os.path.expanduser(config.tools.filesystem.root)  # noqa: ASYNC240
        ).resolve()
        if not cwd_path.is_relative_to(fs_root):
            return f"[BLOCKED] Working directory {working_dir} is outside allowed root {fs_root}"

    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
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
