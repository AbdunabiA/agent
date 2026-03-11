"""Python code execution tool."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from agent.tools.registry import ToolTier, tool


@tool(
    name="python_exec",
    description=(
        "Execute a Python code snippet and return the output. "
        "The code runs in an isolated subprocess. "
        "Use this for calculations, data processing, quick scripts, "
        "and testing code ideas. Print statements are captured as output."
    ),
    tier=ToolTier.MODERATE,
)
async def python_exec(code: str, timeout: int = 30) -> str:  # noqa: ASYNC109
    """Execute Python code.

    Args:
        code: Python code to execute. Can be multi-line.
        timeout: Maximum execution time in seconds.

    Returns:
        Combined stdout and stderr from the code execution.
    """
    # Write code to a temp file
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        )
        tmp.write(code)
        tmp.close()

        # Execute with subprocess
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            tmp.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return f"[ERROR] Python execution timed out after {timeout}s"

        output_parts = []
        if stdout:
            output_parts.append(stdout.decode(errors="replace"))
        if stderr:
            output_parts.append(f"[STDERR]\n{stderr.decode(errors='replace')}")

        result = "\n".join(output_parts) if output_parts else "[No output]"

        if process.returncode != 0:
            result += f"\n[Exit code: {process.returncode}]"

        return result

    except Exception as e:
        return f"[ERROR] Failed to execute Python code: {e}"
    finally:
        # Clean up temp file
        if tmp:
            import contextlib

            with contextlib.suppress(OSError):
                Path(tmp.name).unlink(missing_ok=True)  # noqa: ASYNC240
