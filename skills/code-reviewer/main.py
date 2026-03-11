"""Code Reviewer skill — review code files for quality and bugs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent.skills.base import Skill


def _sync_read_for_review(file_path: str) -> str:
    """Read a file for code review (sync, run via to_thread)."""
    path = Path(file_path)
    if not path.is_file():
        return f"Error: File not found: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: Cannot read binary file: {file_path}"

    if not content.strip():
        return f"File is empty: {file_path}"

    line_count = len(content.splitlines())
    suffix = path.suffix or "unknown"

    return (
        f"## Code Review: {path.name}\n\n"
        f"- **Path**: {file_path}\n"
        f"- **Type**: {suffix}\n"
        f"- **Lines**: {line_count}\n\n"
        f"```{suffix.lstrip('.')}\n{content}\n```\n\n"
        "Please review this code for:\n"
        "1. Bugs and potential errors\n"
        "2. Code quality and readability\n"
        "3. Performance concerns\n"
        "4. Security issues\n"
        "5. Suggested improvements"
    )


class CodeReviewerSkill(Skill):
    """Review code files and provide feedback."""

    async def setup(self) -> None:
        """Register the code review tool."""
        self.register_tool(
            name="review",
            description=(
                "Review a code file for quality, bugs, and improvements. "
                "Args: file_path (path to the file to review)."
            ),
            function=self._review,
            tier="safe",
        )

    def get_system_prompt_extension(self) -> str | None:
        """Inform the LLM about code review capabilities."""
        return (
            "**Code Reviewer**: You can review code files for quality "
            "and bugs using the code-reviewer.review tool."
        )

    async def _review(self, file_path: str) -> str:
        """Review a code file.

        Reads the file and returns it with a review prompt header.
        The LLM itself performs the actual review by analyzing the content.
        """
        return await asyncio.to_thread(_sync_read_for_review, file_path)
