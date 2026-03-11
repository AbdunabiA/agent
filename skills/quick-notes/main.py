"""Quick Notes skill — save, search, and list markdown notes."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

from agent.skills.base import Skill

NOTES_DIR = Path("data/notes")


def _sync_save_note(title: str, content: str) -> str:
    """Save a note (sync, run via to_thread)."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '-').lower()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-{safe_title}.md"
    filepath = NOTES_DIR / filename

    note_content = f"# {title}\n\n*Created: {datetime.now().isoformat()}*\n\n{content}\n"
    filepath.write_text(note_content, encoding="utf-8")

    return f"Note saved: {filepath}"


def _sync_search_notes(query: str) -> str:
    """Search notes by keyword (sync, run via to_thread)."""
    if not NOTES_DIR.is_dir():
        return "No notes directory found."

    results: list[str] = []
    query_lower = query.lower()

    for note_file in sorted(NOTES_DIR.glob("*.md")):
        content = note_file.read_text(encoding="utf-8")
        if query_lower in content.lower():
            first_line = content.split("\n")[0].lstrip("# ").strip()
            preview = content[:200].replace("\n", " ").strip()
            results.append(f"- **{first_line}** ({note_file.name})\n  {preview}...")

    if not results:
        return f"No notes found matching '{query}'."

    return f"**Found {len(results)} note(s):**\n\n" + "\n\n".join(results)


def _sync_list_notes() -> str:
    """List all saved notes (sync, run via to_thread)."""
    if not NOTES_DIR.is_dir():
        return "No notes directory found."

    notes = sorted(NOTES_DIR.glob("*.md"))
    if not notes:
        return "No notes saved yet."

    lines = [f"**{len(notes)} note(s):**\n"]
    for note_file in notes:
        content = note_file.read_text(encoding="utf-8")
        first_line = content.split("\n")[0].lstrip("# ").strip()
        lines.append(f"- {first_line} ({note_file.name})")

    return "\n".join(lines)


class QuickNotesSkill(Skill):
    """Save and retrieve quick markdown notes."""

    async def setup(self) -> None:
        """Register note management tools."""
        self.register_tool(
            name="save",
            description="Save a new note. Args: title (str), content (str).",
            function=self._save,
            tier="moderate",
        )
        self.register_tool(
            name="search",
            description="Search notes by keyword. Args: query (str).",
            function=self._search,
            tier="safe",
        )
        self.register_tool(
            name="list",
            description="List all saved notes.",
            function=self._list,
            tier="safe",
        )

    def get_system_prompt_extension(self) -> str | None:
        """Inform the LLM about note-taking capabilities."""
        return (
            "**Quick Notes**: You can save, search, and list markdown notes "
            "using the quick-notes tools (save, search, list)."
        )

    async def _save(self, title: str, content: str) -> str:
        """Save a note as a markdown file."""
        return await asyncio.to_thread(_sync_save_note, title, content)

    async def _search(self, query: str) -> str:
        """Search notes by keyword."""
        return await asyncio.to_thread(_sync_search_notes, query)

    async def _list(self) -> str:
        """List all saved notes."""
        return await asyncio.to_thread(_sync_list_notes)
