"""Memory export and import.

Exports facts and soul.md to JSON or Markdown for backup/migration.
Vectors are NOT exported — they are regenerated from text on import.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.memory.soul import SoulLoader
    from agent.memory.store import FactStore

logger = structlog.get_logger(__name__)

EXPORT_VERSION = "1.0"


class MemoryExporter:
    """Export and import agent memory."""

    def __init__(
        self,
        fact_store: FactStore | None = None,
        soul_loader: SoulLoader | None = None,
    ) -> None:
        self.fact_store = fact_store
        self.soul_loader = soul_loader

    async def export_json(self, output_path: str) -> dict[str, Any]:
        """Export all memory to a JSON file.

        Args:
            output_path: Path to write the JSON file.

        Returns:
            Dict with export stats.
        """
        data: dict[str, Any] = {
            "version": EXPORT_VERSION,
            "exported_at": datetime.now().isoformat(),
            "facts": [],
            "soul": "",
        }

        # Export facts
        facts_count = 0
        if self.fact_store:
            facts = await self.fact_store.get_all(limit=100000)
            data["facts"] = [
                {
                    "key": f.key,
                    "value": f.value,
                    "category": f.category,
                    "confidence": f.confidence,
                    "source": f.source,
                    "created_at": f.created_at.isoformat(),
                    "updated_at": f.updated_at.isoformat(),
                }
                for f in facts
            ]
            facts_count = len(data["facts"])

        # Export soul
        if self.soul_loader:
            data["soul"] = self.soul_loader.content or ""

        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        await asyncio.to_thread(
            Path(output_path).write_text, json_str, encoding="utf-8",
        )

        logger.info("memory_exported", path=output_path, facts=facts_count)
        return {"facts_exported": facts_count, "path": output_path}

    async def import_json(
        self, input_path: str, *, merge: bool = True
    ) -> dict[str, Any]:
        """Import memory from a JSON file.

        Args:
            input_path: Path to the JSON export file.
            merge: If True, merge with existing (update on conflict).
                   If False, clear existing facts first.

        Returns:
            Dict with import stats.
        """
        content = await asyncio.to_thread(
            Path(input_path).read_text, encoding="utf-8",
        )
        data = json.loads(content)

        if data.get("version") != EXPORT_VERSION:
            logger.warning(
                "import_version_mismatch",
                expected=EXPORT_VERSION,
                got=data.get("version"),
            )

        stats = {"facts_imported": 0, "soul_updated": False}

        if self.fact_store and data.get("facts"):
            if not merge:
                # Clear existing facts
                existing = await self.fact_store.get_all(limit=100000)
                for fact in existing:
                    await self.fact_store.delete(fact.key)

            for fact_data in data["facts"]:
                await self.fact_store.set(
                    key=fact_data["key"],
                    value=fact_data["value"],
                    category=fact_data.get("category", "general"),
                    source=fact_data.get("source", "import"),
                    confidence=fact_data.get("confidence", 1.0),
                )
                stats["facts_imported"] += 1

        if self.soul_loader and data.get("soul"):
            await asyncio.to_thread(self.soul_loader.update, data["soul"])
            stats["soul_updated"] = True

        logger.info("memory_imported", path=input_path, **stats)
        return stats

    async def export_markdown(self, output_path: str) -> None:
        """Export memory as human-readable Markdown.

        Args:
            output_path: Path to write the Markdown file.
        """
        lines: list[str] = [
            "# Agent Memory Export",
            "",
            f"*Exported: {datetime.now().isoformat()}*",
            "",
        ]

        # Soul
        if self.soul_loader and self.soul_loader.content:
            lines.extend([
                "## Soul (Personality)",
                "",
                self.soul_loader.content,
                "",
            ])

        # Facts
        if self.fact_store:
            facts = await self.fact_store.get_all(limit=100000)
            if facts:
                lines.extend(["## Facts", "", f"Total: {len(facts)}", ""])

                # Group by category
                categories: dict[str, list[Any]] = {}
                for f in facts:
                    categories.setdefault(f.category, []).append(f)

                for cat, cat_facts in sorted(categories.items()):
                    lines.append(f"### {cat.title()}")
                    lines.append("")
                    for f in cat_facts:
                        lines.append(
                            f"- **{f.key}**: {f.value} "
                            f"(confidence: {f.confidence:.2f}, source: {f.source})"
                        )
                    lines.append("")

        md_content = "\n".join(lines)
        await asyncio.to_thread(
            Path(output_path).write_text, md_content, encoding="utf-8",
        )
        logger.info("memory_exported_markdown", path=output_path)
