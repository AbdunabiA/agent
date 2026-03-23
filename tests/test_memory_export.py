"""Tests for memory export/import."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.memory.export import EXPORT_VERSION, MemoryExporter


@pytest.fixture
def exporter() -> MemoryExporter:
    return MemoryExporter()


def test_export_version() -> None:
    assert EXPORT_VERSION == "1.0"


@pytest.mark.asyncio
async def test_export_json_empty(exporter: MemoryExporter, tmp_path: Path) -> None:
    output = str(tmp_path / "export.json")
    stats = await exporter.export_json(output)

    assert stats["facts_exported"] == 0
    assert Path(output).exists()  # noqa: ASYNC240

    data = json.loads(Path(output).read_text(encoding="utf-8"))  # noqa: ASYNC240
    assert data["version"] == EXPORT_VERSION
    assert data["facts"] == []
    assert data["soul"] == ""
    assert "exported_at" in data


@pytest.mark.asyncio
async def test_export_markdown_empty(exporter: MemoryExporter, tmp_path: Path) -> None:
    output = str(tmp_path / "export.md")
    await exporter.export_markdown(output)

    content = Path(output).read_text(encoding="utf-8")  # noqa: ASYNC240
    assert "# Agent Memory Export" in content


@pytest.mark.asyncio
async def test_import_json_empty_facts(exporter: MemoryExporter, tmp_path: Path) -> None:
    export_file = tmp_path / "import.json"
    export_data = {
        "version": EXPORT_VERSION,
        "exported_at": "2025-01-01T00:00:00",
        "facts": [],
        "soul": "",
    }
    export_file.write_text(json.dumps(export_data), encoding="utf-8")

    stats = await exporter.import_json(str(export_file))
    assert stats["facts_imported"] == 0
    assert stats["soul_updated"] is False


@pytest.mark.asyncio
async def test_export_with_soul_loader(tmp_path: Path) -> None:
    """Test export includes soul content."""

    class MockSoulLoader:
        content = "I am a helpful assistant."

    exporter = MemoryExporter(soul_loader=MockSoulLoader())
    output = str(tmp_path / "export.json")
    await exporter.export_json(output)

    data = json.loads(Path(output).read_text(encoding="utf-8"))  # noqa: ASYNC240
    assert data["soul"] == "I am a helpful assistant."


@pytest.mark.asyncio
async def test_export_markdown_with_soul(tmp_path: Path) -> None:
    """Test markdown export includes soul section."""

    class MockSoulLoader:
        content = "I am a helpful assistant."

    exporter = MemoryExporter(soul_loader=MockSoulLoader())
    output = str(tmp_path / "export.md")
    await exporter.export_markdown(output)

    content = Path(output).read_text(encoding="utf-8")  # noqa: ASYNC240
    assert "## Soul (Personality)" in content
    assert "I am a helpful assistant." in content
