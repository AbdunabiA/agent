"""Tests for skill CLI commands (scaffolding and listing)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app

runner = CliRunner()


def test_skills_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test skill scaffolding creates correct files."""
    monkeypatch.chdir(tmp_path)

    # Create a minimal agent.yaml so config loads
    (tmp_path / "agent.yaml").write_text(
        "skills:\n  directory: skills\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["skills", "create", "test-skill", "--config", "agent.yaml"])
    assert result.exit_code == 0
    assert "Skill scaffolded" in result.stdout

    skill_dir = tmp_path / "skills" / "test-skill"
    assert skill_dir.is_dir()
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "main.py").exists()

    # Verify SKILL.md content
    skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "name: test-skill" in skill_md
    assert "permissions:" in skill_md

    # Verify main.py content
    main_py = (skill_dir / "main.py").read_text(encoding="utf-8")
    assert "class TestSkillSkill(Skill):" in main_py
    assert "async def setup" in main_py


def test_skills_create_already_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test skill create fails if directory exists."""
    monkeypatch.chdir(tmp_path)

    (tmp_path / "agent.yaml").write_text(
        "skills:\n  directory: skills\n", encoding="utf-8"
    )
    (tmp_path / "skills" / "existing").mkdir(parents=True)

    result = runner.invoke(app, ["skills", "create", "existing", "--config", "agent.yaml"])
    assert result.exit_code == 1
    assert "already exists" in result.stdout
