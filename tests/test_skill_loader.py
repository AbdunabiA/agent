"""Tests for SkillLoader — discovery, metadata parsing, class loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.skills.base import Skill
from agent.skills.loader import SkillLoader, SkillLoadError


@pytest.fixture
def loader() -> SkillLoader:
    return SkillLoader()


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Create a valid skill directory."""
    skill = tmp_path / "test-skill"
    skill.mkdir()

    skill_md = skill / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: test-skill\n"
        "description: A test skill\n"
        "version: '1.0.0'\n"
        "permissions:\n"
        "  - safe\n"
        "---\n"
        "# Test Skill\n"
        "Documentation here.\n",
        encoding="utf-8",
    )

    main_py = skill / "main.py"
    main_py.write_text(
        "from agent.skills.base import Skill\n\n"
        "class TestSkill(Skill):\n"
        "    async def setup(self):\n"
        "        pass\n",
        encoding="utf-8",
    )

    return tmp_path


def test_discover_finds_valid_skills(loader: SkillLoader, skill_dir: Path) -> None:
    discovered = loader.discover(skill_dir)
    assert len(discovered) == 1
    assert discovered[0].name == "test-skill"


def test_discover_skips_incomplete(loader: SkillLoader, tmp_path: Path) -> None:
    # Directory with only SKILL.md (no main.py)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "SKILL.md").write_text("---\nname: incomplete\n---\n")

    discovered = loader.discover(tmp_path)
    assert len(discovered) == 0


def test_discover_skips_hidden_dirs(loader: SkillLoader, tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text("---\nname: hidden\n---\n")
    (hidden / "main.py").write_text("pass")

    discovered = loader.discover(tmp_path)
    assert len(discovered) == 0


def test_discover_skips_underscore_dirs(loader: SkillLoader, tmp_path: Path) -> None:
    internal = tmp_path / "_internal"
    internal.mkdir()
    (internal / "SKILL.md").write_text("---\nname: internal\n---\n")
    (internal / "main.py").write_text("pass")

    discovered = loader.discover(tmp_path)
    assert len(discovered) == 0


def test_discover_nonexistent_dir(loader: SkillLoader, tmp_path: Path) -> None:
    discovered = loader.discover(tmp_path / "nonexistent")
    assert discovered == []


def test_parse_metadata(loader: SkillLoader, skill_dir: Path) -> None:
    meta = loader.parse_metadata(skill_dir / "test-skill")
    assert meta.name == "test-skill"
    assert meta.display_name == "Test Skill"  # Auto-generated from name
    assert meta.description == "A test skill"
    assert meta.version == "1.0.0"
    assert meta.permissions == ["safe"]


def test_parse_metadata_defaults(loader: SkillLoader, tmp_path: Path) -> None:
    skill = tmp_path / "minimal"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\n---\nMinimal skill.\n")
    (skill / "main.py").write_text("pass")

    meta = loader.parse_metadata(skill)
    assert meta.name == "minimal"  # Falls back to directory name
    assert meta.display_name == "Minimal"  # Auto-generated from name
    assert meta.version == "0.1.0"
    assert meta.permissions == ["safe"]


def test_parse_metadata_explicit_display_name(loader: SkillLoader, tmp_path: Path) -> None:
    skill = tmp_path / "my-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndisplay_name: My Cool Skill\n---\n"
    )
    (skill / "main.py").write_text("pass")

    meta = loader.parse_metadata(skill)
    assert meta.display_name == "My Cool Skill"


def test_parse_metadata_missing_file(loader: SkillLoader, tmp_path: Path) -> None:
    with pytest.raises(SkillLoadError, match="SKILL.md not found"):
        loader.parse_metadata(tmp_path / "nonexistent")


def test_parse_metadata_invalid_frontmatter(
    loader: SkillLoader, tmp_path: Path
) -> None:
    skill = tmp_path / "bad"
    skill.mkdir()
    (skill / "SKILL.md").write_text("No frontmatter here")
    (skill / "main.py").write_text("pass")

    with pytest.raises(SkillLoadError, match="Invalid SKILL.md frontmatter"):
        loader.parse_metadata(skill)


def test_load_skill_class(loader: SkillLoader, skill_dir: Path) -> None:
    cls = loader.load_skill_class(skill_dir / "test-skill")
    assert issubclass(cls, Skill)
    assert cls is not Skill


def test_load_skill_class_missing_main(loader: SkillLoader, tmp_path: Path) -> None:
    skill = tmp_path / "no-main"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: no-main\n---\n")

    with pytest.raises(SkillLoadError, match="main.py not found"):
        loader.load_skill_class(skill)


def test_load_skill_class_no_subclass(loader: SkillLoader, tmp_path: Path) -> None:
    skill = tmp_path / "no-subclass"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: no-subclass\n---\n")
    (skill / "main.py").write_text("class NotASkill:\n    pass\n")

    with pytest.raises(SkillLoadError, match="No Skill subclass found"):
        loader.load_skill_class(skill)


def test_check_dependencies_all_present(loader: SkillLoader) -> None:
    from agent.skills.base import SkillMetadata

    meta = SkillMetadata(name="test", dependencies=["os", "sys"])
    missing = loader.check_dependencies(meta)
    assert missing == []


def test_check_dependencies_missing(loader: SkillLoader) -> None:
    from agent.skills.base import SkillMetadata

    meta = SkillMetadata(
        name="test", dependencies=["nonexistent_package_xyz_12345"]
    )
    missing = loader.check_dependencies(meta)
    assert "nonexistent_package_xyz_12345" in missing


def test_check_dependencies_empty(loader: SkillLoader) -> None:
    from agent.skills.base import SkillMetadata

    meta = SkillMetadata(name="test", dependencies=[])
    missing = loader.check_dependencies(meta)
    assert missing == []
