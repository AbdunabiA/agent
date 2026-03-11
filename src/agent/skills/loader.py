"""Skill discovery and loading.

Scans the skills directory for subdirectories containing SKILL.md + main.py,
parses YAML frontmatter from SKILL.md, and dynamically imports skill classes.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import structlog
import yaml

from agent.skills.base import Skill, SkillMetadata

logger = structlog.get_logger(__name__)


class SkillLoadError(Exception):
    """Error during skill loading."""


class SkillLoader:
    """Discovers and loads skills from the filesystem."""

    def discover(self, skills_dir: Path) -> list[Path]:
        """Scan skills directory for valid skill subdirectories.

        A valid skill directory contains both SKILL.md and main.py.

        Args:
            skills_dir: Root skills directory to scan.

        Returns:
            List of paths to valid skill directories.
        """
        if not skills_dir.is_dir():
            logger.warning("skills_dir_not_found", path=str(skills_dir))
            return []

        discovered: list[Path] = []
        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith((".", "_")):
                continue
            skill_md = entry / "SKILL.md"
            main_py = entry / "main.py"
            if skill_md.is_file() and main_py.is_file():
                discovered.append(entry)
                logger.debug("skill_discovered", name=entry.name, path=str(entry))
            else:
                logger.debug(
                    "skill_skipped_incomplete",
                    name=entry.name,
                    has_skill_md=skill_md.is_file(),
                    has_main_py=main_py.is_file(),
                )

        return discovered

    def parse_metadata(self, skill_dir: Path) -> SkillMetadata:
        """Parse SKILL.md YAML frontmatter into SkillMetadata.

        The frontmatter is the content between the first two ``---`` delimiters.

        Args:
            skill_dir: Path to the skill directory.

        Returns:
            Parsed SkillMetadata.

        Raises:
            SkillLoadError: If SKILL.md is missing or frontmatter is invalid.
        """
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise SkillLoadError(f"SKILL.md not found in {skill_dir}")

        content = skill_md.read_text(encoding="utf-8")
        parts = content.split("---")

        if len(parts) < 3:
            raise SkillLoadError(
                f"Invalid SKILL.md frontmatter in {skill_dir}: "
                "expected YAML between --- delimiters"
            )

        frontmatter_text = parts[1].strip()
        try:
            data: dict[str, Any] = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            raise SkillLoadError(f"Invalid YAML in {skill_md}: {e}") from e

        # Use directory name as default skill name
        name = data.get("name", skill_dir.name)
        display_name = data.get(
            "display_name", name.replace("-", " ").replace("_", " ").title()
        )

        return SkillMetadata(
            name=name,
            display_name=display_name,
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            author=data.get("author", ""),
            permissions=data.get("permissions", ["safe"]),
            dependencies=data.get("dependencies", []),
            triggers=data.get("triggers", []),
            enabled=data.get("enabled", True),
        )

    def load_skill_class(self, skill_dir: Path) -> type[Skill]:
        """Dynamically import main.py and find the Skill subclass.

        Args:
            skill_dir: Path to the skill directory.

        Returns:
            The Skill subclass found in main.py.

        Raises:
            SkillLoadError: If main.py can't be loaded or no Skill subclass found.
        """
        main_py = skill_dir / "main.py"
        if not main_py.is_file():
            raise SkillLoadError(f"main.py not found in {skill_dir}")

        module_name = f"agent_skill_{skill_dir.name.replace('-', '_')}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, main_py)
            if spec is None or spec.loader is None:
                raise SkillLoadError(f"Cannot create module spec for {main_py}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            raise SkillLoadError(f"Failed to import {main_py}: {e}") from e

        # Find the Skill subclass
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Skill) and obj is not Skill:
                return obj

        raise SkillLoadError(f"No Skill subclass found in {main_py}")

    def check_dependencies(self, metadata: SkillMetadata) -> list[str]:
        """Check which dependencies are missing.

        Args:
            metadata: Skill metadata with dependencies list.

        Returns:
            List of missing package names (empty if all present).
        """
        missing: list[str] = []
        for dep in metadata.dependencies:
            # Normalize package name for import check
            import_name = dep.replace("-", "_").split(">=")[0].split("==")[0].strip()
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(dep)
        return missing
