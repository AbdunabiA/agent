"""Skill lifecycle manager.

Orchestrates discovery, loading, unloading, hot-reload, and shutdown
of all skills in the configured skills directory.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from agent.config import SkillsConfig
from agent.core.events import EventBus, Events
from agent.core.scheduler import TaskScheduler
from agent.skills.base import Skill, SkillMetadata
from agent.skills.loader import SkillLoader, SkillLoadError
from agent.skills.permissions import SkillPermissionManager
from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

WATCHER_INTERVAL_SECONDS = 5.0


@dataclass
class LoadedSkill:
    """Tracks a loaded skill instance and its source path."""

    metadata: SkillMetadata
    instance: Skill
    path: Path
    mtime: float  # main.py modification time


class SkillManager:
    """Manages the full lifecycle of skills.

    Handles discovery, loading, unloading, hot-reload via filesystem polling,
    and clean shutdown.
    """

    def __init__(
        self,
        config: SkillsConfig,
        tool_registry: ToolRegistry,
        event_bus: EventBus,
        scheduler: TaskScheduler,
    ) -> None:
        self.config = config
        self.tool_registry = tool_registry
        self.event_bus = event_bus
        self.scheduler = scheduler

        self._loader = SkillLoader()
        self._permissions = SkillPermissionManager()
        self._loaded: dict[str, LoadedSkill] = {}
        self._watcher_task: asyncio.Task[None] | None = None

    @property
    def skills_dir(self) -> Path:
        """Resolved skills directory path."""
        return Path(self.config.directory).resolve()

    async def discover_and_load(self) -> list[str]:
        """Discover and load all eligible skills.

        Returns:
            List of successfully loaded skill names.
        """
        if not self.config.auto_discover:
            logger.info("skill_auto_discover_disabled")
            return []

        skill_dirs = self._loader.discover(self.skills_dir)
        loaded_names: list[str] = []

        for skill_dir in skill_dirs:
            try:
                metadata = self._loader.parse_metadata(skill_dir)
            except SkillLoadError as e:
                logger.warning("skill_metadata_error", path=str(skill_dir), error=str(e))
                continue

            # Check enabled/disabled filters
            if not self._is_skill_allowed(metadata.name):
                logger.debug("skill_filtered_out", name=metadata.name)
                continue

            try:
                await self._load_skill(skill_dir, metadata)
                loaded_names.append(metadata.name)
            except SkillLoadError as e:
                logger.warning("skill_load_failed", name=metadata.name, error=str(e))

        logger.info("skills_loaded", count=len(loaded_names), names=loaded_names)
        return loaded_names

    async def _load_skill(self, skill_dir: Path, metadata: SkillMetadata) -> None:
        """Import, instantiate, and set up a single skill.

        Args:
            skill_dir: Path to the skill directory.
            metadata: Parsed skill metadata.

        Raises:
            SkillLoadError: If loading fails at any step.
        """
        # Check dependencies
        missing = self._loader.check_dependencies(metadata)
        if missing:
            raise SkillLoadError(
                f"Skill '{metadata.name}' missing dependencies: {', '.join(missing)}"
            )

        # Load class
        skill_class = self._loader.load_skill_class(skill_dir)

        # Load skill-specific config if config.yaml exists
        skill_config: dict[str, Any] | None = None
        config_file = skill_dir / "config.yaml"
        if config_file.is_file():
            import yaml

            def _load_yaml() -> dict[str, Any] | None:
                with open(config_file, encoding="utf-8") as f:
                    return yaml.safe_load(f)

            skill_config = await asyncio.to_thread(_load_yaml)

        # Instantiate
        instance = skill_class(
            metadata=metadata,
            tool_registry=self.tool_registry,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
            config=skill_config,
        )

        # Setup (registers tools/events)
        await instance.setup()

        # Track
        main_py = skill_dir / "main.py"
        self._loaded[metadata.name] = LoadedSkill(
            metadata=metadata,
            instance=instance,
            path=skill_dir,
            mtime=main_py.stat().st_mtime,
        )

        # Emit event
        await self.event_bus.emit(Events.SKILL_LOADED, {
            "name": metadata.name,
            "version": metadata.version,
            "tools": instance._registered_tools,
        })

        logger.info(
            "skill_loaded",
            name=metadata.name,
            version=metadata.version,
            tools=len(instance._registered_tools),
        )

    async def unload_skill(self, name: str) -> bool:
        """Unload a skill by name.

        Calls teardown and removes from the loaded registry.

        Args:
            name: Skill name to unload.

        Returns:
            True if unloaded, False if not found.
        """
        loaded = self._loaded.pop(name, None)
        if not loaded:
            return False

        await loaded.instance.teardown()
        logger.info("skill_unloaded", name=name)
        return True

    async def reload_skill(self, name: str) -> bool:
        """Hot-reload a skill (unload + load).

        Args:
            name: Skill name to reload.

        Returns:
            True if reloaded successfully, False if skill not found.
        """
        loaded = self._loaded.get(name)
        if not loaded:
            return False

        skill_dir = loaded.path
        await self.unload_skill(name)

        try:
            metadata = self._loader.parse_metadata(skill_dir)
            await self._load_skill(skill_dir, metadata)
            logger.info("skill_reloaded", name=name)
            return True
        except SkillLoadError as e:
            logger.error("skill_reload_failed", name=name, error=str(e))
            return False

    def list_skills(self) -> list[dict[str, Any]]:
        """List all discovered skills with their loaded status.

        Returns:
            List of dicts with skill name, loaded status, and metadata.
        """
        result: list[dict[str, Any]] = []

        # Show loaded skills
        for name, loaded in self._loaded.items():
            result.append({
                "name": name,
                "display_name": loaded.metadata.display_name,
                "loaded": True,
                "version": loaded.metadata.version,
                "author": loaded.metadata.author,
                "description": loaded.metadata.description,
                "permissions": loaded.metadata.permissions,
                "tools": loaded.instance._registered_tools,
                "path": str(loaded.path),
            })

        # Also show discovered-but-not-loaded
        if self.skills_dir.is_dir():
            for skill_dir in self._loader.discover(self.skills_dir):
                if skill_dir.name not in self._loaded:
                    try:
                        meta = self._loader.parse_metadata(skill_dir)
                        result.append({
                            "name": meta.name,
                            "display_name": meta.display_name,
                            "loaded": False,
                            "version": meta.version,
                            "author": meta.author,
                            "description": meta.description,
                            "permissions": meta.permissions,
                            "tools": [],
                            "path": str(skill_dir),
                        })
                    except SkillLoadError:
                        pass

        return result

    def get_system_prompt_extensions(self) -> list[str]:
        """Collect system prompt extensions from all loaded skills.

        Returns:
            List of non-empty prompt extension strings.
        """
        extensions: list[str] = []
        for loaded in self._loaded.values():
            ext = loaded.instance.get_system_prompt_extension()
            if ext:
                extensions.append(ext)
        return extensions

    async def start_watcher(self) -> None:
        """Start the filesystem watcher for hot-reload."""
        if self._watcher_task is not None:
            return
        self._watcher_task = asyncio.create_task(self._watch_loop())
        logger.info("skill_watcher_started", interval=WATCHER_INTERVAL_SECONDS)

    async def stop_watcher(self) -> None:
        """Stop the filesystem watcher."""
        if self._watcher_task is not None:
            self._watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher_task
            self._watcher_task = None
            logger.info("skill_watcher_stopped")

    async def _watch_loop(self) -> None:
        """Poll for filesystem changes and trigger reloads."""
        while True:
            await asyncio.sleep(WATCHER_INTERVAL_SECONDS)
            try:
                await self._check_for_changes()
            except Exception as e:
                logger.error("skill_watcher_error", error=str(e))

    async def _check_for_changes(self) -> None:
        """Detect new, removed, and modified skills."""
        if not self.skills_dir.is_dir():
            return

        current_dirs = {d.name: d for d in self._loader.discover(self.skills_dir)}

        # Detect removed skills
        for name in list(self._loaded.keys()):
            if name not in current_dirs:
                logger.info("skill_removed_detected", name=name)
                await self.unload_skill(name)

        for dir_name, skill_dir in current_dirs.items():
            main_py = skill_dir / "main.py"

            if dir_name in self._loaded:
                # Check for modification
                loaded = self._loaded[dir_name]
                current_mtime = main_py.stat().st_mtime
                if current_mtime > loaded.mtime:
                    logger.info("skill_modified_detected", name=dir_name)
                    await self.reload_skill(dir_name)
            else:
                # New skill detected
                try:
                    metadata = self._loader.parse_metadata(skill_dir)
                    if self._is_skill_allowed(metadata.name):
                        logger.info("skill_new_detected", name=metadata.name)
                        await self._load_skill(skill_dir, metadata)
                except SkillLoadError as e:
                    logger.warning("skill_auto_load_failed", name=dir_name, error=str(e))

    def _is_skill_allowed(self, name: str) -> bool:
        """Check if a skill is allowed by enabled/disabled filters.

        Args:
            name: Skill name to check.

        Returns:
            True if the skill should be loaded.
        """
        # If disabled list contains the skill, skip it
        if name in self.config.disabled:
            return False

        # If enabled list is set and non-empty, only allow listed skills
        return not (self.config.enabled and name not in self.config.enabled)

    async def shutdown(self) -> None:
        """Teardown all skills and stop the watcher."""
        await self.stop_watcher()

        for name in list(self._loaded.keys()):
            await self.unload_skill(name)

        logger.info("skill_manager_shutdown")
