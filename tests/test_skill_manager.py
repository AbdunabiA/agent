"""Tests for SkillManager — lifecycle, filtering, reload, list."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import SkillsConfig
from agent.core.events import EventBus
from agent.core.scheduler import TaskScheduler
from agent.skills.manager import SkillManager
from agent.tools.registry import ToolRegistry


def _create_skill(
    skills_dir: Path,
    name: str,
    permissions: str = "safe",
    tier: str = "safe",
) -> Path:
    """Helper: create a valid skill directory."""
    skill = skills_dir / name
    skill.mkdir(parents=True, exist_ok=True)

    (skill / "SKILL.md").write_text(
        f"---\n"
        f"name: {name}\n"
        f"description: Test {name}\n"
        f"version: '0.1.0'\n"
        f"permissions:\n"
        f"  - {permissions}\n"
        f"---\n"
        f"# {name}\n",
        encoding="utf-8",
    )

    (skill / "main.py").write_text(
        "from agent.skills.base import Skill\n\n"
        f"class {name.replace('-', '_').title().replace('_', '')}Skill(Skill):\n"
        "    async def setup(self):\n"
        "        self.register_tool(\n"
        f"            name='do_thing',\n"
        f"            description='Does a thing',\n"
        f"            function=self._do_thing,\n"
        f"            tier='{tier}',\n"
        "        )\n\n"
        "    async def _do_thing(self, param: str = 'default') -> str:\n"
        "        return f'done: {{param}}'\n",
        encoding="utf-8",
    )

    return skill


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def scheduler(event_bus: EventBus) -> TaskScheduler:
    return TaskScheduler(event_bus)


@pytest.mark.asyncio
async def test_discover_and_load(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")
    _create_skill(tmp_path, "beta")

    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    loaded = await manager.discover_and_load()

    assert "alpha" in loaded
    assert "beta" in loaded
    assert tool_registry.get_tool("alpha.do_thing") is not None
    assert tool_registry.get_tool("beta.do_thing") is not None


@pytest.mark.asyncio
async def test_disabled_skill_not_loaded(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")
    _create_skill(tmp_path, "beta")

    config = SkillsConfig(directory=str(tmp_path), disabled=["beta"])
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    loaded = await manager.discover_and_load()

    assert "alpha" in loaded
    assert "beta" not in loaded


@pytest.mark.asyncio
async def test_enabled_filter(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")
    _create_skill(tmp_path, "beta")
    _create_skill(tmp_path, "gamma")

    config = SkillsConfig(directory=str(tmp_path), enabled=["alpha", "gamma"])
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    loaded = await manager.discover_and_load()

    assert "alpha" in loaded
    assert "gamma" in loaded
    assert "beta" not in loaded


@pytest.mark.asyncio
async def test_auto_discover_disabled(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")

    config = SkillsConfig(directory=str(tmp_path), auto_discover=False)
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    loaded = await manager.discover_and_load()

    assert loaded == []


@pytest.mark.asyncio
async def test_unload_skill(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")

    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    await manager.discover_and_load()

    assert tool_registry.get_tool("alpha.do_thing") is not None

    result = await manager.unload_skill("alpha")
    assert result is True
    assert tool_registry.get_tool("alpha.do_thing") is None


@pytest.mark.asyncio
async def test_unload_nonexistent(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)

    result = await manager.unload_skill("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_reload_skill(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")

    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    await manager.discover_and_load()

    result = await manager.reload_skill("alpha")
    assert result is True
    # Tool should still be registered after reload
    assert tool_registry.get_tool("alpha.do_thing") is not None


@pytest.mark.asyncio
async def test_reload_nonexistent(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)

    result = await manager.reload_skill("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_list_skills(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")
    _create_skill(tmp_path, "beta")

    config = SkillsConfig(directory=str(tmp_path), disabled=["beta"])
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    await manager.discover_and_load()

    skills = manager.list_skills()
    names = {s["name"] for s in skills}
    assert "alpha" in names
    assert "beta" in names  # Listed but not loaded

    alpha = next(s for s in skills if s["name"] == "alpha")
    assert alpha["loaded"] is True

    beta = next(s for s in skills if s["name"] == "beta")
    assert beta["loaded"] is False


@pytest.mark.asyncio
async def test_shutdown(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    _create_skill(tmp_path, "alpha")

    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    await manager.discover_and_load()

    await manager.shutdown()
    assert tool_registry.get_tool("alpha.do_thing") is None
    assert len(manager._loaded) == 0


@pytest.mark.asyncio
async def test_get_system_prompt_extensions(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    # Create a skill with a prompt extension
    skill = tmp_path / "ext-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: ext-skill\ndescription: Test\npermissions:\n  - safe\n---\n"
    )
    (skill / "main.py").write_text(
        "from agent.skills.base import Skill\n\n"
        "class ExtSkill(Skill):\n"
        "    async def setup(self):\n"
        "        pass\n\n"
        "    def get_system_prompt_extension(self):\n"
        "        return 'I am the ext skill.'\n",
    )

    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    await manager.discover_and_load()

    extensions = manager.get_system_prompt_extensions()
    assert len(extensions) == 1
    assert "ext skill" in extensions[0]


@pytest.mark.asyncio
async def test_missing_dependencies_skipped(
    tmp_path: Path,
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    skill = tmp_path / "needs-dep"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: needs-dep\ndescription: Test\n"
        "dependencies:\n  - nonexistent_xyz_pkg\n---\n"
    )
    (skill / "main.py").write_text(
        "from agent.skills.base import Skill\n\n"
        "class NeedsDepSkill(Skill):\n"
        "    async def setup(self):\n"
        "        pass\n",
    )

    config = SkillsConfig(directory=str(tmp_path))
    manager = SkillManager(config, tool_registry, event_bus, scheduler)
    loaded = await manager.discover_and_load()

    assert "needs-dep" not in loaded
