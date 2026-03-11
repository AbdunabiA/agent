"""Tests for the Skill base class and SkillMetadata."""

from __future__ import annotations

import pytest

from agent.core.events import EventBus
from agent.core.scheduler import TaskScheduler
from agent.skills.base import Skill, SkillMetadata
from agent.skills.permissions import SkillPermissionError
from agent.tools.registry import ToolRegistry


@pytest.fixture
def tool_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def scheduler(event_bus: EventBus) -> TaskScheduler:
    return TaskScheduler(event_bus)


@pytest.fixture
def sample_metadata() -> SkillMetadata:
    return SkillMetadata(
        name="test-skill",
        description="A test skill",
        version="1.0.0",
        permissions=["safe", "moderate"],
    )


class ConcreteSkill(Skill):
    """Concrete implementation for testing."""

    async def setup(self) -> None:
        self.register_tool(
            name="greet",
            description="Greet someone",
            function=self._greet,
            tier="safe",
        )

    def get_system_prompt_extension(self) -> str | None:
        return "Test skill extension."

    async def _greet(self, name: str) -> str:
        return f"Hello, {name}!"


class NoExtensionSkill(Skill):
    """Skill without prompt extension."""

    async def setup(self) -> None:
        pass


def test_skill_metadata_defaults() -> None:
    meta = SkillMetadata(name="my-skill")
    assert meta.name == "my-skill"
    assert meta.display_name == ""
    assert meta.description == ""
    assert meta.version == "0.1.0"
    assert meta.author == ""
    assert meta.permissions == ["safe"]
    assert meta.dependencies == []
    assert meta.triggers == []
    assert meta.enabled is True


def test_skill_metadata_custom() -> None:
    meta = SkillMetadata(
        name="custom",
        display_name="Custom Skill",
        description="Custom skill",
        version="2.0.0",
        author="Test",
        permissions=["safe", "dangerous"],
        dependencies=["httpx"],
        triggers=["test"],
        enabled=False,
    )
    assert meta.name == "custom"
    assert meta.display_name == "Custom Skill"
    assert meta.version == "2.0.0"
    assert meta.permissions == ["safe", "dangerous"]
    assert meta.enabled is False


@pytest.mark.asyncio
async def test_skill_register_tool_prefixes_name(
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
    sample_metadata: SkillMetadata,
) -> None:
    skill = ConcreteSkill(
        metadata=sample_metadata,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    await skill.setup()

    # Tool should be prefixed with skill name
    assert "test-skill.greet" in [t.name for t in tool_registry.list_tools()]
    assert tool_registry.get_tool("test-skill.greet") is not None


@pytest.mark.asyncio
async def test_skill_teardown_unregisters_tools(
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
    sample_metadata: SkillMetadata,
) -> None:
    skill = ConcreteSkill(
        metadata=sample_metadata,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    await skill.setup()
    assert tool_registry.get_tool("test-skill.greet") is not None

    await skill.teardown()
    assert tool_registry.get_tool("test-skill.greet") is None
    assert skill._registered_tools == []


@pytest.mark.asyncio
async def test_skill_teardown_unregisters_events(
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
    sample_metadata: SkillMetadata,
) -> None:
    skill = ConcreteSkill(
        metadata=sample_metadata,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    handler_called = False

    async def handler(data: object) -> None:
        nonlocal handler_called
        handler_called = True

    skill.register_event("test.event", handler)
    assert len(skill._registered_events) == 1

    await skill.teardown()
    assert skill._registered_events == []

    # Handler should no longer fire
    await event_bus.emit("test.event", {})
    assert handler_called is False


@pytest.mark.asyncio
async def test_skill_prompt_extension(
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
    sample_metadata: SkillMetadata,
) -> None:
    skill = ConcreteSkill(
        metadata=sample_metadata,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    assert skill.get_system_prompt_extension() == "Test skill extension."


@pytest.mark.asyncio
async def test_skill_no_prompt_extension(
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
    sample_metadata: SkillMetadata,
) -> None:
    skill = NoExtensionSkill(
        metadata=sample_metadata,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )
    assert skill.get_system_prompt_extension() is None


@pytest.mark.asyncio
async def test_register_tool_permission_enforcement(
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    """Skill with only safe permissions cannot register moderate tools."""
    safe_only = SkillMetadata(name="safe-skill", permissions=["safe"])
    skill = NoExtensionSkill(
        metadata=safe_only,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    async def dummy(param: str) -> str:
        return param

    with pytest.raises(SkillPermissionError, match="moderate"):
        skill.register_tool(
            name="bad_tool",
            description="Should fail",
            function=dummy,
            tier="moderate",
        )


@pytest.mark.asyncio
async def test_register_tool_permission_allowed(
    tool_registry: ToolRegistry,
    event_bus: EventBus,
    scheduler: TaskScheduler,
) -> None:
    """Skill with moderate permissions can register safe and moderate tools."""
    meta = SkillMetadata(name="mod-skill", permissions=["safe", "moderate"])
    skill = NoExtensionSkill(
        metadata=meta,
        tool_registry=tool_registry,
        event_bus=event_bus,
        scheduler=scheduler,
    )

    async def dummy(param: str) -> str:
        return param

    # Should not raise
    skill.register_tool(name="safe_tool", description="OK", function=dummy, tier="safe")
    assert tool_registry.get_tool("mod-skill.safe_tool") is not None
