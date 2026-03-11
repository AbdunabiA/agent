"""Skill base class and metadata model.

Provides the abstract base class that all skills must extend,
and the SkillMetadata dataclass parsed from SKILL.md frontmatter.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from agent.tools.registry import ToolTier

if TYPE_CHECKING:
    from agent.core.events import EventBus, EventHandler
    from agent.core.scheduler import TaskScheduler
    from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

_TIER_LEVELS: dict[str, int] = {"safe": 0, "moderate": 1, "dangerous": 2}


@dataclass
class SkillMetadata:
    """Metadata parsed from a skill's SKILL.md frontmatter."""

    name: str
    display_name: str = ""
    description: str = ""
    version: str = "0.1.0"
    author: str = ""
    permissions: list[str] = field(default_factory=lambda: ["safe"])
    dependencies: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    enabled: bool = True


class Skill(abc.ABC):
    """Abstract base class for all skills.

    Subclasses must implement ``setup()`` to register tools and events.
    ``teardown()`` automatically cleans up registered tools and event handlers.
    """

    def __init__(
        self,
        metadata: SkillMetadata,
        tool_registry: ToolRegistry,
        event_bus: EventBus,
        scheduler: TaskScheduler,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = metadata
        self.tool_registry = tool_registry
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.config = config or {}

        # Track registrations for cleanup
        self._registered_tools: list[str] = []
        self._registered_events: list[tuple[str, EventHandler]] = []

    @abc.abstractmethod
    async def setup(self) -> None:
        """Initialize the skill — register tools and event handlers."""

    async def teardown(self) -> None:
        """Clean up all registered tools and event handlers."""
        for tool_name in self._registered_tools:
            self.tool_registry.unregister_tool(tool_name)
            logger.debug("skill_tool_unregistered", skill=self.metadata.name, tool=tool_name)

        for event_name, handler in self._registered_events:
            self.event_bus.off(event_name, handler)
            logger.debug(
                "skill_event_unregistered", skill=self.metadata.name, event_name=event_name
            )

        self._registered_tools.clear()
        self._registered_events.clear()

    def register_tool(
        self,
        name: str,
        description: str,
        function: Callable[..., Coroutine[Any, Any, Any]],
        tier: str = "safe",
    ) -> None:
        """Register a tool under this skill's namespace.

        The tool name is prefixed with the skill name to avoid collisions.

        Args:
            name: Tool name (will be prefixed with skill name).
            description: Human-readable description for the LLM.
            function: Async callable implementing the tool.
            tier: Permission tier as string ("safe", "moderate", "dangerous").
        """
        prefixed_name = f"{self.metadata.name}.{name}"
        tool_tier = ToolTier(tier)

        # Validate permission tier
        max_level = max(
            (_TIER_LEVELS.get(p, 0) for p in self.metadata.permissions),
            default=0,
        )
        requested_level = _TIER_LEVELS.get(tier, 0)
        if requested_level > max_level:
            from agent.skills.permissions import SkillPermissionError

            raise SkillPermissionError(
                f"Skill '{self.metadata.name}' tried to register tool '{prefixed_name}' "
                f"with tier '{tier}', but its max declared tier is "
                f"'{self.metadata.permissions}'. Add '{tier}' to the skill's "
                f"permissions in SKILL.md."
            )

        # Use the registry's decorator-based registration
        decorator = self.tool_registry.tool(
            name=prefixed_name,
            description=description,
            tier=tool_tier,
            category="skill",
        )
        decorator(function)

        self._registered_tools.append(prefixed_name)
        logger.debug(
            "skill_tool_registered",
            skill=self.metadata.name,
            tool=prefixed_name,
            tier=tier,
        )

    def register_event(self, event_name: str, handler: EventHandler) -> None:
        """Register an event handler for this skill.

        Args:
            event_name: Event name to subscribe to.
            handler: Async callable to invoke when event fires.
        """
        self.event_bus.on(event_name, handler)
        self._registered_events.append((event_name, handler))
        logger.debug(
            "skill_event_registered", skill=self.metadata.name, event_name=event_name
        )

    def get_system_prompt_extension(self) -> str | None:
        """Return optional text to append to the system prompt.

        Override in subclass to provide skill-specific instructions
        to the LLM.

        Returns:
            Prompt extension string, or None.
        """
        return None
