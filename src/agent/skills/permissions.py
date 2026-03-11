"""Skill permission scoping.

Validates that skills only register tools within their declared permission tier.
"""

from __future__ import annotations

from agent.skills.base import SkillMetadata


class SkillPermissionError(Exception):
    """A skill tried to exceed its declared permissions."""


TIER_HIERARCHY: dict[str, int] = {
    "safe": 0,
    "moderate": 1,
    "dangerous": 2,
}


class SkillPermissionManager:
    """Validates skill tool registrations against declared permissions."""

    def get_max_tier(self, metadata: SkillMetadata) -> str:
        """Get the highest permission tier declared by the skill.

        Args:
            metadata: Skill metadata with permissions list.

        Returns:
            The highest tier string (e.g. "moderate").
        """
        max_level = 0
        max_tier = "safe"
        for perm in metadata.permissions:
            level = TIER_HIERARCHY.get(perm, 0)
            if level > max_level:
                max_level = level
                max_tier = perm
        return max_tier

    def validate_tool_registration(
        self,
        metadata: SkillMetadata,
        tool_name: str,
        requested_tier: str,
    ) -> None:
        """Validate that a tool's tier doesn't exceed the skill's declared permissions.

        Args:
            metadata: Skill metadata with declared permissions.
            tool_name: Name of the tool being registered.
            requested_tier: The tier the tool wants to register at.

        Raises:
            SkillPermissionError: If the requested tier exceeds the skill's max tier.
        """
        max_tier = self.get_max_tier(metadata)
        max_level = TIER_HIERARCHY.get(max_tier, 0)
        requested_level = TIER_HIERARCHY.get(requested_tier, 0)

        if requested_level > max_level:
            raise SkillPermissionError(
                f"Skill '{metadata.name}' tried to register tool '{tool_name}' "
                f"with tier '{requested_tier}', but its max declared tier is "
                f"'{max_tier}'. Add '{requested_tier}' to the skill's permissions "
                f"in SKILL.md."
            )
