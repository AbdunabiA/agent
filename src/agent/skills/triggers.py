"""Skill trigger matching.

Matches incoming messages against loaded skills' trigger keywords
for automatic skill hint injection into the LLM context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.skills.base import SkillMetadata

logger = structlog.get_logger(__name__)


class TriggerMatcher:
    """Matches incoming text against skill trigger keywords.

    Skills define trigger keywords in their SKILL.md frontmatter.
    This matcher checks incoming messages for those keywords and
    returns matching skills so the agent loop can inject hints.
    """

    def __init__(self) -> None:
        self._skills: list[SkillMetadata] = []

    def register_skill(self, metadata: SkillMetadata) -> None:
        """Register a skill's triggers for matching.

        Args:
            metadata: Skill metadata with trigger keywords.
        """
        if metadata.triggers:
            self._skills.append(metadata)

    def unregister_skill(self, name: str) -> None:
        """Remove a skill from trigger matching.

        Args:
            name: Skill name to remove.
        """
        self._skills = [s for s in self._skills if s.name != name]

    def match(self, text: str) -> list[tuple[SkillMetadata, str]]:
        """Check text against all registered skill triggers.

        Uses case-insensitive substring matching.

        Args:
            text: Incoming message text to check.

        Returns:
            List of (metadata, matched_trigger) tuples for matching skills.
        """
        if not text:
            return []

        text_lower = text.lower()
        matches: list[tuple[SkillMetadata, str]] = []

        for skill in self._skills:
            for trigger in skill.triggers:
                if trigger.lower() in text_lower:
                    matches.append((skill, trigger))
                    break  # One match per skill is enough

        return matches
