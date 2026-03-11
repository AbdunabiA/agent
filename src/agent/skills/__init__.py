"""Skills/plugin system for extending the agent with self-contained skill directories."""

from agent.skills.base import Skill, SkillMetadata
from agent.skills.loader import SkillLoader, SkillLoadError
from agent.skills.manager import SkillManager
from agent.skills.permissions import SkillPermissionError, SkillPermissionManager

__all__ = [
    "Skill",
    "SkillLoadError",
    "SkillLoader",
    "SkillManager",
    "SkillMetadata",
    "SkillPermissionError",
    "SkillPermissionManager",
]
