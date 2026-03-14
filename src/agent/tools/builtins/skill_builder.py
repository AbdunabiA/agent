"""Skill builder tools — let the LLM create new skills at runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools.registry import ToolTier, tool

if TYPE_CHECKING:
    from agent.skills.builder import SkillBuilder

_global_skill_builder: SkillBuilder | None = None


def set_skill_builder(builder: SkillBuilder) -> None:
    """Set the global SkillBuilder instance (called during agent startup).

    Args:
        builder: The initialized SkillBuilder.
    """
    global _global_skill_builder
    _global_skill_builder = builder


def get_skill_builder() -> SkillBuilder:
    """Get the global SkillBuilder instance.

    Returns:
        The shared SkillBuilder.

    Raises:
        RuntimeError: If set_skill_builder() hasn't been called yet.
    """
    if _global_skill_builder is None:
        raise RuntimeError(
            "SkillBuilder not initialized. "
            "Enable skill_builder in config and restart."
        )
    return _global_skill_builder


@tool(
    name="build_skill",
    description=(
        "Create a new skill from a natural language description. "
        "This generates SKILL.md + main.py, validates the code, "
        "tests it in a sandbox, and stages it for user approval. "
        "The user must approve the skill before it becomes active. "
        "Provide a clear description of what the skill should do."
    ),
    tier=ToolTier.DANGEROUS,
)
async def build_skill(
    description: str,
    name: str = "",
    permissions: str = "",
) -> str:
    """Build a new skill from description.

    Args:
        description: What the skill should do.
        name: Optional skill name (auto-generated if empty).
        permissions: Comma-separated permission tiers (e.g., "safe,moderate").

    Returns:
        Build result with staging path or error.
    """
    builder = get_skill_builder()

    perm_list = [p.strip() for p in permissions.split(",") if p.strip()] or None
    skill_name = name.strip() or None

    result = await builder.build_skill(
        description=description,
        name=skill_name,
        permissions=perm_list,
    )

    if result.success:
        lines = [
            f"Skill '{result.skill_name}' built successfully!",
            f"Staged at: {result.staging_path}",
        ]
        if result.test:
            lines.append(f"Test: {result.test.output} ({result.test.duration_ms}ms)")
        if result.validation and result.validation.warnings:
            lines.append(f"Warnings: {', '.join(result.validation.warnings)}")
        if result.retries > 0:
            lines.append(f"Took {result.retries + 1} attempt(s)")
        lines.append(
            "\nUse approve_skill to activate it, or reject_skill to discard it."
        )
        return "\n".join(lines)
    else:
        return f"Failed to build skill '{result.skill_name}': {result.error}"


@tool(
    name="approve_skill",
    description=(
        "Approve a staged skill and move it to the active skills directory. "
        "The skill will be automatically loaded within ~5 seconds."
    ),
    tier=ToolTier.MODERATE,
)
async def approve_skill(name: str) -> str:
    """Approve a staged skill.

    Args:
        name: Name of the staged skill to approve.

    Returns:
        Status message.
    """
    builder = get_skill_builder()
    return await builder.approve_skill(name)


@tool(
    name="reject_skill",
    description="Delete a staged skill that was not approved.",
    tier=ToolTier.SAFE,
)
async def reject_skill(name: str) -> str:
    """Reject and delete a staged skill.

    Args:
        name: Name of the staged skill to reject.

    Returns:
        Status message.
    """
    builder = get_skill_builder()
    return await builder.reject_skill(name)


@tool(
    name="list_staged_skills",
    description="List all skills that have been built but not yet approved.",
    tier=ToolTier.SAFE,
)
async def list_staged_skills() -> str:
    """List staged skills.

    Returns:
        Formatted list of staged skills.
    """
    builder = get_skill_builder()
    staged = builder.list_staged()

    if not staged:
        return "No staged skills."

    lines = [f"Staged Skills ({len(staged)}):"]
    for s in staged:
        desc = s.get("description", "")
        created = s.get("created_at", "")
        lines.append(f"  - {s['name']}: {desc}")
        if created:
            lines.append(f"    Created: {created}")

    return "\n".join(lines)
