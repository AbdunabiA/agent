# Skills

Skills are self-contained plugins that extend the agent with new tools, event handlers, and system prompt extensions.

## Directory Structure

Each skill lives in its own directory under `skills/`:

```
skills/
├── my-skill/
│   ├── SKILL.md          # Required: metadata + documentation
│   ├── main.py           # Required: Skill subclass
│   ├── requirements.txt  # Optional: pip dependencies
│   └── config.yaml       # Optional: skill-specific config
```

## SKILL.md Format

The `SKILL.md` file must start with YAML frontmatter between `---` delimiters:

```markdown
---
name: my-skill
description: What this skill does.
version: "0.1.0"
author: Your Name
permissions:
  - safe
  - moderate
dependencies:
  - httpx
triggers:
  - keyword1
  - keyword2
enabled: true
---

# My Skill

Documentation for the skill goes here (after the frontmatter).
```

### Frontmatter Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | No | Directory name | Unique skill identifier |
| `description` | No | `""` | Human-readable description |
| `version` | No | `"0.1.0"` | Semantic version |
| `author` | No | `""` | Skill author |
| `permissions` | No | `["safe"]` | Max tool tiers: `safe`, `moderate`, `dangerous` |
| `dependencies` | No | `[]` | Required pip packages |
| `triggers` | No | `[]` | Keywords that hint at this skill |
| `enabled` | No | `true` | Whether the skill is enabled |

## main.py Template

```python
from __future__ import annotations

from agent.skills.base import Skill


class MySkill(Skill):
    """Description of what this skill does."""

    async def setup(self) -> None:
        """Register tools and event handlers."""
        self.register_tool(
            name="my_tool",
            description="What this tool does. Args: param (str).",
            function=self._my_tool,
            tier="safe",  # Must be within declared permissions
        )

    def get_system_prompt_extension(self) -> str | None:
        """Optional: add instructions to the system prompt."""
        return "**My Skill**: You can do X using my-skill.my_tool."

    async def _my_tool(self, param: str) -> str:
        """Tool implementation."""
        return f"Result for {param}"
```

### Key Points

- Your class **must** extend `Skill` and implement `setup()`
- Tool names are auto-prefixed with the skill name (e.g., `my-skill.my_tool`)
- Tool functions **must** be async
- The `tier` must not exceed the skill's declared `permissions` in SKILL.md
- `teardown()` is called automatically — it unregisters all tools and events

## Permission Tiers

| Tier | Level | Description |
|------|-------|-------------|
| `safe` | 0 | Read-only operations, no side effects |
| `moderate` | 1 | Write operations (files, HTTP), configurable approval |
| `dangerous` | 2 | System changes, always requires confirmation |

A skill can only register tools up to its maximum declared permission tier.

## Hot Reload

The skill manager watches the `skills/` directory for changes:

- **New skill directory** → automatically loaded
- **Modified `main.py`** → automatically reloaded (unload + load)
- **Removed directory** → automatically unloaded

The watcher polls every 5 seconds.

## Configuration

In `agent.yaml`:

```yaml
skills:
  directory: skills          # Skills directory path
  enabled: []               # Empty = all discovered skills
  disabled: [my-skill]      # Skills to skip
  auto_discover: true       # Scan directory on startup
```

## Available Hooks

Skills can use:

- `self.tool_registry` — Register/unregister tools
- `self.event_bus` — Subscribe to agent events
- `self.scheduler` — Schedule tasks and reminders
- `self.config` — Skill-specific configuration dict
- `self.metadata` — Parsed SKILL.md metadata
