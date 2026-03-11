# Skills

Skills are drop-in plugins that extend Agent with new tools, event handlers, and scheduled tasks.

## Using Skills

### Discovery

Agent auto-discovers skills in the `skills/` directory (configurable via `skills.directory`).

```bash
# List discovered skills
agent skills list

# Show skill details
agent skills info github-monitor

# Enable/disable
agent skills enable daily-digest
agent skills disable code-reviewer
```

### Bundled Skills

| Skill | Description |
|-------|-------------|
| `github-monitor` | Monitor GitHub repos for new issues and PRs |
| `daily-digest` | Generate daily summary of activities |
| `code-reviewer` | Review code changes and provide feedback |
| `quick-notes` | Store and retrieve quick notes |

## Creating a Skill

### Scaffold

```bash
agent skills create my-skill
```

This creates:

```
skills/my-skill/
  SKILL.md    # Metadata
  main.py     # Implementation
```

### SKILL.md Format

```markdown
---
name: my-skill
display_name: My Skill
description: What this skill does
version: '0.1.0'
author: Your Name
permissions:
  - safe
  - moderate
triggers:
  - pattern: "^/mycommand"
dependencies:
  - requests>=2.0
---

# My Skill

Extended description and usage instructions.
```

#### Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier |
| `display_name` | No | Human-readable name |
| `description` | Yes | Short description |
| `version` | Yes | Semantic version |
| `author` | No | Author name |
| `permissions` | Yes | Required permission tiers |
| `triggers` | No | Regex patterns that activate the skill |
| `dependencies` | No | pip packages required |

### main.py

```python
from agent.skills.base import Skill


class MySkill(Skill):
    """Your skill implementation."""

    async def setup(self) -> None:
        """Called when the skill is loaded. Register tools here."""
        self.register_tool(
            name="my_tool",
            description="What this tool does",
            function=self._my_tool,
            tier="safe",
        )

    async def teardown(self) -> None:
        """Called when the skill is unloaded. Cleanup here."""
        pass

    async def _my_tool(self, param: str, count: int = 5) -> str:
        """Tool implementation. Parameters auto-generate JSON Schema."""
        return f"Result for {param} (count={count})"
```

### Skill Base Class API

```python
class Skill:
    # Properties
    self.metadata: SkillMetadata    # Parsed SKILL.md
    self.event_bus: EventBus        # Emit/subscribe to events
    self.scheduler: TaskScheduler   # Schedule tasks
    self.tool_registry: ToolRegistry

    # Methods
    def register_tool(name, description, function, tier="safe")
    async def setup() -> None       # Override: initialization
    async def teardown() -> None    # Override: cleanup
```

### Event Handling

```python
async def setup(self) -> None:
    self.event_bus.on("message.incoming", self._on_message)

async def _on_message(self, data: dict) -> None:
    message = data.get("message", "")
    if "deploy" in message.lower():
        # React to deployment-related messages
        ...
```

### Scheduled Tasks

```python
async def setup(self) -> None:
    await self.scheduler.add_cron(
        description="Daily digest",
        cron_expression="0 9 * * *",  # 9 AM daily
        channel="telegram",
    )
```

## Hot Reload

Skills support hot-reload during runtime:

```bash
# Via CLI (requires running agent)
agent skills reload my-skill

# Via API
POST /api/v1/skills/my-skill/reload
```

File changes in the skills directory are also watched automatically.

## Permissions

Skills declare required permissions in SKILL.md. The agent validates that a skill only registers tools within its declared permission tiers. A skill requesting `safe` permissions cannot register `dangerous` tools.

See [Security](security.md) for the full permission model.
