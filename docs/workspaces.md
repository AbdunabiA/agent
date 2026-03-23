# Workspaces

Workspaces provide isolated environments within a single Agent instance. Each workspace has its own memory, session history, soul.md personality, and configuration overrides.

## Why Workspaces?

- **Project isolation**: Keep work context separate (e.g., "project-alpha" vs "personal")
- **Multi-user**: Route different Telegram users to different workspaces
- **Personality switching**: Each workspace can have its own soul.md
- **Memory boundaries**: Facts and conversations don't leak between workspaces

## Quick Start

```bash
# Create a workspace
agent workspace create my-project

# Switch to it
agent workspace switch my-project

# Check current workspace
agent workspace current

# List all workspaces
agent workspace list
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `agent workspace list` | List all workspaces |
| `agent workspace create <name>` | Create a new workspace |
| `agent workspace switch <name>` | Set active workspace |
| `agent workspace current` | Show active workspace |
| `agent workspace info <name>` | Show workspace details |
| `agent workspace delete <name>` | Delete a workspace |

## Workspace Structure

Each workspace is a directory under the workspaces root:

```
workspaces/
  default/
    config.yaml       # Workspace-specific config overrides
    soul.md            # Workspace personality
    data/
      agent.db         # Isolated SQLite database
      memory/          # Isolated memory files
      sessions/        # Isolated session state
  project-alpha/
    config.yaml
    soul.md
    data/
      ...
```

## Configuration

### Workspace Settings

In `agent.yaml`:

```yaml
workspaces:
  directory: "workspaces"       # Where workspaces are stored
  default: "default"            # Default workspace name
  auto_create_default: true     # Create default workspace on startup
  routing:
    default: "default"          # Fallback workspace
    rules: []                   # Routing rules (see below)
```

### Workspace-Specific Config

Each workspace can have a `config.yaml` that overrides the global config:

```yaml
# workspaces/project-alpha/config.yaml
name: project-alpha
display_name: "Project Alpha"
```

### Workspace soul.md

Each workspace gets its own `soul.md` for personality customization:

```markdown
# Project Alpha Assistant

You are a development assistant for the Alpha project.
You specialize in Python backend development and PostgreSQL.
```

## Routing

Routing determines which workspace handles incoming messages based on channel and user.

### Routing Rules

```yaml
workspaces:
  routing:
    default: "default"
    rules:
      # Route specific Telegram user to a workspace
      - channel: "telegram"
        user_id: "123456789"
        workspace: "project-alpha"

      # Route all webchat to a workspace
      - channel: "webchat"
        workspace: "web-workspace"

      # Pattern matching on message content
      - channel: "*"
        pattern: "alpha:.*"
        workspace: "project-alpha"
```

### Rule Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `"*"` | Channel to match (`telegram`, `webchat`, `*` for all) |
| `workspace` | string | Required | Target workspace name |
| `user_id` | string | `null` | Match specific user ID |
| `pattern` | string | `null` | Regex pattern to match message content |

Rules are evaluated in order. First match wins. If no rule matches, the `routing.default` workspace is used.

## Shared Memory

By default, workspaces are fully isolated. To share specific facts across workspaces, use the shared memory layer:

```python
# Programmatic API
from agent.workspaces.shared_memory import SharedMemory

shared = SharedMemory(workspace_manager)
await shared.share_fact("api_key_rotation_date", "2026-04-01",
                         from_workspace="ops", to_workspaces=["dev", "staging"])
```

## Data Isolation

| Data Type | Isolated? | Notes |
|-----------|-----------|-------|
| SQLite database | Yes | Separate `agent.db` per workspace |
| Memory facts | Yes | Facts stored in workspace DB |
| Vector store | Yes | Separate ChromaDB collection |
| Sessions | Yes | Conversation history per workspace |
| soul.md | Yes | Different personality per workspace |
| Tools | Shared | Same tool registry across workspaces |
| Skills | Shared | Skills available to all workspaces |
| Config | Merged | Workspace config overrides global |
