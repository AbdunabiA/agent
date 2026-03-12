# Configuration

## Setup

The easiest way to create your config is the interactive wizard:

```bash
agent init
```

This creates `agent.yaml` and `.env` in your agent home directory (`~/.config/agent/`).

## Config Loading Order

Agent loads configuration from multiple sources in priority order:

1. CLI `--config` flag
2. `AGENT_CONFIG` environment variable
3. `./agent.yaml` (current directory)
4. `$AGENT_HOME/agent.yaml` (defaults to `~/.config/agent/agent.yaml`)
5. Built-in defaults

Secrets come from `.env` files and are interpolated via `${VAR_NAME}` syntax. The `.env` file is loaded from `$AGENT_HOME/.env` first, then the current directory (already-set values are not overridden).

### Agent Home Directory

By default, Agent stores config and runtime files in `~/.config/agent/`. Override with:

```bash
export AGENT_HOME=/path/to/custom/agent/home
```

This directory contains:

| File | Purpose |
|------|---------|
| `agent.yaml` | Configuration |
| `.env` | API keys and secrets |
| `agent.pid` | PID of running agent (for `agent stop`) |

## Full Reference

### `agent`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"Agent"` | Agent display name |
| `persona` | string | (default prompt) | Base system prompt |
| `max_iterations` | int | `10` | Max tool-call iterations per request |
| `heartbeat_interval` | string | `"30m"` | Heartbeat tick interval |

### `models`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default` | string | `"claude-sonnet-4-5-20250929"` | Primary model |
| `fallback` | string | `"gpt-4o"` | Fallback model on failure |
| `providers` | dict | `{}` | Provider configurations |

#### `models.providers.<name>`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api_key` | string | `null` | API key (or use env var) |
| `base_url` | string | `null` | Custom endpoint URL |

### `channels.telegram`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable Telegram channel |
| `token` | string | `null` | Bot token from BotFather |
| `allowed_users` | list[int] | `[]` | Telegram user IDs (empty = all) |

### `channels.webchat`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable WebSocket chat |
| `port` | int | `8080` | WebChat listen port |

### `tools`

#### `tools.shell`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable shell tool |
| `sandbox` | bool | `false` | Run in Docker sandbox |
| `allowed_commands` | list[str] | `["*"]` | Allowed command patterns |

#### `tools.browser`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable browser tool |
| `headless` | bool | `true` | Run browser headless |

#### `tools.filesystem`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable filesystem tool |
| `root` | string | `"~/"` | Root directory for file access |

### `memory`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `db_path` | string | `"./data/agent.db"` | SQLite database path |
| `markdown_dir` | string | `"./data/memory/"` | Memory files directory |
| `auto_extract` | bool | `true` | Auto-extract facts from conversations |
| `max_facts_in_context` | int | `15` | Max facts injected per LLM call |
| `max_vectors_in_context` | int | `5` | Max vector results per LLM call |
| `summarize_threshold` | int | `20` | Messages before auto-summarize |
| `soul_path` | string | `null` | Path to soul.md (auto-detected) |

### `skills`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `directory` | string | `"skills"` | Skills directory path |
| `enabled` | list[str] | `[]` | Enabled skills (empty = all) |
| `disabled` | list[str] | `[]` | Explicitly disabled skills |
| `auto_discover` | bool | `true` | Auto-discover skills on startup |

### `gateway`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"127.0.0.1"` | Bind address |
| `port` | int | `8765` | Listen port |
| `auth_token` | string | `null` | Bearer token (null = no auth) |
| `cors_origins` | list[str] | `["http://localhost:5173"]` | Allowed CORS origins |

### `logging`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | string | `"INFO"` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `format` | string | `"console"` | Output format: `console` or `json` |

## Environment Variables

| Variable | Used For |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic/Claude API |
| `OPENAI_API_KEY` | OpenAI API |
| `GEMINI_API_KEY` | Google Gemini API |
| `TELEGRAM_BOT_TOKEN` | Telegram bot |
| `GATEWAY_TOKEN` | Gateway authentication |
| `AGENT_CONFIG` | Config file path override |
| `AGENT_HOME` | Agent home directory (default: `~/.config/agent`) |

## Example

```yaml
agent:
  name: "MyAgent"
  max_iterations: 10
  heartbeat_interval: "30m"

models:
  default: "claude-sonnet-4-5-20250929"
  fallback: "gpt-4o"
  providers:
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
    openai:
      api_key: "${OPENAI_API_KEY}"
    ollama:
      base_url: "http://localhost:11434"

channels:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    allowed_users: [123456789]

tools:
  shell:
    enabled: true
  filesystem:
    root: "~/projects"

memory:
  auto_extract: true
  soul_path: "./soul.md"

gateway:
  port: 8765
  auth_token: "${GATEWAY_TOKEN}"
  cors_origins:
    - "http://localhost:5173"

logging:
  level: "INFO"
  format: "console"
```
