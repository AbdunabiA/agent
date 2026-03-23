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
| `max_request_body_mb` | int | `10` | Maximum request body size in MB |
| `request_timeout_seconds` | int | `60` | Request processing timeout (504 on expiry) |
| `max_ws_connections` | int | `10` | Maximum concurrent WebSocket connections |
| `max_ws_message_size` | int | `100000` | Maximum WebSocket message size in bytes |
| `rate_limit_per_minute` | int | `60` | Maximum requests per minute per IP |
| `auth_lockout_attempts` | int | `5` | Failed auth attempts before lockout |
| `auth_lockout_minutes` | int | `5` | Lockout duration in minutes |

### `logging`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | string | `"INFO"` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `format` | string | `"console"` | Output format: `console` or `json` |
| `log_file` | string | `null` | Path to log file. If set, enables file logging with rotation |
| `log_max_bytes` | int | `52428800` | Max log file size in bytes before rotation (50MB) |
| `log_backup_count` | int | `5` | Number of rotated log files to keep |

### `orchestration`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable multi-agent orchestration |
| `max_concurrent_agents` | int | `5` | Max parallel sub-agents |
| `default_max_iterations` | int | `200` | Default iteration budget per agent |
| `subagent_timeout` | int | `1800` | Timeout per agent in seconds |
| `teams_directory` | string | `"teams"` | Directory containing team YAML files |
| `use_controller` | bool | `false` | Enable the controller agent |
| `controller_model` | string | `null` | Model override for controller |
| `controller_max_turns` | int | `200` | Max turns for controller |
| `teams` | list | `[]` | Inline team definitions (directory takes precedence) |

See [Orchestration](orchestration.md) for team/project YAML format.

### `workspaces`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `directory` | string | `"workspaces"` | Where workspaces are stored |
| `default` | string | `"default"` | Default workspace name |
| `auto_create_default` | bool | `true` | Create default workspace on startup |

#### `workspaces.routing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default` | string | `"default"` | Fallback workspace for unmatched messages |
| `rules` | list | `[]` | Routing rules (see below) |

Each rule in `rules`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `channel` | string | `"*"` | Channel to match (`telegram`, `webchat`, `*`) |
| `workspace` | string | Required | Target workspace name |
| `user_id` | string | `null` | Match specific user ID |
| `pattern` | string | `null` | Regex pattern to match message content |

See [Workspaces](workspaces.md) for full guide.

### `voice`

#### `voice.stt`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | `"llm_native"` | STT provider: `llm_native`, `whisper_api`, `whisper_local`, `deepgram` |
| `whisper_model` | string | `"whisper-1"` | Whisper API model name |
| `whisper_local_model` | string | `"base"` | Local whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `whisper_local_device` | string | `"cpu"` | Device for local whisper: `cpu` or `cuda` |
| `deepgram_model` | string | `"nova-2"` | Deepgram model |
| `language` | string | `""` | Language code (empty = auto-detect). ISO 639-1 (`en`, `uz`, `ru`) |

#### `voice.tts`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable text-to-speech |
| `provider` | string | `"edge_tts"` | TTS provider: `edge_tts` or `openai` |
| `edge_voice` | string | `"en-US-AriaNeural"` | Edge TTS voice name |
| `edge_rate` | string | `"+0%"` | Speech rate adjustment |
| `edge_pitch` | string | `"+0Hz"` | Pitch adjustment |
| `openai_model` | string | `"tts-1"` | OpenAI TTS model: `tts-1` or `tts-1-hd` |
| `openai_voice` | string | `"alloy"` | OpenAI voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `output_format` | string | `"opus"` | Audio format: `opus`, `mp3`, `wav` |
| `max_text_length` | int | `4000` | Max text length for TTS |

#### `voice` (top-level)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_voice_reply` | bool | `true` | Auto-reply with voice to voice messages |
| `voice_reply_channels` | list[str] | `["telegram"]` | Channels that get voice replies |
| `voice_transcription_prefix` | bool | `true` | Show transcription before response |

### `desktop`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable desktop control tools |
| `screenshot_scale` | float | `0.75` | Scale screenshots to save LLM tokens |
| `mouse_move_duration` | float | `0.3` | Mouse animation speed in seconds |
| `typing_interval` | float | `0.02` | Delay between keystrokes in seconds |
| `failsafe` | bool | `true` | pyautogui failsafe (move to corner to abort) |
| `max_screenshot_size_kb` | int | `500` | Max screenshot size sent to LLM |

### `prompts`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `shortcuts` | list | `[]` | Prompt shortcuts for `/run` command |

Each shortcut:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `alias` | string | Yes | Short name (e.g., `"review"`) |
| `template` | string | Yes | Prompt template (can use `{input}` placeholder) |
| `description` | string | No | Help text |

Example:

```yaml
prompts:
  shortcuts:
    - alias: "review"
      template: "Review this code for bugs and security issues: {input}"
      description: "Quick code review"
    - alias: "explain"
      template: "Explain this in simple terms: {input}"
      description: "Explain code or concepts"
```

### `tools.filesystem` (additional fields)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `write_root` | string | `"~"` | Root directory for file writes |
| `max_file_size` | int | `10485760` | Max file size in bytes (10MB) |
| `deny_paths` | list[str] | `["/proc/kcore", ...]` | Paths always denied |

### `skills.builder`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable self-building skills |
| `staging_dir` | string | `"skills/_staging"` | Staging directory for new skills |
| `max_retries` | int | `3` | Max build retries |
| `auto_approve` | bool | `false` | Auto-approve built skills |
| `max_permissions` | list[str] | `["safe", "moderate"]` | Max permission tier for built skills |

### `models.claude_sdk`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `claude_auth_dir` | string | `"~/.claude"` | Claude auth directory |
| `working_dir` | string | `"."` | Working directory for SDK |
| `max_turns` | int | `50` | Max conversation turns |
| `permission_mode` | string | `null` | SDK permission mode |
| `model` | string | `null` | Override model for SDK |
| `idle_timeout` | int | `1800` | Idle client disconnect timeout (seconds) |

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
