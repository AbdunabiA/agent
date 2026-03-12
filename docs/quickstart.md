# Quick Start

Get Agent running in 5 minutes.

## Install

### From PyPI

```bash
pip install agent-ai
```

### From Source

```bash
git clone https://github.com/OWNER/agent.git
cd agent
pip install -e .
```

### Docker

```bash
docker run -e ANTHROPIC_API_KEY=sk-... ghcr.io/OWNER/agent
```

## Set Up

The easiest way to configure Agent is the interactive setup wizard:

```bash
agent init
```

This walks you through:
1. **LLM backend** — API keys (Anthropic/OpenAI/Gemini) or Claude SDK (local subscription)
2. **Telegram bot** — optional, paste your BotFather token
3. **Gateway** — port and auth token (auto-generated)

Config files are created in `~/.config/agent/`:

| File | Purpose |
|------|---------|
| `agent.yaml` | All settings (models, channels, tools, etc.) |
| `.env` | API keys and secrets |

You can override the config location with the `AGENT_HOME` environment variable.

### Minimal Setup (No Wizard)

If you just want to chat, set one API key and go:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
agent chat
```

## Run

### Interactive Chat

```bash
agent chat
```

Chat commands: `/help`, `/exit`, `/model <name>`, `/tools`, `/memory`, `/soul`, `/clear`.

### Full Agent Mode

Starts the gateway, heartbeat, and all enabled channels (Telegram, webchat):

```bash
agent start
```

Stop the agent from any terminal:

```bash
agent stop
```

Or press `Ctrl+C` in the terminal where it's running.

### Dashboard

Once the agent is running, open:

```
http://localhost:8765/dashboard
```

If a gateway token is configured, the dashboard will ask you to log in. Paste the `GATEWAY_TOKEN` from your `~/.config/agent/.env` file.

### Health Check

```bash
agent doctor
```

Runs checks on: Python version, API keys, model connectivity, tools, memory, channels, security.

### Other Commands

```bash
agent version          # Show version info
agent config show      # Show resolved config (secrets masked)
agent tools list       # List registered tools
agent skills list      # List discovered skills
agent memory stats     # Memory system statistics
agent memory export    # Export memory to JSON
```

## Connect Telegram

If you didn't set up Telegram during `agent init`, you can add it manually:

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Add the token to `~/.config/agent/.env`:
   ```
   TELEGRAM_BOT_TOKEN=your-token
   ```
3. Enable in `~/.config/agent/agent.yaml`:
   ```yaml
   channels:
     telegram:
       enabled: true
       token: "${TELEGRAM_BOT_TOKEN}"
       allowed_users: [your_telegram_id]
   ```
4. Restart the agent: `agent stop && agent start`

See [Telegram guide](telegram.md) for details.

## Config File Locations

Agent searches for config in this order (first found wins):

1. `--config` flag — explicit path
2. `AGENT_CONFIG` env var — explicit path
3. `./agent.yaml` — current directory (useful for development)
4. `~/.config/agent/agent.yaml` — user home (default for installed agents)

The `.env` file is loaded from both `~/.config/agent/.env` and the current directory.

## Next Steps

- [Configuration](configuration.md) — Full config reference
- [Tools](tools.md) — What tools are available
- [Skills](skills.md) — Extend with plugins
- [Memory](memory.md) — How memory works
- [Dashboard](dashboard.md) — Web UI guide
