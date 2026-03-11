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

## Configure

Set at least one API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

That's it. Agent works with zero config.

For more options, create `agent.yaml`:

```bash
cp agent.yaml.example agent.yaml
# Edit agent.yaml to customize
```

Or create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-your-key
OPENAI_API_KEY=sk-your-key
```

## Run

### Interactive Chat

```bash
agent chat
```

Chat commands: `/help`, `/exit`, `/model <name>`, `/tools`, `/memory`, `/soul`, `/clear`.

### Full Agent Mode

Starts the gateway, heartbeat, and all enabled channels:

```bash
agent start
```

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

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Add the token to `.env`: `TELEGRAM_BOT_TOKEN=your-token`
3. Configure in `agent.yaml`:
   ```yaml
   channels:
     telegram:
       enabled: true
       token: "${TELEGRAM_BOT_TOKEN}"
       allowed_users: [your_telegram_id]
   ```
4. Run `agent start`

See [Telegram guide](telegram.md) for details.

## Next Steps

- [Configuration](configuration.md) — Customize everything
- [Tools](tools.md) — What tools are available
- [Skills](skills.md) — Extend with plugins
- [Memory](memory.md) — How memory works
