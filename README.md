# Agent

[![PyPI version](https://img.shields.io/pypi/v/agent-ai.svg)](https://pypi.org/project/agent-ai/)
[![CI](https://github.com/OWNER/agent/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/agent/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Open-source autonomous AI assistant that runs locally, remembers conversations, executes tools, and acts proactively.**

Agent connects to LLM providers (Claude, OpenAI, Gemini, Ollama), acts on your behalf through messaging channels (Telegram, web chat), executes tools (shell, files, Python, browser), and maintains a three-layer memory system. Think of it as your personal AI that can read files, run commands, browse the web, and remember what you tell it.

## Quick Start

### Install

```bash
pip install agent-ai
```

### Set Up

```bash
agent init
```

Interactive wizard that creates your config and API keys in `~/.config/agent/`. You can also skip this and just set an API key — the agent works with **zero config**:

```bash
export ANTHROPIC_API_KEY=sk-...
agent chat
```

### Full Agent Mode

```bash
agent start       # Start gateway + heartbeat + channels
agent stop        # Stop the running agent (from any terminal)

agent doctor      # Check your setup
agent config show # Show config (secrets masked)
```

### Docker

```bash
docker run -e ANTHROPIC_API_KEY=sk-... ghcr.io/OWNER/agent
```

## Features

| | Feature | Description |
|---|---------|-------------|
| **Brain** | Multi-provider LLM | Claude, OpenAI, Gemini, Ollama via LiteLLM + Claude Agent SDK |
| **Memory** | Three-layer memory | SQLite facts + ChromaDB vectors + soul.md personality |
| **Tools** | Tool execution | Shell, filesystem, Python, HTTP, browser (Playwright), web search, file sending |
| **Channels** | Telegram + Web | Telegram bot with voice/photo/video/file support, WebSocket chat |
| **Heartbeat** | Proactive actions | Scheduled checks, reminders, cron jobs via HEARTBEAT.md |
| **Planning** | Task decomposition | Multi-step plans with progress tracking |
| **Skills** | Plugin system | Drop-in skill directories with SKILL.md metadata and hot-reload |
| **Safety** | Guardrails | Three-tier permissions (safe/moderate/dangerous), audit log, cost tracking |
| **Gateway** | REST + WebSocket API | FastAPI with auth, CORS, rate limiting |
| **Dashboard** | React UI | Real-time config editing, monitoring |
| **Voice** | TTS + STT | edge-tts, Whisper, Deepgram, LLM-native audio processing |
| **Desktop** | Screen + input control | Screenshot capture, mouse, keyboard, window management |
| **Workspaces** | Isolation | Multiple agent personas with per-workspace memory and routing |

## Architecture

```
Channels (Telegram, WebChat)
         |
    Gateway (FastAPI) + Workspace Router
         |
    Agent Core (loop, planner, heartbeat, cost tracker)
    |    |    |      |      |       |       |
   LLM  Tools Memory Skills Browser Voice  Desktop
```

- **Agent Loop** processes messages through LLM with tool calling
- **Heartbeat** runs scheduled checks from HEARTBEAT.md
- **Memory** assembles context from facts (SQLite) + vectors (ChromaDB) + personality (soul.md)
- **Tools** execute with permission checks, guardrails, and audit logging
- **Skills** extend the agent with custom tools and event handlers
- **Voice** handles speech-to-text and text-to-speech across channels
- **Desktop** captures screenshots, controls mouse/keyboard, manages windows
- **Workspaces** isolate agent personas with separate memory and config

## Configuration

Minimal `agent.yaml` (or set `ANTHROPIC_API_KEY` and use defaults):

```yaml
agent:
  name: "Agent"
  max_iterations: 10

models:
  default: "claude-sonnet-4-5-20250929"
  fallback: "gpt-4o"

channels:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    allowed_users: [123456789]

gateway:
  port: 8765
  auth_token: "${GATEWAY_TOKEN}"
```

See [docs/configuration.md](docs/configuration.md) for the full reference.

## Documentation

- [Quick Start](docs/quickstart.md) — Install and run in 5 minutes
- [Configuration](docs/configuration.md) — Full config reference
- [Tools](docs/tools.md) — Built-in tools reference
- [Skills](docs/skills.md) — Creating and using skills
- [Telegram](docs/telegram.md) — Telegram bot setup
- [Dashboard](docs/dashboard.md) — React dashboard guide
- [Memory](docs/memory.md) — Memory system explained
- [Security](docs/security.md) — Security model and permissions
- [API](docs/api.md) — REST and WebSocket API reference
- [Deployment](docs/deployment.md) — Docker, systemd, reverse proxy

## Development

```bash
git clone https://github.com/OWNER/agent.git
cd agent
pip install -e ".[dev]"

make test         # Tests with coverage
make lint         # Lint with ruff
make format       # Auto-format
make type-check   # mypy strict
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.
