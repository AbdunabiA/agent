# CLAUDE.md — Agent Project Guide

## What is this project?

**Agent** is an open-source autonomous AI assistant inspired by OpenClaw. It runs locally on the user's machine, connects to LLM providers (Claude, OpenAI, Gemini, Ollama), and acts on behalf of the user through messaging channels (Telegram, web chat), tool execution (shell, files, browser), and a proactive heartbeat system.

The project was built in 7 phases — all phases are now implemented. New work focuses on polish, fixes, and extensions.

---

## Quick Commands

```bash
# Install
pip install -e ".[dev]"

# Run
agent chat              # Interactive terminal chat
agent start             # Full agent (gateway + channels + heartbeat)
agent version           # Version info
agent doctor            # Health check
agent models            # List available models

# Config
agent config show       # Show resolved config (secrets masked)

# Tools
agent tools list        # List registered tools with tier/status
agent tools enable X    # Enable a tool
agent tools disable X   # Disable a tool

# Memory
agent memory stats      # Memory system statistics
agent memory export     # Export memory to file
agent memory import F   # Import memory from file

# Skills
agent skills list       # List discovered skills
agent skills info X     # Skill details
agent skills enable X   # Enable a skill
agent skills disable X  # Disable a skill
agent skills reload X   # Hot-reload a skill
agent skills create X   # Scaffold a new skill

# Workspaces
agent workspace list    # List workspaces
agent workspace create  # Create workspace
agent workspace switch  # Set active workspace
agent workspace current # Show active workspace
agent workspace info X  # Workspace details
agent workspace delete  # Delete workspace

# Heartbeat
agent heartbeat start   # Start heartbeat in foreground
agent heartbeat status  # Show heartbeat status

# Audit
agent audit             # Show recent audit log
agent audit stats       # Audit statistics

# Voice
agent voice list-voices # List TTS voices
agent voice test "text" # Test TTS synthesis
agent voice config      # Show voice config

# Development
make test               # Run tests with coverage
make lint               # Lint with ruff
make format             # Auto-format with ruff
make type-check         # Type check with mypy
pytest -v -x            # Tests, stop on first failure
pytest -k "test_name"   # Run specific test
ruff check --fix .      # Auto-fix lint issues
```

---

## Project Structure

```
agent/
├── pyproject.toml              # Project metadata, deps, entry points
├── Makefile                    # Dev shortcuts
├── agent.yaml.example          # Example config
├── .env.example                # API key template
├── HEARTBEAT.md                # Heartbeat checklist
├── soul.md                     # Agent personality
├── CLAUDE.md                   # This file
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guidelines
│
├── src/agent/                  # Main package
│   ├── __init__.py             # __version__
│   ├── __main__.py             # python -m agent
│   ├── cli.py                  # Typer CLI (all commands)
│   ├── config.py               # YAML + .env config with Pydantic
│   │
│   ├── core/                   # Agent brain
│   │   ├── agent_loop.py       # Main reasoning loop
│   │   ├── events.py           # Async event bus (pub/sub)
│   │   ├── session.py          # Conversation session manager
│   │   ├── planner.py          # Planning engine
│   │   ├── heartbeat.py        # Heartbeat daemon
│   │   ├── scheduler.py        # Task scheduler
│   │   ├── context.py          # Context window assembler
│   │   ├── guardrails.py       # Safety guardrails
│   │   ├── permissions.py      # Tiered permissions
│   │   ├── audit.py            # Action audit log
│   │   ├── recovery.py         # Error recovery
│   │   ├── rollback.py         # Undo system
│   │   ├── cost_tracker.py     # LLM cost tracking
│   │   ├── doctor.py           # Health check logic
│   │   └── startup.py          # Startup sequencing
│   │
│   ├── llm/                    # LLM integration
│   │   ├── provider.py         # LiteLLM wrapper + failover
│   │   ├── claude_sdk.py       # Claude Agent SDK backend
│   │   ├── prompts.py          # System prompt builder
│   │   └── tools_schema.py     # Tool definitions for function calling
│   │
│   ├── memory/                 # Three-layer memory
│   │   ├── store.py            # SQLite facts store
│   │   ├── database.py         # Database schema and migrations
│   │   ├── vectors.py          # ChromaDB vector store
│   │   ├── embeddings.py       # Local embedding model
│   │   ├── soul.py             # soul.md loader + watcher
│   │   ├── extraction.py       # Fact extraction pipeline
│   │   ├── summarizer.py       # Conversation summarizer
│   │   ├── decay.py            # Memory confidence decay
│   │   ├── export.py           # Memory export/import
│   │   └── models.py           # Memory data models
│   │
│   ├── tools/                  # Tool execution
│   │   ├── registry.py         # @tool decorator + registry
│   │   ├── executor.py         # Tool dispatcher
│   │   └── builtins/
│   │       ├── shell.py        # Shell commands
│   │       ├── filesystem.py   # File read/write/list
│   │       ├── python_exec.py  # Python code execution
│   │       ├── http.py         # HTTP requests
│   │       ├── browser.py      # Playwright browser control
│   │       ├── web_search.py   # Web search (DuckDuckGo)
│   │       ├── desktop.py      # Desktop control tools
│   │       ├── system.py       # System info tools
│   │       ├── memory.py       # Memory lookup tools
│   │       └── send_file.py    # Send files/images/videos to users
│   │
│   ├── channels/               # Messaging channels
│   │   ├── base.py             # Abstract channel interface
│   │   ├── telegram.py         # aiogram 3.x adapter
│   │   └── webchat.py          # WebSocket chat
│   │
│   ├── gateway/                # API gateway
│   │   ├── app.py              # FastAPI app factory (+ serves dashboard)
│   │   ├── middleware.py        # Auth, CORS, rate limit
│   │   ├── dependencies.py     # FastAPI dependency injection
│   │   └── routes/
│   │       ├── api.py          # REST endpoints
│   │       └── ws.py           # WebSocket hub
│   │
│   ├── skills/                 # Plugin system
│   │   ├── base.py             # Abstract skill interface
│   │   ├── loader.py           # Skill discovery + loading
│   │   ├── manager.py          # Skill lifecycle manager
│   │   └── permissions.py      # Skill permission checks
│   │
│   ├── voice/                  # Voice pipeline
│   │   ├── config.py           # STT/TTS configuration models
│   │   ├── pipeline.py         # Voice processing pipeline
│   │   ├── stt.py              # Speech-to-text providers
│   │   └── tts.py              # Text-to-speech providers
│   │
│   ├── desktop/                # Desktop control
│   │   ├── screen.py           # Screenshot capture
│   │   ├── vision.py           # Vision analysis
│   │   ├── mouse.py            # Mouse control
│   │   ├── keyboard.py         # Keyboard control
│   │   ├── apps.py             # Application management
│   │   ├── windows.py          # Window management
│   │   └── platform_utils.py   # OS-specific utilities
│   │
│   ├── workspaces/             # Workspace isolation
│   │   ├── config.py           # Workspace configuration
│   │   ├── manager.py          # Workspace lifecycle
│   │   ├── router.py           # Channel→workspace routing
│   │   ├── isolation.py        # Data isolation layer
│   │   ├── delegation.py       # Cross-workspace delegation
│   │   └── shared_memory.py    # Shared memory across workspaces
│   │
│   └── utils/
│       ├── logging.py          # structlog setup
│       └── helpers.py          # Common utilities
│
├── dashboard/                  # React SPA (Vite + TS + Tailwind)
├── docs/                       # Documentation
│   ├── quickstart.md           # Quick start guide
│   ├── configuration.md        # Full config reference
│   ├── tools.md                # Built-in tools reference
│   ├── skills.md               # Skills/plugins guide
│   ├── telegram.md             # Telegram bot setup
│   ├── dashboard.md            # Dashboard guide
│   ├── memory.md               # Memory system docs
│   ├── security.md             # Security model
│   ├── api.md                  # REST/WebSocket API
│   ├── deployment.md           # Docker, systemd setup
│   └── index.md                # Documentation index
├── skills/                     # User custom skills
├── workspaces/                 # Workspace data directories
├── data/                       # Runtime data (git-ignored)
│   ├── agent.db                # SQLite database
│   ├── memory/                 # Markdown memory files
│   └── sessions/               # Session state
│
└── tests/                      # 70+ test files
    ├── conftest.py             # Shared fixtures
    ├── test_config.py
    ├── test_agent_loop.py
    ├── test_llm_provider.py
    ├── test_events.py
    ├── test_session.py
    └── ...                     # Tests for all modules
```

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.12+ | asyncio-native, type hints everywhere |
| Gateway | FastAPI | REST + WebSocket + serves dashboard |
| Telegram | aiogram 3.x | Async Telegram bot framework (optional dep) |
| LLM | LiteLLM + Claude SDK | Unified API for 100+ providers + native Claude |
| Database | SQLite (aiosqlite) | Facts, audit log, tasks, conversations |
| Vectors | ChromaDB | Local embeddings, all-MiniLM-L6-v2 (optional dep) |
| Personality | soul.md | Markdown file, editable from everywhere |
| Scheduler | APScheduler | Heartbeat + cron jobs |
| Browser | Playwright | Async Python bindings (optional dep) |
| Desktop | pyautogui + Pillow | Screen capture, mouse, keyboard control |
| CLI | Typer + Rich | Beautiful terminal UI |
| TTS | edge-tts | Microsoft Edge voices, free, async |
| STT | LLM native + Whisper | Multiple providers (llm_native, whisper, deepgram) |
| Dashboard | React + Vite + TS + Tailwind | shadcn/ui components |
| Testing | pytest + pytest-asyncio | Target >80% coverage |
| Linting | ruff | Replaces black, isort, flake8 |
| Types | mypy (strict) | All public APIs typed |
| CI/CD | GitHub Actions | Lint → Test → Publish to PyPI + GHCR |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    CHANNEL LAYER                          │
│   ┌───────────┐  ┌───────────┐  ┌─────────────────┐     │
│   │ Telegram  │  │  WebChat  │  │ Future Channels │     │
│   │ (aiogram) │  │(WebSocket)│  │ (WhatsApp etc)  │     │
│   └─────┬─────┘  └─────┬─────┘  └────────┬────────┘     │
└─────────┼───────────────┼─────────────────┼──────────────┘
          │               │                 │
          ▼               ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│              GATEWAY (FastAPI) + Workspace Router         │
│  REST + WebSocket + Session Management + Auth            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                    AGENT CORE                             │
│  Agent Loop → Planner → Tool Dispatcher → Error Recovery │
│  Heartbeat Daemon (APScheduler) + Cost Tracker           │
│  Guardrails + Permissions + Audit Log                    │
└──┬──────┬───────┬──────┬──────┬───────┬───────┬────────┘
   │      │       │      │      │       │       │
   ▼      ▼       ▼      ▼      ▼       ▼       ▼
┌──────┐┌──────┐┌──────┐┌─────┐┌──────┐┌─────┐┌──────┐
│ LLM  ││Tools ││Memory││Skills││Browser││Voice││Desktop│
│Layer ││Exec  ││Store ││Engine││Ctrl  ││Pipe ││Ctrl  │
│LiteLLM│subproc│SQLite ││Plugin││Play- ││STT/ ││pyauto│
│+Claude│       │ChromaDB│      ││wright││TTS  ││gui   │
│SDK   │       │soul.md │      │└──────┘└─────┘└──────┘
└──────┘└──────┘└──────┘└─────┘
```

---

## Three-Layer Memory System

```
Every LLM call assembles context from all three:

1. soul.md        → System prompt (always included, defines personality)
2. SQLite Facts   → Top N relevant key-value facts (user.name, preferences)
3. ChromaDB       → Top K semantically similar conversation chunks

Query: "What did we discuss about deploying my project?"
  ├─→ SQLite:   "user deploys with Docker + GitHub Actions"
  ├─→ ChromaDB: [3 most similar past conversation summaries]
  └─→ soul.md:  (always present as system prompt)
  │
  ▼
  Combined context → LLM call
```

- **SQLite**: Structured facts. Fast exact lookups. `key: "user.name", value: "Abduvohid"`
- **ChromaDB**: Conversation summaries + key messages. Semantic search. Local embeddings (all-MiniLM-L6-v2, ~80MB).
- **soul.md**: Static personality. Editable from file, dashboard, or Telegram `/soul` command.

---

## Code Conventions

### Must Follow

1. **All I/O is async** — use `async def` + `await`. Never block the event loop.
2. **Type hints on everything** — all function params, return types, class attributes.
3. **Pydantic for structured data** — config, API models, memory models. Not raw dicts.
4. **structlog for logging** — never `print()` for operational output. `get_logger(__name__)`.
5. **Rich for CLI output** — all terminal output through Rich (console, tables, panels, markdown).
6. **Specific exception handling** — catch specific exceptions, log with context, provide helpful messages.
7. **No global mutable state** — dependency injection. Only singleton is config.
8. **Docstrings on all public APIs** — classes, methods, functions.
9. **100 char line limit** — enforced by ruff.
10. **Import order** — stdlib → third-party → local (enforced by ruff `I` rules).

### Naming

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_prefixed`
- Type variables: `T`, `ResponseT`, etc.

### Error Handling Pattern

```python
# DO THIS:
try:
    result = await some_async_operation()
except SpecificError as e:
    logger.error("operation_failed", error=str(e), context="relevant_info")
    raise AgentError(f"Could not complete operation: {e}") from e

# NOT THIS:
try:
    result = await some_async_operation()
except Exception:
    pass
```

### Logging Pattern

```python
import structlog

logger = structlog.get_logger(__name__)

async def process_something(item_id: str) -> Result:
    logger.info("processing_started", item_id=item_id)
    # ... do work ...
    logger.info("processing_complete", item_id=item_id, duration_ms=elapsed)
```

---

## Configuration

Config is loaded from (in priority order):
1. CLI `--config` flag
2. `AGENT_CONFIG` environment variable
3. `./agent.yaml` (current directory)
4. `~/.config/agent/agent.yaml`
5. Built-in defaults (everything has a default)

Secrets come from `.env` file and are interpolated into YAML via `${VAR_NAME}` syntax.

The agent should work with ZERO config if `ANTHROPIC_API_KEY` is in the environment.

---

## Event Bus

Internal async pub/sub for component communication. Events:

```python
class Events:
    MESSAGE_INCOMING   = "message.incoming"    # User sent a message
    MESSAGE_OUTGOING   = "message.outgoing"    # Agent sending response
    TOOL_EXECUTE       = "tool.execute"        # Tool being called
    TOOL_RESULT        = "tool.result"         # Tool returned result
    HEARTBEAT_TICK     = "heartbeat.tick"      # Heartbeat fired
    HEARTBEAT_ACTION   = "heartbeat.action"    # Heartbeat taking action
    MEMORY_UPDATE      = "memory.update"       # Memory changed
    SKILL_LOADED       = "skill.loaded"        # Skill registered
    AGENT_ERROR        = "agent.error"         # Error occurred
    AGENT_STARTED      = "agent.started"       # Agent initialized
    AGENT_STOPPED      = "agent.stopped"       # Agent shutting down
    VOICE_TRANSCRIBED  = "voice.transcribed"   # Voice message transcribed
    VOICE_SYNTHESIZED  = "voice.synthesized"   # TTS audio generated
```

---

## Development Phases

| Phase | Status | Focus |
|-------|--------|-------|
| **1. Foundation** | ✅ Done | CLI, config, LLM, agent loop |
| **2. Autonomy** | ✅ Done | Tools, heartbeat, planning, safety |
| **3. Telegram + Gateway** | ✅ Done | FastAPI, Telegram adapter, streaming |
| **4. Memory** | ✅ Done | SQLite + ChromaDB + soul.md |
| **5. Dashboard & Browser** | ✅ Done | React UI, Playwright, desktop control |
| **6. Skills & Launch** | ✅ Done | Plugin system, skill manager, hot-reload |
| **7. Advanced** | ✅ Done | Voice pipeline (edge-tts + STT), workspaces |

All phases are implemented. New work focuses on bug fixes, polish, and extensions.

---

## Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=agent --cov-report=term-missing

# Run specific test file
pytest tests/test_config.py -v

# Run specific test
pytest -k "test_config_loads_defaults"

# Run only fast tests (skip integration)
pytest -m "not integration"
```

### Test file naming

- Unit tests: `tests/test_<module>.py`
- Integration tests: `tests/integration/test_<feature>.py`
- Fixtures go in `tests/conftest.py`

### Mocking LLM calls

Always mock LiteLLM in tests — never make real API calls:

```python
from unittest.mock import AsyncMock, patch

@patch("agent.llm.provider.litellm.acompletion")
async def test_completion(mock_acompletion):
    mock_acompletion.return_value = MockResponse(
        choices=[MockChoice(message=MockMessage(content="Hello!"))],
        usage=MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    # ... test code ...
```

---

## Security Model

Three-tier tool permissions:
- 🟢 **Safe**: Read files, web search, memory lookup → auto-approve
- 🟡 **Moderate**: Write files, shell commands, HTTP requests → configurable
- 🔴 **Dangerous**: Delete files, arbitrary code, system config → always confirm

All tool executions are logged in the audit table.

Resource limits: 10 iterations/request, 5 min timeout, daily cost budget.

Circuit breaker: heartbeat auto-disables after 3 consecutive failures.

---

## Common Patterns

### Adding a new CLI command

```python
# In cli.py
@app.command()
def my_command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Description of what this command does."""
    config = get_config()
    setup_logging(config.logging)
    # ... implementation ...
```

### Adding a new tool

```python
# In tools/builtins/my_tool.py
from agent.tools.registry import tool

@tool(
    name="my_tool",
    description="What this tool does",
    tier="safe",  # or "moderate" or "dangerous"
)
async def my_tool(param: str, count: int = 5) -> str:
    """Tool implementation. Params auto-generate JSON Schema."""
    # ... implementation ...
    return result
```

### Adding a new channel

```python
# In channels/my_channel.py
from agent.channels.base import BaseChannel

class MyChannel(BaseChannel):
    async def start(self): ...
    async def stop(self): ...
    async def send_message(self, user_id: str, text: str, **kwargs): ...
```

### Adding a new skill

```
skills/my-skill/
├── SKILL.md          # name, description, permissions, triggers
├── main.py           # exports a class extending Skill base
├── requirements.txt  # optional dependencies
└── config.yaml       # optional skill-specific config
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'agent'"
Run `pip install -e ".[dev]"` from the project root.

### "Config file not found"
Copy `agent.yaml.example` to `agent.yaml` and edit it. Or set `ANTHROPIC_API_KEY` in `.env` — the agent works with zero config.

### LLM returns errors
Run `agent doctor` to check API connectivity. Verify API keys in `.env`.

### Tests fail with async errors
Make sure `pytest-asyncio` is installed and `asyncio_mode = "auto"` is in `pyproject.toml`.

---

## Key Decisions Log

| Decision | Choice | Reason |
|----------|--------|--------|
| Language | Python 3.12+ | Developer expertise + AI ecosystem dominance |
| LLM layer | LiteLLM | Unified API for 100+ providers, battle-tested |
| Config | YAML + .env + Pydantic | Human-readable, validated, secret-safe |
| CLI | Typer + Rich | Modern, type-hint based, beautiful output |
| Database | SQLite | Zero-config, local-first, single file |
| Vectors | ChromaDB | Pure Python, local embeddings, no infra |
| Embeddings | all-MiniLM-L6-v2 (local) | Free, ~80MB, good quality, offline |
| TTS | edge-tts | Free, 300+ voices, async, multi-language |
| STT | LLM native audio | No extra model, LLM processes audio directly |
| Distribution | pip + Docker | CLI-first for devs, Docker for easy deploy |
| License | MIT | Maximum community adoption |
