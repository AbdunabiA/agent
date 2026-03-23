# Troubleshooting

Common issues and how to resolve them.

## ModuleNotFoundError: No module named 'agent'

The package is not installed in your Python environment.

```bash
pip install -e ".[dev]"
```

For all optional features (memory, voice, desktop, browser):

```bash
pip install -e ".[all]"
```

## Config file not found

Run the setup wizard to generate a config file:

```bash
agent init
```

This creates `agent.yaml` in the current directory. Alternatively, copy `agent.yaml.example` and edit it manually. The agent works with zero config if `ANTHROPIC_API_KEY` is set in the environment.

## LLM API errors

Run the built-in health check:

```bash
agent doctor
```

Verify your API keys are set in `.env`:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Common causes:
- Expired or invalid API key
- Rate limit exceeded on the provider side
- Network connectivity issues
- Wrong model name in `agent.yaml`

## Database locked

SQLite only supports one writer at a time. This happens when multiple agent instances try to access the same database.

```
sqlite3.OperationalError: database is locked
```

Fix: ensure only one agent instance is running:

```bash
agent stop
agent start
```

The database file is located at `data/agent.db`.

## Port already in use

```
OSError: [Errno 48] Address already in use
```

Another process is using the gateway port. Either stop that process or change the port in `agent.yaml`:

```yaml
gateway:
  port: 8766  # default is 8765
```

## Telegram bot not responding

1. Verify the bot token is correct in `.env`:

```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
```

2. Check that `allowed_users` is set in `agent.yaml`. An empty list allows everyone (insecure):

```yaml
channels:
  telegram:
    allowed_users: [123456789]  # your Telegram user ID
```

3. Run diagnostics:

```bash
agent doctor
```

4. Make sure only one instance of the bot is running (Telegram only allows one connection per token).

## ChromaDB / memory errors

Memory features require the `memory` extra:

```bash
pip install -e ".[memory]"
```

If ChromaDB fails to initialize, try removing the existing data and letting it rebuild:

```bash
rm -rf data/memory/chroma/
agent start
```

## Tests failing

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Requirements:
- Python 3.12 or newer
- `pytest-asyncio` installed
- `asyncio_mode = "auto"` in `pyproject.toml`

Run tests with verbose output to identify the failure:

```bash
pytest -v -x
```

## Voice not working

Voice features require the `voice` extra:

```bash
pip install -e ".[voice]"
```

Check your voice configuration:

```bash
agent voice config
```

Test TTS synthesis:

```bash
agent voice test "Hello, world"
```

## Desktop control not working

Desktop features require the `desktop` extra:

```bash
pip install -e ".[desktop]"
```

Platform support:
- **macOS**: Requires accessibility permissions (System Settings → Privacy & Security → Accessibility)
- **Linux**: Requires X11 (Wayland is not fully supported)
- **Windows**: Should work out of the box

## Still stuck?

1. Run `agent doctor` for a full health check
2. Check logs in `data/logs/` (if file logging is enabled)
3. Set `logging.level: debug` in `agent.yaml` for verbose output
4. Open an issue on GitHub with the error output
