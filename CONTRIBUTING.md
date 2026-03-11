# Contributing to Agent

Thanks for your interest in contributing! This guide covers the basics.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/OWNER/agent.git
cd agent

# Install in dev mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

- **Python 3.12+** with type hints on all public APIs
- **Async-first** — use `async def` + `await` for all I/O
- **Pydantic** for structured data, not raw dicts
- **structlog** for logging, never `print()`
- **Rich** for CLI output
- **100 char line limit** (enforced by ruff)

## Running Checks

```bash
make lint         # Lint with ruff
make format       # Auto-format with ruff
make type-check   # Type check with mypy
make test         # Run tests with coverage
```

All checks must pass before merging.

## Testing

- Write tests for new features in `tests/test_<module>.py`
- Always mock LLM calls — never make real API requests in tests
- Use `pytest-asyncio` for async tests (`asyncio_mode = "auto"`)
- Target >80% coverage

```bash
pytest -v -x                  # Stop on first failure
pytest -k "test_name"         # Run specific test
pytest --cov=agent            # With coverage
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all checks pass (`make lint && make type-check && make test`)
4. Open a PR with a clear description of what and why
5. Address review feedback

## Project Structure

See [CLAUDE.md](CLAUDE.md) for the full project structure and architecture.

## Adding a Tool

```python
# src/agent/tools/builtins/my_tool.py
from agent.tools.registry import tool

@tool(name="my_tool", description="What it does", tier="safe")
async def my_tool(param: str) -> str:
    return result
```

## Adding a Skill

```bash
agent skills create my-skill
# Edit skills/my-skill/SKILL.md and skills/my-skill/main.py
```

See [docs/skills.md](docs/skills.md) for the full guide.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
