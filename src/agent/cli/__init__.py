"""Typer CLI application for Agent.

Provides commands: chat, start, version, doctor, config show,
tools list/enable/disable, audit, heartbeat start/status.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import signal
from pathlib import Path
from typing import Any

import structlog
import typer
import yaml
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent import __version__
from agent.cli._helpers import (
    _apply_log_level_flags,
    _init_agent_stack,
    _is_process_running,
    _load_config,
    _read_pid_file,
    _remove_pid_file,
    _resolve_workspace,
    _write_pid_file,
    console,
    err_console,
)
from agent.cli.memory_workspace import memory_app, workspace_app
from agent.cli.skills import skills_app
from agent.cli.teams_projects import project_app, teams_app
from agent.cli.voice_daemon import daemon_app, voice_app
from agent.config import (
    AgentConfig,
    config_to_dict_masked,
    get_agent_home,
    get_available_models,
)
from agent.core.agent_loop import AgentLoop
from agent.core.events import EventBus as EventBus
from agent.core.planner import Planner
from agent.core.session import Session
from agent.llm.provider import LLMProvider
from agent.tools.registry import registry
from agent.utils.helpers import get_system_info
from agent.utils.logging import setup_logging

logger = structlog.get_logger(__name__)

app = typer.Typer(
    name="agent",
    help="Agent — An open-source autonomous AI assistant.",
    no_args_is_help=True,
)

config_app = typer.Typer(help="Configuration commands.")
tools_app = typer.Typer(help="Tool management commands.")
audit_app = typer.Typer(help="Audit log commands.")
heartbeat_app = typer.Typer(help="Heartbeat daemon commands.")

# Register all sub-apps
app.add_typer(config_app, name="config")
app.add_typer(tools_app, name="tools")
app.add_typer(audit_app, name="audit", invoke_without_command=True)
app.add_typer(heartbeat_app, name="heartbeat")
app.add_typer(skills_app, name="skills")
app.add_typer(memory_app, name="memory")
app.add_typer(workspace_app, name="workspace")
app.add_typer(voice_app, name="voice")
app.add_typer(daemon_app, name="daemon")
app.add_typer(teams_app, name="teams")
app.add_typer(project_app, name="project")


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@app.command()
def chat(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override default model"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace to use"),
    backend: str | None = typer.Option(
        None, "--backend", "-b", help="LLM backend: litellm or claude-sdk"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only"),
) -> None:
    """Interactive terminal chat mode."""
    cfg = _load_config(config)
    _apply_log_level_flags(cfg, verbose=verbose, quiet=quiet)
    setup_logging(cfg.logging)
    cfg = _resolve_workspace(cfg, workspace)

    if model:
        cfg.models.default = model
    if backend:
        cfg.models.backend = backend

    if cfg.models.backend == "claude-sdk":
        asyncio.run(_chat_loop_sdk(cfg))
    else:
        asyncio.run(_chat_loop(cfg))


async def _chat_loop(cfg: AgentConfig) -> None:
    """Run the interactive chat REPL.

    Args:
        cfg: Validated agent configuration.
    """
    agent_loop, event_bus, audit, permissions, guardrails, recovery, planner = _init_agent_stack(
        cfg
    )
    session = Session()

    # Phase 4: Initialize memory components (async)
    database = None
    fact_store = None
    vector_store = None
    summarizer = None

    try:
        from agent.memory.database import Database
        from agent.memory.extraction import FactExtractor
        from agent.memory.soul import SoulLoader
        from agent.memory.store import FactStore

        database = Database(cfg.memory.db_path)
        await database.connect()
        fact_store = FactStore(database)

        # Wire into memory tools
        from agent.tools.builtins.memory import set_fact_store

        set_fact_store(fact_store)

        # Wire session store for message persistence
        from agent.core.session import SessionStore

        session_store = SessionStore(db=database)
        agent_loop.session_store = session_store

        soul_loader = SoulLoader(cfg.memory.soul_path)
        agent_loop.soul_loader = soul_loader
        agent_loop.fact_store = fact_store

        # Rebuild system prompt with soul content
        from agent.llm.prompts import build_system_prompt

        agent_loop.system_prompt = build_system_prompt(cfg.agent, soul_content=soul_loader.load())

        # Fact extractor
        fact_extractor = FactExtractor(
            llm=agent_loop.llm,
            fact_store=fact_store,
            enabled=cfg.memory.auto_extract,
        )
        agent_loop.fact_extractor = fact_extractor

        # Vector store (optional — requires chromadb)
        try:
            from agent.memory.summarizer import ConversationSummarizer
            from agent.memory.vectors import VectorStore

            vector_store = VectorStore(
                persist_dir=cfg.memory.markdown_dir + "chroma",
            )
            await vector_store.initialize()
            agent_loop.vector_store = vector_store

            summarizer = ConversationSummarizer(llm=agent_loop.llm, vector_store=vector_store)
            agent_loop.summarizer = summarizer
        except Exception as e:
            logger.debug("vector_store_unavailable", error=str(e))

    except Exception as e:
        logger.debug("memory_init_skipped", error=str(e))

    # Welcome banner
    tool_count = len(registry.list_tools())
    console.print()
    console.print(
        Panel(
            f"[bold cyan]{cfg.agent.name}[/bold cyan] v{__version__}\n"
            f"Model: [green]{cfg.models.default}[/green]\n"
            f"Tools: [green]{tool_count} registered[/green]\n"
            f"Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
            title="Welcome",
            border_style="cyan",
        )
    )
    console.print()

    current_model = cfg.models.default

    while True:
        try:
            user_input = console.input("[bold green]You > [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            # Summarize session on exit
            if summarizer and session.message_count > 0:
                with contextlib.suppress(Exception):
                    await summarizer.summarize_session(session)
            if database:
                await database.close()
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            command = user_input.lower().split()
            cmd = command[0]

            if cmd in ("/exit", "/quit"):
                if summarizer and session.message_count > 0:
                    with contextlib.suppress(Exception):
                        await summarizer.summarize_session(session)
                if database:
                    await database.close()
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/clear":
                session.clear()
                console.print("[dim]Conversation cleared.[/dim]")
                continue
            elif cmd == "/model":
                if len(command) > 1:
                    current_model = command[1]
                    cfg.models.default = current_model
                    agent_loop.llm = LLMProvider(cfg.models)
                    console.print(f"[dim]Model switched to: {current_model}[/dim]")
                else:
                    console.print(f"[dim]Current model: {current_model}[/dim]")
                    _print_available_models(cfg)
                continue
            elif cmd == "/models":
                _print_available_models(cfg)
                continue
            elif cmd == "/config":
                _print_config_summary(cfg)
                continue
            elif cmd == "/tools":
                _print_tools_list()
                continue
            elif cmd == "/audit":
                await _print_audit_entries(audit)
                continue
            elif cmd == "/plan":
                _print_active_plan(planner)
                continue
            elif cmd == "/memory":
                await _print_memory_facts(fact_store)
                continue
            elif cmd == "/soul":
                _print_soul(agent_loop)
                continue
            elif cmd == "/heartbeat":
                console.print("[dim]Heartbeat is available via 'agent heartbeat start'[/dim]")
                continue
            elif cmd == "/help":
                _print_help()
                continue
            else:
                console.print(f"[dim]Unknown command: {cmd}. Type /help for help.[/dim]")
                continue

        # Send message to agent
        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]"):
                response = await agent_loop.process_message(user_input, session)

            console.print()
            console.print(
                Panel(
                    Markdown(response.content),
                    title=f"[bold cyan]{cfg.agent.name}[/bold cyan]",
                    border_style="cyan",
                )
            )
            console.print(
                f"[dim]Tokens: {response.usage.input_tokens} in / "
                f"{response.usage.output_tokens} out | "
                f"Model: {response.model}[/dim]"
            )
            console.print()
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}\n")


async def _chat_loop_sdk(cfg: AgentConfig) -> None:
    """Run the interactive chat REPL using Claude Agent SDK.

    Uses local Claude Code subscription instead of API keys.

    Args:
        cfg: Validated agent configuration.
    """
    from agent.llm.claude_sdk import ClaudeSDKService, sdk_available

    if not sdk_available():
        console.print(
            "[red]Error:[/red] claude-agent-sdk is not installed.\n"
            "Install it with: [bold]pip install claude-agent-sdk[/bold]"
        )
        return

    sdk_cfg = cfg.models.claude_sdk
    working_dir = sdk_cfg.working_dir or str(Path.cwd())

    # Initialize memory components for SDK chat (personality + facts)
    soul_loader = None
    fact_store = None
    vector_store = None
    fact_extractor = None
    db = None
    try:
        from agent.memory.soul import SoulLoader

        soul_loader = SoulLoader(cfg.memory.soul_path)
    except Exception as e:
        logger.debug("soul_loader_init_failed", error=str(e))

    try:
        from agent.memory.database import Database
        from agent.memory.store import FactStore

        db = Database(cfg.memory.db_path)
        await db.connect()
        fact_store = FactStore(db)
    except Exception as e:
        logger.debug("fact_store_init_failed", error=str(e))

    try:
        from agent.memory.vectors import VectorStore

        vector_store = VectorStore(
            persist_dir=str(Path(cfg.memory.markdown_dir) / "chroma"),
        )
        await vector_store.initialize()
    except Exception as e:
        logger.debug("vector_store_init_failed", error=str(e))

    if fact_store:
        try:
            from agent.llm.provider import LLMProvider as _LLMProvider
            from agent.memory.extraction import FactExtractor

            llm = _LLMProvider(cfg.models)
            fact_extractor = FactExtractor(
                llm=llm,
                fact_store=fact_store,
                enabled=cfg.memory.auto_extract,
            )
        except Exception as e:
            logger.debug("fact_extractor_init_failed", error=str(e))

    sdk = ClaudeSDKService(
        working_dir=working_dir,
        max_turns=sdk_cfg.max_turns,
        permission_mode=sdk_cfg.permission_mode,
        model=sdk_cfg.model,
        claude_auth_dir=sdk_cfg.claude_auth_dir,
        soul_loader=soul_loader,
        fact_store=fact_store,
        vector_store=vector_store,
        fact_extractor=fact_extractor,
    )

    ok, msg = await sdk.check_available()
    if not ok:
        console.print(f"[red]Error:[/red] {msg}")
        return

    session_id: str | None = None

    # Permission callback — asks user in terminal
    async def on_permission(
        tool_name: str,
        details: str,
        tool_input: dict,  # noqa: ARG001
    ) -> bool:
        console.print(f"\n[bold yellow]Permission required:[/bold yellow] {tool_name}")
        console.print(f"[dim]{details}[/dim]")
        try:
            answer = console.input("[bold yellow]Allow? (y/n) > [/bold yellow]").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        return answer in ("y", "yes")

    # Question callback — asks user in terminal
    async def on_question(question: str, options: list[str]) -> str:
        console.print(f"\n[bold cyan]Claude asks:[/bold cyan] {question}")
        if options:
            for i, opt in enumerate(options, 1):
                console.print(f"  [cyan]{i}.[/cyan] {opt}")
            console.print("[dim]Enter number or type your answer:[/dim]")
        try:
            answer = console.input("[bold cyan]Answer > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            return ""
        # If user typed a number, map to option
        if options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(options):
                return options[idx]
        return answer

    # Welcome banner
    console.print()
    console.print(
        Panel(
            f"[bold cyan]{cfg.agent.name}[/bold cyan] v{__version__}\n"
            f"Backend: [green]Claude Agent SDK[/green] (local subscription)\n"
            f"Working dir: [green]{working_dir}[/green]\n"
            f"Model: [green]{sdk_cfg.model or 'default'}[/green]\n"
            f"Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
            title="Welcome (SDK Mode)",
            border_style="cyan",
        )
    )
    console.print()

    try:
        while True:
            try:
                user_input = console.input("[bold green]You > [/bold green]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                if cmd in ("/exit", "/quit"):
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif cmd == "/session":
                    console.print(f"[dim]Session: {session_id or 'none (new)'}[/dim]")
                    continue
                elif cmd == "/new":
                    session_id = None
                    console.print("[dim]New session started.[/dim]")
                    continue
                elif cmd == "/help":
                    table = Table(
                        title="Commands (SDK Mode)",
                        show_header=True,
                        header_style="bold cyan",
                    )
                    table.add_column("Command", style="green")
                    table.add_column("Description")
                    table.add_row("/help", "Show this help")
                    table.add_row("/exit, /quit", "Exit chat")
                    table.add_row("/session", "Show current session ID")
                    table.add_row("/new", "Start a new session")
                    console.print(table)
                    continue
                else:
                    console.print(f"[dim]Unknown command: {cmd}. Type /help for help.[/dim]")
                    continue

            # Send to Claude SDK
            console.print()
            result_text = ""
            total_cost = 0.0
            total_input = 0
            total_output = 0

            try:
                async for event in sdk.run_task_stream(
                    prompt=user_input,
                    session_id=session_id,
                    working_dir=working_dir,
                    on_permission=on_permission,
                    on_question=on_question,
                ):
                    if event.type == "text":
                        is_subagent = event.data and event.data.get("subagent")
                        if not is_subagent:
                            console.print(event.content, end="")
                            result_text += event.content
                    elif event.type == "thinking":
                        console.print(f"[dim italic]{event.content}[/dim italic]", end="")
                    elif event.type == "tool_use":
                        tool_name = event.data.get("tool", "?")
                        console.print(f"\n[bold yellow]> {tool_name}[/bold yellow]", end="")
                        if tool_name == "Bash":
                            cmd = event.data.get("input", {}).get("command", "")
                            if cmd:
                                console.print(f" [dim]$ {cmd}[/dim]", end="")
                        elif tool_name in ("Write", "Edit", "Read"):
                            path = event.data.get("input", {}).get("file_path", "")
                            if path:
                                console.print(f" [dim]{path}[/dim]", end="")
                        console.print()
                    elif event.type == "result":
                        session_id = event.data.get("session_id")
                        total_cost = event.data.get("cost_usd", 0.0)
                        total_input = event.data.get("input_tokens", 0)
                        total_output = event.data.get("output_tokens", 0)
                    elif event.type == "error":
                        console.print(f"\n[red]Error:[/red] {event.content}")

                console.print()
                cost_str = f"${total_cost:.4f}" if total_cost else "N/A"
                console.print(
                    f"[dim]Tokens: {total_input} in / {total_output} out | "
                    f"Cost: {cost_str} | "
                    f"Session: {session_id or 'N/A'}[/dim]"
                )
                console.print()

            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}\n")
    finally:
        # Clean up memory resources
        if db is not None:
            with contextlib.suppress(Exception):
                await db.close()


def _print_available_models(cfg: AgentConfig) -> None:
    """Print available models based on configured API keys."""
    available = get_available_models(cfg)
    if not available:
        console.print("[yellow]No API keys configured. Add keys to .env file.[/yellow]")
        return

    table = Table(title="Available Models", show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Active", justify="center")

    for provider, models in available.items():
        for i, model in enumerate(models):
            active = "[bold green]\u2713[/bold green]" if model == cfg.models.default else ""
            prov_label = provider if i == 0 else ""
            table.add_row(prov_label, model, active)

    console.print(table)
    console.print("[dim]Usage: /model <model_name>[/dim]")


def _print_help() -> None:
    """Print available chat commands."""
    table = Table(title="Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="green")
    table.add_column("Description")
    table.add_row("/help", "Show this help message")
    table.add_row("/exit, /quit", "Exit the chat")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/model <name>", "Switch the LLM model")
    table.add_row("/models", "List available models")
    table.add_row("/config", "Show current configuration summary")
    table.add_row("/tools", "List available tools")
    table.add_row("/memory", "Show stored facts about you")
    table.add_row("/soul", "Show agent personality (soul.md)")
    table.add_row("/audit", "Show last 10 audit entries")
    table.add_row("/plan", "Show current plan (if any)")
    table.add_row("/heartbeat", "Show heartbeat status")
    console.print(table)


def _print_config_summary(cfg: AgentConfig) -> None:
    """Print a brief configuration summary."""
    table = Table(title="Configuration", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("Agent Name", cfg.agent.name)
    table.add_row("Default Model", cfg.models.default)
    table.add_row("Fallback Model", cfg.models.fallback or "(none)")
    table.add_row("Log Level", cfg.logging.level)
    table.add_row("Log Format", cfg.logging.format)
    console.print(table)


def _print_tools_list() -> None:
    """Print list of registered tools."""
    tools = registry.list_tools()
    if not tools:
        console.print("[dim]No tools registered.[/dim]")
        return

    table = Table(title="Registered Tools", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Tier")
    table.add_column("Status")
    table.add_column("Description", max_width=50)

    tier_colors = {"safe": "green", "moderate": "yellow", "dangerous": "red"}

    for t in tools:
        tier_color = tier_colors.get(t.tier.value, "white")
        status = "[green]enabled[/green]" if t.enabled else "[red]disabled[/red]"
        table.add_row(
            t.name,
            f"[{tier_color}]{t.tier.value}[/{tier_color}]",
            status,
            t.description[:50] + ("..." if len(t.description) > 50 else ""),
        )

    console.print(table)


async def _print_audit_entries(audit: object, limit: int = 10) -> None:
    """Print recent audit log entries."""
    entries = await audit.get_entries(limit=limit)
    if not entries:
        console.print("[dim]No audit entries yet.[/dim]")
        return

    table = Table(title=f"Last {limit} Audit Entries", show_header=True, header_style="bold cyan")
    table.add_column("Time", style="dim")
    table.add_column("Tool", style="green")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Trigger")

    status_colors = {
        "success": "green",
        "error": "red",
        "timeout": "yellow",
        "denied": "red",
        "blocked": "red",
    }

    for entry in entries:
        color = status_colors.get(entry.status, "white")
        table.add_row(
            entry.timestamp.strftime("%H:%M:%S"),
            entry.tool_name,
            f"[{color}]{entry.status}[/{color}]",
            f"{entry.duration_ms}ms",
            entry.trigger,
        )

    console.print(table)


def _print_active_plan(planner: Planner) -> None:
    """Print the currently active plan."""
    plan = planner.get_active_plan()
    if not plan:
        console.print("[dim]No active plan.[/dim]")
        return

    console.print(Panel(plan.to_context_string(), title="Active Plan", border_style="cyan"))


async def _print_memory_facts(fact_store: object | None) -> None:
    """Print top 10 stored facts."""
    if not fact_store:
        console.print("[dim]Memory not available (no database).[/dim]")
        return

    from agent.memory.store import FactStore

    assert isinstance(fact_store, FactStore)
    facts = await fact_store.get_relevant(limit=10)
    if not facts:
        console.print("[dim]No facts stored yet.[/dim]")
        return

    table = Table(title="Stored Facts", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="green")
    table.add_column("Value")
    table.add_column("Category", style="dim")
    table.add_column("Confidence", style="dim")

    for fact in facts:
        table.add_row(
            fact.key,
            fact.value,
            fact.category,
            f"{fact.confidence:.2f}",
        )

    console.print(table)


def _print_soul(agent_loop: AgentLoop) -> None:
    """Print the current soul.md content."""
    if agent_loop.soul_loader:
        content = agent_loop.soul_loader.content
        console.print(Panel(Markdown(content), title="Soul", border_style="cyan"))
    else:
        console.print("[dim]Soul.md not loaded.[/dim]")


@app.command()
def init() -> None:
    """Set up Agent configuration (interactive).

    Creates agent.yaml and .env in the agent home directory
    (~/.config/agent/) so the agent can be started from anywhere.
    """
    import secrets

    agent_home = get_agent_home()
    config_path = agent_home / "agent.yaml"
    env_path = agent_home / ".env"

    console.print(
        Panel(
            f"This will create configuration files in:\n" f"[cyan]{agent_home}[/cyan]",
            title="Agent Setup",
            border_style="cyan",
        )
    )

    # Check for existing config
    if config_path.exists():
        overwrite = typer.confirm(
            f"Config already exists at {config_path}. Overwrite?",
            default=False,
        )
        if not overwrite:
            console.print("[yellow]Keeping existing config.[/yellow]")
            return

    # --- Gather settings ---

    # 1. LLM Backend
    console.print("\n[bold]1. LLM Backend[/bold]")
    console.print(
        "  [cyan]litellm[/cyan]    — Use API keys (Anthropic, OpenAI, Gemini, Ollama)\n"
        "  [cyan]claude-sdk[/cyan] — Use your local Claude Max/Pro subscription (no API key needed)"
    )
    backend = typer.prompt(
        "Backend [litellm/claude-sdk]",
        default="litellm",
        type=str,
    ).strip()
    if backend not in ("litellm", "claude-sdk"):
        console.print(f"[yellow]Unknown backend '{backend}', using litellm.[/yellow]")
        backend = "litellm"

    # 2. API keys (for litellm backend)
    anthropic_key = ""
    openai_key = ""
    gemini_key = ""
    if backend == "litellm":
        console.print("\n[bold]2. API Keys[/bold]")
        console.print("  [dim]Paste your keys below. Leave blank to skip a provider.[/dim]")
        console.print("  [dim]You can always add or change keys later in the .env file.[/dim]")
        anthropic_key = typer.prompt(
            "  Anthropic API key",
            default="",
            show_default=False,
        ).strip()
        openai_key = typer.prompt(
            "  OpenAI API key",
            default="",
            show_default=False,
        ).strip()
        gemini_key = typer.prompt(
            "  Gemini API key",
            default="",
            show_default=False,
        ).strip()

        if not any([anthropic_key, openai_key, gemini_key]):
            console.print(
                "\n  [yellow]No API keys provided.[/yellow] "
                "You can add them later in "
                f"[cyan]{env_path}[/cyan]"
            )
    else:
        console.print(
            "\n[dim]2. API Keys — skipped (Claude SDK uses your local subscription)[/dim]"
        )

    # 3. Telegram
    console.print("\n[bold]3. Telegram Bot[/bold] [dim](optional — press Enter to skip)[/dim]")
    console.print(
        "  [dim]To create a bot: open Telegram, message @BotFather, send /newbot,[/dim]\n"
        "  [dim]follow the prompts, and copy the token it gives you.[/dim]"
    )
    telegram_token = typer.prompt(
        "  Bot token",
        default="",
        show_default=False,
    ).strip()
    telegram_enabled = bool(telegram_token)

    telegram_users_str = ""
    if telegram_enabled:
        console.print(
            "\n  [dim]To find your Telegram user ID: message @userinfobot on Telegram.[/dim]\n"
            "  [dim]This restricts who can use the bot. Use * to allow everyone.[/dim]"
        )
        telegram_users_str = typer.prompt(
            "  Allowed user IDs (comma-separated, or * for all)",
            default="*",
        ).strip()

    # 4. Gateway
    console.print("\n[bold]4. Gateway & Dashboard[/bold]")
    console.print("  [dim]The gateway serves the API and web dashboard.[/dim]")
    gateway_port = typer.prompt("  Port", default=8765, type=int)
    gateway_token = secrets.token_urlsafe(32)

    # --- Build .env ---
    env_lines = ["# Agent environment variables", ""]

    if anthropic_key:
        env_lines.append(f"ANTHROPIC_API_KEY={anthropic_key}")
    else:
        env_lines.append("# ANTHROPIC_API_KEY=sk-ant-your-key-here")

    if openai_key:
        env_lines.append(f"OPENAI_API_KEY={openai_key}")
    else:
        env_lines.append("# OPENAI_API_KEY=sk-your-key-here")

    if gemini_key:
        env_lines.append(f"GEMINI_API_KEY={gemini_key}")
    else:
        env_lines.append("# GEMINI_API_KEY=your-key-here")

    env_lines.append("")

    if telegram_token:
        env_lines.append(f"TELEGRAM_BOT_TOKEN={telegram_token}")
    else:
        env_lines.append("# TELEGRAM_BOT_TOKEN=your-bot-token")

    env_lines.append("")
    env_lines.append(f"GATEWAY_TOKEN={gateway_token}")
    env_lines.append("")

    env_path.write_text("\n".join(env_lines), encoding="utf-8")

    # --- Build agent.yaml ---

    # Parse allowed users
    allowed_users: list[int] | list[str] = []
    if telegram_users_str and telegram_users_str != "*":
        for uid in telegram_users_str.split(","):
            uid = uid.strip()
            if uid.isdigit():
                allowed_users.append(int(uid))

    # Pick default model
    if backend == "litellm":
        if anthropic_key:
            default_model = "claude-sonnet-4-5-20250929"
        elif openai_key:
            default_model = "gpt-4o"
        elif gemini_key:
            default_model = "gemini/gemini-2.5-flash"
        else:
            default_model = "claude-sonnet-4-5-20250929"
    else:
        default_model = "claude-sonnet-4-5-20250929"

    config_data: dict[str, Any] = {
        "agent": {
            "name": "Agent",
            "persona": (
                "You are a helpful autonomous AI assistant running on the user's "
                "local machine.\nYou are proactive, concise, and always try to be "
                "helpful.\nWhen you don't know something, you say so honestly.\n"
            ),
            "max_iterations": 10,
            "heartbeat_interval": "30m",
        },
        "models": {
            "backend": backend,
            "default": default_model,
            "providers": {
                "anthropic": {"api_key": "${ANTHROPIC_API_KEY}"},
                "openai": {"api_key": "${OPENAI_API_KEY}"},
                "ollama": {"base_url": "http://localhost:11434"},
            },
        },
        "channels": {
            "telegram": {
                "enabled": telegram_enabled,
                "token": "${TELEGRAM_BOT_TOKEN}",
                "allowed_users": allowed_users if allowed_users else [],
            },
            "webchat": {
                "enabled": True,
                "port": 8080,
            },
        },
        "gateway": {
            "host": "127.0.0.1",
            "port": gateway_port,
            "auth_token": "${GATEWAY_TOKEN}",
            "cors_origins": ["http://localhost:5173"],
        },
        "tools": {
            "shell": {"enabled": True, "sandbox": False, "allowed_commands": ["*"]},
            "browser": {"enabled": True, "headless": True},
            "filesystem": {
                "enabled": True,
                "root": "~/",
                "write_root": "~",
                "max_file_size": 10485760,
            },
        },
        "memory": {
            "db_path": "./data/agent.db",
            "markdown_dir": "./data/memory/",
            "auto_extract": True,
        },
        "logging": {
            "level": "INFO",
            "format": "console",
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    # --- Done ---
    console.print()

    next_steps = [
        f"[green]Config:[/green]  {config_path}",
        f"[green]Secrets:[/green] {env_path}",
        "",
        "[bold]Next steps:[/bold]",
        "  [cyan]agent start[/cyan]   — Start the agent",
        "  [cyan]agent stop[/cyan]    — Stop the agent",
        "  [cyan]agent doctor[/cyan]  — Verify your setup",
        "  [cyan]agent chat[/cyan]    — Quick terminal chat",
    ]

    if telegram_enabled:
        next_steps.append("")
        next_steps.append("  Telegram bot is enabled. Start the agent and message your bot!")

    next_steps.extend(
        [
            "",
            f"  Dashboard: [blue]http://127.0.0.1:{gateway_port}/dashboard[/blue]",
            f"  [dim]Dashboard login token is in {env_path} (GATEWAY_TOKEN)[/dim]",
            "",
            f"  [dim]Edit config anytime: {config_path}[/dim]",
        ]
    )

    console.print(
        Panel(
            "\n".join(next_steps),
            title="Setup Complete",
            border_style="green",
        )
    )


@app.command()
def start(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    host: str | None = typer.Option(None, "--host", help="Override bind host"),
    port: int | None = typer.Option(None, "--port", "-p", help="Override bind port"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only"),
) -> None:
    """Start the full agent (gateway + heartbeat)."""
    cfg = _load_config(config)
    _apply_log_level_flags(cfg, verbose=verbose, quiet=quiet)
    setup_logging(cfg.logging)
    cfg = _resolve_workspace(cfg, workspace)

    if host:
        cfg.gateway.host = host
    if port:
        cfg.gateway.port = port

    asyncio.run(_run_gateway(cfg))


async def _run_gateway(cfg: AgentConfig) -> None:
    """Run the full agent via Application lifecycle."""
    from agent.core.startup import Application

    application = Application(cfg)
    shutdown_triggered = False

    async def _graceful_shutdown(sig: signal.Signals) -> None:
        nonlocal shutdown_triggered
        if shutdown_triggered:
            return
        shutdown_triggered = True
        logger.info("shutdown_signal_received", signal=sig.name)
        _remove_pid_file()
        await application.shutdown()

    try:
        await application.initialize()
        _write_pid_file()

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(_graceful_shutdown(s)),
            )

        console.print(
            Panel(
                f"[bold cyan]{cfg.agent.name}[/bold cyan] Gateway v{__version__}\n"
                f"Listening on [green]http://{cfg.gateway.host}:{cfg.gateway.port}[/green]\n"
                f"Auth: {'[green]enabled[/green]' if cfg.gateway.auth_token else '[yellow]open[/yellow]'}\n"  # noqa: E501
                f"Docs: [blue]http://{cfg.gateway.host}:{cfg.gateway.port}/docs[/blue]\n"
                f"Stop with: [yellow]agent stop[/yellow] or [yellow]Ctrl+C[/yellow]",
                title="Gateway Started",
                border_style="cyan",
            )
        )

        await application.start()
    except KeyboardInterrupt:
        pass
    finally:
        _remove_pid_file()
        if not shutdown_triggered:
            await application.shutdown()


@app.command()
def stop() -> None:
    """Stop a running agent process."""
    pid = _read_pid_file()
    if pid is None:
        err_console.print("[yellow]No running agent found[/yellow] (no PID file)")
        raise typer.Exit(1)

    if not _is_process_running(pid):
        console.print(f"[yellow]Agent process (PID {pid}) is not running. Cleaning up.[/yellow]")
        _remove_pid_file()
        raise typer.Exit(0)

    console.print(f"Stopping agent (PID {pid})...")
    try:
        if platform.system() == "Windows":
            # On Windows, SIGTERM calls TerminateProcess (no cleanup).
            # Use CTRL_C_EVENT to trigger KeyboardInterrupt for graceful shutdown.
            # Temporarily ignore SIGINT in this process so the console-wide
            # CTRL_C_EVENT doesn't kill the `agent stop` process itself.
            prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            os.kill(pid, signal.CTRL_C_EVENT)
            signal.signal(signal.SIGINT, prev_handler)
        else:
            os.kill(pid, signal.SIGTERM)
        console.print("[green]Agent stopped.[/green]")
        # Safety net: remove PID file in case the target process didn't clean up
        _remove_pid_file()
    except OSError as e:
        err_console.print(f"[red]Failed to stop agent:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def models(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List available models based on configured API keys."""
    cfg = _load_config(config)
    _print_available_models(cfg)


@app.command()
def version() -> None:
    """Print version and system info."""
    info = get_system_info()
    config_path = _find_config_display_path()

    console.print(f"[bold]Agent[/bold] v{__version__}")
    console.print(f"Python {info['python_version']}")
    console.print(f"OS: {platform.system()} {info['architecture']}")
    console.print(f"Config: {config_path}")


def _find_config_display_path() -> str:
    """Find the config file path for display."""
    if Path("agent.yaml").exists():
        return str(Path("agent.yaml").resolve())
    home_config = get_agent_home() / "agent.yaml"
    if home_config.exists():
        return str(home_config)
    return "(using defaults)"


@app.command()
def doctor(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    security: bool = typer.Option(False, "--security", help="Run only security checks"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace to check"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Check configuration health and connectivity."""
    cfg = _load_config(config)
    _apply_log_level_flags(cfg, verbose=verbose, quiet=False)
    setup_logging(cfg.logging)
    cfg = _resolve_workspace(cfg, workspace)
    asyncio.run(_run_doctor(cfg, security_only=security))


async def _run_doctor(cfg: AgentConfig, *, security_only: bool = False) -> None:
    """Run diagnostic checks and display results."""
    from agent.core.doctor import run_all_checks, run_security_checks

    status_styles = {
        "pass": "[green]PASS[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }

    with console.status("[bold cyan]Running health checks...[/bold cyan]"):
        if security_only:
            checks = run_security_checks(cfg)
        else:
            checks = await run_all_checks(cfg)

    # Group by category
    categories: dict[str, list] = {}
    for check in checks:
        categories.setdefault(check.category, []).append(check)

    pass_count = sum(1 for c in checks if c.status == "pass")
    warn_count = sum(1 for c in checks if c.status == "warn")
    fail_count = sum(1 for c in checks if c.status == "fail")

    for category, cat_checks in categories.items():
        table = Table(title=category, show_header=True, header_style="bold cyan")
        table.add_column("Check", style="bold", min_width=20)
        table.add_column("Status", min_width=6)
        table.add_column("Details")

        for check in cat_checks:
            table.add_row(
                check.name,
                status_styles.get(check.status, check.status),
                check.message,
            )

        console.print(table)
        console.print()

    # Summary
    console.print(
        f"  [green]{pass_count} passed[/green]  "
        f"[yellow]{warn_count} warnings[/yellow]  "
        f"[red]{fail_count} failed[/red]  "
        f"({len(checks)} total checks)"
    )
    console.print()


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Print resolved configuration with secrets masked."""
    cfg = _load_config(config)
    masked = config_to_dict_masked(cfg)
    yaml_str = yaml.dump(masked, default_flow_style=False, sort_keys=False)
    console.print(Panel(yaml_str, title="Resolved Configuration", border_style="cyan"))


# ---------------------------------------------------------------------------
# Tools subcommands
# ---------------------------------------------------------------------------


@tools_app.command("list")
def tools_list(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List all registered tools with tier and status."""
    _load_config(config)
    import agent.tools.builtins  # noqa: F401

    _print_tools_list()


@tools_app.command("enable")
def tools_enable(
    name: str = typer.Argument(help="Tool name to enable"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Enable a tool."""
    _load_config(config)
    import agent.tools.builtins  # noqa: F401

    try:
        registry.enable_tool(name)
        console.print(f"[green]Tool '{name}' enabled.[/green]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@tools_app.command("disable")
def tools_disable(
    name: str = typer.Argument(help="Tool name to disable"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Disable a tool."""
    _load_config(config)
    import agent.tools.builtins  # noqa: F401

    try:
        registry.disable_tool(name)
        console.print(f"[yellow]Tool '{name}' disabled.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


# ---------------------------------------------------------------------------
# Audit subcommands
# ---------------------------------------------------------------------------


@audit_app.callback(invoke_without_command=True)
def audit_default(
    ctx: typer.Context,
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of entries to show"),
) -> None:
    """Show recent audit log entries."""
    if ctx.invoked_subcommand is not None:
        return
    console.print(
        "[dim]Audit log is session-based in Phase 2." " Start a chat to generate entries.[/dim]"
    )


@audit_app.command("stats")
def audit_stats() -> None:
    """Show audit statistics."""
    console.print(
        "[dim]Audit stats are session-based in Phase 2." " Start a chat to generate entries.[/dim]"
    )


# ---------------------------------------------------------------------------
# Heartbeat subcommands
# ---------------------------------------------------------------------------


@heartbeat_app.command("start")
def heartbeat_start(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """Start heartbeat in foreground (for testing)."""
    cfg = _load_config(config)
    setup_logging(cfg.logging)
    asyncio.run(_run_heartbeat(cfg))


async def _run_heartbeat(cfg: AgentConfig) -> None:
    """Run the heartbeat daemon in foreground."""
    from agent.core.heartbeat import HeartbeatDaemon

    agent_loop, event_bus, *_ = _init_agent_stack(cfg)

    heartbeat = HeartbeatDaemon(
        agent_loop=agent_loop,
        config=cfg.agent,
        event_bus=event_bus,
    )

    console.print(f"[cyan]Starting heartbeat (interval: {cfg.agent.heartbeat_interval})...[/cyan]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    await heartbeat.start()

    stop_event = asyncio.Event()
    try:
        # Keep running until signaled
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        await heartbeat.stop()
        console.print("\n[dim]Heartbeat stopped.[/dim]")


@heartbeat_app.command("status")
def heartbeat_status() -> None:
    """Show heartbeat status."""
    console.print("[dim]Heartbeat status is only available while running.[/dim]")
    console.print("[dim]Use 'agent heartbeat start' to start the heartbeat daemon.[/dim]")
