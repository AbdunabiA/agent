"""Shared CLI helpers: config loading, logging, PID files, console instances."""

from __future__ import annotations

import contextlib
import os
import platform
from pathlib import Path

import structlog
import typer
from rich.console import Console

from agent.config import (
    AgentConfig,
    get_agent_home,
    get_config,
)
from agent.core.agent_loop import AgentLoop
from agent.core.audit import AuditLog
from agent.core.events import EventBus
from agent.core.guardrails import Guardrails
from agent.core.permissions import PermissionManager
from agent.core.planner import Planner
from agent.core.recovery import ErrorRecovery
from agent.llm.provider import LLMProvider
from agent.tools.executor import ToolExecutor
from agent.tools.registry import registry
from agent.workspaces.manager import WorkspaceManager, WorkspaceNotFoundError

logger = structlog.get_logger(__name__)

console = Console()
err_console = Console(stderr=True)


def _load_config(config_path: str | None = None) -> AgentConfig:
    """Load config with error handling."""
    try:
        return get_config(config_path)
    except Exception as e:
        err_console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from e


def _apply_log_level_flags(cfg: AgentConfig, *, verbose: bool, quiet: bool) -> None:
    """Override config log level from CLI flags.

    --verbose sets DEBUG, --quiet sets WARNING. If both are given, verbose wins.
    """
    if verbose:
        cfg.logging.level = "DEBUG"
    elif quiet:
        cfg.logging.level = "WARNING"


def _pid_file_path() -> Path:
    """Return the path to the agent PID file."""
    return get_agent_home() / "agent.pid"


def _write_pid_file() -> None:
    """Write current process PID to the PID file."""
    pid_path = _pid_file_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def _remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    pid_path = _pid_file_path()
    with contextlib.suppress(FileNotFoundError):
        pid_path.unlink()


def _read_pid_file() -> int | None:
    """Read the PID from the PID file, or None if missing/invalid."""
    pid_path = _pid_file_path()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if platform.system() == "Windows":
        # On Windows, os.kill(pid, 0) sends CTRL_C_EVENT (== 0) which actually
        # interrupts the process instead of just checking existence.
        # Use ctypes OpenProcess to safely check without side effects.
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        synchronize = 0x00100000
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _resolve_workspace(cfg: AgentConfig, workspace_name: str | None = None) -> AgentConfig:
    """Resolve workspace and apply config overrides.

    If workspace_name is given, use that workspace. Otherwise use the default.
    Returns a new AgentConfig with workspace overrides applied.
    """
    manager = WorkspaceManager(cfg)
    manager.ensure_default()
    ws_name = workspace_name or cfg.workspaces.default
    try:
        workspace = manager.resolve(ws_name)
    except WorkspaceNotFoundError:
        err_console.print(f"[red]Workspace '{ws_name}' not found.[/red]")
        err_console.print(f"[dim]Available: {', '.join(manager.discover()) or '(none)'}[/dim]")
        raise typer.Exit(1) from None
    resolved = manager.apply_overrides(cfg, workspace)
    logger.info("workspace_active", workspace=workspace.name)
    return resolved


def _init_agent_stack(
    cfg: AgentConfig,
) -> tuple[AgentLoop, EventBus, AuditLog, PermissionManager, Guardrails, ErrorRecovery, Planner]:
    """Initialize the full Phase 2 agent stack.

    Returns:
        Tuple of (agent_loop, event_bus, audit, permissions, guardrails, recovery, planner).
    """
    event_bus = EventBus()
    llm = LLMProvider(cfg.models)

    # Register built-in tools
    import agent.tools.builtins  # noqa: F401

    # Initialize Phase 2 components
    guardrails = Guardrails(cfg.tools)
    permissions = PermissionManager(cfg.tools)
    audit = AuditLog()
    recovery = ErrorRecovery()

    tool_executor = ToolExecutor(
        registry=registry,
        config=cfg.tools,
        event_bus=event_bus,
        audit=audit,
        permissions=permissions,
        guardrails=guardrails,
    )

    planner = Planner(llm=llm, config=cfg.agent)

    agent_loop = AgentLoop(
        llm=llm,
        config=cfg.agent,
        event_bus=event_bus,
        tool_executor=tool_executor,
        planner=planner,
        recovery=recovery,
        guardrails=guardrails,
        orchestration_enabled=cfg.orchestration.enabled,
        skill_builder_enabled=cfg.skills.builder.enabled,
    )

    return agent_loop, event_bus, audit, permissions, guardrails, recovery, planner
