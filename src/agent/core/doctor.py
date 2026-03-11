"""Comprehensive health diagnostic for the agent.

Runs checks across all subsystems and returns structured results.
Each check is fault-tolerant — failures are reported, never crash the diagnostic.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from agent import __version__
from agent.utils.helpers import get_system_info

if TYPE_CHECKING:
    from agent.config import AgentConfig

logger = structlog.get_logger(__name__)


@dataclass
class HealthCheck:
    """Result of a single health check."""

    name: str
    category: str
    status: str  # "pass", "warn", "fail"
    message: str
    details: str = ""


async def run_all_checks(config: AgentConfig) -> list[HealthCheck]:
    """Run all health checks and return results.

    Args:
        config: Agent configuration.

    Returns:
        List of HealthCheck results across all categories.
    """
    checks: list[HealthCheck] = []
    checks.extend(check_core(config))
    checks.extend(await check_llm(config))
    checks.extend(check_tools(config))
    checks.extend(check_skills(config))
    checks.extend(check_memory(config))
    checks.extend(check_channels(config))
    checks.extend(check_heartbeat(config))
    checks.extend(check_resources(config))
    checks.extend(check_security(config))
    return checks


def check_core(config: AgentConfig) -> list[HealthCheck]:
    """Check core system health."""
    checks: list[HealthCheck] = []
    info = get_system_info()

    checks.append(HealthCheck(
        name="Python Version",
        category="Core",
        status="pass",
        message=info["python_version"],
    ))
    checks.append(HealthCheck(
        name="Agent Version",
        category="Core",
        status="pass",
        message=__version__,
    ))
    checks.append(HealthCheck(
        name="Operating System",
        category="Core",
        status="pass",
        message=f"{info['os']} {info['architecture']}",
    ))

    # Config file
    if Path("agent.yaml").exists():
        checks.append(HealthCheck(
            name="Config File",
            category="Core",
            status="pass",
            message=str(Path("agent.yaml").resolve()),
        ))
    else:
        checks.append(HealthCheck(
            name="Config File",
            category="Core",
            status="warn",
            message="Using defaults (no agent.yaml)",
        ))

    # Data directory
    if Path("data").is_dir():
        checks.append(HealthCheck(
            name="Data Directory",
            category="Core",
            status="pass",
            message=str(Path("data").resolve()),
        ))
    else:
        checks.append(HealthCheck(
            name="Data Directory",
            category="Core",
            status="warn",
            message="Not found (will be created on first use)",
        ))

    return checks


async def check_llm(config: AgentConfig) -> list[HealthCheck]:
    """Check LLM provider connectivity."""
    checks: list[HealthCheck] = []

    # API keys
    providers = config.models.providers
    for name in ("anthropic", "openai"):
        provider = providers.get(name)
        if provider and provider.api_key:
            checks.append(HealthCheck(
                name=f"{name.title()} API Key",
                category="LLM Providers",
                status="pass",
                message="Configured",
            ))
        else:
            checks.append(HealthCheck(
                name=f"{name.title()} API Key",
                category="LLM Providers",
                status="fail",
                message=f"Not set (set {name.upper()}_API_KEY)",
            ))

    # Model connectivity
    try:
        from agent.llm.provider import LLMProvider

        llm = LLMProvider(config.models)
        ok = await llm.test_connection(config.models.default)
        checks.append(HealthCheck(
            name="Default Model",
            category="LLM Providers",
            status="pass" if ok else "fail",
            message=f"{config.models.default} — {'connected' if ok else 'not accessible'}",
        ))
    except Exception as e:
        checks.append(HealthCheck(
            name="Default Model",
            category="LLM Providers",
            status="fail",
            message=f"{config.models.default} — error: {e}",
        ))

    # Claude Agent SDK check
    try:
        from agent.llm.claude_sdk import ClaudeSDKService, sdk_available

        if sdk_available():
            sdk_cfg = config.models.claude_sdk
            sdk_service = ClaudeSDKService(
                claude_auth_dir=sdk_cfg.claude_auth_dir,
            )
            ok, msg = await sdk_service.check_available()
            checks.append(HealthCheck(
                name="Claude Agent SDK",
                category="LLM Providers",
                status="pass" if ok else "warn",
                message=msg,
            ))
        else:
            checks.append(HealthCheck(
                name="Claude Agent SDK",
                category="LLM Providers",
                status="warn",
                message="Not installed (pip install claude-agent-sdk)",
            ))
    except Exception as e:
        checks.append(HealthCheck(
            name="Claude Agent SDK",
            category="LLM Providers",
            status="warn",
            message=f"Check failed: {e}",
        ))

    if config.models.fallback:
        try:
            ok = await llm.test_connection(config.models.fallback)
            checks.append(HealthCheck(
                name="Fallback Model",
                category="LLM Providers",
                status="pass" if ok else "warn",
                message=f"{config.models.fallback} — {'connected' if ok else 'not accessible'}",
            ))
        except Exception as e:
            checks.append(HealthCheck(
                name="Fallback Model",
                category="LLM Providers",
                status="warn",
                message=f"{config.models.fallback} — error: {e}",
            ))

    return checks


def check_tools(config: AgentConfig) -> list[HealthCheck]:
    """Check registered tools."""
    checks: list[HealthCheck] = []

    try:
        import agent.tools.builtins  # noqa: F401
        from agent.tools.registry import registry

        tools = registry.list_tools()
        enabled = [t for t in tools if t.enabled]
        builtin = [t for t in tools if t.category == "builtin"]
        skill_tools = [t for t in tools if t.category == "skill"]

        checks.append(HealthCheck(
            name="Registered Tools",
            category="Tools",
            status="pass",
            message=(
                f"{len(enabled)}/{len(tools)} enabled"
                f" ({len(builtin)} built-in, {len(skill_tools)} from skills)"
            ),
        ))

        for t in tools:
            checks.append(HealthCheck(
                name=t.name,
                category="Tools",
                status="pass" if t.enabled else "warn",
                message=f"{'enabled' if t.enabled else 'disabled'}, {t.tier.value} tier",
            ))
    except Exception as e:
        checks.append(HealthCheck(
            name="Tool Registry",
            category="Tools",
            status="fail",
            message=f"Error loading tools: {e}",
        ))

    return checks


def check_skills(config: AgentConfig) -> list[HealthCheck]:
    """Check skill system."""
    checks: list[HealthCheck] = []

    skills_dir = Path(config.skills.directory)
    if not skills_dir.is_dir():
        checks.append(HealthCheck(
            name="Skills Directory",
            category="Skills",
            status="warn",
            message=f"Not found: {skills_dir}",
        ))
        return checks

    try:
        from agent.skills.loader import SkillLoader

        loader = SkillLoader()
        discovered = loader.discover(skills_dir)

        loaded_count = 0
        for skill_dir in discovered:
            try:
                meta = loader.parse_metadata(skill_dir)
                is_disabled = meta.name in config.skills.disabled
                status = "warn" if is_disabled else "pass"
                label = "disabled" if is_disabled else "available"

                # Check dependencies
                missing = loader.check_dependencies(meta)
                if missing:
                    status = "warn"
                    label = f"missing deps: {', '.join(missing)}"
                elif not is_disabled:
                    loaded_count += 1

                checks.append(HealthCheck(
                    name=meta.display_name or meta.name,
                    category="Skills",
                    status=status,
                    message=f"{label} — v{meta.version}",
                ))
            except Exception as e:
                checks.append(HealthCheck(
                    name=skill_dir.name,
                    category="Skills",
                    status="fail",
                    message=f"Load error: {e}",
                ))

        # Summary at the top
        checks.insert(
            len(checks) - len(discovered),
            HealthCheck(
                name="Skills Summary",
                category="Skills",
                status="pass",
                message=f"{loaded_count} loadable / {len(discovered)} discovered",
            ),
        )
    except Exception as e:
        checks.append(HealthCheck(
            name="Skill Loader",
            category="Skills",
            status="fail",
            message=str(e),
        ))

    return checks


def check_memory(config: AgentConfig) -> list[HealthCheck]:
    """Check memory subsystem."""
    checks: list[HealthCheck] = []

    # SQLite database
    db_path = Path(config.memory.db_path)
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        checks.append(HealthCheck(
            name="SQLite Database",
            category="Memory",
            status="pass",
            message=f"{db_path} ({size_mb:.1f} MB)",
        ))
    else:
        checks.append(HealthCheck(
            name="SQLite Database",
            category="Memory",
            status="warn",
            message="Not created yet (will initialize on first use)",
        ))

    # soul.md
    soul_path = config.memory.soul_path
    if soul_path and Path(soul_path).exists():
        size_kb = Path(soul_path).stat().st_size / 1024
        checks.append(HealthCheck(
            name="soul.md",
            category="Memory",
            status="pass",
            message=f"Loaded ({size_kb:.1f} KB)",
        ))
    elif Path("soul.md").exists():
        checks.append(HealthCheck(
            name="soul.md",
            category="Memory",
            status="pass",
            message="Found in project root",
        ))
    else:
        checks.append(HealthCheck(
            name="soul.md",
            category="Memory",
            status="warn",
            message="Not found (using default personality)",
        ))

    # ChromaDB directory
    chroma_dir = Path(config.memory.markdown_dir + "chroma")
    if chroma_dir.is_dir():
        checks.append(HealthCheck(
            name="ChromaDB",
            category="Memory",
            status="pass",
            message=f"Directory exists: {chroma_dir}",
        ))
    else:
        checks.append(HealthCheck(
            name="ChromaDB",
            category="Memory",
            status="warn",
            message="Not initialized yet",
        ))

    return checks


def check_channels(config: AgentConfig) -> list[HealthCheck]:
    """Check messaging channels."""
    checks: list[HealthCheck] = []

    # Telegram
    tg = config.channels.telegram
    if tg.enabled and tg.token:
        users = len(tg.allowed_users) if tg.allowed_users else 0
        checks.append(HealthCheck(
            name="Telegram",
            category="Channels",
            status="pass",
            message=f"Enabled — {users} allowed user(s)",
        ))
    elif tg.enabled and not tg.token:
        checks.append(HealthCheck(
            name="Telegram",
            category="Channels",
            status="fail",
            message="Enabled but no token set",
        ))
    else:
        checks.append(HealthCheck(
            name="Telegram",
            category="Channels",
            status="warn",
            message="Disabled",
        ))

    # WebChat
    wc = config.channels.webchat
    if wc.enabled:
        checks.append(HealthCheck(
            name="WebChat",
            category="Channels",
            status="pass",
            message=f"Enabled on port {wc.port}",
        ))
    else:
        checks.append(HealthCheck(
            name="WebChat",
            category="Channels",
            status="warn",
            message="Disabled",
        ))

    return checks


def check_heartbeat(config: AgentConfig) -> list[HealthCheck]:
    """Check heartbeat configuration."""
    checks: list[HealthCheck] = []

    if Path("HEARTBEAT.md").exists():
        size = Path("HEARTBEAT.md").stat().st_size
        checks.append(HealthCheck(
            name="HEARTBEAT.md",
            category="Heartbeat",
            status="pass",
            message=f"Found ({size} bytes)",
        ))
    else:
        checks.append(HealthCheck(
            name="HEARTBEAT.md",
            category="Heartbeat",
            status="warn",
            message="Not found",
        ))

    checks.append(HealthCheck(
        name="Interval",
        category="Heartbeat",
        status="pass",
        message=config.agent.heartbeat_interval,
    ))

    return checks


def check_resources(config: AgentConfig) -> list[HealthCheck]:
    """Check disk and resource usage."""
    checks: list[HealthCheck] = []

    # Data directory size
    data_dir = Path("data")
    if data_dir.is_dir():
        total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        checks.append(HealthCheck(
            name="Data Directory Size",
            category="Resources",
            status="pass",
            message=f"{size_mb:.1f} MB used",
        ))

    # Disk space
    try:
        usage = shutil.disk_usage(Path.cwd())
        free_gb = usage.free / (1024 ** 3)
        status = "pass" if free_gb > 1 else "warn"
        checks.append(HealthCheck(
            name="Disk Space",
            category="Resources",
            status=status,
            message=f"{free_gb:.1f} GB available",
        ))
    except Exception:
        pass

    return checks


def check_security(config: AgentConfig) -> list[HealthCheck]:
    """Check security configuration."""
    checks: list[HealthCheck] = []

    # .env file
    env_path = Path(".env")
    if env_path.exists():
        checks.append(HealthCheck(
            name=".env File",
            category="Security",
            status="pass",
            message="Present",
        ))
    else:
        checks.append(HealthCheck(
            name=".env File",
            category="Security",
            status="warn",
            message="Not found (using environment variables)",
        ))

    # Gateway auth
    if config.gateway.auth_token:
        checks.append(HealthCheck(
            name="Gateway Auth",
            category="Security",
            status="pass",
            message="Auth token configured",
        ))
    else:
        checks.append(HealthCheck(
            name="Gateway Auth",
            category="Security",
            status="warn",
            message="No auth token — API is open",
        ))

    # Filesystem root
    fs_root = config.tools.filesystem.root
    if fs_root and fs_root != "~/" and fs_root != str(Path.home()):
        checks.append(HealthCheck(
            name="Filesystem Root",
            category="Security",
            status="pass",
            message=f"Restricted to: {fs_root}",
        ))
    else:
        checks.append(HealthCheck(
            name="Filesystem Root",
            category="Security",
            status="warn",
            message=f"Broad access: {fs_root or '~'}",
        ))

    # Telegram allowlist
    tg = config.channels.telegram
    if tg.enabled and not tg.allowed_users:
        checks.append(HealthCheck(
            name="Telegram Allowlist",
            category="Security",
            status="warn",
            message="Telegram enabled but allowed_users is empty — anyone can use the bot",
        ))
    elif tg.enabled and tg.allowed_users:
        checks.append(HealthCheck(
            name="Telegram Allowlist",
            category="Security",
            status="pass",
            message=f"{len(tg.allowed_users)} allowed user(s)",
        ))

    # Hardcoded secrets scan
    checks.extend(_check_hardcoded_secrets())

    # .env file permissions (Unix only)
    if env_path.exists() and os.name != "nt":
        try:
            mode = oct(env_path.stat().st_mode)[-3:]
            if mode in ("600", "400"):
                checks.append(HealthCheck(
                    name=".env Permissions",
                    category="Security",
                    status="pass",
                    message=f"File mode: {mode}",
                ))
            else:
                checks.append(HealthCheck(
                    name=".env Permissions",
                    category="Security",
                    status="warn",
                    message=f"File mode: {mode} (recommend 600)",
                ))
        except Exception:
            pass

    # CORS origins check
    origins = config.gateway.cors_origins
    if "*" in origins:
        checks.append(HealthCheck(
            name="CORS Origins",
            category="Security",
            status="warn",
            message="Wildcard '*' allows all origins",
        ))
    else:
        checks.append(HealthCheck(
            name="CORS Origins",
            category="Security",
            status="pass",
            message=f"{len(origins)} origin(s) configured",
        ))

    return checks


def _check_hardcoded_secrets() -> list[HealthCheck]:
    """Scan source files for potential hardcoded secrets."""
    import re

    checks: list[HealthCheck] = []
    secret_patterns = [
        re.compile(r'sk-ant-[a-zA-Z0-9]{20,}'),
        re.compile(r'sk-[a-zA-Z0-9]{20,}'),
        re.compile(r'AIza[a-zA-Z0-9_-]{35}'),
    ]

    src_dir = Path("src")
    if not src_dir.is_dir():
        return checks

    found_secrets = False
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in secret_patterns:
                if pattern.search(content):
                    found_secrets = True
                    checks.append(HealthCheck(
                        name="Hardcoded Secret",
                        category="Security",
                        status="fail",
                        message=f"Potential secret in {py_file}",
                    ))
                    break
        except Exception:
            continue

    if not found_secrets:
        checks.append(HealthCheck(
            name="Hardcoded Secrets",
            category="Security",
            status="pass",
            message="No hardcoded secrets found in source",
        ))

    return checks


def run_security_checks(config: AgentConfig) -> list[HealthCheck]:
    """Run only security-related checks.

    Used by ``agent doctor --security``.

    Args:
        config: Agent configuration.

    Returns:
        List of security-focused HealthCheck results.
    """
    return check_security(config)
