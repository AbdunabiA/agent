"""Configuration system for Agent.

Loads configuration from YAML files with environment variable interpolation,
validates with Pydantic, and provides a singleton accessor.
"""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from agent.voice.config import VoiceConfig

logger = structlog.get_logger(__name__)

# Singleton instance
_config_instance: AgentConfig | None = None
_config_path: str | None = None


def get_agent_home() -> Path:
    """Return the agent home directory.

    Priority:
    1. AGENT_HOME environment variable
    2. ~/.config/agent (default)

    The directory is created if it does not exist.
    """
    env_home = os.environ.get("AGENT_HOME")
    home = Path(env_home) if env_home else Path.home() / ".config" / "agent"
    home.mkdir(parents=True, exist_ok=True)
    return home


class ModelProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    api_key: str | None = None
    base_url: str | None = None


class ClaudeSDKConfig(BaseModel):
    """Configuration for Claude Agent SDK backend."""

    claude_auth_dir: str = "~/.claude"
    working_dir: str = "."
    max_turns: int = 50
    permission_mode: str | None = None
    model: str | None = None


class ModelsConfig(BaseModel):
    """LLM model selection and provider configuration."""

    backend: str = "litellm"  # "litellm" or "claude-sdk"
    default: str = "claude-sonnet-4-5-20250929"
    fallback: str | None = "gpt-4o"
    providers: dict[str, ModelProviderConfig] = {}
    claude_sdk: ClaudeSDKConfig = ClaudeSDKConfig()


class TelegramConfig(BaseModel):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str | None = None
    allowed_users: list[int] = []


class WebChatConfig(BaseModel):
    """Web chat channel configuration."""

    enabled: bool = False
    port: int = 8080


class ChannelsConfig(BaseModel):
    """Messaging channels configuration."""

    telegram: TelegramConfig = TelegramConfig()
    webchat: WebChatConfig = WebChatConfig()


class ToolsShellConfig(BaseModel):
    """Shell tool configuration."""

    enabled: bool = True
    sandbox: bool = False
    allowed_commands: list[str] = ["*"]


class ToolsBrowserConfig(BaseModel):
    """Browser tool configuration."""

    enabled: bool = True
    headless: bool = True


class ToolsFilesystemConfig(BaseModel):
    """Filesystem tool configuration."""

    enabled: bool = True
    root: str = "/"
    write_root: str = "~"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    deny_paths: list[str] = [
        "/proc/kcore",
        "/dev/sda",
        "/dev/nvme",
        "/boot/efi",
    ]


class ToolsConfig(BaseModel):
    """Tool execution configuration."""

    shell: ToolsShellConfig = ToolsShellConfig()
    browser: ToolsBrowserConfig = ToolsBrowserConfig()
    filesystem: ToolsFilesystemConfig = ToolsFilesystemConfig()


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    db_path: str = "./data/agent.db"
    markdown_dir: str = "./data/memory/"
    auto_extract: bool = True
    max_facts_in_context: int = 15
    max_vectors_in_context: int = 5
    summarize_threshold: int = 20
    soul_path: str | None = None


class SkillsConfig(BaseModel):
    """Skills/plugin configuration."""

    directory: str = "skills"
    enabled: list[str] = []  # Empty = all discovered
    disabled: list[str] = []
    auto_discover: bool = True


class GatewayConfig(BaseModel):
    """API gateway configuration."""

    host: str = "127.0.0.1"
    port: int = 8765
    auth_token: str | None = None
    cors_origins: list[str] = ["http://localhost:5173"]


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "console"  # "console" or "json"


class AgentPersonaConfig(BaseModel):
    """Agent identity and behavior configuration."""

    name: str = "Agent"
    persona: str = "You are a helpful autonomous AI assistant."
    max_iterations: int = 10
    heartbeat_interval: str = "30m"


class RoutingRuleConfig(BaseModel):
    """A single routing rule from agent.yaml."""

    channel: str = "*"
    workspace: str = "default"
    user_id: str | None = None
    pattern: str | None = None


class RoutingConfig(BaseModel):
    """Routing section inside workspaces config."""

    default: str = "default"
    rules: list[RoutingRuleConfig] = []


class WorkspacesSection(BaseModel):
    """Top-level workspaces config in agent.yaml."""

    directory: str = "workspaces"
    default: str = "default"
    auto_create_default: bool = True
    routing: RoutingConfig = RoutingConfig()


class DesktopConfig(BaseModel):
    """Desktop control configuration."""

    enabled: bool = True
    screenshot_scale: float = 0.75  # Scale screenshots to save LLM tokens
    mouse_move_duration: float = 0.3  # Mouse animation speed
    typing_interval: float = 0.02  # Delay between keystrokes
    vision_model: str = ""  # Override model for vision (empty = use default)
    failsafe: bool = True  # pyautogui failsafe (corner escape)
    max_screenshot_size_kb: int = 500  # Max screenshot size to send to LLM


class ShortcutConfig(BaseModel):
    """A single prompt shortcut for /run command."""

    alias: str
    template: str
    description: str = ""


class PromptsConfig(BaseModel):
    """Prompt shortcuts configuration."""

    shortcuts: list[ShortcutConfig] = []


class AgentConfig(BaseModel):
    """Root configuration model."""

    agent: AgentPersonaConfig = AgentPersonaConfig()
    models: ModelsConfig = ModelsConfig()
    channels: ChannelsConfig = ChannelsConfig()
    tools: ToolsConfig = ToolsConfig()
    memory: MemoryConfig = MemoryConfig()
    skills: SkillsConfig = SkillsConfig()
    gateway: GatewayConfig = GatewayConfig()
    logging: LoggingConfig = LoggingConfig()
    workspaces: WorkspacesSection = WorkspacesSection()
    voice: VoiceConfig = VoiceConfig()
    desktop: DesktopConfig = DesktopConfig()
    prompts: PromptsConfig = PromptsConfig()


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in a data structure."""
    if isinstance(data, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def _replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        return pattern.sub(_replacer, data)
    elif isinstance(data, dict):
        return {key: _resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def _find_config_path(override: str | None = None) -> Path | None:
    """Find the configuration file path.

    Priority:
    1. Explicit override (CLI --config flag)
    2. AGENT_CONFIG environment variable
    3. ./agent.yaml (current directory)
    4. $AGENT_HOME/agent.yaml (defaults to ~/.config/agent/agent.yaml)
    """
    if override:
        path = Path(override)
        if path.exists():
            return path
        logger.warning("config_override_not_found", path=str(path))
        return None

    env_path = os.environ.get("AGENT_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        logger.warning("env_config_not_found", path=str(path))

    local_path = Path("agent.yaml")
    if local_path.exists():
        return local_path

    home_path = get_agent_home() / "agent.yaml"
    if home_path.exists():
        return home_path

    return None


def load_config(config_path: str | None = None) -> AgentConfig:
    """Load and validate configuration from YAML + .env.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Validated AgentConfig instance.
    """
    # Load .env files — local first, then agent home for fallback values.
    # load_dotenv does NOT override already-set vars, so first call wins.
    # This matches config priority: ./agent.yaml > $AGENT_HOME/agent.yaml.
    load_dotenv()
    agent_home = get_agent_home()
    load_dotenv(agent_home / ".env")

    # Find config file
    path = _find_config_path(config_path)

    if path is not None:
        logger.info("loading_config", path=str(path))
        with open(path) as f:
            raw_data = yaml.safe_load(f) or {}

        # Resolve environment variables
        resolved_data = _resolve_env_vars(raw_data)

        # Validate with Pydantic
        config = AgentConfig.model_validate(resolved_data)
    else:
        logger.info("using_default_config")
        config = AgentConfig()

    # Apply API keys from environment if not set in config
    _apply_env_api_keys(config)

    # Auto-select models based on available API keys
    _auto_select_models(config)

    return config


def _apply_env_api_keys(config: AgentConfig) -> None:
    """Apply API keys and SDK settings from environment variables."""
    env_key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    for provider_name, env_var in env_key_map.items():
        env_value = os.environ.get(env_var)
        if env_value:
            if provider_name not in config.models.providers:
                config.models.providers[provider_name] = ModelProviderConfig()
            provider = config.models.providers[provider_name]
            if not provider.api_key:
                provider.api_key = env_value

    # Claude SDK settings from environment
    sdk = config.models.claude_sdk
    sdk_defaults = ClaudeSDKConfig()
    sdk_env_map: dict[str, str] = {
        "CLAUDE_AUTH_DIR": "claude_auth_dir",
        "CLAUDE_WORKING_DIR": "working_dir",
        "CLAUDE_SDK_MODEL": "model",
        "CLAUDE_SDK_PERMISSION_MODE": "permission_mode",
    }
    for env_var, field_name in sdk_env_map.items():
        env_value = os.environ.get(env_var)
        if env_value:
            current = getattr(sdk, field_name, None)
            default = getattr(sdk_defaults, field_name, None)
            if current == default:
                setattr(sdk, field_name, env_value)

    max_turns_env = os.environ.get("CLAUDE_SDK_MAX_TURNS")
    if max_turns_env and sdk.max_turns == 50:  # Only override default
        with contextlib.suppress(ValueError):
            sdk.max_turns = int(max_turns_env)

    # Backend selection from env
    backend_env = os.environ.get("AGENT_LLM_BACKEND")
    if backend_env and config.models.backend == "litellm":
        config.models.backend = backend_env

    # Voice settings from environment
    voice_env_map: dict[str, tuple[str, str]] = {
        "VOICE_STT_PROVIDER": ("stt", "provider"),
        "VOICE_TTS_PROVIDER": ("tts", "provider"),
        "VOICE_TTS_VOICE": ("tts", "edge_voice"),
    }
    for env_var, (sub_obj, field_name) in voice_env_map.items():
        env_value = os.environ.get(env_var)
        if env_value:
            sub = getattr(config.voice, sub_obj)
            default_sub = type(sub)()
            if getattr(sub, field_name) == getattr(default_sub, field_name):
                setattr(sub, field_name, env_value)

    voice_auto_env = os.environ.get("VOICE_AUTO_REPLY")
    if voice_auto_env is not None and config.voice.auto_voice_reply is True:
        config.voice.auto_voice_reply = voice_auto_env.lower() in ("1", "true", "yes")


def _provider_for_model(model: str) -> str | None:
    """Detect which provider a model string belongs to."""
    lower = model.lower()
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if lower.startswith("gemini/") or lower.startswith("gemini-"):
        return "gemini"
    if lower.startswith("ollama/"):
        return "ollama"
    return None


# Default model per provider (used when auto-selecting)
_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
    "gemini": "gemini/gemini-2.5-flash",
}

# Known models per provider (for listing available options)
PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o3",
        "o3-mini",
        "o4-mini",
    ],
    "gemini": [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
    ],
    "ollama": [
        "ollama/llama3",
        "ollama/mistral",
        "ollama/codellama",
        "ollama/deepseek-r1",
    ],
}


def get_available_models(config: AgentConfig) -> dict[str, list[str]]:
    """Return models grouped by provider, only for providers that have API keys."""
    available: dict[str, list[str]] = {}
    for name, prov in config.models.providers.items():
        if prov.api_key and name in PROVIDER_MODELS:
            available[name] = PROVIDER_MODELS[name]
    # Ollama doesn't need a key
    if "ollama" in config.models.providers:
        available["ollama"] = PROVIDER_MODELS["ollama"]
    return available


def _auto_select_models(config: AgentConfig) -> None:
    """Auto-select default/fallback models based on available API keys.

    If the configured default model's provider has no key, switch to a
    provider that does. Same for fallback.
    """
    available = {
        name
        for name, prov in config.models.providers.items()
        if prov.api_key
    }

    if not available:
        return

    default_provider = _provider_for_model(config.models.default)

    # If the default model's provider has no key, pick one that does
    if default_provider and default_provider not in available:
        for prov_name in available:
            if prov_name in _PROVIDER_DEFAULT_MODELS:
                new_model = _PROVIDER_DEFAULT_MODELS[prov_name]
                logger.info(
                    "auto_selected_model",
                    reason=f"no {default_provider} API key",
                    old_model=config.models.default,
                    new_model=new_model,
                )
                config.models.default = new_model
                break

    # Auto-select fallback from a different available provider
    if config.models.fallback:
        fallback_provider = _provider_for_model(config.models.fallback)
        default_provider = _provider_for_model(config.models.default)

        if fallback_provider and fallback_provider not in available:
            # Pick a different available provider for fallback
            found = False
            for prov_name in available:
                if prov_name != default_provider and prov_name in _PROVIDER_DEFAULT_MODELS:
                    new_fallback = _PROVIDER_DEFAULT_MODELS[prov_name]
                    logger.info(
                        "auto_selected_fallback",
                        old_fallback=config.models.fallback,
                        new_fallback=new_fallback,
                    )
                    config.models.fallback = new_fallback
                    found = True
                    break
            if not found:
                # No other provider available, disable fallback
                config.models.fallback = None


def get_config(config_path: str | None = None) -> AgentConfig:
    """Get the singleton configuration instance.

    Creates the config on first call, returns cached instance after that.

    Args:
        config_path: Optional path to config file (only used on first call).

    Returns:
        The global AgentConfig instance.
    """
    global _config_instance, _config_path

    if _config_instance is None or config_path != _config_path:
        _config_instance = load_config(config_path)
        _config_path = config_path

    return _config_instance


def reset_config() -> None:
    """Reset the singleton config. Useful for testing."""
    global _config_instance, _config_path
    _config_instance = None
    _config_path = None


def get_config_path() -> Path | None:
    """Return the path of the currently loaded config file, or None if using defaults."""
    return Path(_config_path) if _config_path else None


# Sections that can be edited via the dashboard API.
EDITABLE_SECTIONS: dict[str, type[BaseModel]] = {
    "agent": AgentPersonaConfig,
    "models": ModelsConfig,
    "channels": ChannelsConfig,
    "tools": ToolsConfig,
    "memory": MemoryConfig,
    "skills": SkillsConfig,
    "gateway": GatewayConfig,
    "logging": LoggingConfig,
    "desktop": DesktopConfig,
    "voice": VoiceConfig,
    "workspaces": WorkspacesSection,
    "prompts": PromptsConfig,
}

# Fields that contain secrets and must NOT be overwritten via the API.
SECRET_FIELD_INDICATORS = {"key", "token", "secret", "password", "auth"}


def _is_secret_field(field_name: str) -> bool:
    """Check whether a field name looks like a secret."""
    lower = field_name.lower()
    return any(ind in lower for ind in SECRET_FIELD_INDICATORS)


def _strip_secret_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove secret fields from a dict so they can't be overwritten."""
    cleaned: dict[str, Any] = {}
    for k, v in data.items():
        if _is_secret_field(k):
            continue
        if isinstance(v, dict):
            cleaned[k] = _strip_secret_fields(v)
        else:
            cleaned[k] = v
    return cleaned


def update_config_section(section: str, data: dict[str, Any]) -> AgentConfig:
    """Update a single config section, validate, persist, and reload the singleton.

    Args:
        section: Top-level config key (e.g. "models", "logging").
        data: New values for that section. Secret fields are stripped.

    Returns:
        The updated AgentConfig singleton.

    Raises:
        ValueError: If the section name is unknown.
        ValidationError: If the new data fails Pydantic validation.
    """
    if section not in EDITABLE_SECTIONS:
        raise ValueError(
            f"Unknown config section: {section}. "
            f"Valid: {', '.join(EDITABLE_SECTIONS)}"
        )

    # Strip secret fields so they can't be set from the dashboard
    safe_data = _strip_secret_fields(data)

    # Validate against the section model
    model_cls = EDITABLE_SECTIONS[section]
    # Merge with current values so partial updates work
    current_config = get_config()
    current_section = getattr(current_config, section)
    merged = {**current_section.model_dump(), **safe_data}
    validated = model_cls.model_validate(merged)

    # Apply to the live singleton
    setattr(current_config, section, validated)

    # Persist to disk
    _save_config_to_disk(current_config)

    logger.info("config_section_updated", section=section)
    return current_config


def _save_config_to_disk(config: AgentConfig) -> None:
    """Persist config to the YAML file, preserving env-var references for secrets.

    If no config file was loaded (pure defaults + env), creates agent.yaml in AGENT_HOME.
    """
    path = _find_config_path(_config_path)
    if path is None:
        path = get_agent_home() / "agent.yaml"

    # Load existing raw YAML to preserve ${ENV_VAR} references
    existing_raw: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            existing_raw = yaml.safe_load(f) or {}

    # Build new data from the config model
    new_data = config.model_dump()

    # Merge: for every secret field, keep the raw value from existing YAML
    merged = _merge_preserving_secrets(existing_raw, new_data)

    with open(path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    logger.info("config_saved", path=str(path))


def _merge_preserving_secrets(
    existing: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Merge new config into existing, keeping raw secret values (e.g. ${ENV_VAR})."""
    result: dict[str, Any] = {}
    for key in set(existing) | set(new):
        old_val = existing.get(key)
        new_val = new.get(key)

        if isinstance(old_val, dict) and isinstance(new_val, dict):
            result[key] = _merge_preserving_secrets(old_val, new_val)
        elif _is_secret_field(key) and old_val is not None:
            # Preserve the raw YAML value (which may be "${SOME_KEY}")
            result[key] = old_val
        elif new_val is not None:
            result[key] = new_val
        elif old_val is not None:
            result[key] = old_val
    return result


def _build_field_meta(
    field_name: str,
    field_value: Any,
    section_name: str,
    section_obj: BaseModel,
) -> dict[str, Any]:
    """Build metadata for a single config field.

    For nested Pydantic models, recursively exposes sub-fields via a `fields` key
    so the dashboard can render editable sub-fields.
    """
    is_secret = _is_secret_field(field_name)
    field_type = _python_type_name(field_value)

    masked = mask_secret(str(field_value)) if is_secret and field_value else None
    field_info: dict[str, Any] = {
        "value": masked if masked is not None else field_value,
        "type": field_type,
        "editable": not is_secret,
    }

    # Add options for known enum-like fields
    if field_name == "level" and section_name == "logging":
        field_info["options"] = ["DEBUG", "INFO", "WARNING", "ERROR"]
    elif field_name == "format" and section_name == "logging":
        field_info["options"] = ["console", "json", "clean"]
    elif field_name == "default" and section_name == "models":
        field_info["options"] = _all_known_models()
    elif field_name == "backend" and section_name == "models":
        field_info["options"] = ["litellm", "claude-sdk"]
    elif field_name == "output_format" and section_name == "voice":
        field_info["options"] = ["opus", "mp3", "wav"]

    # Expand nested Pydantic models into sub-fields for the dashboard
    if isinstance(field_value, dict) and field_type == "object":
        nested_obj = getattr(section_obj, field_name, None)
        if isinstance(nested_obj, BaseModel):
            sub_fields: dict[str, Any] = {}
            for sub_name, sub_value in field_value.items():
                sub_is_secret = _is_secret_field(sub_name)
                sub_masked = (
                    mask_secret(str(sub_value))
                    if sub_is_secret and sub_value
                    else None
                )
                sub_meta: dict[str, Any] = {
                    "value": sub_masked if sub_masked is not None else sub_value,
                    "type": _python_type_name(sub_value),
                    "editable": not sub_is_secret,
                }
                if sub_name == "permission_mode":
                    sub_meta["options"] = ["", "acceptEdits", "bypassPermissions"]
                elif sub_name == "provider" and field_name == "stt":
                    sub_meta["options"] = [
                        "llm_native", "whisper_api", "whisper_local", "deepgram",
                    ]
                elif sub_name == "provider" and field_name == "tts":
                    sub_meta["options"] = ["edge_tts", "openai"]
                elif sub_name == "output_format" and field_name == "tts":
                    sub_meta["options"] = ["opus", "mp3", "wav"]
                sub_fields[sub_name] = sub_meta
            field_info["fields"] = sub_fields
            field_info["editable"] = True

    return field_info


def get_editable_config_meta(config: AgentConfig) -> dict[str, Any]:
    """Return config metadata describing which fields are editable.

    Each section contains fields with: value, type, editable, options (if enum-like).
    Secret fields are marked read-only with masked values.
    Nested Pydantic models include a ``fields`` dict with sub-field metadata.
    """
    meta: dict[str, Any] = {}

    for section_name in EDITABLE_SECTIONS:
        section_obj = getattr(config, section_name)
        section_data = section_obj.model_dump()
        fields_meta: dict[str, Any] = {}

        for field_name, field_value in section_data.items():
            fields_meta[field_name] = _build_field_meta(
                field_name, field_value, section_name, section_obj
            )

        meta[section_name] = fields_meta

    return meta


def _python_type_name(value: Any) -> str:
    """Map a Python value to a simple type name for the frontend."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _all_known_models() -> list[str]:
    """Return flat list of all known model IDs."""
    models: list[str] = []
    for provider_models in PROVIDER_MODELS.values():
        models.extend(provider_models)
    return models


def mask_secret(value: str) -> str:
    """Mask a secret value, showing first 3 and last 4 characters.

    Args:
        value: The secret string to mask.

    Returns:
        Masked string like 'sk-****...****' or '***' for short values.
    """
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}****...{value[-4:]}"


def config_to_dict_masked(config: AgentConfig) -> dict[str, Any]:
    """Convert config to dict with secrets masked.

    Any value that looks like an API key (contains 'key', 'token', 'secret')
    will be masked.
    """
    data = config.model_dump()
    return _mask_secrets_recursive(data)


def _mask_secrets_recursive(data: Any, key: str = "") -> Any:
    """Recursively mask secret values in a dictionary."""
    secret_indicators = {"key", "token", "secret", "password", "auth"}

    if isinstance(data, dict):
        return {k: _mask_secrets_recursive(v, k) for k, v in data.items()}
    elif isinstance(data, list):
        return [_mask_secrets_recursive(item, key) for item in data]
    elif isinstance(data, str) and data:
        key_lower = key.lower()
        if any(indicator in key_lower for indicator in secret_indicators):
            return mask_secret(data)
    return data
