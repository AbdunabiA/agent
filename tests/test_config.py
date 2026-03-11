"""Tests for the configuration system."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import (
    AgentConfig,
    AgentPersonaConfig,
    LoggingConfig,
    ModelsConfig,
    _resolve_env_vars,
    config_to_dict_masked,
    get_config,
    load_config,
    mask_secret,
    reset_config,
)


class TestDefaultConfig:
    """Test that defaults load correctly."""

    def test_default_config_loads(self) -> None:
        config = AgentConfig()
        assert config.agent.name == "Agent"
        assert config.models.default == "claude-sonnet-4-5-20250929"
        assert config.logging.level == "INFO"
        assert config.logging.format == "console"

    def test_default_model_fallback(self) -> None:
        config = AgentConfig()
        assert config.models.fallback == "gpt-4o"

    def test_default_gateway(self) -> None:
        config = AgentConfig()
        assert config.gateway.host == "127.0.0.1"
        assert config.gateway.port == 8765

    def test_default_tools_enabled(self) -> None:
        config = AgentConfig()
        assert config.tools.shell.enabled is True
        assert config.tools.browser.enabled is True
        assert config.tools.filesystem.enabled is True

    def test_default_channels_disabled(self) -> None:
        config = AgentConfig()
        assert config.channels.telegram.enabled is False
        assert config.channels.webchat.enabled is False


class TestYAMLParsing:
    """Test YAML file loading."""

    @patch("agent.config._auto_select_models")
    def test_loads_yaml_file(self, _mock_auto: object, tmp_path: Path) -> None:
        yaml_content = """
agent:
  name: "TestBot"
  persona: "A test assistant."
models:
  default: "gpt-4o-mini"
logging:
  level: "DEBUG"
"""
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.agent.name == "TestBot"
        assert config.models.default == "gpt-4o-mini"
        assert config.logging.level == "DEBUG"

    @patch("agent.config._auto_select_models")
    def test_partial_yaml_uses_defaults(self, _mock_auto: object, tmp_path: Path) -> None:
        yaml_content = """
agent:
  name: "PartialBot"
"""
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config.agent.name == "PartialBot"
        assert config.models.default == "claude-sonnet-4-5-20250929"  # default

    def test_missing_config_uses_defaults(self) -> None:
        reset_config()
        config = load_config("/nonexistent/path/agent.yaml")
        assert config.agent.name == "Agent"


class TestEnvVarInterpolation:
    """Test ${VAR} replacement in YAML values."""

    def test_resolve_simple_var(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "my_secret"}):
            result = _resolve_env_vars("${TEST_KEY}")
            assert result == "my_secret"

    def test_resolve_missing_var_empty_string(self) -> None:
        result = _resolve_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == ""

    def test_resolve_in_dict(self) -> None:
        with patch.dict(os.environ, {"MY_KEY": "value123"}):
            data = {"key": "${MY_KEY}", "other": "static"}
            result = _resolve_env_vars(data)
            assert result["key"] == "value123"
            assert result["other"] == "static"

    def test_resolve_in_list(self) -> None:
        with patch.dict(os.environ, {"LIST_VAR": "item1"}):
            data = ["${LIST_VAR}", "static"]
            result = _resolve_env_vars(data)
            assert result[0] == "item1"
            assert result[1] == "static"

    def test_resolve_nested(self) -> None:
        with patch.dict(os.environ, {"NESTED": "deep_value"}):
            data = {"outer": {"inner": "${NESTED}"}}
            result = _resolve_env_vars(data)
            assert result["outer"]["inner"] == "deep_value"

    def test_non_string_passthrough(self) -> None:
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(True) is True
        assert _resolve_env_vars(None) is None

    def test_yaml_with_env_vars(self, tmp_path: Path) -> None:
        yaml_content = """
models:
  providers:
    anthropic:
      api_key: "${TEST_ANTHROPIC_KEY}"
"""
        config_file = tmp_path / "agent.yaml"
        config_file.write_text(yaml_content)

        with patch.dict(os.environ, {"TEST_ANTHROPIC_KEY": "sk-ant-test123"}):
            config = load_config(str(config_file))
            anthropic = config.models.providers.get("anthropic")
            assert anthropic is not None
            assert anthropic.api_key == "sk-ant-test123"


class TestConfigSingleton:
    """Test the get_config() singleton behavior."""

    def test_returns_same_instance(self) -> None:
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_clears_instance(self) -> None:
        reset_config()
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2


class TestSecretMasking:
    """Test API key masking."""

    def test_mask_long_secret(self) -> None:
        result = mask_secret("sk-ant-very-long-secret-key-here")
        assert result == "sk-****...here"

    def test_mask_short_secret(self) -> None:
        result = mask_secret("short")
        assert result == "***"

    def test_mask_exact_boundary(self) -> None:
        result = mask_secret("12345678")
        assert result == "***"

    def test_mask_nine_chars(self) -> None:
        result = mask_secret("123456789")
        assert result == "123****...6789"

    def test_config_to_dict_masks_keys(self) -> None:
        config = AgentConfig(
            models=ModelsConfig(
                providers={
                    "anthropic": {"api_key": "sk-ant-very-long-key-12345"},
                }
            ),
            gateway={"auth_token": "super-secret-token-value"},
        )
        masked = config_to_dict_masked(config)

        # API key should be masked
        anthropic_key = masked["models"]["providers"]["anthropic"]["api_key"]
        assert "****" in anthropic_key

        # Auth token should be masked
        auth_token = masked["gateway"]["auth_token"]
        assert "****" in auth_token

    def test_config_to_dict_preserves_non_secrets(self) -> None:
        config = AgentConfig(
            agent=AgentPersonaConfig(name="TestBot"),
        )
        masked = config_to_dict_masked(config)
        assert masked["agent"]["name"] == "TestBot"


class TestInvalidConfig:
    """Test validation error handling."""

    def test_invalid_port_type(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(gateway={"port": "not_a_number"})

    def test_valid_custom_config(self) -> None:
        config = AgentConfig(
            agent=AgentPersonaConfig(name="Custom", max_iterations=20),
            logging=LoggingConfig(level="WARNING"),
        )
        assert config.agent.max_iterations == 20
        assert config.logging.level == "WARNING"
