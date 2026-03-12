"""Tests for the Claude Agent SDK integration."""

from __future__ import annotations

import pytest

from agent.llm.claude_sdk import (
    ClaudeSDKService,
    SDKStreamEvent,
    SDKTaskResult,
    SDKTaskStatus,
    _format_tool_details,
    sdk_available,
)


class TestSDKAvailability:
    """Test SDK availability detection."""

    def test_sdk_available_returns_bool(self):
        result = sdk_available()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_check_available_returns_tuple(self):
        service = ClaudeSDKService()
        ok, msg = await service.check_available()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)


class TestSDKTaskStatus:
    """Test SDKTaskStatus enum."""

    def test_all_statuses_exist(self):
        assert SDKTaskStatus.IDLE == "idle"
        assert SDKTaskStatus.RUNNING == "running"
        assert SDKTaskStatus.WAITING_PERMISSION == "waiting_permission"
        assert SDKTaskStatus.WAITING_ANSWER == "waiting_answer"
        assert SDKTaskStatus.COMPLETED == "completed"
        assert SDKTaskStatus.FAILED == "failed"
        assert SDKTaskStatus.CANCELLED == "cancelled"


class TestSDKStreamEvent:
    """Test SDKStreamEvent dataclass."""

    def test_default_values(self):
        event = SDKStreamEvent(type="text")
        assert event.type == "text"
        assert event.content == ""
        assert event.data == {}

    def test_with_data(self):
        event = SDKStreamEvent(
            type="tool_use",
            content="Using Bash",
            data={"tool": "Bash", "input": {"command": "ls"}},
        )
        assert event.data["tool"] == "Bash"


class TestSDKTaskResult:
    """Test SDKTaskResult dataclass."""

    def test_default_values(self):
        result = SDKTaskResult(success=True, output="done")
        assert result.success
        assert result.output == "done"
        assert result.session_id is None
        assert result.cost_usd == 0.0
        assert result.error is None

    def test_failure_result(self):
        result = SDKTaskResult(success=False, output="", error="timeout")
        assert not result.success
        assert result.error == "timeout"


class TestMultimodalMCPHandler:
    """Test MCP handler detection of MultimodalToolOutput."""

    def test_multimodal_output_detected(self):
        from agent.tools.executor import ImageContent, MultimodalToolOutput

        result = MultimodalToolOutput(
            text="Screenshot: 800x600",
            images=[ImageContent(base64_data="abc123", media_type="image/png")],
        )
        assert isinstance(result, MultimodalToolOutput)
        assert result.text == "Screenshot: 800x600"
        assert len(result.images) == 1
        assert result.images[0].base64_data == "abc123"
        assert result.images[0].media_type == "image/png"

    def test_multimodal_output_content_blocks(self):
        """MultimodalToolOutput should produce the right MCP content blocks."""
        from agent.tools.executor import ImageContent, MultimodalToolOutput

        result = MultimodalToolOutput(
            text="Captured",
            images=[
                ImageContent(base64_data="img1", media_type="image/png"),
                ImageContent(base64_data="img2", media_type="image/jpeg"),
            ],
        )
        # Simulate what the MCP handler does
        content_blocks = [{"type": "text", "text": result.text}]
        for img in result.images:
            content_blocks.append({
                "type": "image",
                "data": img.base64_data,
                "mimeType": img.media_type,
            })

        assert len(content_blocks) == 3
        assert content_blocks[0] == {"type": "text", "text": "Captured"}
        assert content_blocks[1] == {
            "type": "image", "data": "img1", "mimeType": "image/png",
        }
        assert content_blocks[2] == {
            "type": "image", "data": "img2", "mimeType": "image/jpeg",
        }


class TestFormatToolDetails:
    """Test _format_tool_details helper."""

    def test_bash_command(self):
        details = _format_tool_details("Bash", {"command": "ls -la", "description": "List files"})
        assert "ls -la" in details
        assert "List files" in details

    def test_bash_no_description(self):
        details = _format_tool_details("Bash", {"command": "pwd"})
        assert "$ pwd" in details

    def test_write_file(self):
        details = _format_tool_details("Write", {"file_path": "/tmp/test.py"})
        assert "/tmp/test.py" in details

    def test_edit_file(self):
        details = _format_tool_details("Edit", {"file_path": "/tmp/test.py"})
        assert "/tmp/test.py" in details

    def test_unknown_tool(self):
        details = _format_tool_details("Unknown", {"key": "value"})
        assert "value" in details


class TestClaudeSDKService:
    """Test ClaudeSDKService initialization and status tracking."""

    def test_init(self):
        service = ClaudeSDKService(
            working_dir="/tmp",
            max_turns=10,
            model="claude-sonnet-4-6",
        )
        assert service.working_dir == "/tmp"
        assert service.max_turns == 10
        assert service.model == "claude-sonnet-4-6"

    def test_get_status_default_idle(self):
        service = ClaudeSDKService()
        assert service.get_status("nonexistent") == SDKTaskStatus.IDLE

    def test_approve_permission(self):
        service = ClaudeSDKService()
        # Should not raise even with no pending request
        service.approve_permission("test", approved=True)

    def test_answer_question(self):
        service = ClaudeSDKService()
        # Should not raise even with no pending question
        service.answer_question("test", answer="yes")


class TestConfigIntegration:
    """Test config integration with claude_sdk settings."""

    def test_models_config_has_backend(self):
        from agent.config import ModelsConfig
        cfg = ModelsConfig()
        assert cfg.backend == "litellm"

    def test_models_config_sdk_backend(self):
        from agent.config import ModelsConfig
        cfg = ModelsConfig(backend="claude-sdk")
        assert cfg.backend == "claude-sdk"

    def test_models_config_has_claude_sdk(self):
        from agent.config import ClaudeSDKConfig, ModelsConfig
        cfg = ModelsConfig()
        assert isinstance(cfg.claude_sdk, ClaudeSDKConfig)
        assert cfg.claude_sdk.max_turns == 50
        assert cfg.claude_sdk.working_dir == "."

    def test_claude_sdk_config_custom(self):
        from agent.config import ClaudeSDKConfig
        cfg = ClaudeSDKConfig(
            claude_auth_dir="/custom/.claude",
            working_dir="/home/user/project",
            max_turns=100,
            permission_mode="acceptEdits",
            model="claude-opus-4-6",
        )
        assert cfg.claude_auth_dir == "/custom/.claude"
        assert cfg.working_dir == "/home/user/project"
        assert cfg.max_turns == 100
        assert cfg.permission_mode == "acceptEdits"
        assert cfg.model == "claude-opus-4-6"

    def test_claude_sdk_config_default_auth_dir(self):
        from agent.config import ClaudeSDKConfig
        cfg = ClaudeSDKConfig()
        assert cfg.claude_auth_dir == "~/.claude"


class TestEnvVarLoading:
    """Test environment variable loading for SDK settings."""

    def test_env_vars_apply_to_sdk_config(self, monkeypatch):
        from agent.config import AgentConfig, _apply_env_api_keys
        monkeypatch.setenv("CLAUDE_WORKING_DIR", "/from/env")
        monkeypatch.setenv("CLAUDE_AUTH_DIR", "/custom/auth")
        monkeypatch.setenv("CLAUDE_SDK_MAX_TURNS", "100")
        monkeypatch.setenv("AGENT_LLM_BACKEND", "claude-sdk")

        cfg = AgentConfig()
        _apply_env_api_keys(cfg)
        assert cfg.models.claude_sdk.working_dir == "/from/env"
        assert cfg.models.claude_sdk.claude_auth_dir == "/custom/auth"
        assert cfg.models.claude_sdk.max_turns == 100
        assert cfg.models.backend == "claude-sdk"

    def test_env_vars_dont_override_yaml(self, monkeypatch):
        from agent.config import AgentConfig, ClaudeSDKConfig, ModelsConfig, _apply_env_api_keys
        monkeypatch.setenv("CLAUDE_WORKING_DIR", "/from/env")

        # Simulate YAML-configured value (non-default)
        cfg = AgentConfig(
            models=ModelsConfig(
                claude_sdk=ClaudeSDKConfig(working_dir="/from/yaml")
            )
        )
        _apply_env_api_keys(cfg)
        # Env var should NOT override existing non-empty value
        assert cfg.models.claude_sdk.working_dir == "/from/yaml"

    def test_invalid_max_turns_ignored(self, monkeypatch):
        from agent.config import AgentConfig, _apply_env_api_keys
        monkeypatch.setenv("CLAUDE_SDK_MAX_TURNS", "not-a-number")

        cfg = AgentConfig()
        _apply_env_api_keys(cfg)
        assert cfg.models.claude_sdk.max_turns == 50  # stays default


class TestEditableConfigMeta:
    """Test editable config metadata generation for dashboard."""

    def test_claude_sdk_has_fields(self):
        from agent.config import AgentConfig, get_editable_config_meta
        cfg = AgentConfig()
        meta = get_editable_config_meta(cfg)

        models_section = meta["models"]
        sdk_meta = models_section["claude_sdk"]

        assert sdk_meta["type"] == "object"
        assert sdk_meta["editable"] is True
        assert "fields" in sdk_meta

        fields = sdk_meta["fields"]
        assert "claude_auth_dir" in fields
        assert "working_dir" in fields
        assert "max_turns" in fields
        assert "permission_mode" in fields
        assert "model" in fields

    def test_claude_sdk_subfields_types(self):
        from agent.config import AgentConfig, get_editable_config_meta
        cfg = AgentConfig()
        meta = get_editable_config_meta(cfg)

        fields = meta["models"]["claude_sdk"]["fields"]
        assert fields["working_dir"]["type"] == "string"
        assert fields["working_dir"]["editable"] is True
        assert fields["max_turns"]["type"] == "number"
        assert fields["claude_auth_dir"]["type"] == "string"

    def test_backend_has_options(self):
        from agent.config import AgentConfig, get_editable_config_meta
        cfg = AgentConfig()
        meta = get_editable_config_meta(cfg)

        backend_meta = meta["models"]["backend"]
        assert backend_meta["options"] == ["litellm", "claude-sdk"]
