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
            content_blocks.append(
                {
                    "type": "image",
                    "data": img.base64_data,
                    "mimeType": img.media_type,
                }
            )

        assert len(content_blocks) == 3
        assert content_blocks[0] == {"type": "text", "text": "Captured"}
        assert content_blocks[1] == {
            "type": "image",
            "data": "img1",
            "mimeType": "image/png",
        }
        assert content_blocks[2] == {
            "type": "image",
            "data": "img2",
            "mimeType": "image/jpeg",
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
        cfg = AgentConfig(models=ModelsConfig(claude_sdk=ClaudeSDKConfig(working_dir="/from/yaml")))
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


# -----------------------------------------------------------------------
# _safe_receive tests
# -----------------------------------------------------------------------


class TestSafeReceive:
    """Tests for ClaudeSDKService._safe_receive — unknown message handling."""

    async def _mock_client(self, messages, error_at=None, error_msg=None):
        """Create a mock client whose receive_messages() yields messages.

        If error_at is set, raises an exception at that index.
        """

        class _MockStream:
            def __init__(self):
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(messages):
                    raise StopAsyncIteration
                if error_at is not None and self._index == error_at:
                    self._index += 1
                    raise Exception(error_msg or "Unknown message type: rate_limit_event")
                msg = messages[self._index]
                self._index += 1
                return msg

        class _MockClient:
            def receive_messages(self):
                return _MockStream()

        return _MockClient()

    async def test_normal_messages_yielded(self):
        """Normal messages pass through."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        assistant = AssistantMessage(
            content=[TextBlock(text="Hello")],
            model="test",
        )
        result = ResultMessage(
            subtype="result",
            session_id="s1",
            num_turns=1,
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            total_cost_usd=0.0,
            usage={},
        )

        client = await self._mock_client([assistant, result])
        received = []
        async for msg in ClaudeSDKService._safe_receive(client):
            received.append(msg)

        assert len(received) == 2
        assert isinstance(received[0], AssistantMessage)
        assert isinstance(received[1], ResultMessage)

    async def test_unknown_message_type_skipped(self):
        """Unknown message types are skipped, iteration continues."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        assistant = AssistantMessage(
            content=[TextBlock(text="Hello")],
            model="test",
        )
        result = ResultMessage(
            subtype="result",
            session_id="s1",
            num_turns=1,
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            total_cost_usd=0.0,
            usage={},
        )

        # Error at index 1 (between assistant and result)
        client = await self._mock_client(
            [assistant, None, result],  # None is placeholder, error_at skips it
            error_at=1,
            error_msg="Unknown message type: rate_limit_event",
        )
        received = []
        async for msg in ClaudeSDKService._safe_receive(client):
            received.append(msg)

        # Should still get both real messages
        assert len(received) == 2
        assert isinstance(received[0], AssistantMessage)
        assert isinstance(received[1], ResultMessage)

    async def test_unknown_error_still_raised(self):
        """Non-'Unknown message type' errors still propagate."""
        from claude_code_sdk import AssistantMessage, TextBlock

        assistant = AssistantMessage(
            content=[TextBlock(text="Hi")],
            model="test",
        )

        client = await self._mock_client(
            [assistant, None],
            error_at=1,
            error_msg="Connection reset by peer",
        )

        with pytest.raises(Exception, match="Connection reset"):
            async for _ in ClaudeSDKService._safe_receive(client):
                pass

    async def test_stale_result_skipped(self):
        """ResultMessage before any AssistantMessage is skipped as stale."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        stale_result = ResultMessage(
            subtype="result",
            session_id="old",
            num_turns=0,
            duration_ms=0,
            duration_api_ms=0,
            is_error=False,
            total_cost_usd=0.0,
            usage={},
        )
        assistant = AssistantMessage(
            content=[TextBlock(text="Real response")],
            model="test",
        )
        real_result = ResultMessage(
            subtype="result",
            session_id="new",
            num_turns=1,
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            total_cost_usd=0.0,
            usage={},
        )

        client = await self._mock_client([stale_result, assistant, real_result])
        received = []
        async for msg in ClaudeSDKService._safe_receive(client):
            received.append(msg)

        # Stale result skipped, got assistant + real result
        assert len(received) == 2
        assert isinstance(received[0], AssistantMessage)
        assert isinstance(received[1], ResultMessage)
        assert received[1].session_id == "new"

    async def test_multiple_unknown_messages_skipped(self):
        """Multiple unknown messages in a row are all skipped."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        assistant = AssistantMessage(
            content=[TextBlock(text="Done")],
            model="test",
        )
        result = ResultMessage(
            subtype="result",
            session_id="s1",
            num_turns=1,
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            total_cost_usd=0.0,
            usage={},
        )

        # Two consecutive errors then real messages
        class _MultiErrorClient:
            def receive_messages(self):
                return self._stream()

            async def _stream(self):
                raise Exception("Unknown message type: rate_limit_event")

        # Build a custom client with multiple errors
        class _Stream:
            def __init__(self):
                self._items = [
                    ("error", "Unknown message type: rate_limit_event"),
                    ("error", "Unknown message type: system_event"),
                    ("msg", assistant),
                    ("msg", result),
                ]
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._i >= len(self._items):
                    raise StopAsyncIteration
                kind, val = self._items[self._i]
                self._i += 1
                if kind == "error":
                    raise Exception(val)
                return val

        class _Client:
            def receive_messages(self):
                return _Stream()

        received = []
        async for msg in ClaudeSDKService._safe_receive(_Client()):
            received.append(msg)

        assert len(received) == 2


# -----------------------------------------------------------------------
# Subagent filtering tests
# -----------------------------------------------------------------------


class TestSubagentFiltering:
    """Tests for subagent text filtering in _run_task_locked."""

    def test_assistant_message_subagent_flag(self):
        """AssistantMessage with parent_tool_use_id is a subagent message."""
        from claude_code_sdk import AssistantMessage, TextBlock

        # Main agent message
        main_msg = AssistantMessage(
            content=[TextBlock(text="Main response")],
            model="test",
            parent_tool_use_id=None,
        )
        assert getattr(main_msg, "parent_tool_use_id", None) is None

        # Subagent message
        sub_msg = AssistantMessage(
            content=[TextBlock(text="Subagent research")],
            model="test",
            parent_tool_use_id="toolu_abc123",
        )
        assert getattr(sub_msg, "parent_tool_use_id", None) == "toolu_abc123"

    def test_subagent_text_not_in_result(self):
        """Verify the logic: subagent text should not be accumulated."""
        from claude_code_sdk import AssistantMessage, TextBlock

        main_msg = AssistantMessage(
            content=[TextBlock(text="Final answer")],
            model="test",
            parent_tool_use_id=None,
        )
        sub_msg = AssistantMessage(
            content=[TextBlock(text="Internal research notes")],
            model="test",
            parent_tool_use_id="toolu_abc",
        )

        # Simulate accumulation logic from _run_task_locked
        result_text = ""
        for message in [sub_msg, main_msg]:
            is_subagent = getattr(message, "parent_tool_use_id", None) is not None
            for block in message.content:
                if isinstance(block, TextBlock) and not is_subagent:
                    result_text += block.text

        assert result_text == "Final answer"
        assert "Internal research" not in result_text

    def test_sdk_stream_event_carries_subagent_flag(self):
        """SDKStreamEvent should carry subagent flag in data."""
        event_main = SDKStreamEvent(
            type="text",
            content="Hello",
            data={"subagent": False},
        )
        event_sub = SDKStreamEvent(
            type="text",
            content="Research",
            data={"subagent": True},
        )

        assert not event_main.data.get("subagent")
        assert event_sub.data.get("subagent")

    def test_telegram_accumulation_skips_subagent(self):
        """Simulate telegram's accumulation logic with subagent events."""
        events = [
            SDKStreamEvent(type="text", content="Sub work", data={"subagent": True}),
            SDKStreamEvent(type="text", content="Main answer", data={"subagent": False}),
            SDKStreamEvent(type="text", content="More sub", data={"subagent": True}),
        ]

        accumulated = ""
        for event in events:
            if event.type == "text" and not (event.data and event.data.get("subagent")):
                accumulated += event.content

        assert accumulated == "Main answer"
        assert "Sub work" not in accumulated
        assert "More sub" not in accumulated
