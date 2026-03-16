"""Tests for runtime context injection (channel awareness, capabilities)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.config import AgentPersonaConfig
from agent.core.events import EventBus
from agent.core.session import Session
from agent.llm.prompts import build_runtime_context

# =====================================================================
# build_runtime_context unit tests
# =====================================================================


class TestBuildRuntimeContext:
    """Tests for the build_runtime_context function."""

    def test_telegram_channel(self) -> None:
        """Telegram channel should be mentioned with formatting guidance."""
        ctx = build_runtime_context(channel="telegram")
        assert "Telegram" in ctx
        assert "Markdown" in ctx
        assert "voice" in ctx.lower()

    def test_webchat_channel(self) -> None:
        """WebChat channel should mention browser and WebSocket."""
        ctx = build_runtime_context(channel="webchat")
        assert "Web Chat" in ctx
        assert "browser" in ctx.lower()

    def test_cli_default(self) -> None:
        """Default channel should be CLI."""
        ctx = build_runtime_context()
        assert "terminal" in ctx.lower() or "CLI" in ctx

    def test_unknown_channel(self) -> None:
        """Unknown channel should get a generic message."""
        ctx = build_runtime_context(channel="discord")
        assert "discord" in ctx.lower()

    def test_capabilities_listed(self) -> None:
        """Enabled capabilities should appear in the context."""
        ctx = build_runtime_context(
            channel="cli",
            enabled_tools=["shell_exec", "file_read", "web_search"],
            has_memory=True,
            has_browser=True,
        )
        assert "Shell command" in ctx
        assert "File system" in ctx
        assert "Web search" in ctx
        assert "Persistent memory" in ctx
        assert "Browser automation" in ctx

    def test_no_capabilities(self) -> None:
        """No capabilities enabled should produce minimal output."""
        ctx = build_runtime_context(channel="cli")
        assert "Current Session Context" in ctx
        # No capabilities section when nothing is enabled
        assert "active capabilities" not in ctx.lower()

    def test_model_name_included(self) -> None:
        """Model name should appear when provided."""
        ctx = build_runtime_context(model_name="claude-sonnet-4-5-20250929")
        assert "claude-sonnet-4-5-20250929" in ctx

    def test_heartbeat_capability(self) -> None:
        """Heartbeat should appear when enabled."""
        ctx = build_runtime_context(has_heartbeat=True)
        assert "heartbeat" in ctx.lower()

    def test_voice_capability(self) -> None:
        """Voice pipeline should appear when enabled."""
        ctx = build_runtime_context(has_voice=True)
        assert "Voice pipeline" in ctx

    def test_skills_capability(self) -> None:
        """Skills should appear when enabled."""
        ctx = build_runtime_context(has_skills=True)
        assert "Skill" in ctx

    def test_orchestration_capability(self) -> None:
        """Orchestration should appear when enabled."""
        ctx = build_runtime_context(has_orchestration=True)
        assert "orchestration" in ctx.lower()

    def test_desktop_capability(self) -> None:
        """Desktop control should appear when enabled."""
        ctx = build_runtime_context(has_desktop=True)
        assert "Desktop control" in ctx

    def test_tool_categories_deduped(self) -> None:
        """Multiple file tools should produce a single filesystem entry."""
        ctx = build_runtime_context(
            enabled_tools=["file_read", "file_write", "file_delete", "list_directory"],
        )
        # Should have one filesystem entry, not four
        assert ctx.count("File system") == 1

    def test_send_file_capability(self) -> None:
        """send_file tool should produce a capability entry."""
        ctx = build_runtime_context(enabled_tools=["send_file"])
        assert "Send files" in ctx


# =====================================================================
# Agent loop integration — runtime context in system prompt
# =====================================================================


class TestAgentLoopRuntimeContext:
    """Verify that _build_messages injects runtime context."""

    @pytest.fixture
    def agent_loop(self) -> Any:
        from agent.core.agent_loop import AgentLoop
        from agent.llm.provider import LLMProvider

        config = AgentPersonaConfig(name="TestBot")
        event_bus = EventBus()
        llm = LLMProvider.__new__(LLMProvider)
        llm.config = MagicMock()
        llm.config.default = "test-model-v1"

        loop = AgentLoop(
            llm=llm,
            config=config,
            event_bus=event_bus,
        )
        return loop

    def test_system_prompt_includes_channel(self, agent_loop: Any) -> None:
        """System prompt should include channel from session metadata."""
        session = Session()
        session.metadata["channel"] = "telegram"

        messages = agent_loop._build_messages(session, plan=None)

        system_content = messages[0]["content"]
        assert "Telegram" in system_content
        assert "Current Session Context" in system_content

    def test_system_prompt_defaults_to_cli(self, agent_loop: Any) -> None:
        """When no channel in metadata, should default to CLI."""
        session = Session()

        messages = agent_loop._build_messages(session, plan=None)

        system_content = messages[0]["content"]
        assert "Current Session Context" in system_content
        assert "terminal" in system_content.lower() or "CLI" in system_content

    def test_system_prompt_includes_model(self, agent_loop: Any) -> None:
        """Model name should be included in runtime context."""
        session = Session()

        messages = agent_loop._build_messages(session, plan=None)

        system_content = messages[0]["content"]
        assert "test-model-v1" in system_content

    def test_webchat_channel_context(self, agent_loop: Any) -> None:
        """WebChat channel should get webchat-specific context."""
        session = Session()
        session.metadata["channel"] = "webchat"

        messages = agent_loop._build_messages(session, plan=None)

        system_content = messages[0]["content"]
        assert "Web Chat" in system_content


# =====================================================================
# SDK system prompt — channel injection
# =====================================================================


class TestSDKChannelContext:
    """Verify that the SDK system prompt includes channel context."""

    def test_sdk_build_system_prompt_with_channel(self) -> None:
        """SDK _build_system_prompt should include channel context."""
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk = ClaudeSDKService.__new__(ClaudeSDKService)
        sdk.soul_loader = None
        sdk.fact_store = None
        sdk.vector_store = None
        sdk.model = "test-model"

        prompt = sdk._build_system_prompt(channel="telegram")

        assert prompt is not None
        assert "Telegram" in prompt
        assert "Current Session Context" in prompt

    def test_sdk_build_system_prompt_defaults_to_cli(self) -> None:
        """SDK _build_system_prompt without channel should default to CLI."""
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk = ClaudeSDKService.__new__(ClaudeSDKService)
        sdk.soul_loader = None
        sdk.fact_store = None
        sdk.vector_store = None
        sdk.model = "test-model"

        prompt = sdk._build_system_prompt(channel=None)

        assert prompt is not None
        assert "Current Session Context" in prompt

    def test_sdk_build_system_prompt_webchat(self) -> None:
        """SDK _build_system_prompt with webchat channel."""
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk = ClaudeSDKService.__new__(ClaudeSDKService)
        sdk.soul_loader = None
        sdk.fact_store = None
        sdk.vector_store = None
        sdk.model = "test-model"

        prompt = sdk._build_system_prompt(channel="webchat")

        assert prompt is not None
        assert "Web Chat" in prompt
