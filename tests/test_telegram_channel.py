"""Tests for the Telegram channel adapter."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.config import TelegramConfig
from agent.core.events import EventBus
from agent.core.session import SessionStore

if TYPE_CHECKING:
    from agent.channels.telegram import TelegramChannel


def _make_tg_message(
    user_id: int = 111,
    text: str | None = "hello",
) -> MagicMock:
    """Create a mock aiogram Message."""
    msg = AsyncMock()
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.text = text
    msg.answer = AsyncMock()
    return msg


async def _drain_background_tasks(channel: Any) -> None:
    """Await all background tasks spawned by the channel."""
    # Snapshot to avoid dict-changed-size-during-iteration
    all_tasks = [
        task
        for tasks in list(channel._background_tasks.values())
        for task, _desc in list(tasks)
    ]
    for task in all_tasks:
        if not task.done():
            with contextlib.suppress(Exception):
                await task


@pytest.fixture
def config() -> TelegramConfig:
    return TelegramConfig(enabled=True, token="fake-token", allowed_users=[])


@pytest.fixture
def config_restricted() -> TelegramConfig:
    return TelegramConfig(enabled=True, token="fake-token", allowed_users=[111, 222])


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def session_store() -> SessionStore:
    return SessionStore()


@pytest.fixture
def mock_agent_loop() -> AsyncMock:
    loop = AsyncMock()
    response = AsyncMock()
    response.content = "Agent reply"
    loop.process_message.return_value = response
    return loop


@pytest.fixture
def mock_heartbeat() -> MagicMock:
    hb = MagicMock()
    hb.is_enabled = True
    hb.last_tick = None
    hb.enable = MagicMock()
    hb.disable = MagicMock()
    return hb


@pytest.fixture
def channel(
    config: TelegramConfig,
    event_bus: EventBus,
    session_store: SessionStore,
    mock_agent_loop: AsyncMock,
    mock_heartbeat: MagicMock,
) -> Any:
    from agent.channels.telegram import TelegramChannel

    with patch("agent.channels.telegram.Bot"), \
         patch("agent.channels.telegram.Dispatcher"), \
         patch("agent.channels.telegram.Router"):
        ch = TelegramChannel(
            config=config,
            event_bus=event_bus,
            session_store=session_store,
            agent_loop=mock_agent_loop,
            heartbeat=mock_heartbeat,
        )
        # Give it mock bot/dispatcher so send_message / send_typing / stop work
        ch._bot = AsyncMock()
        ch._dispatcher = AsyncMock()
    return ch


@pytest.fixture
def restricted_channel(
    config_restricted: TelegramConfig,
    event_bus: EventBus,
    session_store: SessionStore,
    mock_agent_loop: AsyncMock,
) -> Any:
    from agent.channels.telegram import TelegramChannel

    with patch("agent.channels.telegram.Bot"), \
         patch("agent.channels.telegram.Dispatcher"), \
         patch("agent.channels.telegram.Router"):
        ch = TelegramChannel(
            config=config_restricted,
            event_bus=event_bus,
            session_store=session_store,
            agent_loop=mock_agent_loop,
            heartbeat=None,
        )
        ch._bot = AsyncMock()
    return ch


# =====================================================================
# Allowlist
# =====================================================================

class TestAllowlist:
    """Allowlist security checks."""

    def test_empty_allowlist_allows_all(self, channel: TelegramChannel) -> None:
        assert channel._is_allowed(999) is True

    def test_restricted_allows_listed(self, restricted_channel: TelegramChannel) -> None:
        assert restricted_channel._is_allowed(111) is True
        assert restricted_channel._is_allowed(222) is True

    def test_restricted_blocks_unlisted(self, restricted_channel: TelegramChannel) -> None:
        assert restricted_channel._is_allowed(333) is False


# =====================================================================
# Handle text
# =====================================================================

class TestHandleText:
    """Text message processing."""

    async def test_processes_message(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        msg = _make_tg_message(text="What is 2+2?")
        await channel._handle_text(msg)
        await _drain_background_tasks(channel)

        mock_agent_loop.process_message.assert_called_once()
        call_args = mock_agent_loop.process_message.call_args
        assert call_args[0][0] == "What is 2+2?"

    async def test_sends_response(
        self,
        channel: TelegramChannel,
    ) -> None:
        msg = _make_tg_message(text="Hi")
        # msg.answer returns a status message mock that supports edit_text
        status_msg = AsyncMock()
        msg.answer.return_value = status_msg
        await channel._handle_text(msg)
        await _drain_background_tasks(channel)

        # Status message should be sent, then edited with the response
        msg.answer.assert_called()
        status_msg.edit_text.assert_called()

    async def test_emits_incoming_event(
        self,
        channel: TelegramChannel,
        event_bus: EventBus,
    ) -> None:
        events: list[dict[str, Any]] = []

        async def handler(data: dict[str, Any]) -> None:
            events.append(data)

        event_bus.on("message.incoming", handler)

        msg = _make_tg_message(text="test event")
        await channel._handle_text(msg)

        # Event is emitted synchronously before spawning background task
        assert len(events) == 1
        assert events[0]["content"] == "test event"

    async def test_reuses_session_for_same_user(
        self,
        channel: TelegramChannel,
        session_store: SessionStore,
    ) -> None:
        msg1 = _make_tg_message(user_id=42, text="first")
        msg2 = _make_tg_message(user_id=42, text="second")

        await channel._handle_text(msg1)
        await _drain_background_tasks(channel)
        await channel._handle_text(msg2)
        await _drain_background_tasks(channel)

        # Should have one session for user 42
        session = await session_store.get("telegram:42")
        assert session is not None

    async def test_handles_agent_error(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        mock_agent_loop.process_message.side_effect = RuntimeError("LLM down")
        msg = _make_tg_message(text="boom")
        status_msg = AsyncMock()
        msg.answer.return_value = status_msg
        await channel._handle_text(msg)
        await _drain_background_tasks(channel)

        # Error should be shown by editing the status message
        status_msg.edit_text.assert_called_with(
            "Sorry, something went wrong processing your message."
        )

    async def test_blocks_when_paused(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        channel.pause()
        msg = _make_tg_message(text="ignored")
        await channel._handle_text(msg)

        mock_agent_loop.process_message.assert_not_called()
        msg.answer.assert_called_once()

    async def test_ignores_no_from_user(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        msg = _make_tg_message(text="anon")
        msg.from_user = None
        await channel._handle_text(msg)

        mock_agent_loop.process_message.assert_not_called()

    async def test_ignores_empty_text(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        msg = _make_tg_message(text=None)
        await channel._handle_text(msg)

        mock_agent_loop.process_message.assert_not_called()

    async def test_handler_returns_immediately(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """_handle_text should return before processing finishes (background task)."""
        processing_started = asyncio.Event()
        processing_gate = asyncio.Event()

        original = mock_agent_loop.process_message

        async def slow_process(*args: Any, **kwargs: Any) -> Any:
            processing_started.set()
            await processing_gate.wait()
            return original.return_value

        mock_agent_loop.process_message.side_effect = slow_process

        msg = _make_tg_message(text="slow task")
        await channel._handle_text(msg)

        # Handler returned, but processing hasn't finished yet
        assert not mock_agent_loop.process_message.return_value.content == "done"
        await processing_started.wait()  # task is running
        assert len(channel._get_active_tasks()) == 1

        # Let it finish
        processing_gate.set()
        await _drain_background_tasks(channel)

    async def test_concurrent_messages_from_same_user(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Multiple messages from the same user should run concurrently."""
        gate = asyncio.Event()
        call_count = 0

        async def counting_process(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            await gate.wait()
            return mock_agent_loop.process_message.return_value

        mock_agent_loop.process_message.side_effect = counting_process

        msg1 = _make_tg_message(user_id=42, text="first")
        msg2 = _make_tg_message(user_id=42, text="second")

        await channel._handle_text(msg1)
        await channel._handle_text(msg2)

        # Both should be registered
        await asyncio.sleep(0.01)
        active = channel._get_active_tasks("42")
        assert len(active) == 2

        gate.set()
        await _drain_background_tasks(channel)
        assert call_count == 2

    async def test_concurrent_messages_from_different_users(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Messages from different users should run independently."""
        gate = asyncio.Event()

        async def gated_process(*args: Any, **kwargs: Any) -> Any:
            await gate.wait()
            return mock_agent_loop.process_message.return_value

        mock_agent_loop.process_message.side_effect = gated_process

        msg1 = _make_tg_message(user_id=10, text="user10")
        msg2 = _make_tg_message(user_id=20, text="user20")

        await channel._handle_text(msg1)
        await channel._handle_text(msg2)
        await asyncio.sleep(0.01)

        assert len(channel._get_active_tasks("10")) == 1
        assert len(channel._get_active_tasks("20")) == 1
        assert len(channel._get_active_tasks()) == 2

        gate.set()
        await _drain_background_tasks(channel)

    async def test_status_message_send_failure(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Processing should continue even if status message send fails."""
        msg = _make_tg_message(text="test")
        msg.answer.side_effect = Exception("Telegram API down")

        await channel._handle_text(msg)
        await _drain_background_tasks(channel)

        # Should still have processed the message
        mock_agent_loop.process_message.assert_called_once()

    async def test_error_without_status_message(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Error handling when status message couldn't be sent."""
        mock_agent_loop.process_message.side_effect = RuntimeError("boom")
        msg = _make_tg_message(text="fail")
        # Make status message send fail → status_message is None
        msg.answer.side_effect = Exception("can't send")

        await channel._handle_text(msg)
        await _drain_background_tasks(channel)

        # Should have tried to send error via msg.answer (2nd call)
        # First call: status message (failed), second: error fallback
        assert msg.answer.call_count >= 1

    async def test_task_unregistered_after_completion(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Background task should be cleaned up after it finishes."""
        msg = _make_tg_message(text="done")
        await channel._handle_text(msg)
        await _drain_background_tasks(channel)

        assert len(channel._get_active_tasks()) == 0
        assert "111" not in channel._background_tasks

    async def test_task_unregistered_after_error(
        self,
        channel: TelegramChannel,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """Background task should be cleaned up even on error."""
        mock_agent_loop.process_message.side_effect = RuntimeError("crash")
        msg = _make_tg_message(text="err")
        msg.answer.return_value = AsyncMock()  # status msg mock

        await channel._handle_text(msg)
        await _drain_background_tasks(channel)

        assert len(channel._get_active_tasks()) == 0


# =====================================================================
# Tool explanation
# =====================================================================

class TestToolExplanation:
    """Tests for _tool_explanation approval message builder."""

    def _explain(self, tool_name: str, args: dict[str, Any]) -> str:
        from agent.channels.telegram import _tool_explanation
        return _tool_explanation(tool_name, args)

    def test_bash_with_description(self) -> None:
        result = self._explain("Bash", {
            "command": "ls -la /tmp",
            "description": "List files in tmp directory",
        })
        assert "List files in tmp directory" in result
        assert "ls -la /tmp" in result
        assert "shell command" in result.lower()

    def test_bash_without_description(self) -> None:
        result = self._explain("Bash", {"command": "rm -rf /tmp/junk"})
        assert "Run a shell command" in result
        assert "rm -rf /tmp/junk" in result

    def test_shell_exec_maps_to_bash(self) -> None:
        result = self._explain("shell_exec", {"command": "echo hi"})
        assert "echo hi" in result
        assert "shell command" in result.lower()

    def test_write_file(self) -> None:
        result = self._explain("Write", {
            "file_path": "/home/user/app.py",
            "content": "x" * 200,
        })
        assert "/home/user/app.py" in result
        assert "200 characters" in result
        assert "write" in result.lower() or "overwrite" in result.lower()

    def test_file_write_uses_path_key(self) -> None:
        result = self._explain("file_write", {"path": "/tmp/out.txt", "content": "hi"})
        assert "/tmp/out.txt" in result

    def test_edit_shows_diff_preview(self) -> None:
        result = self._explain("Edit", {
            "file_path": "/src/main.py",
            "old_string": "def foo():",
            "new_string": "def bar():",
        })
        assert "/src/main.py" in result
        assert "def foo():" in result
        assert "def bar():" in result
        assert "modify" in result.lower()

    def test_edit_truncates_long_strings(self) -> None:
        long_old = "x" * 200
        result = self._explain("Edit", {
            "file_path": "/f.py",
            "old_string": long_old,
            "new_string": "short",
        })
        assert "…" in result  # truncated

    def test_file_delete(self) -> None:
        result = self._explain("file_delete", {"path": "/home/user/secret.txt"})
        assert "/home/user/secret.txt" in result
        assert "permanently delete" in result.lower()

    def test_python_exec(self) -> None:
        result = self._explain("python_exec", {
            "code": "import os; os.listdir('/')",
        })
        assert "import os" in result
        assert "Python" in result

    def test_python_exec_truncates_long_code(self) -> None:
        result = self._explain("python_exec", {"code": "x = 1\n" * 200})
        assert "…" in result

    def test_http_request(self) -> None:
        result = self._explain("http_request", {
            "method": "POST",
            "url": "https://api.example.com/data",
        })
        assert "POST" in result
        assert "https://api.example.com/data" in result
        assert "network request" in result.lower()

    def test_http_request_defaults_to_get(self) -> None:
        result = self._explain("http_request", {"url": "https://example.com"})
        assert "GET" in result

    def test_browser_navigate(self) -> None:
        result = self._explain("browser_navigate", {"url": "https://google.com"})
        assert "https://google.com" in result
        assert "browser" in result.lower()

    def test_browser_action(self) -> None:
        result = self._explain("browser_action", {"action": "click_element"})
        assert "click_element" in result

    def test_desktop_click(self) -> None:
        result = self._explain("desktop_click", {"x": 100, "y": 200})
        assert "100" in result
        assert "200" in result
        assert "desktop" in result.lower()

    def test_desktop_type(self) -> None:
        result = self._explain("desktop_type", {"text": "Hello World"})
        assert "Hello World" in result
        assert "desktop" in result.lower()

    def test_desktop_type_truncates_long_text(self) -> None:
        result = self._explain("desktop_type", {"text": "a" * 200})
        # Should truncate at 100
        assert len(result) < 500

    def test_notebook_edit(self) -> None:
        result = self._explain("NotebookEdit", {
            "notebook_path": "/notebooks/analysis.ipynb",
        })
        assert "analysis.ipynb" in result
        assert "notebook" in result.lower()

    def test_unknown_tool_generic_fallback(self) -> None:
        result = self._explain("my_custom_tool", {
            "param1": "value1",
            "param2": "value2",
        })
        assert "my_custom_tool" in result
        assert "param1=value1" in result

    def test_unknown_tool_with_description(self) -> None:
        result = self._explain("my_tool", {
            "description": "Custom explanation",
            "param": "val",
        })
        assert "Custom explanation" in result
        # description key should be excluded from params
        assert "description=" not in result

    def test_unknown_tool_empty_args(self) -> None:
        result = self._explain("empty_tool", {})
        assert "empty_tool" in result

    def test_all_outputs_have_html_structure(self) -> None:
        """Every explanation should have the AI-wants-to and risk sections."""
        cases = [
            ("Bash", {"command": "ls"}),
            ("Write", {"file_path": "/f", "content": "x"}),
            ("Edit", {"file_path": "/f", "old_string": "a", "new_string": "b"}),
            ("file_delete", {"path": "/f"}),
            ("python_exec", {"code": "x=1"}),
            ("http_request", {"url": "http://x.com"}),
            ("browser_navigate", {"url": "http://x.com"}),
            ("desktop_click", {"x": 1, "y": 2}),
            ("desktop_type", {"text": "hi"}),
            ("NotebookEdit", {"notebook_path": "/n.ipynb"}),
            ("unknown_tool", {"a": "b"}),
        ]
        for tool_name, args in cases:
            result = self._explain(tool_name, args)
            assert "<b>AI wants to:</b>" in result, f"Missing header for {tool_name}"
            assert "<i>" in result, f"Missing risk note for {tool_name}"


# =====================================================================
# Response positioning (approval-aware)
# =====================================================================

class TestReplaceStatusWithResponse:
    """Tests for _replace_status_with_response — approval-aware positioning."""

    async def test_no_approvals_edits_status_message(
        self, channel: TelegramChannel,
    ) -> None:
        """Without approvals, response should edit the status message in-place."""
        status_msg = AsyncMock()
        channel._had_approvals["42"] = False

        await channel._replace_status_with_response(status_msg, "42", "The answer")

        status_msg.edit_text.assert_called()
        status_msg.delete.assert_not_called()

    async def test_with_approvals_deletes_status_and_sends_new(
        self, channel: TelegramChannel,
    ) -> None:
        """With approvals, status message should be deleted and response sent fresh."""
        status_msg = AsyncMock()
        channel._had_approvals["42"] = True

        await channel._replace_status_with_response(status_msg, "42", "The answer")

        status_msg.delete.assert_called_once()
        # Should NOT edit the old message
        status_msg.edit_text.assert_not_called()

    async def test_empty_response_shows_no_response_message(
        self, channel: TelegramChannel,
    ) -> None:
        """Empty response should show 'No response' in the status message."""
        status_msg = AsyncMock()
        await channel._replace_status_with_response(status_msg, "42", "")

        status_msg.edit_text.assert_called_once_with(
            "No response was generated. Please try again."
        )

    async def test_empty_response_no_status_message(
        self, channel: TelegramChannel,
    ) -> None:
        """Empty response with no status message should not crash."""
        # Should not raise
        await channel._replace_status_with_response(None, "42", "")

    async def test_no_bot_returns_silently(
        self, channel: TelegramChannel,
    ) -> None:
        """If bot is None, should return without error."""
        channel._bot = None
        # Should not raise
        await channel._replace_status_with_response(AsyncMock(), "42", "hello")

    async def test_no_status_message_sends_normally(
        self, channel: TelegramChannel,
    ) -> None:
        """Without a status message, response is sent as a new message."""
        with patch.object(channel, "send_streamed_response", new_callable=AsyncMock) as mock_send:
            await channel._replace_status_with_response(None, "42", "fresh response")
            mock_send.assert_called_once_with("42", "fresh response")

    async def test_edit_markdown_failure_falls_back_to_plain(
        self, channel: TelegramChannel,
    ) -> None:
        """If Markdown edit fails, should retry without parse_mode."""
        status_msg = AsyncMock()
        # First call (with Markdown) fails, second (plain) succeeds
        status_msg.edit_text.side_effect = [Exception("Markdown error"), None]
        channel._had_approvals["42"] = False

        await channel._replace_status_with_response(status_msg, "42", "simple text")

        assert status_msg.edit_text.call_count == 2
        # Second call should be plain text
        assert status_msg.edit_text.call_args_list[1] == ((("simple text",),))

    async def test_edit_both_fail_deletes_and_sends_fresh(
        self, channel: TelegramChannel,
    ) -> None:
        """If both edit attempts fail, delete status and send fresh."""
        status_msg = AsyncMock()
        status_msg.edit_text.side_effect = Exception("all edits fail")
        channel._had_approvals["42"] = False

        with patch.object(channel, "send_message", new_callable=AsyncMock) as mock_send:
            await channel._replace_status_with_response(status_msg, "42", "fallback")

            status_msg.delete.assert_called_once()
            mock_send.assert_called_once()

    async def test_long_response_sends_remaining_chunks(
        self, channel: TelegramChannel,
    ) -> None:
        """Long responses should send additional chunks as new messages."""
        status_msg = AsyncMock()
        channel._had_approvals["42"] = False
        long_text = "x" * 5000  # will be split into chunks

        await channel._replace_status_with_response(status_msg, "42", long_text)

        # First chunk goes into status edit, rest via send_message
        status_msg.edit_text.assert_called_once()
        assert channel._bot.send_message.call_count >= 1

    async def test_approvals_flag_per_user_isolation(
        self, channel: TelegramChannel,
    ) -> None:
        """Approvals flag for one user should not affect another."""
        status_msg_a = AsyncMock()
        status_msg_b = AsyncMock()
        channel._had_approvals["10"] = True
        channel._had_approvals["20"] = False

        await channel._replace_status_with_response(status_msg_a, "10", "for A")
        await channel._replace_status_with_response(status_msg_b, "20", "for B")

        # User 10 had approvals: should delete status
        status_msg_a.delete.assert_called_once()
        status_msg_a.edit_text.assert_not_called()

        # User 20 no approvals: should edit in place
        status_msg_b.edit_text.assert_called()
        status_msg_b.delete.assert_not_called()


# =====================================================================
# Background task management
# =====================================================================

class TestBackgroundTaskManagement:
    """Tests for task registration, unregistration, and listing."""

    def test_register_task(self, channel: TelegramChannel) -> None:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[None] = loop.create_future()
        task = asyncio.ensure_future(fut)

        channel._register_background_task("42", task, "test task")

        assert "42" in channel._background_tasks
        assert len(channel._background_tasks["42"]) == 1
        assert channel._background_tasks["42"][0][1] == "test task"
        fut.set_result(None)

    def test_register_multiple_tasks_same_user(
        self, channel: TelegramChannel,
    ) -> None:
        loop = asyncio.get_event_loop()
        f1: asyncio.Future[None] = loop.create_future()
        f2: asyncio.Future[None] = loop.create_future()
        t1, t2 = asyncio.ensure_future(f1), asyncio.ensure_future(f2)

        channel._register_background_task("42", t1, "task 1")
        channel._register_background_task("42", t2, "task 2")

        assert len(channel._background_tasks["42"]) == 2
        f1.set_result(None)
        f2.set_result(None)

    async def test_unregister_removes_done_tasks(
        self, channel: TelegramChannel,
    ) -> None:
        async def noop() -> None:
            pass

        task = asyncio.create_task(noop())
        channel._register_background_task("42", task, "quick")
        await task  # let it finish

        channel._unregister_background_task("42")

        # Should be cleaned up since task is done
        assert "42" not in channel._background_tasks

    async def test_unregister_keeps_running_tasks(
        self, channel: TelegramChannel,
    ) -> None:
        gate = asyncio.Event()

        async def blocked() -> None:
            await gate.wait()

        task = asyncio.create_task(blocked())
        channel._register_background_task("42", task, "slow")

        channel._unregister_background_task("42")

        # Still running, should be kept
        assert "42" in channel._background_tasks
        assert len(channel._background_tasks["42"]) == 1

        gate.set()
        await task

    def test_get_active_tasks_empty(self, channel: TelegramChannel) -> None:
        assert channel._get_active_tasks() == []
        assert channel._get_active_tasks("42") == []

    async def test_get_active_tasks_filters_done(
        self, channel: TelegramChannel,
    ) -> None:
        gate = asyncio.Event()

        async def blocked() -> None:
            await gate.wait()

        async def fast() -> None:
            pass

        running = asyncio.create_task(blocked())
        done = asyncio.create_task(fast())
        await done

        channel._register_background_task("42", running, "running")
        channel._register_background_task("42", done, "done")

        active = channel._get_active_tasks("42")
        assert len(active) == 1
        assert active[0][1] == "running"

        gate.set()
        await running

    async def test_get_active_tasks_all_users(
        self, channel: TelegramChannel,
    ) -> None:
        gate = asyncio.Event()

        async def blocked() -> None:
            await gate.wait()

        t1 = asyncio.create_task(blocked())
        t2 = asyncio.create_task(blocked())
        channel._register_background_task("10", t1, "user10-task")
        channel._register_background_task("20", t2, "user20-task")

        all_tasks = channel._get_active_tasks()
        assert len(all_tasks) == 2
        descs = {d for _, d in all_tasks}
        assert "user10-task" in descs
        assert "user20-task" in descs

        gate.set()
        await t1
        await t2


# =====================================================================
# /tasks command
# =====================================================================

class TestCmdTasks:
    """Tests for the /tasks command."""

    async def test_no_running_tasks(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        await channel._cmd_tasks(msg)

        text = msg.answer.call_args[0][0]
        assert "no tasks" in text.lower() or "free" in text.lower()

    async def test_shows_running_tasks(self, channel: TelegramChannel) -> None:
        gate = asyncio.Event()

        async def blocked() -> None:
            await gate.wait()

        task = asyncio.create_task(blocked())
        channel._register_background_task("111", task, "Doing research")

        msg = _make_tg_message()
        await channel._cmd_tasks(msg)

        text = msg.answer.call_args[0][0]
        assert "Doing research" in text
        assert "1" in text  # count

        gate.set()
        await task

    async def test_tasks_requires_auth(self, channel: TelegramChannel) -> None:
        """Should check _check_message for authorization."""
        msg = _make_tg_message()
        msg.from_user = None  # will fail _check_message

        await channel._cmd_tasks(msg)
        msg.answer.assert_not_called()


# =====================================================================
# Commands
# =====================================================================

class TestCommands:
    """Bot command handlers."""

    async def test_cmd_start(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        await channel._cmd_start(msg)
        msg.answer.assert_called_once()
        assert "Hello" in msg.answer.call_args[0][0]

    async def test_cmd_help(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        await channel._cmd_help(msg)
        text = msg.answer.call_args[0][0]
        assert "/start" in text
        assert "/help" in text
        assert "/mute" in text

    async def test_cmd_status(
        self,
        channel: TelegramChannel,
        mock_heartbeat: MagicMock,
    ) -> None:
        msg = _make_tg_message()
        with patch("agent.tools.registry.registry") as mock_reg:
            mock_reg.list_tools.return_value = [MagicMock(), MagicMock()]
            await channel._cmd_status(msg)

        text = msg.answer.call_args[0][0]
        assert "enabled" in text
        assert "Tools: 2" in text

    async def test_cmd_tools_empty(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        with patch("agent.tools.registry.registry") as mock_reg:
            mock_reg.list_tools.return_value = []
            await channel._cmd_tools(msg)

        msg.answer.assert_called_with("No tools registered.")

    async def test_cmd_tools_with_tools(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()

        mock_tool = MagicMock()
        mock_tool.name = "shell_exec"
        mock_tool.description = "Run shell commands"
        mock_tool.tier.value = "moderate"
        mock_tool.enabled = True

        with patch("agent.tools.registry.registry") as mock_reg:
            mock_reg.list_tools.return_value = [mock_tool]
            await channel._cmd_tools(msg)

        text = msg.answer.call_args[0][0]
        assert "shell_exec" in text
        assert "[on]" in text

    async def test_cmd_pause(self, channel: TelegramChannel) -> None:
        msg = _make_tg_message()
        await channel._cmd_pause(msg)
        assert channel._paused is True
        assert "paused" in msg.answer.call_args[0][0].lower()

    async def test_cmd_resume(self, channel: TelegramChannel) -> None:
        channel.pause()
        msg = _make_tg_message()
        await channel._cmd_resume(msg)
        assert channel._paused is False
        assert "resumed" in msg.answer.call_args[0][0].lower()

    async def test_cmd_mute(
        self,
        channel: TelegramChannel,
        mock_heartbeat: MagicMock,
    ) -> None:
        msg = _make_tg_message()
        await channel._cmd_mute(msg)
        mock_heartbeat.disable.assert_called_once()
        assert "muted" in msg.answer.call_args[0][0].lower()

    async def test_cmd_unmute(
        self,
        channel: TelegramChannel,
        mock_heartbeat: MagicMock,
    ) -> None:
        msg = _make_tg_message()
        await channel._cmd_unmute(msg)
        mock_heartbeat.enable.assert_called_once()
        assert "unmuted" in msg.answer.call_args[0][0].lower()

    async def test_cmd_mute_no_heartbeat(
        self,
        config: TelegramConfig,
        event_bus: EventBus,
        session_store: SessionStore,
        mock_agent_loop: AsyncMock,
    ) -> None:
        from agent.channels.telegram import TelegramChannel

        with patch("agent.channels.telegram.Bot"), \
             patch("agent.channels.telegram.Dispatcher"), \
             patch("agent.channels.telegram.Router"):
            ch = TelegramChannel(
                config=config,
                event_bus=event_bus,
                session_store=session_store,
                agent_loop=mock_agent_loop,
                heartbeat=None,
            )
            ch._bot = AsyncMock()

        msg = _make_tg_message()
        await ch._cmd_mute(msg)
        assert "not configured" in msg.answer.call_args[0][0].lower()


# =====================================================================
# Split message
# =====================================================================

class TestSplitMessage:
    """Message splitting logic."""

    def test_short_message_unchanged(self) -> None:
        from agent.channels.telegram import TelegramChannel

        result = TelegramChannel._split_message("short text")
        assert result == ["short text"]

    def test_splits_at_newline(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "a" * 50 + "\n" + "b" * 50
        result = TelegramChannel._split_message(text, max_length=60)
        assert len(result) == 2
        assert result[0] == "a" * 50

    def test_splits_at_space(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "a" * 50 + " " + "b" * 50
        result = TelegramChannel._split_message(text, max_length=60)
        assert len(result) == 2
        assert result[0] == "a" * 50

    def test_hard_split(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "a" * 100
        result = TelegramChannel._split_message(text, max_length=60)
        assert len(result) == 2
        assert len(result[0]) == 60
        assert len(result[1]) == 40

    def test_multiple_chunks(self) -> None:
        from agent.channels.telegram import TelegramChannel

        text = "word " * 100  # 500 chars
        result = TelegramChannel._split_message(text, max_length=50)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 50


# =====================================================================
# Lifecycle
# =====================================================================

class TestLifecycle:
    """Start/stop behavior."""

    def test_name(self, channel: TelegramChannel) -> None:
        assert channel.name == "telegram"

    async def test_stop_without_start(self, channel: TelegramChannel) -> None:
        """stop() should not raise if never started."""
        await channel.stop()

    async def test_start_no_token(
        self,
        event_bus: EventBus,
        session_store: SessionStore,
        mock_agent_loop: AsyncMock,
    ) -> None:
        """start() should log warning and return if no token."""
        from agent.channels.telegram import TelegramChannel

        no_token_config = TelegramConfig(enabled=True, token=None)
        ch = TelegramChannel(
            config=no_token_config,
            event_bus=event_bus,
            session_store=session_store,
            agent_loop=mock_agent_loop,
        )
        await ch.start()
        assert ch.is_running is False


# =====================================================================
# Keep typing
# =====================================================================

class TestKeepTyping:
    """Typing indicator loop."""

    async def test_keep_typing_sends_and_cancels(
        self,
        channel: TelegramChannel,
    ) -> None:
        task = asyncio.create_task(channel._keep_typing("42"))

        # Let it run for a short while
        await asyncio.sleep(0.05)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should not raise — task cancelled cleanly
        assert task.done()
