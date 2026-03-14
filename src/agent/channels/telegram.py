"""Telegram channel adapter using aiogram 3.x.

Handles text messages, voice/photo/document media, bot commands,
streaming via message edits, and inline keyboard approval for tools.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from agent.channels.base import BaseChannel, OutgoingMessage
from agent.core.events import Events

if TYPE_CHECKING:
    from agent.config import TelegramConfig
    from agent.core.agent_loop import AgentLoop
    from agent.core.audit import AuditLog
    from agent.core.cost_tracker import CostTracker
    from agent.core.events import EventBus
    from agent.core.heartbeat import HeartbeatDaemon
    from agent.core.scheduler import TaskScheduler
    from agent.core.session import SessionStore
    from agent.voice.pipeline import VoicePipeline
    from agent.workspaces.router import WorkspaceRouter

logger = structlog.get_logger(__name__)

# Attempt to import aiogram at module level; flag availability
try:
    from aiogram import Bot, Dispatcher, F, Router
    from aiogram.enums import ChatAction, ContentType
    from aiogram.types import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
    )

    AIOGRAM_AVAILABLE = True
except ImportError:
    AIOGRAM_AVAILABLE = False

# Telegram max message length
_TG_MAX_LENGTH = 4096

# Typing refresh interval in seconds
_TYPING_INTERVAL = 4

# Streaming: responses longer than this get the edit-based preview
_STREAM_THRESHOLD = 500

# Approval timeout in seconds
_APPROVAL_TIMEOUT = 300

# Upload directory
_UPLOAD_DIR = Path("data/uploads")

# Default prompt shortcuts for /run command
_DEFAULT_SHORTCUTS = [
    {
        "alias": "review",
        "template": (
            "Review the latest git diff and provide feedback on"
            " code quality, potential bugs, and suggestions."
        ),
        "description": "Review latest git diff",
    },
    {
        "alias": "test",
        "template": (
            "Run the test suite and summarize the results,"
            " highlighting any failures or warnings."
        ),
        "description": "Run tests and summarize",
    },
    {
        "alias": "explain",
        "template": "Read and explain the file: {args}",
        "description": "Explain a file",
    },
    {
        "alias": "commit",
        "template": (
            "Look at staged git changes and suggest a clear,"
            " concise commit message following conventional"
            " commit format."
        ),
        "description": "Suggest commit message",
    },
    {
        "alias": "fix",
        "template": "Debug and fix this issue: {args}",
        "description": "Debug an issue",
    },
    {
        "alias": "summarize",
        "template": (
            "Summarize our conversation so far, highlighting"
            " key decisions and action items."
        ),
        "description": "Summarize conversation",
    },
]


def _tool_explanation(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a human-readable approval message explaining the AI's intent.

    Uses the AI's ``description`` field (available on Bash/shell tools) and
    constructs clear, natural-language explanations for all tool types so the
    user understands *what* will happen and *why* it needs approval.
    """
    # The AI's own explanation (Claude fills 'description' for Bash-like tools)
    ai_desc = arguments.get("description", "")

    # --- Build a natural-language "what & why" block per tool type ----------

    if tool_name in ("Bash", "shell_exec"):
        cmd = arguments.get("command", "unknown command")
        if ai_desc:
            intent = f"{ai_desc}"
        else:
            intent = "Run a shell command"
        detail = f"Command:\n<code>{cmd}</code>"
        risk = "This will execute a shell command on your machine."

    elif tool_name in ("Write", "file_write"):
        path = arguments.get("file_path") or arguments.get("path", "?")
        size = len(arguments.get("content", ""))
        intent = ai_desc or f"Create or overwrite a file"
        detail = f"File: <code>{path}</code> ({size} characters)"
        risk = "This will write to your filesystem — the file will be created or overwritten."

    elif tool_name == "Edit":
        path = arguments.get("file_path", "?")
        old = arguments.get("old_string", "")
        new = arguments.get("new_string", "")
        intent = ai_desc or "Modify an existing file"
        # Show a compact diff preview
        old_preview = (old[:120] + "…") if len(old) > 120 else old
        new_preview = (new[:120] + "…") if len(new) > 120 else new
        detail = (
            f"File: <code>{path}</code>\n"
            f"Replace:\n<code>{old_preview}</code>\n"
            f"With:\n<code>{new_preview}</code>"
        )
        risk = "This will modify the contents of a file on disk."

    elif tool_name == "file_delete":
        path = arguments.get("path", "?")
        intent = ai_desc or f"Delete a file"
        detail = f"File: <code>{path}</code>"
        risk = "This will permanently delete a file from disk."

    elif tool_name == "python_exec":
        code = arguments.get("code", "")
        preview = (code[:400] + "…") if len(code) > 400 else code
        intent = ai_desc or "Execute Python code"
        detail = f"Code:\n<code>{preview}</code>"
        risk = "This runs Python with full system access."

    elif tool_name == "http_request":
        method = arguments.get("method", "GET")
        url = arguments.get("url", "?")
        intent = ai_desc or f"Make an HTTP request"
        detail = f"{method} {url}"
        risk = "This sends a network request to an external server."

    elif tool_name in ("browser_navigate", "browser_action"):
        url = arguments.get("url", arguments.get("action", "?"))
        intent = ai_desc or "Control the browser"
        detail = f"URL/Action: {url}"
        risk = "This will open or interact with a website in a browser."

    elif tool_name in ("desktop_click", "desktop_type"):
        intent = ai_desc or "Control your desktop"
        if tool_name == "desktop_click":
            x, y = arguments.get("x", "?"), arguments.get("y", "?")
            detail = f"Click at ({x}, {y})"
        else:
            text = arguments.get("text", "?")
            detail = f"Type: {text[:100]}"
        risk = "This will perform actions on your desktop (mouse/keyboard)."

    elif tool_name == "NotebookEdit":
        path = arguments.get("notebook_path", "?")
        intent = ai_desc or "Edit a Jupyter notebook"
        detail = f"Notebook: <code>{path}</code>"
        risk = "This will modify a Jupyter notebook file."

    else:
        # Generic fallback for unknown tools
        intent = ai_desc or f"Use the {tool_name} tool"
        parts = [
            f"{k}={str(v)[:80]}"
            for k, v in list(arguments.items())[:4]
            if k != "description"
        ]
        detail = ", ".join(parts) if parts else str(arguments)[:200]
        risk = f"This grants access to the '{tool_name}' tool."

    lines = [
        f"<b>AI wants to:</b> {intent}",
        "",
        detail,
        "",
        f"<i>{risk}</i>",
    ]

    return "\n".join(lines)


class TelegramChannel(BaseChannel):
    """Telegram messaging channel via aiogram 3.x.

    Supports text messages, voice/photo/document media, and bot commands
    (/start, /help, /status, /tools, /pause, /resume, /mute, /unmute, /soul,
    /backend, /workdir).
    """

    def __init__(
        self,
        config: TelegramConfig,
        event_bus: EventBus,
        session_store: SessionStore,
        agent_loop: AgentLoop,
        heartbeat: HeartbeatDaemon | None = None,
        workspace_router: WorkspaceRouter | None = None,
        voice_pipeline: VoicePipeline | None = None,
        sdk_service: object | None = None,
        scheduler: TaskScheduler | None = None,
        audit_log: AuditLog | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, session_store=session_store)
        self.agent_loop = agent_loop
        self.heartbeat = heartbeat
        self.workspace_router = workspace_router
        self.voice_pipeline = voice_pipeline
        self.sdk_service = sdk_service
        self.scheduler = scheduler
        self.audit_log = audit_log
        self.cost_tracker = cost_tracker
        self._workspace_agent_loops: dict[str, AgentLoop] = {}
        self._workspace_session_stores: dict[str, SessionStore] = {}
        self._polling_task: asyncio.Task[Any] | None = None
        self._bot: Any = None
        self._dispatcher: Any = None
        self._router: Any = None
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._had_approvals: dict[str, bool] = {}  # per-user approval tracking
        # Background task tracking: user_id → list of (task, description)
        self._background_tasks: dict[str, list[tuple[asyncio.Task[Any], str]]] = {}

        if AIOGRAM_AVAILABLE and config.token:
            self._bot = Bot(token=config.token)
            self._dispatcher = Dispatcher()
            self._router = Router()
            self._dispatcher.include_router(self._router)
            self._setup_handlers()

    @property
    def name(self) -> str:
        """Channel identifier."""
        return "telegram"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start polling for Telegram updates."""
        if not AIOGRAM_AVAILABLE:
            logger.warning(
                "telegram_disabled",
                reason="aiogram not installed. Install with: pip install aiogram",
            )
            return

        if not self.config.token:
            logger.warning("telegram_disabled", reason="No bot token configured")
            return

        # Validate token by deleting any stale webhook
        try:
            await self._bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            logger.error(
                "telegram_start_failed",
                error=str(e),
                hint="Check that TELEGRAM_BOT_TOKEN is valid",
            )
            return

        # Set bot command menu (the "/" button near the input field)
        await self._set_bot_commands()

        # Subscribe to file send and channel posting events
        self.event_bus.on(Events.FILE_SEND, self._on_file_send)
        self.event_bus.on(Events.CHANNEL_POST, self._on_channel_post)
        self.event_bus.on(Events.CHANNEL_SEND_MESSAGE, self._on_send_message)

        self._running = True
        self._polling_task = asyncio.create_task(self._run_polling())
        logger.info("telegram_started")

    async def _set_bot_commands(self) -> None:
        """Register bot commands so they appear in the Telegram menu button."""
        try:
            from aiogram.types import BotCommand

            commands = [
                BotCommand(command="help", description="Show available commands"),
                BotCommand(command="new", description="Start a new conversation"),
                BotCommand(command="remind", description="Set a reminder"),
                BotCommand(command="reminders", description="List pending reminders"),
                BotCommand(command="status", description="Agent status"),
                BotCommand(command="tools", description="List available tools"),
                BotCommand(command="session", description="Current session info"),
                BotCommand(command="soul", description="View/edit personality"),
                BotCommand(command="backend", description="View/switch LLM backend"),
                BotCommand(command="workdir", description="View/change working directory"),
                BotCommand(command="pause", description="Pause message processing"),
                BotCommand(command="resume", description="Resume processing"),
                BotCommand(command="run", description="Run a prompt shortcut (/run list)"),
                BotCommand(command="mute", description="Disable heartbeat"),
                BotCommand(command="unmute", description="Enable heartbeat"),
            ]
            await self._bot.set_my_commands(commands)
            logger.info("bot_commands_set", count=len(commands))
        except Exception as e:
            logger.warning("bot_commands_set_failed", error=str(e))

    async def _run_polling(self) -> None:
        """Run dispatcher polling in a background task."""
        try:
            await self._dispatcher.start_polling(self._bot)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("telegram_polling_error", error=str(e))
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop polling and close bot session."""
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._polling_task

        if self._dispatcher:
            with contextlib.suppress(RuntimeError):
                await self._dispatcher.stop_polling()

        if self._bot:
            await self._bot.session.close()

        self._running = False
        logger.info("telegram_stopped")

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    def _is_allowed(self, user_id: int) -> bool:
        """Check if a Telegram user is in the allowlist.

        An empty allowlist means everyone is allowed.

        Args:
            user_id: Telegram user ID to check.

        Returns:
            True if the user is allowed.
        """
        allowed: list[int] = self.config.allowed_users
        if not allowed:
            return True
        return user_id in allowed

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _setup_handlers(self) -> None:
        """Register command and message handlers on the router."""
        if not self._router:
            return

        from aiogram.filters import Command

        self._router.message(Command("start"))(self._cmd_start)
        self._router.message(Command("help"))(self._cmd_help)
        self._router.message(Command("status"))(self._cmd_status)
        self._router.message(Command("tools"))(self._cmd_tools)
        self._router.message(Command("pause"))(self._cmd_pause)
        self._router.message(Command("resume"))(self._cmd_resume)
        self._router.message(Command("mute"))(self._cmd_mute)
        self._router.message(Command("unmute"))(self._cmd_unmute)
        self._router.message(Command("soul"))(self._cmd_soul)
        self._router.message(Command("backend"))(self._cmd_backend)
        self._router.message(Command("workdir"))(self._cmd_workdir)
        self._router.message(Command("new"))(self._cmd_new)
        self._router.message(Command("session"))(self._cmd_session)
        self._router.message(Command("remind"))(self._cmd_remind)
        self._router.message(Command("reminders"))(self._cmd_reminders)
        self._router.message(Command("run"))(self._cmd_run)
        self._router.message(Command("tasks"))(self._cmd_tasks)

        # Media handlers (before catch-all text)
        self._router.message(F.content_type == ContentType.VOICE)(self._handle_voice)
        self._router.message(F.content_type == ContentType.PHOTO)(self._handle_photo)
        self._router.message(F.content_type == ContentType.DOCUMENT)(self._handle_document)

        # Callback query handler for inline buttons (approval + navigation)
        self._router.callback_query()(self._handle_callback)

        # Catch-all text handler (must be registered last)
        self._router.message()(self._handle_text)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def _cmd_start(self, message: Any) -> None:
        """Handle /start command."""
        if not self._check_message(message):
            return
        await message.answer(
            "Hello! I'm your AI assistant. Send me a message and I'll help you out.\n"
            "Use /help to see available commands."
        )

    async def _cmd_help(self, message: Any) -> None:
        """Handle /help command."""
        if not self._check_message(message):
            return
        await message.answer(
            "Available commands:\n"
            "/start — Welcome message\n"
            "/help — This help text\n"
            "/status — Agent status\n"
            "/tools — List available tools\n"
            "/new — Start a new conversation\n"
            "/session — Show current session info\n"
            "/run <shortcut> — Run a prompt shortcut (/run list for all)\n"
            "/soul — View/edit agent personality\n"
            "/backend — View/switch LLM backend\n"
            "/workdir — View/change working directory\n"
            "/remind <delay> <text> — Set a reminder\n"
            "/reminders — List pending reminders\n"
            "/tasks — Show running background tasks\n"
            "/pause — Pause message processing\n"
            "/resume — Resume message processing\n"
            "/mute — Disable heartbeat\n"
            "/unmute — Enable heartbeat"
        )

    async def _cmd_status(self, message: Any) -> None:
        """Handle /status command — show agent status."""
        if not self._check_message(message):
            return

        text = await self._build_status_text()

        if AIOGRAM_AVAILABLE:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="Costs", callback_data="nav:status:costs"),
                    InlineKeyboardButton(text="Health", callback_data="nav:status:health"),
                    InlineKeyboardButton(text="Audit", callback_data="nav:status:audit"),
                ],
            ])
            await message.answer(text, reply_markup=keyboard)
        else:
            await message.answer(text)

    async def _build_status_text(self) -> str:
        """Build the status text with cost and audit data."""
        from agent.tools.registry import registry

        heartbeat_status = "not configured"
        last_tick = "never"
        if self.heartbeat:
            heartbeat_status = "enabled" if self.heartbeat.is_enabled else "disabled"
            if self.heartbeat.last_tick:
                last_tick = self.heartbeat.last_tick.strftime("%Y-%m-%d %H:%M:%S")

        tools_count = len(registry.list_tools())
        sessions_count = self.session_store.active_count

        lines = [
            "Agent Status\n",
            f"Heartbeat: {heartbeat_status}",
            f"Last tick: {last_tick}\n",
            f"Tools: {tools_count} registered",
            f"Sessions: {sessions_count} active",
        ]

        # Cost summary
        try:
            if self.cost_tracker:
                stats = self.cost_tracker.get_stats("day")
                total_cost = stats.get("total_cost", 0)
                total_calls = stats.get("total_calls", 0)
                tokens = stats.get("total_tokens", {})
                total_tokens = tokens.get("input", 0) + tokens.get("output", 0)
                token_display = f"{total_tokens / 1000:.1f}K" if total_tokens else "0"
                lines.append(
                    f"\nToday: {total_calls} calls | ${total_cost:.2f} | {token_display} tokens"
                )
        except Exception:
            pass

        # Audit summary
        try:
            if self.audit_log:
                audit_stats = await self.audit_log.get_stats()
                tool_calls = audit_stats.get("total_calls", 0)
                success_rate = audit_stats.get("success_rate", 0) * 100
                avg_ms = audit_stats.get("avg_duration_ms", 0)
                if tool_calls > 0:
                    lines.append(
                        f"Tools: {tool_calls} calls | {success_rate:.1f}% success | avg {avg_ms}ms"
                    )
        except Exception:
            pass

        return "\n".join(lines)

    async def _cmd_tools(self, message: Any) -> None:
        """Handle /tools command — list registered tools."""
        if not self._check_message(message):
            return

        text, keyboard = await self._build_tools_text_and_keyboard()
        if text is None:
            await message.answer("No tools registered.")
            return

        if keyboard and AIOGRAM_AVAILABLE:
            await message.answer(text, reply_markup=keyboard)
        else:
            await message.answer(text)

    async def _build_tools_text_and_keyboard(
        self,
    ) -> tuple[str | None, Any]:
        """Build tools list text with usage counts and toggle keyboard."""
        from agent.tools.registry import registry

        tools = registry.list_tools()
        if not tools:
            return None, None

        # Get per-tool usage counts from audit log
        tools_used: dict[str, int] = {}
        try:
            if self.audit_log:
                audit_stats = await self.audit_log.get_stats()
                tools_used = audit_stats.get("tools_used", {})
        except Exception:
            pass

        tier_emoji = {"safe": "\U0001f7e2", "moderate": "\U0001f7e1", "dangerous": "\U0001f534"}
        lines: list[str] = ["Available tools:\n"]
        for t in tools:
            emoji = tier_emoji.get(t.tier.value, "\u2753")
            status = "on" if t.enabled else "off"
            count = tools_used.get(t.name, 0)
            usage = f" ({count}x)" if count > 0 else ""
            lines.append(f"{emoji} {t.name} [{status}] — {t.description}{usage}")

        text = "\n".join(lines)

        # Build toggle buttons for first 8 tools
        keyboard = None
        if AIOGRAM_AVAILABLE:
            buttons: list[list[Any]] = []
            row: list[Any] = []
            for t in tools[:8]:
                label = f"{'Disable' if t.enabled else 'Enable'} {t.name}"
                row.append(
                    InlineKeyboardButton(
                        text=label, callback_data=f"nav:tools:toggle:{t.name}",
                    )
                )
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)
            if buttons:
                keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

        return text, keyboard

    async def _cmd_pause(self, message: Any) -> None:
        """Handle /pause command."""
        if not self._check_message(message):
            return
        self.pause()
        await message.answer("Message processing paused. Use /resume to continue.")

    async def _cmd_resume(self, message: Any) -> None:
        """Handle /resume command."""
        if not self._check_message(message):
            return
        self.resume()
        await message.answer("Message processing resumed.")

    async def _cmd_mute(self, message: Any) -> None:
        """Handle /mute command — disable heartbeat."""
        if not self._check_message(message):
            return
        if self.heartbeat:
            self.heartbeat.disable()
            await message.answer("Heartbeat muted.")
        else:
            await message.answer("Heartbeat is not configured.")

    async def _cmd_unmute(self, message: Any) -> None:
        """Handle /unmute command — enable heartbeat."""
        if not self._check_message(message):
            return
        if self.heartbeat:
            self.heartbeat.enable()
            await message.answer("Heartbeat unmuted.")
        else:
            await message.answer("Heartbeat is not configured.")

    async def _cmd_soul(self, message: Any) -> None:
        """Handle /soul command — view or edit soul.md.

        /soul — display current soul.md content
        /soul edit <text> — overwrite soul.md with new content
        """
        if not self._check_message(message):
            return

        text = message.text or ""
        parts = text.split(maxsplit=2)

        soul_path = Path("soul.md")

        if len(parts) >= 3 and parts[1].lower() == "edit":
            # /soul edit <new content>
            new_content = parts[2]
            try:
                await asyncio.to_thread(
                    soul_path.write_text, new_content, encoding="utf-8"
                )
                await message.answer("soul.md updated successfully.")
            except OSError as e:
                await message.answer(f"Failed to update soul.md: {e}")
            return

        # /soul — show current content
        exists = await asyncio.to_thread(soul_path.exists)
        if exists:
            try:
                content = await asyncio.to_thread(
                    soul_path.read_text, encoding="utf-8"
                )
                if len(content) > _TG_MAX_LENGTH - 50:
                    content = content[: _TG_MAX_LENGTH - 50] + "\n...(truncated)"
                await message.answer(f"Current soul.md:\n\n{content}")
            except OSError as e:
                await message.answer(f"Failed to read soul.md: {e}")
        else:
            await message.answer(
                "soul.md not found. Use /soul edit <text> to create it."
            )

    async def _cmd_backend(self, message: Any) -> None:
        """Handle /backend command — view or switch LLM backend.

        /backend — show current backend
        /backend litellm — switch to litellm
        /backend claude-sdk — switch to claude-sdk
        """
        if not self._check_message(message):
            return

        from agent.config import get_config, update_config_section

        config = get_config()
        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            await message.answer(
                f"Current backend: {config.models.backend}\n\n"
                "Usage: /backend <litellm|claude-sdk>"
            )
            return

        new_backend = parts[1].strip().lower()
        valid = ("litellm", "claude-sdk")
        if new_backend not in valid:
            await message.answer(
                f"Invalid backend '{new_backend}'. Choose: {', '.join(valid)}"
            )
            return

        try:
            await asyncio.to_thread(
                update_config_section, "models", {"backend": new_backend}
            )
            await message.answer(f"Backend switched to: {new_backend}")
        except Exception as e:
            await message.answer(f"Failed to switch backend: {e}")

    async def _cmd_workdir(self, message: Any) -> None:
        """Handle /workdir command — view or change working directory.

        /workdir — show current working directory
        /workdir <path> — change working directory
        """
        if not self._check_message(message):
            return

        from agent.config import get_config, update_config_section

        config = get_config()
        text = message.text or ""
        parts = text.split(maxsplit=1)

        if len(parts) < 2:
            wd = config.models.claude_sdk.working_dir
            await message.answer(
                f"Current working directory: {wd}\n\n"
                "Usage: /workdir <path>"
            )
            return

        new_dir = parts[1].strip()
        resolved = await asyncio.to_thread(
            lambda: Path(new_dir).expanduser().resolve()
        )
        if not await asyncio.to_thread(resolved.is_dir):
            await message.answer(f"Directory not found: {resolved}")
            return

        try:
            await asyncio.to_thread(
                update_config_section,
                "models",
                {"claude_sdk": {"working_dir": str(resolved)}},
            )
            await message.answer(f"Working directory changed to: {resolved}")
        except Exception as e:
            await message.answer(f"Failed to change working directory: {e}")

    async def _cmd_new(self, message: Any) -> None:
        """Handle /new command — start a fresh conversation.

        Clears the current SDK session so the next message
        begins a new conversation without prior context.
        """
        if not self._check_message(message):
            return

        user_id = str(message.from_user.id)
        session_id = self._make_session_id(user_id)

        # Disconnect persistent SDK client and clear session
        if self.sdk_service is not None:
            from agent.llm.claude_sdk import ClaudeSDKService
            sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
            await sdk.disconnect_client(session_id)

        session = await self.session_store.get(session_id)
        if session:
            session.metadata.pop("sdk_session_id", None)
            session.clear()
            logger.info("session_cleared", user_id=user_id)

        await message.answer(
            "New conversation started. "
            "Your next message will begin a fresh session."
        )

    async def _cmd_session(self, message: Any) -> None:
        """Handle /session command — show current session info."""
        if not self._check_message(message):
            return

        user_id = str(message.from_user.id)
        session_id = self._make_session_id(user_id)

        session = await self.session_store.get(session_id)
        sdk_sid = session.metadata.get("sdk_session_id") if session else None

        backend = "claude-sdk" if self.sdk_service else "litellm"
        status = "active" if sdk_sid else "none"

        lines = [
            "Session Info\n",
            f"Backend: {backend}",
            f"Agent session: {session_id}",
            f"SDK session: {sdk_sid or 'none (next message starts new)'}",
            f"Status: {status}",
        ]

        if self.sdk_service:
            from agent.llm.claude_sdk import ClaudeSDKService

            sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
            task_status = sdk.get_status(session_id)
            lines.append(f"Task status: {task_status}")

        # Add message/token info from session
        if session:
            msg_count = getattr(session, "message_count", 0)
            tokens_in = getattr(session, "tokens_in", 0)
            tokens_out = getattr(session, "tokens_out", 0)
            total_tokens = tokens_in + tokens_out
            if msg_count or total_tokens:
                lines.append("")
            if msg_count:
                lines.append(f"Messages: {msg_count}")
            if total_tokens:
                lines.append(
                    f"Tokens: {total_tokens:,} (in: {tokens_in:,} + out: {tokens_out:,})"
                )

        if AIOGRAM_AVAILABLE:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Clear Session", callback_data="nav:session:clear",
                    ),
                    InlineKeyboardButton(
                        text="New Session", callback_data="nav:session:new",
                    ),
                ],
            ])
            await message.answer("\n".join(lines), reply_markup=keyboard)
        else:
            await message.answer("\n".join(lines))

    async def _cmd_remind(self, message: Any) -> None:
        """Handle /remind command — set a reminder.

        Usage: /remind <delay> <description>
        Examples:
            /remind 5m Check the deployment
            /remind 1h Review the pull request
            /remind 30s Test notification
        """
        if not self._check_message(message):
            return

        if not self.scheduler:
            await message.answer("Scheduler is not configured.")
            return

        text = message.text or ""
        parts = text.split(maxsplit=2)

        if len(parts) < 3:
            await message.answer(
                "Usage: /remind <delay> <description>\n\n"
                "Examples:\n"
                "/remind 5m Check the deployment\n"
                "/remind 1h30m Review the pull request\n"
                "/remind 2h Call the client\n"
                "/remind 45s Quick test\n\n"
                "Any combination works: s=seconds, m=minutes, h=hours, d=days\n"
                "e.g. 90s, 15m, 2h, 1d, 1h45m, 3h30m"
            )
            return

        delay_str = parts[1]
        description = parts[2]
        user_id = str(message.from_user.id)

        from agent.tools.builtins.scheduler import _parse_delay

        delta = _parse_delay(delay_str)
        if delta is None:
            await message.answer(
                f"Could not parse delay '{delay_str}'. "
                "Use formats like 5m, 1h, 30s, 2h30m."
            )
            return

        if delta.total_seconds() < 10:
            await message.answer("Reminder delay must be at least 10 seconds.")
            return

        from datetime import datetime

        run_at = datetime.now() + delta
        task = await self.scheduler.add_reminder(
            description=description,
            run_at=run_at,
            channel="telegram",
            user_id=user_id,
        )

        await message.answer(
            f"Reminder set (id={task.id}).\n"
            f"I'll remind you about \"{description}\" at "
            f"{run_at.strftime('%H:%M:%S')} (in {delay_str})."
        )

    async def _cmd_reminders(self, message: Any) -> None:
        """Handle /reminders command — list pending reminders."""
        if not self._check_message(message):
            return

        if not self.scheduler:
            await message.answer("Scheduler is not configured.")
            return

        tasks = self.scheduler.list_tasks()
        if not tasks:
            await message.answer("No scheduled reminders.")
            return

        lines: list[str] = ["Scheduled reminders:\n"]
        for t in tasks:
            status_icon = {
                "pending": "\u23f3", "running": "\U0001f504",
                "completed": "\u2705", "failed": "\u274c",
            }
            icon = status_icon.get(t.status, "\u2753")
            time_info = (
                t.next_run.strftime("%H:%M:%S") if t.next_run else t.schedule
            )
            lines.append(f"{icon} [{t.id}] {t.description} @ {time_info}")

        await message.answer("\n".join(lines))

    def _get_shortcuts(self) -> list[dict[str, str]]:
        """Resolve prompt shortcuts — merge defaults with config overrides."""
        from agent.config import get_config

        try:
            cfg = get_config()
            config_shortcuts = [
                {"alias": s.alias, "template": s.template, "description": s.description}
                for s in cfg.prompts.shortcuts
            ]
        except Exception:
            config_shortcuts = []

        # Config overrides defaults by alias
        merged: dict[str, dict[str, str]] = {}
        for s in _DEFAULT_SHORTCUTS:
            merged[s["alias"]] = s
        for s in config_shortcuts:
            merged[s["alias"]] = s
        return list(merged.values())

    async def _cmd_run(self, message: Any) -> None:
        """Handle /run command — run a prompt shortcut.

        Usage:
            /run list           — show all shortcuts
            /run <alias> [args] — run a shortcut
        """
        if not self._check_message(message):
            return

        if self._paused:
            await message.answer("I'm currently paused. Use /resume to re-enable me.")
            return

        text = (message.text or "").strip()
        parts = text.split(maxsplit=2)

        shortcuts = self._get_shortcuts()

        # /run (no args) or /run list
        if len(parts) < 2 or parts[1].lower() == "list":
            lines = ["Prompt shortcuts:\n"]
            for s in shortcuts:
                desc = f" — {s['description']}" if s.get("description") else ""
                lines.append(f"  /run {s['alias']}{desc}")
            lines.append("\nUsage: /run <shortcut> [args]")
            await message.answer("\n".join(lines))
            return

        alias = parts[1].lower()
        args = parts[2] if len(parts) > 2 else ""

        # Find matching shortcut
        shortcut = next((s for s in shortcuts if s["alias"] == alias), None)
        if not shortcut:
            await message.answer(
                f"Unknown shortcut: {alias}\n"
                "Use /run list to see available shortcuts."
            )
            return

        # Interpolate {args}
        prompt = shortcut["template"]
        if "{args}" in prompt:
            if not args:
                await message.answer(
                    f"Shortcut '{alias}' requires arguments.\n"
                    f"Usage: /run {alias} <args>"
                )
                return
            prompt = prompt.replace("{args}", args)

        # Process through agent loop (same pattern as _handle_text)
        user_id = message.from_user.id

        from agent.tools.builtins.scheduler import set_context
        set_context(channel="telegram", user_id=str(user_id))

        from agent.tools.builtins.send_file import set_file_send_context
        set_file_send_context(channel="telegram", user_id=str(user_id))
        from agent.tools.builtins.telegram_post import set_telegram_post_context
        set_telegram_post_context(channel="telegram", user_id=str(user_id))

        agent_loop, session_store = self._resolve_components(str(user_id))
        session_id = self._make_session_id(str(user_id))
        session = await session_store.get_or_create(
            session_id=session_id, channel="telegram"
        )

        status_message = None
        try:
            status_message = await message.answer("\u23f3 Processing\u2026")
        except Exception:
            pass

        # Dispatch to background
        task = asyncio.create_task(
            self._run_text_task(
                user_text=prompt,
                session=session,
                agent_loop=agent_loop,
                status_message=status_message,
                user_id=str(user_id),
                message=message,
            ),
            name=f"run:{user_id}:{alias}",
        )
        self._register_background_task(str(user_id), task, f"/run {alias}")

    async def _cmd_tasks(self, message: Any) -> None:
        """Handle /tasks — show running background tasks."""
        if not self._check_message(message):
            return

        active = self._get_active_tasks()
        if not active:
            await message.answer("No tasks running right now. I'm free to chat!")
            return

        lines = [f"\u2699\ufe0f <b>Running tasks ({len(active)}):</b>\n"]
        for uid, desc in active:
            lines.append(f"\u2022 {desc}")
        await message.answer("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # Media handlers
    # ------------------------------------------------------------------

    async def _handle_voice(self, message: Any) -> None:
        """Handle voice messages — download, transcribe, dispatch to background."""
        if not self._check_message(message):
            return
        if self._paused:
            await message.answer("I'm currently paused. Use /resume to re-enable me.")
            return

        user_id = message.from_user.id

        # Set file-send context for voice messages
        from agent.tools.builtins.send_file import set_file_send_context
        set_file_send_context(channel="telegram", user_id=str(user_id))
        from agent.tools.builtins.telegram_post import set_telegram_post_context
        set_telegram_post_context(channel="telegram", user_id=str(user_id))

        try:
            file = await self._bot.get_file(message.voice.file_id)
            from io import BytesIO

            buf = BytesIO()
            await self._bot.download_file(file.file_path, buf)
            audio_bytes = buf.getvalue()

            # Resolve workspace-specific components
            agent_loop, session_store = self._resolve_components(str(user_id))

            session_id = self._make_session_id(str(user_id))
            session = await session_store.get_or_create(
                session_id=session_id, channel="telegram"
            )

            # Dispatch to background
            task = asyncio.create_task(
                self._run_voice_task(
                    audio_bytes=audio_bytes,
                    session=session,
                    agent_loop=agent_loop,
                    user_id=str(user_id),
                    message=message,
                ),
                name=f"voice:{user_id}",
            )
            self._register_background_task(str(user_id), task, "[voice message]")

        except Exception as e:
            logger.error("voice_handle_error", error=str(e), user_id=user_id)
            await message.answer(
                "Sorry, I couldn't process your voice message. "
                "Try sending it as text instead."
            )

    async def _run_voice_task(
        self,
        audio_bytes: bytes,
        session: Any,
        agent_loop: Any,
        user_id: str,
        message: Any,
    ) -> None:
        """Background coroutine that processes a voice message."""
        status_message = None
        try:
            response_text: str

            # Standalone STT path (whisper, deepgram, etc.)
            if self.voice_pipeline and not self.voice_pipeline.is_llm_native():
                stt_result = await self.voice_pipeline.transcribe(
                    audio_bytes, "audio/ogg",
                )

                # Show transcription to user
                if self.voice_pipeline.config.voice_transcription_prefix:
                    transcription_msg = f"\U0001f3a4 *Transcription:* {stt_result.text}"
                    await message.reply(transcription_msg, parse_mode="Markdown")

                # Send status message and process via agent loop or SDK
                try:
                    status_message = await message.answer("\u23f3 Processing\u2026")
                except Exception:
                    pass

                response_text = await self._process_via_sdk_or_loop(
                    stt_result.text, session, agent_loop,
                    status_message=status_message,
                    user_id=user_id,
                )

            else:
                # LLM native: send audio directly
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                multimodal_content: list[dict[str, Any]] = [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "ogg"},
                    },
                ]

                from agent.core.session import Message

                session.add_message(Message(role="user", content="[Voice message]"))

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": agent_loop.system_prompt},
                ]
                messages.extend(session.get_history()[:-1])
                messages.append({"role": "user", "content": multimodal_content})

                llm_response = await agent_loop.llm.completion(messages=messages)
                response_text = llm_response.content

                session.add_message(
                    Message(role="assistant", content=response_text)
                )

            # Voice reply if enabled
            if self.voice_pipeline and self.voice_pipeline.should_voice_reply(
                "telegram",
            ):
                tts_result = await self.voice_pipeline.synthesize(response_text)
                if tts_result and AIOGRAM_AVAILABLE:
                    from aiogram.types import BufferedInputFile

                    voice_file = BufferedInputFile(
                        file=tts_result.audio_data,
                        filename="response.ogg",
                    )
                    if status_message is not None:
                        with contextlib.suppress(Exception):
                            await status_message.delete()
                    await message.reply_voice(
                        voice=voice_file,
                        duration=int(tts_result.duration_seconds),
                    )
                    if len(response_text) > 200:
                        short = response_text[:200] + "..."
                        await message.reply(
                            f"\U0001f4dd _{short}_", parse_mode="Markdown",
                        )
                    return

            # Text-only reply
            await self._replace_status_with_response(
                status_message, user_id, response_text
            )

        except Exception as e:
            logger.error("voice_task_error", error=str(e), user_id=user_id)
            with contextlib.suppress(Exception):
                await message.answer(
                    "Sorry, I couldn't process your voice message. "
                    "Try sending it as text instead."
                )
        finally:
            self._unregister_background_task(user_id)

    async def _handle_photo(self, message: Any) -> None:
        """Handle photo messages — download highest-res, base64-encode, send to LLM."""
        if not self._check_message(message):
            return
        if self._paused:
            await message.answer("I'm currently paused. Use /resume to re-enable me.")
            return

        user_id = message.from_user.id
        await self.send_typing(str(user_id))

        try:
            # Get highest resolution photo
            photo = message.photo[-1]
            file = await self._bot.get_file(photo.file_id)
            from io import BytesIO

            buf = BytesIO()
            await self._bot.download_file(file.file_path, buf)
            image_bytes = buf.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            caption = message.caption or ""

            # Build multimodal message with image_url content block
            multimodal_content: list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                    },
                },
            ]
            if caption:
                multimodal_content.append({"type": "text", "text": caption})

            session_id = self._make_session_id(str(user_id))
            session = await self.session_store.get_or_create(
                session_id=session_id, channel="telegram"
            )

            from agent.core.session import Message

            placeholder = f"[Photo]{f': {caption}' if caption else ''}"
            session.add_message(Message(role="user", content=placeholder))

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self.agent_loop.system_prompt},
            ]
            messages.extend(session.get_history()[:-1])
            messages.append({"role": "user", "content": multimodal_content})

            response = await self.agent_loop.llm.completion(messages=messages)

            session.add_message(
                Message(role="assistant", content=response.content)
            )

            await self.send_message(
                OutgoingMessage(
                    content=response.content,
                    channel_user_id=str(user_id),
                    parse_mode="Markdown",
                )
            )

        except Exception as e:
            logger.error("photo_handle_error", error=str(e), user_id=user_id)
            await message.answer(
                "Sorry, I couldn't process your photo. "
                "Try describing it in text instead."
            )

    async def _handle_document(self, message: Any) -> None:
        """Handle document uploads — save to data/uploads/, dispatch to background."""
        if not self._check_message(message):
            return
        if self._paused:
            await message.answer("I'm currently paused. Use /resume to re-enable me.")
            return

        user_id = message.from_user.id

        try:
            doc = message.document
            file = await self._bot.get_file(doc.file_id)
            filename = doc.file_name or f"document_{doc.file_id}"
            mime_type = doc.mime_type or "application/octet-stream"
            file_size = doc.file_size or 0

            # Save to uploads directory
            os.makedirs(_UPLOAD_DIR, exist_ok=True)
            save_path = _UPLOAD_DIR / filename

            from io import BytesIO

            buf = BytesIO()
            await self._bot.download_file(file.file_path, buf)
            save_path.write_bytes(buf.getvalue())

            caption = message.caption or ""
            description = (
                f"User uploaded a file:\n"
                f"Name: {filename}\n"
                f"Size: {file_size} bytes\n"
                f"Type: {mime_type}\n"
                f"Saved to: {save_path}"
            )
            if caption:
                description += f"\nCaption: {caption}"

            session_id = self._make_session_id(str(user_id))
            session = await self.session_store.get_or_create(
                session_id=session_id, channel="telegram"
            )

            status_message = None
            try:
                status_message = await message.answer(
                    "\u23f3 Processing document\u2026",
                )
            except Exception:
                pass

            # Dispatch to background
            task = asyncio.create_task(
                self._run_document_task(
                    description=description,
                    session=session,
                    status_message=status_message,
                    user_id=str(user_id),
                    message=message,
                ),
                name=f"doc:{user_id}:{filename[:30]}",
            )
            self._register_background_task(
                str(user_id), task, f"[document: {filename}]",
            )

        except Exception as e:
            logger.error("document_handle_error", error=str(e), user_id=user_id)
            await message.answer("Sorry, I couldn't process your document.")

    async def _run_document_task(
        self,
        description: str,
        session: Any,
        status_message: Any | None,
        user_id: str,
        message: Any,
    ) -> None:
        """Background coroutine that processes a document upload."""
        try:
            response_text = await self._process_via_sdk_or_loop(
                description, session, self.agent_loop,
                status_message=status_message,
                user_id=user_id,
            )

            await self._replace_status_with_response(
                status_message, user_id, response_text
            )

        except Exception as e:
            logger.error("document_task_error", error=str(e), user_id=user_id)
            if status_message:
                with contextlib.suppress(Exception):
                    await status_message.edit_text(
                        "Sorry, I couldn't process your document."
                    )
            else:
                with contextlib.suppress(Exception):
                    await message.answer(
                        "Sorry, I couldn't process your document."
                    )
        finally:
            self._unregister_background_task(user_id)

    # ------------------------------------------------------------------
    # Callback dispatcher
    # ------------------------------------------------------------------

    async def _handle_callback(self, callback: Any) -> None:
        """Route inline button callbacks to the correct handler."""
        data = callback.data or ""
        if data.startswith("nav:"):
            await self._handle_nav_callback(callback)
        else:
            await self._handle_approval_callback(callback)

    async def _handle_approval_callback(self, callback: Any) -> None:
        """Handle inline button callbacks for tool approval.

        Callback data format: 'approve:{id}', 'deny:{id}', 'approve_session:{id}:{tool}'
        """
        data = callback.data or ""

        if data.startswith("approve_session:"):
            parts = data.split(":", 2)
            if len(parts) == 3:
                request_id = parts[1]
                fut = self._approval_futures.get(request_id)
                if fut and not fut.done():
                    fut.set_result(True)
                await callback.message.edit_text(
                    f"{callback.message.text}\n\nApproved (session)"
                )
        elif data.startswith("approve:"):
            request_id = data.split(":", 1)[1]
            fut = self._approval_futures.get(request_id)
            if fut and not fut.done():
                fut.set_result(True)
            await callback.message.edit_text(
                f"{callback.message.text}\n\nApproved"
            )
        elif data.startswith("deny:"):
            request_id = data.split(":", 1)[1]
            fut = self._approval_futures.get(request_id)
            if fut and not fut.done():
                fut.set_result(False)
            await callback.message.edit_text(
                f"{callback.message.text}\n\nDenied"
            )

        await callback.answer()

    async def _handle_nav_callback(self, callback: Any) -> None:
        """Handle navigation inline button callbacks.

        Routes by callback data prefix:
        - nav:status:costs / nav:status:health / nav:status:audit
        - nav:tools:toggle:<name>
        - nav:session:clear / nav:session:new
        """
        data = callback.data or ""

        try:
            if data == "nav:status:costs":
                await self._nav_status_costs(callback)
            elif data == "nav:status:health":
                await self._nav_status_health(callback)
            elif data == "nav:status:audit":
                await self._nav_status_audit(callback)
            elif data.startswith("nav:tools:toggle:"):
                tool_name = data.split(":", 3)[3]
                await self._nav_tools_toggle(callback, tool_name)
            elif data == "nav:session:clear":
                await self._nav_session_clear(callback)
            elif data == "nav:session:new":
                await self._nav_session_new(callback)
            elif data == "nav:back:status":
                text = await self._build_status_text()
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(text="Costs", callback_data="nav:status:costs"),
                        InlineKeyboardButton(text="Health", callback_data="nav:status:health"),
                        InlineKeyboardButton(text="Audit", callback_data="nav:status:audit"),
                    ],
                ])
                await callback.message.edit_text(text, reply_markup=keyboard)
            elif data == "nav:back:tools":
                text, keyboard = await self._build_tools_text_and_keyboard()
                if text:
                    await callback.message.edit_text(
                        text, reply_markup=keyboard,
                    )
            elif data == "nav:back:session":
                await callback.message.edit_text(
                    "Use /session to view current session info.",
                )
        except Exception as e:
            logger.warning("nav_callback_error", error=str(e), data=data)

        await callback.answer()

    async def _nav_status_costs(self, callback: Any) -> None:
        """Show cost breakdown detail view."""
        if not self.cost_tracker:
            await callback.message.edit_text("Cost tracker not available.")
            return

        stats = self.cost_tracker.get_stats("day")
        total_cost = stats.get("total_cost", 0)
        total_calls = stats.get("total_calls", 0)
        tokens = stats.get("total_tokens", {})

        lines = [
            "Cost Details (Today)\n",
            f"Total cost: ${total_cost:.4f}",
            f"Total calls: {total_calls}",
            f"Tokens in: {tokens.get('input', 0):,}",
            f"Tokens out: {tokens.get('output', 0):,}",
        ]

        by_model = stats.get("by_model", [])
        if by_model:
            lines.append("\nBy model:")
            for m in by_model:
                lines.append(
                    f"  {m['model']}: ${m['cost']:.4f} ({m['percentage']:.0f}%)"
                )

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Back", callback_data="nav:back:status")],
        ])
        await callback.message.edit_text("\n".join(lines), reply_markup=keyboard)

    async def _nav_status_health(self, callback: Any) -> None:
        """Show health check summary."""
        from agent.config import get_config
        from agent.core.doctor import run_all_checks

        config = get_config()
        checks = await run_all_checks(config)

        status_icon = {"pass": "\u2705", "warn": "\u26a0\ufe0f", "fail": "\u274c"}
        lines = ["Health Checks\n"]
        for check in checks:
            icon = status_icon.get(check.status, "\u2753")
            lines.append(f"{icon} {check.name}: {check.message}")

        pass_count = sum(1 for c in checks if c.status == "pass")
        warn_count = sum(1 for c in checks if c.status == "warn")
        fail_count = sum(1 for c in checks if c.status == "fail")
        lines.append(f"\n{pass_count} pass | {warn_count} warn | {fail_count} fail")

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Back", callback_data="nav:back:status")],
        ])
        # Truncate if too long for Telegram
        text = "\n".join(lines)
        if len(text) > _TG_MAX_LENGTH - 100:
            text = text[:_TG_MAX_LENGTH - 150] + "\n\n... (truncated)"
        await callback.message.edit_text(text, reply_markup=keyboard)

    async def _nav_status_audit(self, callback: Any) -> None:
        """Show audit log summary."""
        if not self.audit_log:
            await callback.message.edit_text("Audit log not available.")
            return

        stats = await self.audit_log.get_stats()
        total = stats.get("total_calls", 0)
        success = stats.get("success_count", 0)
        errors = stats.get("error_count", 0)
        rate = stats.get("success_rate", 0) * 100
        avg_ms = stats.get("avg_duration_ms", 0)

        lines = [
            "Audit Summary\n",
            f"Total tool calls: {total}",
            f"Success: {success} | Errors: {errors}",
            f"Success rate: {rate:.1f}%",
            f"Avg duration: {avg_ms}ms",
        ]

        tools_used = stats.get("tools_used", {})
        if tools_used:
            lines.append("\nPer-tool breakdown:")
            for name, count in sorted(tools_used.items(), key=lambda x: -x[1]):
                lines.append(f"  {name}: {count}x")

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Back", callback_data="nav:back:status")],
        ])
        text = "\n".join(lines)
        if len(text) > _TG_MAX_LENGTH - 100:
            text = text[:_TG_MAX_LENGTH - 150] + "\n\n... (truncated)"
        await callback.message.edit_text(text, reply_markup=keyboard)

    async def _nav_tools_toggle(self, callback: Any, tool_name: str) -> None:
        """Toggle a tool on/off and refresh the tools view."""
        from agent.tools.registry import registry

        try:
            tool = next(
                (t for t in registry.list_tools() if t.name == tool_name), None
            )
            if not tool:
                await callback.message.edit_text(f"Tool '{tool_name}' not found.")
                return

            if tool.enabled:
                registry.disable_tool(tool_name)
                action = "disabled"
            else:
                registry.enable_tool(tool_name)
                action = "enabled"

            logger.info("tool_toggled_via_telegram", tool=tool_name, action=action)
        except Exception as e:
            logger.warning("tool_toggle_error", tool=tool_name, error=str(e))
            await callback.message.edit_text(f"Error toggling {tool_name}: {e}")
            return

        # Refresh tools view
        text, keyboard = await self._build_tools_text_and_keyboard()
        if text:
            await callback.message.edit_text(text, reply_markup=keyboard)

    async def _nav_session_clear(self, callback: Any) -> None:
        """Clear conversation history in the current session."""
        user_id = str(callback.from_user.id)
        session_id = self._make_session_id(user_id)

        session = await self.session_store.get(session_id)
        if session:
            session.clear()
            session.metadata.pop("sdk_session_id", None)

        await callback.message.edit_text("Session cleared. Your next message starts fresh.")

    async def _nav_session_new(self, callback: Any) -> None:
        """Create a new session."""
        user_id = str(callback.from_user.id)
        session_id = self._make_session_id(user_id)

        session = await self.session_store.get(session_id)
        if session:
            session.metadata.pop("sdk_session_id", None)

        await callback.message.edit_text(
            "New conversation started. Your next message begins a fresh session."
        )

    # ------------------------------------------------------------------
    # Text message handling
    # ------------------------------------------------------------------

    def set_workspace_components(
        self,
        workspace_name: str,
        agent_loop: AgentLoop,
        session_store: SessionStore,
    ) -> None:
        """Register workspace-specific agent loop and session store.

        Args:
            workspace_name: The workspace name.
            agent_loop: Agent loop for this workspace.
            session_store: Session store for this workspace.
        """
        self._workspace_agent_loops[workspace_name] = agent_loop
        self._workspace_session_stores[workspace_name] = session_store

    def _resolve_components(
        self, user_id: str, message_text: str = "",
    ) -> tuple[AgentLoop, SessionStore]:
        """Resolve the agent loop and session store for a user via routing.

        Falls back to the default agent_loop/session_store when no router
        is configured or no workspace-specific components are registered.

        Args:
            user_id: Telegram user ID as string.
            message_text: Message text for pattern matching (future use).

        Returns:
            Tuple of (agent_loop, session_store) to use.
        """
        if self.workspace_router:
            workspace = self.workspace_router.route(
                channel="telegram",
                user_id=user_id,
                message=message_text,
            )
            ws_name = workspace.name
            if ws_name in self._workspace_agent_loops:
                return (
                    self._workspace_agent_loops[ws_name],
                    self._workspace_session_stores.get(ws_name, self.session_store),
                )
        return self.agent_loop, self.session_store

    async def _try_extract_reminder(
        self, text: str, user_id: str,
    ) -> str | None:
        """Detect natural language reminder requests and schedule them.

        Handles many phrasings in English and Russian:
            "remind me in 5 minutes to check deployment"
            "remind me to check deployment in 5 minutes"
            "can you remind me in 1 hour about the meeting"
            "please remind me in 30m do laundry"
            "напомни через 5 минут проверить деплой"
            "напомни проверить деплой через 5 минут"

        Returns confirmation text if a reminder was created, None otherwise.
        """
        if not self.scheduler:
            return None

        import re
        from datetime import datetime

        from agent.tools.builtins.scheduler import _parse_delay

        lower = text.lower().strip()

        # Quick check: does this look like a reminder request at all?
        en_triggers = ("remind", "reminder", "alert me", "notify me", "wake me")
        ru_triggers = ("напомни", "напоминание", "напомнить")
        is_reminder = any(t in lower for t in en_triggers + ru_triggers)
        if not is_reminder:
            return None

        # Time pattern: matches "5m", "5 min", "5 minutes", "1 hour", "30 seconds", etc.
        time_unit_en = (
            r"(?:seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)"
        )
        time_pattern_en = rf"(\d+\s*{time_unit_en}(?:\s*\d+\s*{time_unit_en})*)"

        # Russian time units
        time_unit_ru = (
            r"(?:секунд[уы]?|сек|минут[уы]?|мин|час[аов]*|дн[яей]*|день)"
        )
        time_pattern_ru = rf"(\d+\s*{time_unit_ru})"

        time_pattern = rf"(?:{time_pattern_en}|{time_pattern_ru})"

        # Try to find a time expression anywhere in the message
        time_match = re.search(time_pattern, lower)
        if not time_match:
            return None

        delay_str = time_match.group(0).strip()
        delay_start = time_match.start()
        delay_end = time_match.end()

        # Russian time word mapping for the delay parser
        ru_time_map = {
            "секунд": "s", "секунды": "s", "секунду": "s", "сек": "s",
            "минут": "m", "минуты": "m", "минуту": "m", "мин": "m",
            "час": "h", "часа": "h", "часов": "h",
            "день": "d", "дня": "d", "дней": "d",
        }

        # Convert Russian time words to parseable format
        parsed_delay = delay_str
        for ru_word, en_unit in ru_time_map.items():
            if ru_word in parsed_delay:
                num_match = re.search(r"(\d+)\s*" + ru_word, parsed_delay)
                if num_match:
                    parsed_delay = num_match.group(1) + en_unit
                    break

        delta = _parse_delay(parsed_delay)
        if delta is None or delta.total_seconds() < 10:
            return None

        # Extract description: everything that's NOT the time expression
        # and NOT filler words (remind me, in, to, about, through, etc.)
        # Use the ORIGINAL text (not lowered) for the description
        before_time = text[:delay_start]
        after_time = text[delay_end:]

        # Clean filler words from both parts
        filler_en = re.compile(
            r"\b(?:please|can you|could you|remind|me|set|a|the|"
            r"reminder|alert|notify|wake|in|after|about|to|that)\b",
            re.IGNORECASE,
        )
        filler_ru = re.compile(
            r"\b(?:пожалуйста|можешь|напомни|мне|поставь|"
            r"напоминание|через|о|об|что|про)\b",
            re.IGNORECASE,
        )

        desc_parts = []
        for part in (before_time, after_time):
            cleaned = filler_en.sub(" ", part)
            cleaned = filler_ru.sub(" ", cleaned)
            cleaned = re.sub(r"[,:;\-]+", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                desc_parts.append(cleaned)

        description = " ".join(desc_parts).strip()
        if not description:
            description = "Reminder"

        run_at = datetime.now() + delta
        task = await self.scheduler.add_reminder(
            description=description,
            run_at=run_at,
            channel="telegram",
            user_id=user_id,
        )

        logger.info(
            "reminder_auto_detected",
            task_id=task.id,
            delay=parsed_delay,
            description=description,
            user_id=user_id,
        )

        return (
            f"Reminder set (id={task.id}). "
            f"I'll remind you about \"{description}\" at "
            f"{run_at.strftime('%H:%M:%S')}."
        )

    async def _process_via_sdk_or_loop(
        self,
        text: str,
        session: Any,
        agent_loop: Any,
        status_message: Any | None = None,
        user_id: str | None = None,
    ) -> str:
        """Route a message through Claude SDK or the LiteLLM agent loop.

        If status_message is provided, it will be edited with live status
        updates as the SDK processes the request (e.g. tool usage).

        Returns the response text.
        """
        if self.sdk_service is not None:
            from agent.llm.claude_sdk import ClaudeSDKService

            sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
            accumulated = ""
            sdk_session_id = session.metadata.get("sdk_session_id")
            last_status_update = 0.0
            self._had_approvals[user_id or ""] = False
            import time as _time

            # Permission callback: ask user via inline keyboard for dangerous tools
            on_permission = None
            if user_id and AIOGRAM_AVAILABLE:
                async def _on_permission(
                    tool_name: str, details: str, tool_input: dict[str, Any],
                ) -> bool:
                    import uuid

                    request_id = str(uuid.uuid4())[:8]
                    return await self.send_approval_request(
                        channel_user_id=user_id,
                        tool_name=tool_name,
                        arguments=tool_input,
                        request_id=request_id,
                    )

                on_permission = _on_permission

            async for event in sdk.run_task_stream(
                prompt=text,
                task_id=session.id,
                session_id=sdk_session_id,
                on_permission=on_permission,
            ):
                if event.type == "text":
                    # Only accumulate main agent text, not subagent output
                    if not (event.data and event.data.get("subagent")):
                        accumulated += event.content
                elif event.type == "tool_use":
                    # Update status message with current action
                    if status_message is not None:
                        now = _time.monotonic()
                        # Throttle edits to avoid Telegram rate limits
                        if now - last_status_update > 2.0:
                            tool_name = event.data.get("tool", "tool")
                            is_sub = event.data.get("subagent", False)
                            prefix = "\U0001f50d Researching" if is_sub else "\u2699\ufe0f Working"
                            status_text = f"{prefix}\u2026 using {tool_name}"
                            with contextlib.suppress(Exception):
                                await status_message.edit_text(status_text)
                            last_status_update = now
                elif event.type == "result":
                    sdk_sid = event.data.get("session_id")
                    if sdk_sid:
                        session.metadata["sdk_session_id"] = sdk_sid
                    # Prefer the result content over accumulated text events
                    # when available — the SDK sometimes sends the full answer
                    # only in the ResultMessage.result field.
                    if event.content and len(event.content) > len(accumulated):
                        accumulated = event.content
                elif event.type == "error":
                    raise RuntimeError(event.content)
            return accumulated or "[No response]"

        response = await agent_loop.process_message(
            text, session, trigger="user_message"
        )
        return response.content

    async def _handle_text(self, message: Any) -> None:
        """Process an incoming text message through the agent loop.

        Processing is dispatched to a background task so the bot stays
        responsive.  The user can send new messages while tasks run.
        """
        if not self._check_message(message):
            return

        user_id = message.from_user.id
        user_text = message.text

        if not user_text:
            return

        # Check if paused
        if self._paused:
            await message.answer("I'm currently paused. Use /resume to re-enable me.")
            return

        # Set scheduler context so set_reminder tool knows who to deliver to
        from agent.tools.builtins.scheduler import set_context
        set_context(channel="telegram", user_id=str(user_id))

        # Set file-send context so send_file tool knows who to deliver to
        from agent.tools.builtins.send_file import set_file_send_context
        set_file_send_context(channel="telegram", user_id=str(user_id))
        from agent.tools.builtins.telegram_post import set_telegram_post_context
        set_telegram_post_context(channel="telegram", user_id=str(user_id))

        # Auto-detect natural language reminder requests
        reminder_confirmation = await self._try_extract_reminder(
            user_text, str(user_id)
        )
        if reminder_confirmation:
            await message.answer(reminder_confirmation)
            return

        # Resolve workspace-specific components
        agent_loop, session_store = self._resolve_components(
            str(user_id), user_text
        )

        # Get or create session
        session_id = self._make_session_id(str(user_id))
        session = await session_store.get_or_create(
            session_id=session_id, channel="telegram"
        )

        # Emit incoming event
        await self.event_bus.emit(
            Events.MESSAGE_INCOMING,
            {
                "content": user_text,
                "channel": "telegram",
                "user_id": str(user_id),
                "session_id": session.id,
            },
        )

        # Send a placeholder message instead of typing indicator
        status_message = None
        try:
            status_message = await message.answer("\u23f3 Processing\u2026")
        except Exception:
            pass

        # Dispatch processing to a background task so the bot stays free
        task = asyncio.create_task(
            self._run_text_task(
                user_text=user_text,
                session=session,
                agent_loop=agent_loop,
                status_message=status_message,
                user_id=str(user_id),
                message=message,
            ),
            name=f"task:{user_id}:{user_text[:40]}",
        )
        self._register_background_task(str(user_id), task, user_text[:80])

    async def _run_text_task(
        self,
        user_text: str,
        session: Any,
        agent_loop: Any,
        status_message: Any | None,
        user_id: str,
        message: Any,
    ) -> None:
        """Background coroutine that processes a text message and delivers the response."""
        try:
            response_text = await self._process_via_sdk_or_loop(
                user_text, session, agent_loop,
                status_message=status_message,
                user_id=user_id,
            )

            # Deliver the response
            await self._replace_status_with_response(
                status_message, user_id, response_text
            )

        except Exception as e:
            logger.error(
                "telegram_handle_error",
                error=str(e),
                user_id=user_id,
            )
            if status_message:
                with contextlib.suppress(Exception):
                    await status_message.edit_text(
                        "Sorry, something went wrong processing your message."
                    )
            else:
                with contextlib.suppress(Exception):
                    await message.answer(
                        "Sorry, something went wrong processing your message."
                    )
        finally:
            self._unregister_background_task(user_id)

    # ------------------------------------------------------------------
    # Background task management
    # ------------------------------------------------------------------

    def _register_background_task(
        self, user_id: str, task: asyncio.Task[Any], description: str,
    ) -> None:
        """Register a background task for a user."""
        if user_id not in self._background_tasks:
            self._background_tasks[user_id] = []
        self._background_tasks[user_id].append((task, description))

    def _unregister_background_task(self, user_id: str) -> None:
        """Remove completed tasks for a user.

        Called from within a task's ``finally`` block, where the calling
        task is still technically "running" (``t.done()`` is False).  We
        also exclude the *current* asyncio task so it gets cleaned up.
        """
        if user_id not in self._background_tasks:
            return
        current = asyncio.current_task()
        self._background_tasks[user_id] = [
            (t, d) for t, d in self._background_tasks[user_id]
            if not t.done() and t is not current
        ]
        if not self._background_tasks[user_id]:
            del self._background_tasks[user_id]

    def _get_active_tasks(self, user_id: str | None = None) -> list[tuple[str, str]]:
        """Get active background tasks as (user_id, description) pairs."""
        result: list[tuple[str, str]] = []
        targets = (
            {user_id: self._background_tasks.get(user_id, [])}
            if user_id
            else self._background_tasks
        )
        for uid, tasks in targets.items():
            for task, desc in tasks:
                if not task.done():
                    result.append((uid, desc))
        return result

    # ------------------------------------------------------------------
    # Status message helpers
    # ------------------------------------------------------------------

    async def _replace_status_with_response(
        self,
        status_message: Any | None,
        channel_user_id: str,
        response_text: str,
    ) -> None:
        """Replace the placeholder status message with the final response.

        If approval prompts were shown during processing, the status message
        is now far up in the chat.  In that case we delete it and send a fresh
        message at the bottom so the user sees the answer immediately.

        Args:
            status_message: The placeholder message to replace (may be None).
            channel_user_id: Telegram user/chat ID as string.
            response_text: The complete response text from the agent.
        """
        if not self._bot:
            return

        if not response_text:
            # Clean up the stale status message instead of leaving it
            if status_message is not None:
                with contextlib.suppress(Exception):
                    await status_message.edit_text(
                        "No response was generated. Please try again."
                    )
            return

        chunks = self._split_message(response_text)
        chat_id = int(channel_user_id)

        # If approvals happened, the status message is buried under approval
        # messages.  Delete it and send a fresh message at the bottom.
        if status_message is not None and self._had_approvals.get(channel_user_id):
            with contextlib.suppress(Exception):
                await status_message.delete()
            status_message = None  # fall through to "no status_message" path

        if status_message is not None:
            # Edit the first chunk into the status message (still at bottom)
            try:
                await status_message.edit_text(
                    chunks[0], parse_mode="Markdown",
                )
            except Exception:
                # Fallback: try without parse_mode
                try:
                    await status_message.edit_text(chunks[0])
                except Exception as e:
                    logger.debug("status_edit_failed", error=str(e))
                    # Last resort: delete status and send fresh
                    with contextlib.suppress(Exception):
                        await status_message.delete()
                    await self.send_message(
                        OutgoingMessage(
                            content=chunks[0],
                            channel_user_id=channel_user_id,
                            parse_mode="Markdown",
                        )
                    )

            # Send remaining chunks as new messages
            for chunk in chunks[1:]:
                try:
                    await self._bot.send_message(
                        chat_id=chat_id, text=chunk, parse_mode="Markdown",
                    )
                except Exception:
                    try:
                        await self._bot.send_message(chat_id=chat_id, text=chunk)
                    except Exception as e:
                        logger.debug("chunk_send_failed", error=str(e))
        else:
            # No status message — send normally
            await self.send_streamed_response(channel_user_id, response_text)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_message(self, message: OutgoingMessage) -> None:
        """Send a text message to a Telegram user.

        Splits long messages and falls back to plain text on parse errors.

        Args:
            message: The outgoing message to send.
        """
        if not self._bot:
            return

        chat_id = int(message.channel_user_id)
        chunks = self._split_message(message.content)

        for chunk in chunks:
            try:
                await self._bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=message.parse_mode,
                )
            except Exception:
                # Fallback: send without parse_mode
                try:
                    await self._bot.send_message(chat_id=chat_id, text=chunk)
                except Exception as e:
                    logger.error(
                        "telegram_send_failed",
                        error=str(e),
                        chat_id=chat_id,
                    )

    async def send_streamed_response(
        self, channel_user_id: str, full_text: str
    ) -> None:
        """Send a response with streaming simulation for long messages.

        Short responses (<500 chars) are sent directly.
        Long responses: send a preview chunk with cursor, wait, then edit with full text.

        Args:
            channel_user_id: Telegram user/chat ID as string.
            full_text: The complete response text.
        """
        if not self._bot:
            return

        if len(full_text) <= _STREAM_THRESHOLD:
            await self.send_message(
                OutgoingMessage(
                    content=full_text,
                    channel_user_id=channel_user_id,
                    parse_mode="Markdown",
                )
            )
            return

        chat_id = int(channel_user_id)

        # Send preview chunk with typing cursor
        preview = full_text[:_STREAM_THRESHOLD] + "\u258c"
        try:
            sent = await self._bot.send_message(chat_id=chat_id, text=preview)
        except Exception:
            # Fallback to normal send
            await self.send_message(
                OutgoingMessage(
                    content=full_text,
                    channel_user_id=channel_user_id,
                    parse_mode="Markdown",
                )
            )
            return

        await asyncio.sleep(1)

        # Edit with full text (split if needed)
        chunks = self._split_message(full_text)
        try:
            await sent.edit_text(chunks[0], parse_mode="Markdown")
        except Exception:
            try:
                await sent.edit_text(chunks[0])
            except Exception as e:
                logger.debug("stream_edit_failed", error=str(e))

        # Send remaining chunks as new messages
        for chunk in chunks[1:]:
            try:
                await self._bot.send_message(
                    chat_id=chat_id, text=chunk, parse_mode="Markdown"
                )
            except Exception:
                try:
                    await self._bot.send_message(chat_id=chat_id, text=chunk)
                except Exception as e:
                    logger.error("telegram_send_failed", error=str(e), chat_id=chat_id)

    async def send_typing(self, channel_user_id: str) -> None:
        """Send a typing indicator to a Telegram user.

        Args:
            channel_user_id: Telegram user/chat ID as string.
        """
        if not self._bot:
            return

        if not AIOGRAM_AVAILABLE:
            return

        try:
            await self._bot.send_chat_action(
                chat_id=int(channel_user_id),
                action=ChatAction.TYPING,
            )
        except Exception as e:
            logger.debug("typing_indicator_failed", error=str(e))

    # Extensions that Telegram can display inline
    _IMAGE_EXTENSIONS = frozenset({
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    })
    _VIDEO_EXTENSIONS = frozenset({
        ".mp4", ".mov", ".avi", ".mkv", ".webm",
    })

    def _media_type(self, file_path: str) -> str:
        """Determine the media type for a file path.

        Args:
            file_path: Path to the file.

        Returns:
            "photo", "video", or "document".
        """
        ext = Path(file_path).suffix.lower()
        if ext in self._IMAGE_EXTENSIONS:
            return "photo"
        if ext in self._VIDEO_EXTENSIONS:
            return "video"
        return "document"

    async def send_file_auto(
        self, channel_user_id: str, file_path: str, caption: str = ""
    ) -> None:
        """Send a file to a Telegram user, choosing the best method.

        Images are sent as photos (inline preview).
        Videos are sent as videos (inline player).
        Everything else is sent as a document attachment.

        Args:
            channel_user_id: Telegram user/chat ID as string.
            file_path: Absolute path to the file to send.
            caption: Optional caption for the file.
        """
        if not self._bot:
            return

        from aiogram.types import FSInputFile

        chat_id = int(channel_user_id)
        input_file = FSInputFile(file_path)
        media_type = self._media_type(file_path)

        try:
            if media_type == "photo":
                await self._bot.send_photo(
                    chat_id=chat_id,
                    photo=input_file,
                    caption=caption or None,
                )
            elif media_type == "video":
                await self._bot.send_video(
                    chat_id=chat_id,
                    video=input_file,
                    caption=caption or None,
                )
            else:
                await self._bot.send_document(
                    chat_id=chat_id,
                    document=input_file,
                    caption=caption or None,
                )
            logger.info(
                "telegram_file_sent",
                chat_id=chat_id,
                file_path=file_path,
                media_type=media_type,
            )
        except Exception as e:
            logger.error(
                "telegram_file_send_failed",
                error=str(e),
                chat_id=chat_id,
                file_path=file_path,
            )
            # Notify user about the failure
            with contextlib.suppress(Exception):
                await self._bot.send_message(
                    chat_id=chat_id,
                    text=f"Failed to send file: {e}",
                )

    async def _on_channel_post(self, data: dict[str, Any]) -> None:
        """Handle CHANNEL_POST events — post to a Telegram channel/group."""
        if data.get("channel") != "telegram" or not self._bot:
            return

        chat_id = data.get("chat_id")
        text = data.get("text", "")
        photo_path = data.get("photo_path", "")
        pin = data.get("pin", False)
        parse_mode = data.get("parse_mode", "Markdown") or None

        if not chat_id or not text:
            logger.warning("channel_post_missing_data", data=data)
            return

        try:
            if photo_path:
                from aiogram.types import FSInputFile

                photo = FSInputFile(photo_path)
                msg = await self._bot.send_photo(
                    chat_id=chat_id, photo=photo,
                    caption=text, parse_mode=parse_mode,
                )
            else:
                msg = await self._bot.send_message(
                    chat_id=chat_id, text=text, parse_mode=parse_mode,
                )

            if pin and msg:
                with contextlib.suppress(Exception):
                    await self._bot.pin_chat_message(
                        chat_id=chat_id, message_id=msg.message_id,
                    )

            logger.info("channel_post_sent", chat_id=chat_id)

        except Exception as e:
            logger.error("channel_post_failed", chat_id=chat_id, error=str(e))

    async def _on_send_message(self, data: dict[str, Any]) -> None:
        """Handle CHANNEL_SEND_MESSAGE events — send DM to a user."""
        if data.get("channel") != "telegram" or not self._bot:
            return

        user_id = data.get("user_id")
        text = data.get("text", "")
        parse_mode = data.get("parse_mode", "Markdown") or None

        if not user_id or not text:
            logger.warning("send_message_missing_data", data=data)
            return

        try:
            await self._bot.send_message(
                chat_id=int(user_id), text=text, parse_mode=parse_mode,
            )
            logger.info("telegram_message_sent", user_id=user_id)
        except Exception as e:
            logger.error(
                "telegram_message_failed", user_id=user_id, error=str(e),
            )

    async def _on_file_send(self, data: dict[str, Any]) -> None:
        """Handle FILE_SEND events from the send_file tool.

        Args:
            data: Event data with file_path, file_name, caption, channel, user_id.
        """
        if data.get("channel") != "telegram":
            return

        user_id = data.get("user_id")
        file_path = data.get("file_path")
        caption = data.get("caption", "")

        if not user_id or not file_path:
            logger.warning("file_send_missing_data", data=data)
            return

        await self.send_file_auto(
            channel_user_id=user_id,
            file_path=file_path,
            caption=caption,
        )

    async def send_approval_request(
        self,
        channel_user_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        request_id: str,
    ) -> bool:
        """Send an inline keyboard approval request for a dangerous tool.

        Args:
            channel_user_id: Telegram user/chat ID.
            tool_name: Name of the tool requesting approval.
            arguments: Tool call arguments.
            request_id: Unique identifier for this approval request.

        Returns:
            True if approved, False if denied or timed out.
        """
        if not self._bot or not AIOGRAM_AVAILABLE:
            return True

        # Build a human-readable explanation of what the tool does and why
        # it needs approval, plus a clear summary of the action.
        explanation = _tool_explanation(tool_name, arguments)
        text = (
            f"\u26a0\ufe0f <b>Permission Required</b>\n\n"
            f"{explanation}\n\n"
            f"Do you approve this action?"
        )

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="\u2705 Approve",
                        callback_data=f"approve:{request_id}",
                    ),
                    InlineKeyboardButton(
                        text="\u274c Deny",
                        callback_data=f"deny:{request_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\U0001f504 Approve All (session)",
                        callback_data=f"approve_session:{request_id}:{tool_name}",
                    ),
                ],
            ]
        )

        chat_id = int(channel_user_id)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._approval_futures[request_id] = future

        # Track that approvals happened so response goes to bottom of chat
        self._had_approvals[channel_user_id] = True

        try:
            await self._bot.send_message(
                chat_id=chat_id, text=text, reply_markup=keyboard,
                parse_mode="HTML",
            )

            result = await asyncio.wait_for(future, timeout=_APPROVAL_TIMEOUT)
            return result

        except TimeoutError:
            logger.warning(
                "approval_timeout",
                request_id=request_id,
                tool=tool_name,
            )
            return False

        except Exception as e:
            logger.error("approval_error", error=str(e), request_id=request_id)
            return False

        finally:
            self._approval_futures.pop(request_id, None)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_message(self, message: Any) -> bool:
        """Validate an incoming Telegram message.

        Checks that from_user exists and user is allowed.

        Args:
            message: aiogram Message object.

        Returns:
            True if message should be processed.
        """
        if not message.from_user:
            return False
        return self._is_allowed(message.from_user.id)

    async def _keep_typing(self, channel_user_id: str) -> None:
        """Continuously send typing indicators until cancelled.

        Sends a typing action every _TYPING_INTERVAL seconds.

        Args:
            channel_user_id: The user to show typing to.
        """
        try:
            while True:
                await asyncio.sleep(_TYPING_INTERVAL)
                await self.send_typing(channel_user_id)
        except asyncio.CancelledError:
            return

    @staticmethod
    def _split_message(text: str, max_length: int = _TG_MAX_LENGTH) -> list[str]:
        """Split a long message into chunks respecting Telegram limits.

        Tries to split at newlines first, then spaces, then hard-cuts.

        Args:
            text: The text to split.
            max_length: Maximum chunk length.

        Returns:
            List of text chunks, each within max_length.
        """
        if len(text) <= max_length:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Try to split at a newline
            split_pos = remaining.rfind("\n", 0, max_length)
            if split_pos == -1:
                # Try to split at a space
                split_pos = remaining.rfind(" ", 0, max_length)
            if split_pos == -1:
                # Hard split
                split_pos = max_length

            chunks.append(remaining[:split_pos])
            remaining = remaining[split_pos:].lstrip("\n")

        return chunks
