"""Telegram channel core — TelegramChannel class and module-level constants.

This is the main module of the telegram package. It contains the class
definition, ``__init__``, lifecycle methods, sending helpers, and the
method-binding block that attaches methods defined in sibling modules.
"""

from __future__ import annotations

import asyncio
import contextlib
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
    from agent.core.orchestrator import SubAgentOrchestrator
    from agent.core.scheduler import TaskScheduler
    from agent.core.session import SessionStore
    from agent.voice.pipeline import VoicePipeline
    from agent.workspaces.router import WorkspaceRouter

logger = structlog.get_logger(__name__)

# Attempt to import aiogram at module level; flag availability
try:
    from aiogram.enums import ChatAction
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
            "Summarize our conversation so far, highlighting" " key decisions and action items."
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
        intent = ai_desc if ai_desc else "Run a shell command"
        detail = f"Command:\n<code>{cmd}</code>"
        risk = "This will execute a shell command on your machine."

    elif tool_name in ("Write", "file_write"):
        path = arguments.get("file_path") or arguments.get("path", "?")
        size = len(arguments.get("content", ""))
        intent = ai_desc or "Create or overwrite a file"
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
        intent = ai_desc or "Delete a file"
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
        intent = ai_desc or "Make an HTTP request"
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
        parts = [f"{k}={str(v)[:80]}" for k, v in list(arguments.items())[:4] if k != "description"]
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
        orchestrator: SubAgentOrchestrator | None = None,
        tracer: object | None = None,
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
        self.orchestrator = orchestrator
        self.tracer = tracer
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
        # Queued file sends: user_id → list of event data dicts.
        # Files are queued (not sent immediately) so that buffered text
        # messages are flushed to the chat *before* the file arrives.
        self._pending_file_sends: dict[str, list[dict[str, Any]]] = {}
        # Track which user spawned which orchestration task for status updates
        self._task_user_map: dict[str, str] = {}

        import agent.channels.telegram as _tg_pkg

        if _tg_pkg.AIOGRAM_AVAILABLE and config.token:
            self._bot = _tg_pkg.Bot(token=config.token)
            self._dispatcher = _tg_pkg.Dispatcher()
            self._router = _tg_pkg.Router()
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

        # Wire up task→user tracking for orchestration status notifications
        from agent.tools.builtins.orchestration import set_task_user_callback

        set_task_user_callback(self._register_task_user)

        # Subscribe to orchestration events for real-time status updates
        self.event_bus.on(Events.SUBAGENT_SPAWNED, self._on_subagent_spawned)
        self.event_bus.on(Events.SUBAGENT_COMPLETED, self._on_subagent_completed)
        self.event_bus.on(Events.SUBAGENT_FAILED, self._on_subagent_failed)
        self.event_bus.on(Events.PROJECT_STARTED, self._on_project_started)
        self.event_bus.on(
            Events.PROJECT_STAGE_STARTED,
            self._on_project_stage_started,
        )
        self.event_bus.on(
            Events.PROJECT_STAGE_COMPLETED,
            self._on_project_stage_completed,
        )
        self.event_bus.on(Events.PROJECT_COMPLETED, self._on_project_completed)
        self.event_bus.on(Events.PROJECT_FAILED, self._on_project_failed)
        self.event_bus.on(
            Events.TASK_COMPLETED_NOTIFY,
            self._on_task_completed_notify,
        )

        # Subscribe to controller events
        self.event_bus.on(
            Events.CONTROLLER_TASK_STARTED,
            self._on_controller_task_started,
        )
        self.event_bus.on(
            Events.CONTROLLER_TASK_PROGRESS,
            self._on_controller_task_progress,
        )
        self.event_bus.on(
            Events.CONTROLLER_TASK_COMPLETED,
            self._on_controller_task_completed,
        )
        self.event_bus.on(
            Events.CONTROLLER_TASK_FAILED,
            self._on_controller_task_failed,
        )
        self.event_bus.on(
            Events.CONTROLLER_TASK_CANCELLED,
            self._on_controller_task_cancelled,
        )

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

        from aiogram import F as _F
        from aiogram.enums import ContentType
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
        self._router.message(Command("stop"))(self._cmd_stop)

        # Media handlers (before catch-all text)
        self._router.message(_F.content_type == ContentType.VOICE)(self._handle_voice)
        self._router.message(_F.content_type == ContentType.PHOTO)(self._handle_photo)
        self._router.message(_F.content_type == ContentType.DOCUMENT)(self._handle_document)

        # Callback query handler for inline buttons (approval + navigation)
        self._router.callback_query()(self._handle_callback)

        # Catch-all text handler (must be registered last)
        self._router.message()(self._handle_text)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    @staticmethod
    def _is_channel_id(chat_id: int | str) -> bool:
        """Detect if chat_id is a Telegram channel (large negative ID or @username)."""
        cid = str(chat_id).strip()
        if cid.startswith("@"):
            return True
        try:
            return int(cid) < -1000000000
        except ValueError:
            return False

    async def send_message(self, message: OutgoingMessage) -> None:
        """Send a text message to a Telegram user.

        Splits long messages and falls back to plain text on parse errors.

        Args:
            message: The outgoing message to send.
        """
        if not self._bot:
            return

        chat_id = int(message.channel_user_id)

        if self._is_channel_id(chat_id):
            logger.info(
                "sending_to_channel",
                chat_id=chat_id,
                note="Target is a Telegram channel, not a direct user chat",
            )
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

    async def send_streamed_response(self, channel_user_id: str, full_text: str) -> None:
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
                await self._bot.send_message(chat_id=chat_id, text=chunk, parse_mode="Markdown")
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
    _IMAGE_EXTENSIONS = frozenset(
        {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
        }
    )
    _VIDEO_EXTENSIONS = frozenset(
        {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
        }
    )

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

    async def send_file_auto(self, channel_user_id: str, file_path: str, caption: str = "") -> None:
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

        if self._is_channel_id(chat_id):
            logger.info(
                "sending_file_to_channel",
                chat_id=chat_id,
                file_path=file_path,
                note="Target is a Telegram channel, not a direct user chat",
            )

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
                    supports_streaming=True,
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
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                parse_mode="HTML",
            )

            import sys as _sys

            _timeout = _sys.modules["agent.channels.telegram"]._APPROVAL_TIMEOUT
            result = await asyncio.wait_for(future, timeout=_timeout)
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

    # ------------------------------------------------------------------
    # Intermediate text helper
    # ------------------------------------------------------------------

    async def _send_intermediate_text(self, chat_id: int, text: str) -> None:
        """Send intermediate/progress text as a new Telegram message.

        Uses Markdown with fallback to plain text, and splits long messages.
        """
        if not self._bot or not text.strip():
            return
        chunks = self._split_message(text.strip())
        for chunk in chunks:
            try:
                await self._bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="Markdown",
                )
            except Exception:
                try:
                    await self._bot.send_message(chat_id=chat_id, text=chunk)
                except Exception as e:
                    logger.warning(
                        "intermediate_text_send_failed",
                        chat_id=chat_id,
                        error=str(e),
                    )

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
                    await status_message.edit_text("No response was generated. Please try again.")
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
                    chunks[0],
                    parse_mode="Markdown",
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
                        chat_id=chat_id,
                        text=chunk,
                        parse_mode="Markdown",
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
    # Workspace support
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
        self,
        user_id: str,
        message_text: str = "",
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

    # ------------------------------------------------------------------
    # Reminder extraction
    # ------------------------------------------------------------------

    async def _try_extract_reminder(
        self,
        text: str,
        user_id: str,
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
        time_unit_en = r"(?:seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)"
        time_pattern_en = rf"(\d+\s*{time_unit_en}(?:\s*\d+\s*{time_unit_en})*)"

        # Russian time units
        time_unit_ru = r"(?:секунд[уы]?|сек|минут[уы]?|мин|час[аов]*|дн[яей]*|день)"
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
            "секунд": "s",
            "секунды": "s",
            "секунду": "s",
            "сек": "s",
            "минут": "m",
            "минуты": "m",
            "минуту": "m",
            "мин": "m",
            "час": "h",
            "часа": "h",
            "часов": "h",
            "день": "d",
            "дня": "d",
            "дней": "d",
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
            r"\b(?:пожалуйста|можешь|напомни|мне|поставь|" r"напоминание|через|о|об|что|про)\b",
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

    # ------------------------------------------------------------------
    # Background task management
    # ------------------------------------------------------------------

    def _register_background_task(
        self,
        user_id: str,
        task: asyncio.Task[Any],
        description: str,
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
            (t, d) for t, d in self._background_tasks[user_id] if not t.done() and t is not current
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
    # File send and channel post event handlers
    # ------------------------------------------------------------------

    async def _on_file_send(self, data: dict[str, Any]) -> None:
        """Handle FILE_SEND events from the send_file tool.

        Files are queued rather than sent immediately so that any buffered
        text messages (from the LLM stream) are flushed to the chat first.
        The queue is drained by ``_drain_pending_files`` which is called
        from the progress callback and after the orchestrator task finishes.

        Args:
            data: Event data with file_path, file_name, caption, channel, user_id.
        """
        if data.get("channel") != "telegram":
            return

        user_id = data.get("user_id")
        file_path = data.get("file_path")

        if not user_id or not file_path:
            logger.warning("file_send_missing_data", data=data)
            return

        self._pending_file_sends.setdefault(user_id, []).append(data)
        logger.debug("file_send_queued", user_id=user_id, file_path=file_path)

    async def _drain_pending_files(self, user_id: str) -> None:
        """Send all queued files for a user, in order."""
        files = self._pending_file_sends.pop(user_id, [])
        for data in files:
            file_path = data.get("file_path", "")
            caption = data.get("caption", "")
            try:
                await self.send_file_auto(
                    channel_user_id=user_id,
                    file_path=file_path,
                    caption=caption,
                )
            except Exception as exc:
                logger.warning(
                    "drain_file_send_failed",
                    user_id=user_id,
                    file_path=file_path,
                    error=str(exc),
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
                    chat_id=chat_id,
                    photo=photo,
                    caption=text,
                    parse_mode=parse_mode,
                )
            else:
                msg = await self._bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )

            if pin and msg:
                with contextlib.suppress(Exception):
                    await self._bot.pin_chat_message(
                        chat_id=chat_id,
                        message_id=msg.message_id,
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
                chat_id=int(user_id),
                text=text,
                parse_mode=parse_mode,
            )
            logger.info("telegram_message_sent", user_id=user_id)
        except Exception as e:
            logger.error(
                "telegram_message_failed",
                user_id=user_id,
                error=str(e),
            )


# ======================================================================
# Method binding from split modules
# ======================================================================

from agent.channels.telegram._commands import (  # noqa: E402
    _build_status_text,
    _build_tools_text_and_keyboard,
    _cancel_user_tasks,
    _cmd_backend,
    _cmd_help,
    _cmd_mute,
    _cmd_new,
    _cmd_pause,
    _cmd_remind,
    _cmd_reminders,
    _cmd_resume,
    _cmd_run,
    _cmd_session,
    _cmd_soul,
    _cmd_start,
    _cmd_status,
    _cmd_stop,
    _cmd_tasks,
    _cmd_tools,
    _cmd_unmute,
    _cmd_workdir,
    _get_shortcuts,
)

TelegramChannel._cmd_start = _cmd_start  # type: ignore[assignment]
TelegramChannel._cmd_help = _cmd_help  # type: ignore[assignment]
TelegramChannel._cmd_status = _cmd_status  # type: ignore[assignment]
TelegramChannel._cmd_tools = _cmd_tools  # type: ignore[assignment]
TelegramChannel._cmd_pause = _cmd_pause  # type: ignore[assignment]
TelegramChannel._cmd_resume = _cmd_resume  # type: ignore[assignment]
TelegramChannel._cmd_mute = _cmd_mute  # type: ignore[assignment]
TelegramChannel._cmd_unmute = _cmd_unmute  # type: ignore[assignment]
TelegramChannel._cmd_soul = _cmd_soul  # type: ignore[assignment]
TelegramChannel._cmd_backend = _cmd_backend  # type: ignore[assignment]
TelegramChannel._cmd_workdir = _cmd_workdir  # type: ignore[assignment]
TelegramChannel._cmd_new = _cmd_new  # type: ignore[assignment]
TelegramChannel._cmd_session = _cmd_session  # type: ignore[assignment]
TelegramChannel._cmd_remind = _cmd_remind  # type: ignore[assignment]
TelegramChannel._cmd_reminders = _cmd_reminders  # type: ignore[assignment]
TelegramChannel._cmd_run = _cmd_run  # type: ignore[assignment]
TelegramChannel._cmd_tasks = _cmd_tasks  # type: ignore[assignment]
TelegramChannel._cmd_stop = _cmd_stop  # type: ignore[assignment]
TelegramChannel._build_status_text = _build_status_text  # type: ignore[assignment]
TelegramChannel._build_tools_text_and_keyboard = _build_tools_text_and_keyboard  # type: ignore[assignment]
TelegramChannel._get_shortcuts = _get_shortcuts  # type: ignore[assignment]
TelegramChannel._cancel_user_tasks = _cancel_user_tasks  # type: ignore[assignment]

from agent.channels.telegram._media import (  # noqa: E402
    _handle_document,
    _handle_photo,
    _handle_voice,
    _run_document_task,
    _run_photo_task,
    _run_voice_task,
)

TelegramChannel._handle_voice = _handle_voice  # type: ignore[assignment]
TelegramChannel._handle_photo = _handle_photo  # type: ignore[assignment]
TelegramChannel._handle_document = _handle_document  # type: ignore[assignment]
TelegramChannel._run_voice_task = _run_voice_task  # type: ignore[assignment]
TelegramChannel._run_photo_task = _run_photo_task  # type: ignore[assignment]
TelegramChannel._run_document_task = _run_document_task  # type: ignore[assignment]

from agent.channels.telegram._callbacks import (  # noqa: E402
    _handle_approval_callback,
    _handle_callback,
    _handle_nav_callback,
    _nav_session_clear,
    _nav_session_new,
    _nav_status_audit,
    _nav_status_costs,
    _nav_status_health,
    _nav_tools_toggle,
)

TelegramChannel._handle_callback = _handle_callback  # type: ignore[assignment]
TelegramChannel._handle_approval_callback = _handle_approval_callback  # type: ignore[assignment]
TelegramChannel._handle_nav_callback = _handle_nav_callback  # type: ignore[assignment]
TelegramChannel._nav_status_costs = _nav_status_costs  # type: ignore[assignment]
TelegramChannel._nav_status_health = _nav_status_health  # type: ignore[assignment]
TelegramChannel._nav_status_audit = _nav_status_audit  # type: ignore[assignment]
TelegramChannel._nav_tools_toggle = _nav_tools_toggle  # type: ignore[assignment]
TelegramChannel._nav_session_clear = _nav_session_clear  # type: ignore[assignment]
TelegramChannel._nav_session_new = _nav_session_new  # type: ignore[assignment]

from agent.channels.telegram._dispatch import (  # noqa: E402
    _dispatch_to_agent,
    _handle_text,
    _process_via_sdk_or_loop,
    _run_text_task,
)

TelegramChannel._process_via_sdk_or_loop = _process_via_sdk_or_loop  # type: ignore[assignment]
TelegramChannel._dispatch_to_agent = _dispatch_to_agent  # type: ignore[assignment]
TelegramChannel._handle_text = _handle_text  # type: ignore[assignment]
TelegramChannel._run_text_task = _run_text_task  # type: ignore[assignment]

from agent.channels.telegram._events import (  # noqa: E402
    _notify_user,
    _on_controller_task_cancelled,
    _on_controller_task_completed,
    _on_controller_task_failed,
    _on_controller_task_progress,
    _on_controller_task_started,
    _on_project_completed,
    _on_project_failed,
    _on_project_stage_completed,
    _on_project_stage_started,
    _on_project_started,
    _on_subagent_completed,
    _on_subagent_failed,
    _on_subagent_spawned,
    _on_task_completed_notify,
    _register_task_user,
    _user_for_task,
)

TelegramChannel._register_task_user = _register_task_user  # type: ignore[assignment]
TelegramChannel._user_for_task = _user_for_task  # type: ignore[assignment]
TelegramChannel._notify_user = _notify_user  # type: ignore[assignment]
TelegramChannel._on_subagent_spawned = _on_subagent_spawned  # type: ignore[assignment]
TelegramChannel._on_subagent_completed = _on_subagent_completed  # type: ignore[assignment]
TelegramChannel._on_subagent_failed = _on_subagent_failed  # type: ignore[assignment]
TelegramChannel._on_project_started = _on_project_started  # type: ignore[assignment]
TelegramChannel._on_project_stage_started = _on_project_stage_started  # type: ignore[assignment]
TelegramChannel._on_project_stage_completed = _on_project_stage_completed  # type: ignore[assignment]
TelegramChannel._on_project_completed = _on_project_completed  # type: ignore[assignment]
TelegramChannel._on_project_failed = _on_project_failed  # type: ignore[assignment]
TelegramChannel._on_task_completed_notify = _on_task_completed_notify  # type: ignore[assignment]
TelegramChannel._on_controller_task_started = _on_controller_task_started  # type: ignore[assignment]
TelegramChannel._on_controller_task_progress = _on_controller_task_progress  # type: ignore[assignment]
TelegramChannel._on_controller_task_completed = _on_controller_task_completed  # type: ignore[assignment]
TelegramChannel._on_controller_task_failed = _on_controller_task_failed  # type: ignore[assignment]
TelegramChannel._on_controller_task_cancelled = _on_controller_task_cancelled  # type: ignore[assignment]
