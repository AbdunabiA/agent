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
    ) -> None:
        super().__init__(config=config, event_bus=event_bus, session_store=session_store)
        self.agent_loop = agent_loop
        self.heartbeat = heartbeat
        self.workspace_router = workspace_router
        self.voice_pipeline = voice_pipeline
        self.sdk_service = sdk_service
        self.scheduler = scheduler
        self._workspace_agent_loops: dict[str, AgentLoop] = {}
        self._workspace_session_stores: dict[str, SessionStore] = {}
        self._polling_task: asyncio.Task[Any] | None = None
        self._bot: Any = None
        self._dispatcher: Any = None
        self._router: Any = None
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}

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

        # Delete any stale webhook so polling works
        await self._bot.delete_webhook(drop_pending_updates=True)

        # Set bot command menu (the "/" button near the input field)
        await self._set_bot_commands()

        # Subscribe to file send events
        self.event_bus.on(Events.FILE_SEND, self._on_file_send)

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

        # Media handlers (before catch-all text)
        self._router.message(F.content_type == ContentType.VOICE)(self._handle_voice)
        self._router.message(F.content_type == ContentType.PHOTO)(self._handle_photo)
        self._router.message(F.content_type == ContentType.DOCUMENT)(self._handle_document)

        # Callback query handler for inline approval buttons
        self._router.callback_query()(self._handle_approval_callback)

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
            "/soul — View/edit agent personality\n"
            "/backend — View/switch LLM backend\n"
            "/workdir — View/change working directory\n"
            "/remind <delay> <text> — Set a reminder\n"
            "/reminders — List pending reminders\n"
            "/pause — Pause message processing\n"
            "/resume — Resume message processing\n"
            "/mute — Disable heartbeat\n"
            "/unmute — Enable heartbeat"
        )

    async def _cmd_status(self, message: Any) -> None:
        """Handle /status command — show agent status."""
        if not self._check_message(message):
            return

        from agent.tools.registry import registry

        heartbeat_status = "not configured"
        last_tick = "never"
        if self.heartbeat:
            heartbeat_status = "enabled" if self.heartbeat.is_enabled else "disabled"
            if self.heartbeat.last_tick:
                last_tick = self.heartbeat.last_tick.strftime("%Y-%m-%d %H:%M:%S")

        tools_count = len(registry.list_tools())
        sessions_count = self.session_store.active_count

        await message.answer(
            f"Agent Status:\n"
            f"Heartbeat: {heartbeat_status}\n"
            f"Last tick: {last_tick}\n"
            f"Tools: {tools_count}\n"
            f"Active sessions: {sessions_count}"
        )

    async def _cmd_tools(self, message: Any) -> None:
        """Handle /tools command — list registered tools."""
        if not self._check_message(message):
            return

        from agent.tools.registry import registry

        tools = registry.list_tools()
        if not tools:
            await message.answer("No tools registered.")
            return

        tier_emoji = {"safe": "\U0001f7e2", "moderate": "\U0001f7e1", "dangerous": "\U0001f534"}
        lines: list[str] = ["Available tools:\n"]
        for t in tools:
            emoji = tier_emoji.get(t.tier.value, "\u2753")
            status = "on" if t.enabled else "off"
            lines.append(f"{emoji} {t.name} [{status}] — {t.description}")

        await message.answer("\n".join(lines))

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

        # Clear the SDK session from the agent session
        session = await self.session_store.get(session_id)
        if session and hasattr(session, "sdk_session_id"):
            old_sid = getattr(session, "sdk_session_id", None)
            session.sdk_session_id = None  # type: ignore[attr-defined]
            logger.info(
                "session_cleared",
                user_id=user_id,
                old_session=old_sid[:16] + "..." if old_sid else None,
            )

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
        sdk_sid = getattr(session, "sdk_session_id", None) if session else None

        backend = "claude-sdk" if self.sdk_service else "litellm"
        status = "active" if sdk_sid else "none"

        lines = [
            f"Session Info:",
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

    # ------------------------------------------------------------------
    # Media handlers
    # ------------------------------------------------------------------

    async def _handle_voice(self, message: Any) -> None:
        """Handle voice messages with full voice pipeline support.

        Flow:
        1. Download voice .ogg file
        2. If STT is llm_native: send audio to LLM directly (legacy behaviour)
        3. Otherwise: transcribe -> show transcription -> process text via agent loop
        4. If voice reply enabled: synthesize response and send as voice message
        """
        if not self._check_message(message):
            return
        if self._paused:
            await message.answer("I'm currently paused. Use /resume to re-enable me.")
            return

        user_id = message.from_user.id
        await self.send_typing(str(user_id))

        # Set file-send context for voice messages
        from agent.tools.builtins.send_file import set_file_send_context
        set_file_send_context(channel="telegram", user_id=str(user_id))

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

            response_text: str

            # Standalone STT path (whisper, deepgram, etc.)
            if self.voice_pipeline and not self.voice_pipeline.is_llm_native():
                stt_result = await self.voice_pipeline.transcribe(audio_bytes, "audio/ogg")

                # Show transcription to user
                if self.voice_pipeline.config.voice_transcription_prefix:
                    transcription_msg = f"🎤 *Transcription:* {stt_result.text}"
                    await message.reply(transcription_msg, parse_mode="Markdown")

                # Process transcribed text through agent loop or SDK
                typing_task = asyncio.create_task(self._keep_typing(str(user_id)))
                try:
                    response_text = await self._process_via_sdk_or_loop(
                        stt_result.text, session, agent_loop,
                    )
                finally:
                    typing_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await typing_task

            else:
                # LLM native: send audio directly (current behaviour)
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
            if self.voice_pipeline and self.voice_pipeline.should_voice_reply("telegram"):
                tts_result = await self.voice_pipeline.synthesize(response_text)
                if tts_result and AIOGRAM_AVAILABLE:
                    from aiogram.types import BufferedInputFile

                    voice_file = BufferedInputFile(
                        file=tts_result.audio_data,
                        filename="response.ogg",
                    )
                    await message.reply_voice(
                        voice=voice_file,
                        duration=int(tts_result.duration_seconds),
                    )
                    # Also send text for accessibility
                    if len(response_text) > 200:
                        short = response_text[:200] + "..."
                        await message.reply(f"📝 _{short}_", parse_mode="Markdown")
                    return

            # Text-only reply
            await self.send_streamed_response(str(user_id), response_text)

        except Exception as e:
            logger.error("voice_handle_error", error=str(e), user_id=user_id)
            await message.answer(
                "Sorry, I couldn't process your voice message. "
                "Try sending it as text instead."
            )

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
        """Handle document uploads — save to data/uploads/, inform agent."""
        if not self._check_message(message):
            return
        if self._paused:
            await message.answer("I'm currently paused. Use /resume to re-enable me.")
            return

        user_id = message.from_user.id
        await self.send_typing(str(user_id))

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

            typing_task = asyncio.create_task(self._keep_typing(str(user_id)))

            try:
                response_text = await self._process_via_sdk_or_loop(
                    description, session, self.agent_loop,
                )

                typing_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task

                await self.send_message(
                    OutgoingMessage(
                        content=response_text,
                        channel_user_id=str(user_id),
                        parse_mode="Markdown",
                    )
                )
            except Exception:
                typing_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task
                raise

        except Exception as e:
            logger.error("document_handle_error", error=str(e), user_id=user_id)
            await message.answer("Sorry, I couldn't process your document.")

    # ------------------------------------------------------------------
    # Approval callback
    # ------------------------------------------------------------------

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
    ) -> str:
        """Route a message through Claude SDK or the LiteLLM agent loop.

        Returns the response text.
        """
        if self.sdk_service is not None:
            from agent.llm.claude_sdk import ClaudeSDKService

            sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
            accumulated = ""
            async for event in sdk.run_task_stream(
                prompt=text,
                task_id=session.id,
                session_id=getattr(session, "sdk_session_id", None),
            ):
                if event.type == "text":
                    accumulated += event.content
                elif event.type == "result":
                    sdk_sid = event.data.get("session_id")
                    if sdk_sid:
                        session.sdk_session_id = sdk_sid
                elif event.type == "error":
                    raise RuntimeError(event.content)
            return accumulated or "[No response]"

        response = await agent_loop.process_message(
            text, session, trigger="user_message"
        )
        return response.content

    async def _handle_text(self, message: Any) -> None:
        """Process an incoming text message through the agent loop."""
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

        # Auto-detect natural language reminder requests
        reminder_confirmation = await self._try_extract_reminder(
            user_text, str(user_id)
        )
        if reminder_confirmation:
            await message.answer(reminder_confirmation)
            return

        # Show typing
        await self.send_typing(str(user_id))

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

        # Keep typing active while processing
        typing_task = asyncio.create_task(self._keep_typing(str(user_id)))

        try:
            response_text = await self._process_via_sdk_or_loop(
                user_text, session, agent_loop,
            )

            # Cancel typing
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

            # Send response (with streaming for long responses)
            await self.send_streamed_response(str(user_id), response_text)

        except Exception as e:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

            logger.error(
                "telegram_handle_error",
                error=str(e),
                user_id=user_id,
            )
            await message.answer("Sorry, something went wrong processing your message.")

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

        args_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        text = (
            f"\u26a0\ufe0f Tool Approval Required\n\n"
            f"Tool: {tool_name}\n"
            f"Arguments: {args_str}\n\n"
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

        try:
            await self._bot.send_message(
                chat_id=chat_id, text=text, reply_markup=keyboard
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
