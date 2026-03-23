"""Telegram media handlers — voice, photo, and document processing."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from agent.channels.telegram._core import TelegramChannel
    from agent.core.agent_loop import AgentLoop

logger = structlog.get_logger(__name__)


async def _handle_voice(self: TelegramChannel, message: Any) -> None:
    """Handle voice messages — download, transcribe, dispatch to background."""
    if not self._check_message(message):
        return
    if self._paused:
        await message.answer("I'm currently paused. Use /resume to re-enable me.")
        return

    user_id = message.from_user.id

    try:
        file = await self._bot.get_file(message.voice.file_id)
        from io import BytesIO

        buf = BytesIO()
        await self._bot.download_file(file.file_path, buf)
        audio_bytes = buf.getvalue()

        # Resolve workspace-specific components
        agent_loop, session_store = self._resolve_components(str(user_id))

        session_id = self._make_session_id(str(user_id))
        session = await session_store.get_or_create(session_id=session_id, channel="telegram")

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
            "Sorry, I couldn't process your voice message. " "Try sending it as text instead."
        )


async def _run_voice_task(
    self: TelegramChannel,
    audio_bytes: bytes,
    session: Any,
    agent_loop: Any,
    user_id: str,
    message: Any,
) -> None:
    """Background coroutine that processes a voice message."""
    # Set per-task context vars (safe for concurrent users in separate tasks)
    from agent.tools.builtins.scheduler import set_context

    set_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.send_file import set_file_send_context

    set_file_send_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.telegram_post import set_telegram_post_context

    set_telegram_post_context(channel="telegram", user_id=user_id)

    status_message = None
    stream_state: dict[str, Any] = {"status_consumed": False}
    try:
        response_text: str

        # Standalone STT path (whisper, deepgram, etc.)
        if self.voice_pipeline and not self.voice_pipeline.is_llm_native():
            stt_result = await self.voice_pipeline.transcribe(
                audio_bytes,
                "audio/ogg",
            )

            # Show transcription to user
            if self.voice_pipeline.config.voice_transcription_prefix:
                transcription_msg = f"\U0001f3a4 *Transcription:* {stt_result.text}"
                await message.reply(transcription_msg, parse_mode="Markdown")

            # Send status message and process via agent loop or SDK
            with contextlib.suppress(Exception):
                status_message = await message.answer("\u23f3 Processing\u2026")

            response_text = await self._dispatch_to_agent(
                stt_result.text,
                session,
                agent_loop,
                status_message=status_message,
                user_id=user_id,
                stream_state=stream_state,
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

            session.add_message(Message(role="assistant", content=response_text))

        # Voice reply if enabled
        from agent.channels.telegram._core import AIOGRAM_AVAILABLE

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
                if status_message is not None and not stream_state["status_consumed"]:
                    with contextlib.suppress(Exception):
                        await status_message.delete()
                await message.reply_voice(
                    voice=voice_file,
                    duration=int(tts_result.duration_seconds),
                )
                if len(response_text) > 200:
                    short = response_text[:200] + "..."
                    await message.reply(
                        f"\U0001f4dd _{short}_",
                        parse_mode="Markdown",
                    )
                return

        # Text-only reply
        effective_status = None if stream_state["status_consumed"] else status_message
        await self._replace_status_with_response(effective_status, user_id, response_text)

    except Exception as e:
        logger.error("voice_task_error", error=str(e), user_id=user_id)
        with contextlib.suppress(Exception):
            await message.answer(
                "Sorry, I couldn't process your voice message. " "Try sending it as text instead."
            )
    finally:
        with contextlib.suppress(Exception):
            await self._drain_pending_files(user_id)
        self._unregister_background_task(user_id)


async def _handle_photo(self: TelegramChannel, message: Any) -> None:
    """Handle photo messages — download, describe as text, dispatch to background."""
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

        # Resolve workspace-aware components
        agent_loop, session_store = self._resolve_components(str(user_id))

        session_id = self._make_session_id(str(user_id))
        session = await session_store.get_or_create(session_id=session_id, channel="telegram")

        # Store the image in session metadata for the LLM to access
        session.metadata["pending_image"] = {
            "base64": image_b64,
            "media_type": "image/jpeg",
        }

        # Build text description for orchestrator dispatch
        text_desc = "[Photo attached]"
        if caption:
            text_desc = f"[Photo attached] {caption}"

        status_message = None
        with contextlib.suppress(Exception):
            status_message = await message.answer("\u2699\ufe0f Processing photo\u2026")

        # Dispatch to background task like text/voice/document handlers
        task = asyncio.create_task(
            self._run_photo_task(
                text_desc=text_desc,
                session=session,
                image_b64=image_b64,
                caption=caption,
                status_message=status_message,
                user_id=str(user_id),
                message=message,
                agent_loop=agent_loop,
            ),
            name=f"photo:{user_id}",
        )
        self._register_background_task(str(user_id), task, "[photo]")

    except Exception as e:
        logger.error("photo_handle_error", error=str(e), user_id=user_id)
        await message.answer(
            "Sorry, I couldn't process your photo. " "Try describing it in text instead."
        )


async def _run_photo_task(
    self: TelegramChannel,
    text_desc: str,
    session: Any,
    image_b64: str,
    caption: str,
    status_message: Any | None,
    user_id: str,
    message: Any,
    agent_loop: AgentLoop | None = None,
) -> None:
    """Background coroutine that processes a photo message."""
    # Set per-task context vars (safe for concurrent users in separate tasks)
    from agent.tools.builtins.scheduler import set_context

    set_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.send_file import set_file_send_context

    set_file_send_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.telegram_post import set_telegram_post_context

    set_telegram_post_context(channel="telegram", user_id=user_id)

    effective_loop = agent_loop or self.agent_loop

    try:
        stream_state: dict[str, Any] = {"status_consumed": False}

        response_text = await self._dispatch_to_agent(
            text=text_desc,
            session=session,
            agent_loop=effective_loop,
            status_message=status_message,
            user_id=user_id,
            stream_state=stream_state,
        )

        # If dispatch didn't produce a result (no orchestrator, no SDK),
        # fall back to direct multimodal LLM call
        if not response_text or response_text == "[No response]":
            if not effective_loop.llm:
                response_text = (
                    "Sorry, I can't process photos right now " "(no LLM provider configured)."
                )
            else:
                from agent.core.session import Message

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

                placeholder = f"[Photo]{f': {caption}' if caption else ''}"
                session.add_message(Message(role="user", content=placeholder))

                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": effective_loop.system_prompt,
                    },
                ]
                messages.extend(session.get_history()[:-1])
                messages.append({"role": "user", "content": multimodal_content})

                response = await effective_loop.llm.completion(messages=messages)
                response_text = response.content

                session.add_message(Message(role="assistant", content=response_text))

        # Clean up status message
        effective_status = None if stream_state["status_consumed"] else status_message
        await self._replace_status_with_response(
            effective_status,
            user_id,
            response_text,
        )

        # Clean up pending image
        session.metadata.pop("pending_image", None)

    except Exception as e:
        # Clean up pending image even on error to avoid stale data
        with contextlib.suppress(Exception):
            session.metadata.pop("pending_image", None)
        logger.error("photo_task_error", error=str(e), user_id=user_id)
        if status_message:
            with contextlib.suppress(Exception):
                await status_message.edit_text(
                    "Sorry, I couldn't process your photo. " "Try describing it in text instead."
                )
        else:
            with contextlib.suppress(Exception):
                await message.answer(
                    "Sorry, I couldn't process your photo. " "Try describing it in text instead."
                )
    finally:
        with contextlib.suppress(Exception):
            await self._drain_pending_files(user_id)
        self._unregister_background_task(user_id)


async def _handle_document(self: TelegramChannel, message: Any) -> None:
    """Handle document uploads — save to data/uploads/, dispatch to background."""
    if not self._check_message(message):
        return
    if self._paused:
        await message.answer("I'm currently paused. Use /resume to re-enable me.")
        return

    # Access _UPLOAD_DIR and os through the package so tests can patch them
    _tg_pkg = sys.modules["agent.channels.telegram"]
    _upload_dir = _tg_pkg._UPLOAD_DIR
    _os = _tg_pkg.os

    user_id = message.from_user.id

    try:
        doc = message.document
        file = await self._bot.get_file(doc.file_id)
        filename = doc.file_name or f"document_{doc.file_id}"
        mime_type = doc.mime_type or "application/octet-stream"
        file_size = doc.file_size or 0

        # Save to uploads directory
        _os.makedirs(_upload_dir, exist_ok=True)
        save_path = _upload_dir / filename

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

        # Resolve workspace-aware components
        agent_loop, session_store = self._resolve_components(str(user_id))

        session_id = self._make_session_id(str(user_id))
        session = await session_store.get_or_create(session_id=session_id, channel="telegram")

        status_message = None
        with contextlib.suppress(Exception):
            status_message = await message.answer(
                "\u23f3 Processing document\u2026",
            )

        # Dispatch to background
        task = asyncio.create_task(
            self._run_document_task(
                description=description,
                session=session,
                status_message=status_message,
                user_id=str(user_id),
                message=message,
                agent_loop=agent_loop,
            ),
            name=f"doc:{user_id}:{filename[:30]}",
        )
        self._register_background_task(
            str(user_id),
            task,
            f"[document: {filename}]",
        )

    except Exception as e:
        logger.error("document_handle_error", error=str(e), user_id=user_id)
        await message.answer("Sorry, I couldn't process your document.")


async def _run_document_task(
    self: TelegramChannel,
    description: str,
    session: Any,
    status_message: Any | None,
    user_id: str,
    message: Any,
    agent_loop: AgentLoop | None = None,
) -> None:
    """Background coroutine that processes a document upload."""
    # Set per-task context vars (safe for concurrent users in separate tasks)
    from agent.tools.builtins.scheduler import set_context

    set_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.send_file import set_file_send_context

    set_file_send_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.telegram_post import set_telegram_post_context

    set_telegram_post_context(channel="telegram", user_id=user_id)

    effective_loop = agent_loop or self.agent_loop

    try:
        stream_state: dict[str, Any] = {"status_consumed": False}
        response_text = await self._dispatch_to_agent(
            description,
            session,
            effective_loop,
            status_message=status_message,
            user_id=user_id,
            stream_state=stream_state,
        )

        effective_status = None if stream_state["status_consumed"] else status_message
        await self._replace_status_with_response(effective_status, user_id, response_text)

    except Exception as e:
        logger.error("document_task_error", error=str(e), user_id=user_id)
        if status_message:
            with contextlib.suppress(Exception):
                await status_message.edit_text("Sorry, I couldn't process your document.")
        else:
            with contextlib.suppress(Exception):
                await message.answer("Sorry, I couldn't process your document.")
    finally:
        with contextlib.suppress(Exception):
            await self._drain_pending_files(user_id)
        self._unregister_background_task(user_id)
