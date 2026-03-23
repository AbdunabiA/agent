"""Telegram dispatch — routing messages to SDK/agent loop/orchestrator."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import Events

if TYPE_CHECKING:
    from agent.channels.telegram._core import TelegramChannel

logger = structlog.get_logger(__name__)


async def _process_via_sdk_or_loop(
    self: TelegramChannel,
    text: str,
    session: Any,
    agent_loop: Any,
    status_message: Any | None = None,
    user_id: str | None = None,
    stream_state: dict[str, Any] | None = None,
) -> str:
    """Route a message through Claude SDK or the LiteLLM agent loop.

    If status_message is provided, it will be edited with live status
    updates as the SDK processes the request (e.g. tool usage).

    When *stream_state* is provided, intermediate text is flushed to
    Telegram on tool_use events, and only the remainder is returned.

    Returns the response text.
    """
    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    if self.sdk_service is not None:
        from agent.llm.claude_sdk import ClaudeSDKService

        sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
        sdk_session_id = session.metadata.get("sdk_session_id")
        last_status_update = 0.0
        self._had_approvals[user_id or ""] = False
        import time as _time

        # Streaming buffer state
        pending_text = [""]
        sent_text_total = [""]
        status_deleted = [False]
        try:
            chat_id = int(user_id) if user_id else 0
        except (ValueError, TypeError):
            chat_id = 0

        # Permission callback: ask user via inline keyboard for dangerous tools
        on_permission = None
        if user_id and AIOGRAM_AVAILABLE:

            async def _on_permission(
                tool_name: str,
                details: str,
                tool_input: dict[str, Any],
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

        # Timeout: don't block Telegram handler for more than 5 minutes
        sdk_stream_timeout = 300  # 5 minutes max per message

        try:
            async with asyncio.timeout(sdk_stream_timeout):
                async for event in sdk.run_task_stream(
                    prompt=text,
                    task_id=session.id,
                    session_id=sdk_session_id,
                    on_permission=on_permission,
                    channel="telegram",
                ):
                    if event.type == "text":
                        # Only accumulate main agent text, not subagent output
                        if not (event.data and event.data.get("subagent")):
                            pending_text[0] += event.content
                    elif event.type == "tool_use":
                        if pending_text[0].strip():
                            # Flush buffered text as intermediate message
                            if status_message is not None and not status_deleted[0]:
                                with contextlib.suppress(Exception):
                                    await status_message.delete()
                                status_deleted[0] = True
                            await self._send_intermediate_text(
                                chat_id,
                                pending_text[0],
                            )
                            sent_text_total[0] += pending_text[0]
                            pending_text[0] = ""
                        elif status_message is not None and not status_deleted[0]:
                            # No pending text — update status with tool name
                            now = _time.monotonic()
                            if now - last_status_update > 2.0:
                                tool_name = event.data.get("tool", "tool")
                                is_sub = event.data.get("subagent", False)
                                prefix = (
                                    "\U0001f50d Researching" if is_sub else "\u2699\ufe0f Working"
                                )
                                status_text = f"{prefix}\u2026 using {tool_name}"
                                with contextlib.suppress(Exception):
                                    await status_message.edit_text(status_text)
                                last_status_update = now
                    elif event.type == "error":
                        raise RuntimeError(event.content)
                    elif event.type == "result":
                        sdk_sid = event.data.get("session_id")
                        if sdk_sid:
                            session.metadata["sdk_session_id"] = sdk_sid
                        # Handle SDK result-override
                        if (
                            event.content
                            and sent_text_total[0]
                            and event.content.startswith(sent_text_total[0])
                        ):
                            pending_text[0] = event.content[len(sent_text_total[0]) :]
                        elif event.content and len(event.content) > len(
                            sent_text_total[0] + pending_text[0]
                        ):
                            # Fallback: prefer result content when it's longer
                            pending_text[0] = event.content[len(sent_text_total[0]) :]

        except TimeoutError:
            logger.warning(
                "sdk_stream_timeout",
                task_id=session.id,
                timeout=sdk_stream_timeout,
            )
            pending_text[0] += (
                "\n\n⏰ Task is taking too long. "
                "Work continues in the background — I'll notify you when done."
            )

        # Drain any queued file sends (no orchestrator to drain them)
        if user_id:
            await self._drain_pending_files(user_id)

        if sent_text_total[0]:
            final = pending_text[0].strip()
            if stream_state is not None:
                stream_state["status_consumed"] = status_deleted[0]
            return final if final else ""
        return pending_text[0] or "[No response]"

    response = await agent_loop.process_message(text, session, trigger="user_message")

    # Drain any queued file sends (no orchestrator to drain them)
    if user_id:
        await self._drain_pending_files(user_id)

    return response.content


async def _dispatch_to_agent(
    self: TelegramChannel,
    text: str,
    session: Any,
    agent_loop: Any,
    status_message: Any | None = None,
    user_id: str | None = None,
    stream_state: dict[str, Any] | None = None,
) -> str:
    """Dispatch a message through the orchestrator or fall back to direct processing.

    When an orchestrator is available, this uses ``run_channel_task``
    which gives concurrency limits, timeouts, cancellation, and event
    tracking for free.  Otherwise it falls back to the direct
    ``_process_via_sdk_or_loop`` path.

    When *stream_state* is provided, intermediate text (text that appears
    before a tool_use event) is flushed to Telegram immediately, and
    only the remaining (final) text is returned.

    Returns the response text.
    """
    from agent.channels.telegram._core import AIOGRAM_AVAILABLE

    if self.orchestrator is not None:
        from agent.core.subagent import SubAgentStatus

        # Streaming state for buffer-and-flush
        pending_text = [""]
        sent_text_total = [""]
        status_deleted = [False]
        try:
            chat_id = int(user_id) if user_id else 0
        except (ValueError, TypeError):
            chat_id = 0

        # Build on_progress callback for status message updates
        on_progress = None
        import time as _time

        last_update = [0.0]

        async def _on_progress(event: Any) -> None:
            if event.type == "text":
                # Buffer text events (skip subagent text)
                if not (event.data and event.data.get("subagent")):
                    pending_text[0] += event.content
            elif event.type == "tool_use":
                if pending_text[0].strip():
                    # Flush buffered text as intermediate message
                    if status_message is not None and not status_deleted[0]:
                        with contextlib.suppress(Exception):
                            await status_message.delete()
                        status_deleted[0] = True
                    await self._send_intermediate_text(
                        chat_id,
                        pending_text[0],
                    )
                    sent_text_total[0] += pending_text[0]
                    pending_text[0] = ""
                elif status_message is not None and not status_deleted[0]:
                    # No pending text — update status with tool name
                    now = _time.monotonic()
                    if now - last_update[0] > 2.0:
                        tool_name = event.data.get("tool", "tool")
                        is_sub = event.data.get("subagent", False)
                        prefix = "\U0001f50d Researching" if is_sub else "\u2699\ufe0f Working"
                        with contextlib.suppress(Exception):
                            await status_message.edit_text(f"{prefix}\u2026 using {tool_name}")
                        last_update[0] = now
                # Drain queued file sends so they arrive after text
                if user_id:
                    await self._drain_pending_files(user_id)
            elif event.type == "result" and event.content:
                if sent_text_total[0] and event.content.startswith(sent_text_total[0]):
                    # Handle SDK result-override: if result content is longer
                    # and starts with what we already sent, extract the tail
                    pending_text[0] = event.content[len(sent_text_total[0]) :]
                elif not sent_text_total[0]:
                    # No text was streamed yet — use result as pending
                    pending_text[0] = event.content
                elif len(event.content) > len(sent_text_total[0] + pending_text[0]):
                    # Fallback: prefer result content when it's longer
                    pending_text[0] = event.content[len(sent_text_total[0]) :]

        on_progress = _on_progress

        # Build on_permission callback
        on_permission = None
        if user_id and AIOGRAM_AVAILABLE:

            async def _on_permission(
                tool_name: str,
                details: str,
                tool_input: dict[str, Any],
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

        # Reset approval tracking for this user
        self._had_approvals[user_id or ""] = False

        result = await self.orchestrator.run_channel_task(
            prompt=text,
            task_id=session.id,
            session=session,
            on_progress=on_progress,
            on_permission=on_permission,
        )

        # Drain any remaining queued file sends before returning
        if user_id:
            await self._drain_pending_files(user_id)

        if result.status == SubAgentStatus.COMPLETED:
            if sent_text_total[0]:
                # Intermediate text was sent — return only remainder
                final = pending_text[0].strip()
                if stream_state is not None:
                    stream_state["status_consumed"] = status_deleted[0]
                return final if final else ""
            # No intermediate text was streamed — check for buffered
            # pending text that was never flushed (no tool_use event
            # triggered a flush).
            if pending_text[0].strip():
                return pending_text[0].strip()
            return result.output or "[No response]"
        elif result.status == SubAgentStatus.CANCELLED:
            return ""  # cancelled tasks don't need a response
        else:
            error_msg = result.error or "Unknown error"
            raise RuntimeError(error_msg)

    # Fallback: direct processing without orchestrator
    return await self._process_via_sdk_or_loop(
        text,
        session,
        agent_loop,
        status_message=status_message,
        user_id=user_id,
        stream_state=stream_state,
    )


async def _handle_text(self: TelegramChannel, message: Any) -> None:
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

    # Natural language stop: cancel running tasks if user says "stop"
    lower = user_text.strip().lower()
    if lower in ("stop", "cancel", "стоп", "отмена") and self._get_active_tasks(
        str(user_id),
    ):
        cancelled = await self._cancel_user_tasks(str(user_id))
        await message.answer(f"\u26d4 Cancelled {cancelled} task(s).")
        return

    # Check if paused
    if self._paused:
        await message.answer("I'm currently paused. Use /resume to re-enable me.")
        return

    # Auto-detect natural language reminder requests
    reminder_confirmation = await self._try_extract_reminder(user_text, str(user_id))
    if reminder_confirmation:
        await message.answer(reminder_confirmation)
        return

    # Resolve workspace-specific components
    agent_loop, session_store = self._resolve_components(str(user_id), user_text)

    # Get or create session
    session_id = self._make_session_id(str(user_id))
    session = await session_store.get_or_create(session_id=session_id, channel="telegram")

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
    with contextlib.suppress(Exception):
        status_message = await message.answer("\u23f3 Processing\u2026")

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
    self: TelegramChannel,
    user_text: str,
    session: Any,
    agent_loop: Any,
    status_message: Any | None,
    user_id: str,
    message: Any,
) -> None:
    """Background coroutine that processes a text message and delivers the response."""
    # Set per-task context vars (safe for concurrent users in separate tasks)
    from agent.tools.builtins.scheduler import set_context

    set_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.send_file import set_file_send_context

    set_file_send_context(channel="telegram", user_id=user_id)

    from agent.tools.builtins.telegram_post import set_telegram_post_context

    set_telegram_post_context(channel="telegram", user_id=user_id)

    try:
        stream_state: dict[str, Any] = {"status_consumed": False}
        response_text = await self._dispatch_to_agent(
            user_text,
            session,
            agent_loop,
            status_message=status_message,
            user_id=user_id,
            stream_state=stream_state,
        )

        # If status was already deleted (intermediate text sent),
        # don't pass it to _replace_status_with_response
        effective_status = None if stream_state["status_consumed"] else status_message

        # Deliver the response
        await self._replace_status_with_response(effective_status, user_id, response_text)

    except Exception as e:
        logger.error(
            "telegram_handle_error",
            error=str(e),
            user_id=user_id,
        )
        # If the status message was already deleted (intermediate text
        # was streamed), we must fall back to message.answer instead.
        error_text = "Sorry, something went wrong processing your message."
        if status_message and not stream_state["status_consumed"]:
            with contextlib.suppress(Exception):
                await status_message.edit_text(error_text)
        else:
            with contextlib.suppress(Exception):
                await message.answer(error_text)
    finally:
        # Drain any orphaned file sends before unregistering
        with contextlib.suppress(Exception):
            await self._drain_pending_files(user_id)
        self._unregister_background_task(user_id)
