"""Main reasoning loop for the agent.

Phase 1: Simple message -> LLM -> response
Phase 2: Multi-step tool calling with ReAct pattern
Phase 4: Adds memory injection (facts, vectors, soul.md)
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from agent.config import AgentPersonaConfig
from agent.core.events import EventBus, Events
from agent.core.session import Message, Session, TokenUsage
from agent.llm.prompts import build_system_prompt
from agent.llm.provider import LLMProvider, LLMResponse
from agent.tools.registry import registry

if TYPE_CHECKING:
    from agent.core.cost_tracker import CostTracker
    from agent.core.guardrails import Guardrails
    from agent.core.planner import Planner
    from agent.core.recovery import ErrorRecovery
    from agent.core.session import SessionStore
    from agent.memory.extraction import FactExtractor
    from agent.memory.soul import SoulLoader
    from agent.memory.store import FactStore
    from agent.memory.summarizer import ConversationSummarizer
    from agent.memory.vectors import VectorStore
    from agent.tools.executor import ToolExecutor, ToolResult

logger = structlog.get_logger(__name__)


@dataclass
class StreamEvent:
    """An event emitted during streaming message processing."""

    type: str  # "chunk", "tool.execute", "tool.result", "done"
    content: str = ""
    data: dict[str, Any] | None = None
    response: LLMResponse | None = None


class AgentLoop:
    """Main reasoning loop for the agent.

    Phase 1: Simple message -> LLM -> response
    Phase 2: Multi-step tool calling with ReAct pattern
    Phase 4: Adds memory injection (facts, vectors, soul.md)
    """

    def __init__(
        self,
        llm: LLMProvider | None,
        config: AgentPersonaConfig,
        event_bus: EventBus,
        tool_executor: ToolExecutor | None = None,
        planner: Planner | None = None,
        recovery: ErrorRecovery | None = None,
        guardrails: Guardrails | None = None,
        fact_store: FactStore | None = None,
        vector_store: VectorStore | None = None,
        soul_loader: SoulLoader | None = None,
        fact_extractor: FactExtractor | None = None,
        summarizer: ConversationSummarizer | None = None,
        cost_tracker: CostTracker | None = None,
        session_store: SessionStore | None = None,
        skill_builder_enabled: bool = False,
        orchestration_enabled: bool = False,
        platform_capabilities: str | None = None,
        use_controller: bool = False,
    ) -> None:
        self.llm = llm
        self.config = config
        self.event_bus = event_bus
        self.tool_executor = tool_executor
        self.planner = planner
        self.recovery = recovery
        self.guardrails = guardrails
        self.fact_store = fact_store
        self.vector_store = vector_store
        self.soul_loader = soul_loader
        self.fact_extractor = fact_extractor
        self.summarizer = summarizer
        self.cost_tracker = cost_tracker
        self.session_store = session_store
        self._skill_builder_enabled = skill_builder_enabled
        self._orchestration_enabled = orchestration_enabled
        self._platform_capabilities = platform_capabilities
        self._use_controller = use_controller
        self._default_tool_registry: Any = None  # Optional ScopedToolRegistry override
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Build system prompt with soul.md content if available
        soul_content = soul_loader.load() if soul_loader else None
        self.system_prompt = build_system_prompt(
            config,
            soul_content=soul_content,
            skill_builder_enabled=skill_builder_enabled,
            orchestration_enabled=orchestration_enabled,
            platform_capabilities=platform_capabilities,
            use_controller=use_controller,
        )

    async def _add_and_persist(self, session: Session, message: Message) -> None:
        """Add message to session and persist to database if available."""
        session.add_message(message)
        if self.session_store:
            try:
                await self.session_store.save_message(session.id, message)
            except Exception as e:
                logger.debug("message_persist_failed", error=str(e))

    async def process_message(
        self,
        user_message: str,
        session: Session,
        trigger: str = "user_message",
        tool_registry_override: Any | None = None,
    ) -> LLMResponse:
        """Process a message with full tool-calling support.

        The ReAct Loop:
        1. Add user message to session
        2. Check if planning is needed -> if yes, create plan
        3. Build context: system prompt + plan (if any) + history + tools
        4. Call LLM
        5. Check response:
           a. If text only (no tool calls) -> return response
           b. If tool calls -> execute tools -> add results to session -> goto 4
        6. If max iterations reached -> force return with partial results
        7. On error -> attempt recovery -> if retry, goto 4

        Args:
            user_message: The user's input text.
            session: The current conversation session.
            trigger: What triggered this message processing.

        Returns:
            LLMResponse with the agent's reply.

        Raises:
            Exception: If LLM call fails (after failover attempt).
        """
        if self.llm is None:
            raise RuntimeError(
                "AgentLoop.process_message() requires an LLM provider. "
                "Configure an API key or use the Claude SDK backend."
            )

        # Add user message to session
        await self._add_and_persist(session, Message(role="user", content=user_message))

        # Emit incoming event
        await self.event_bus.emit(
            Events.MESSAGE_INCOMING,
            {
                "content": user_message,
                "session_id": session.id,
            },
        )

        # Query memory for context injection (parallel)
        relevant_facts, vector_results = await self._query_memory(user_message)

        # Check if we should plan
        plan = None
        if (
            self.planner
            and trigger == "user_message"
            and await self.planner.should_plan(user_message, session)
        ):
            plan = await self.planner.create_plan(user_message, session)
            logger.info("plan_created", goal=plan.goal, steps=len(plan.steps))

        # Get tool schemas (use override if provided, e.g. scoped sub-agent registry)
        effective_registry = tool_registry_override or self._default_tool_registry or registry
        tool_schemas = effective_registry.get_tool_schemas(enabled_only=True)

        # ReAct loop
        iteration = 0
        max_iterations = self.config.max_iterations

        while iteration < max_iterations:
            iteration += 1

            # Build messages
            messages = self._build_messages(
                session, plan, relevant_facts, vector_results, tool_schemas
            )

            # Call LLM
            try:
                kwargs = {}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas
                response = await self.llm.completion(
                    messages=messages,
                    **kwargs,
                )
            except Exception as e:
                logger.error("llm_call_failed", error=str(e), iteration=iteration)
                raise

            # Record cost for every LLM call
            if self.cost_tracker and response.usage:
                channel = str(session.metadata.get("channel", "cli"))
                self.cost_tracker.record(
                    model=response.model or "unknown",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    channel=channel,
                    session_id=session.id,
                )

            # No tool calls -> final response
            if not response.tool_calls:
                await self._add_and_persist(session, Message(
                    role="assistant",
                    content=response.content,
                    model=response.model,
                    usage=response.usage,
                ))
                await self.event_bus.emit(
                    Events.MESSAGE_OUTGOING,
                    {
                        "content": response.content,
                        "session_id": session.id,
                        "model": response.model,
                        "iterations": iteration,
                    },
                )
                logger.info(
                    "llm_response",
                    session_id=session.id,
                    model=response.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    iterations=iteration,
                )

                # Fire-and-forget memory operations (tracked for cleanup)
                if trigger == "user_message":
                    if self.fact_extractor:
                        task = asyncio.create_task(
                            self._safe_extract_facts(session)
                        )
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
                    if self.summarizer:
                        task = asyncio.create_task(
                            self._safe_summarize(session)
                        )
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)

                return response

            # Has tool calls -> execute them
            await self._add_and_persist(session, Message(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls,
                model=response.model,
                usage=response.usage,
            ))

            # Execute tool calls
            for tool_call in response.tool_calls:
                await self.event_bus.emit(
                    Events.TOOL_EXECUTE,
                    {
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "iteration": iteration,
                    },
                )

                result = await self._execute_tool_call(
                    tool_call, session, trigger, plan
                )

                # Add tool result to session (multimodal if images present)
                await self._add_and_persist(session, Message(
                    role="tool",
                    content=self._build_tool_content(result),
                    tool_call_id=tool_call.id,
                ))

                await self.event_bus.emit(
                    Events.TOOL_RESULT,
                    {
                        "tool": tool_call.name,
                        "success": result.success,
                        "duration_ms": result.duration_ms,
                    },
                )

                # Update plan progress if we have a plan
                if plan and hasattr(plan, "status"):
                    from agent.core.planner import PlanStatus

                    if (
                        plan.status == PlanStatus.IN_PROGRESS
                        and plan.current_step < len(plan.steps)
                    ):
                        step = plan.steps[plan.current_step]
                        if result.success:
                            step.status = PlanStatus.COMPLETED
                            step.result = result.output[:500]
                            plan.current_step += 1
                        else:
                            step.status = PlanStatus.FAILED
                            step.error = result.error

            # Continue loop -> next iteration will call LLM with tool results

        # Max iterations reached
        logger.warning("max_iterations_reached", iterations=max_iterations)

        # Force a final response
        await self._add_and_persist(session, Message(
            role="system",
            content=(
                f"You have reached the maximum of {max_iterations} tool-calling "
                "iterations. Summarize what you've accomplished and what remains."
            ),
        ))

        final = await self.llm.completion(
            messages=self._build_messages(
                session, plan, relevant_facts, vector_results, tool_schemas,
            )
        )
        await self._add_and_persist(session, Message(
            role="assistant",
            content=final.content,
            model=final.model,
            usage=final.usage,
        ))
        return final

    async def _execute_tool_call(
        self,
        tool_call: object,
        session: Session,
        trigger: str,
        plan: object | None,
    ) -> ToolResult:
        """Execute a single tool call with error recovery.

        Args:
            tool_call: The tool call from the LLM.
            session: Current session.
            trigger: What triggered this execution.
            plan: Active plan, if any.

        Returns:
            ToolResult from execution.
        """
        from agent.core.session import ToolCall
        from agent.tools.executor import ToolResult

        assert isinstance(tool_call, ToolCall)

        if not self.tool_executor:
            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output="[ERROR] Tool executor not available",
                error="No tool executor configured",
            )

        # Extract channel_user_id from session ID (format: "channel:user_id")
        channel_user_id = None
        if ":" in session.id:
            channel_user_id = session.id.split(":", 1)[1]

        try:
            return await self.tool_executor.execute(
                tool_call=tool_call,
                session_id=session.id,
                trigger=trigger,
                channel_user_id=channel_user_id,
            )
        except Exception as e:
            # Error recovery
            if self.recovery:
                category = self.recovery.classify_error(e, tool_call.name)
                action = self.recovery.get_recovery_action(e, tool_call.name, category)

                if action.action == "retry":
                    result = await self.recovery.execute_recovery(
                        action, tool_call, self.tool_executor, session.id
                    )
                    if result is not None:
                        return result

                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    success=False,
                    output=f"Error: {e}\nRecovery: {action.reason}",
                    error=str(e),
                )

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                success=False,
                output=f"Error: {e}",
                error=str(e),
            )

    def _build_tool_content(
        self, result: ToolResult,
    ) -> str | list[dict[str, Any]]:
        """Build tool message content, with image blocks if present.

        Args:
            result: The tool execution result.

        Returns:
            Plain string or list of content blocks with images.
        """
        if not result.images:
            return result.output
        blocks: list[dict[str, Any]] = []
        if result.output:
            blocks.append({"type": "text", "text": result.output})
        for img in result.images:
            blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.media_type};base64,{img.base64_data}",
                },
            })
        return blocks

    def _build_messages(
        self,
        session: Session,
        plan: object | None,
        facts: list | None = None,
        vector_results: list | None = None,
        tool_schemas: list[dict] | None = None,
    ) -> list[dict]:
        """Build the messages list for the LLM call.

        Includes:
        1. System prompt (always, with soul.md if available)
        2. Memory context: facts + vector results
        3. Active plan context (if any)
        4. Conversation history

        Args:
            session: Current conversation session.
            plan: Active Plan object, if any.
            facts: Relevant facts for context injection.
            vector_results: Semantic search results for context injection.

        Returns:
            List of message dicts for the LLM API.
        """
        # Check if soul.md changed on disk
        if self.soul_loader and self.soul_loader.reload_if_changed():
            self.system_prompt = build_system_prompt(
                self.config,
                soul_content=self.soul_loader.content,
                skill_builder_enabled=self._skill_builder_enabled,
                orchestration_enabled=self._orchestration_enabled,
                platform_capabilities=self._platform_capabilities,
                use_controller=self._use_controller,
            )

        from agent.core.context import build_messages
        from agent.llm.prompts import build_runtime_context

        # Get model name safely (llm may be mocked in tests)
        model = getattr(getattr(self.llm, "config", None), "default", None)
        if not isinstance(model, str):
            model = "claude-sonnet-4-5-20250929"

        # Build runtime context (channel, capabilities, model)
        channel = str(session.metadata.get("channel", "cli"))

        enabled_tools: list[str] = []
        if self.tool_executor:
            with contextlib.suppress(Exception):
                enabled_tools = [
                    t.name for t in self.tool_executor.registry.list_tools()
                    if t.enabled
                ]

        tool_names = set(enabled_tools)
        runtime_ctx = build_runtime_context(
            channel=channel,
            model_name=model,
            enabled_tools=enabled_tools,
            has_memory=self.fact_store is not None,
            has_voice=False,  # not accessible from agent loop
            has_heartbeat=False,  # not accessible from agent loop
            has_browser=bool(tool_names & {"browser_navigate"}),
            has_desktop=bool(tool_names & {"screen_capture", "desktop_capabilities"}),
            has_skills=self._skill_builder_enabled,
            has_orchestration=self._orchestration_enabled,
        )

        system_with_context = f"{self.system_prompt}\n\n{runtime_ctx}"

        return build_messages(
            session=session,
            system_prompt=system_with_context,
            plan=plan,
            tool_schemas=tool_schemas,
            model=model,
            facts=facts,
            vector_results=vector_results,
        )

    async def _safe_extract_facts(self, session: Session) -> None:
        """Extract facts from session, catching all exceptions.

        Args:
            session: The conversation session to extract from.
        """
        try:
            if self.fact_extractor:
                await self.fact_extractor.extract_from_session(session)
        except Exception as e:
            logger.warning("fact_extraction_failed", error=str(e))

    async def _safe_summarize(self, session: Session) -> None:
        """Summarize session if needed, catching all exceptions.

        Args:
            session: The conversation session to summarize.
        """
        try:
            if self.summarizer:
                await self.summarizer.summarize_if_needed(session)
        except Exception as e:
            logger.warning("summarization_failed", error=str(e))

    async def _query_memory(
        self, user_message: str
    ) -> tuple[list | None, list | None]:
        """Query fact store and vector store in parallel.

        Args:
            user_message: The user's message for semantic search.

        Returns:
            Tuple of (relevant_facts, vector_results).
        """
        relevant_facts = None
        vector_results = None

        async def _get_facts() -> list | None:
            if not self.fact_store:
                return None
            try:
                return await self.fact_store.get_relevant(limit=15)
            except Exception as e:
                logger.warning("fact_retrieval_failed", error=str(e))
                return None

        async def _get_vectors() -> list | None:
            if not self.vector_store:
                return None
            try:
                return await self.vector_store.search(user_message, limit=5)
            except Exception as e:
                logger.warning("vector_search_failed", error=str(e))
                return None

        relevant_facts, vector_results = await asyncio.gather(
            _get_facts(), _get_vectors()
        )
        return relevant_facts, vector_results

    async def process_message_stream(
        self,
        user_message: str,
        session: Session,
        trigger: str = "user_message",
        tool_registry_override: Any = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Process a message with streaming LLM responses.

        Same ReAct loop as process_message, but yields StreamEvent objects
        as they occur for real-time streaming to clients.

        Yields:
            StreamEvent with type "chunk", "tool.execute", "tool.result", or "done".
        """
        if self.llm is None:
            raise RuntimeError(
                "AgentLoop.process_message_stream() requires an LLM provider. "
                "Configure an API key or use the Claude SDK backend."
            )

        # Add user message to session
        await self._add_and_persist(
            session, Message(role="user", content=user_message)
        )

        await self.event_bus.emit(
            Events.MESSAGE_INCOMING,
            {"content": user_message, "session_id": session.id},
        )

        # Query memory in parallel
        relevant_facts, vector_results = await self._query_memory(user_message)

        # Check if we should plan
        plan = None
        if (
            self.planner
            and trigger == "user_message"
            and await self.planner.should_plan(user_message, session)
        ):
            plan = await self.planner.create_plan(user_message, session)
            logger.info("plan_created", goal=plan.goal, steps=len(plan.steps))

        # Get tool schemas (use override if provided, e.g. scoped sub-agent registry)
        effective_registry = tool_registry_override or self._default_tool_registry or registry
        tool_schemas = effective_registry.get_tool_schemas(enabled_only=True)

        # ReAct loop with streaming
        iteration = 0
        max_iterations = self.config.max_iterations

        while iteration < max_iterations:
            iteration += 1

            messages = self._build_messages(
                session, plan, relevant_facts, vector_results, tool_schemas
            )

            try:
                kwargs: dict[str, Any] = {}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas

                accumulated_content = ""
                tool_calls = None
                usage: TokenUsage | None = None
                model = ""
                finish_reason = "stop"

                async for chunk in self.llm.stream_completion(
                    messages=messages, **kwargs
                ):
                    if chunk.done:
                        tool_calls = chunk.tool_calls
                        usage = chunk.usage
                        model = chunk.model
                        finish_reason = chunk.finish_reason or "stop"
                    elif chunk.content:
                        accumulated_content += chunk.content
                        yield StreamEvent(
                            type="chunk", content=chunk.content
                        )

            except Exception as e:
                logger.error(
                    "llm_stream_failed", error=str(e), iteration=iteration
                )
                raise

            response = LLMResponse(
                content=accumulated_content,
                model=model,
                tool_calls=tool_calls,
                usage=usage or TokenUsage(0, 0, 0),
                finish_reason=finish_reason,
            )

            # Record cost
            if self.cost_tracker and response.usage:
                channel = str(session.metadata.get("channel", "cli"))
                self.cost_tracker.record(
                    model=response.model or "unknown",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    channel=channel,
                    session_id=session.id,
                )

            # No tool calls -> final response
            if not response.tool_calls:
                await self._add_and_persist(
                    session,
                    Message(
                        role="assistant",
                        content=response.content,
                        model=response.model,
                        usage=response.usage,
                    ),
                )
                await self.event_bus.emit(
                    Events.MESSAGE_OUTGOING,
                    {
                        "content": response.content,
                        "session_id": session.id,
                        "model": response.model,
                        "iterations": iteration,
                    },
                )
                logger.info(
                    "llm_response_streamed",
                    session_id=session.id,
                    model=response.model,
                    input_tokens=response.usage.input_tokens
                    if response.usage
                    else 0,
                    output_tokens=response.usage.output_tokens
                    if response.usage
                    else 0,
                    iterations=iteration,
                )

                # Fire-and-forget memory operations
                if trigger == "user_message":
                    if self.fact_extractor:
                        task = asyncio.create_task(
                            self._safe_extract_facts(session)
                        )
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
                    if self.summarizer:
                        task = asyncio.create_task(
                            self._safe_summarize(session)
                        )
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)

                yield StreamEvent(type="done", response=response)
                return

            # Has tool calls -> execute them
            await self._add_and_persist(
                session,
                Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                    model=response.model,
                    usage=response.usage,
                ),
            )

            for tool_call in response.tool_calls:
                tool_data = {
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "iteration": iteration,
                }
                await self.event_bus.emit(Events.TOOL_EXECUTE, tool_data)
                yield StreamEvent(type="tool.execute", data=tool_data)

                result = await self._execute_tool_call(
                    tool_call, session, trigger, plan
                )

                await self._add_and_persist(
                    session,
                    Message(
                        role="tool",
                        content=self._build_tool_content(result),
                        tool_call_id=tool_call.id,
                    ),
                )

                result_data = {
                    "tool": tool_call.name,
                    "success": result.success,
                    "output": result.output,
                    "duration_ms": result.duration_ms,
                }
                await self.event_bus.emit(Events.TOOL_RESULT, result_data)
                yield StreamEvent(type="tool.result", data=result_data)

                # Update plan progress
                if plan and hasattr(plan, "status"):
                    from agent.core.planner import PlanStatus

                    if (
                        plan.status == PlanStatus.IN_PROGRESS
                        and plan.current_step < len(plan.steps)
                    ):
                        step = plan.steps[plan.current_step]
                        if result.success:
                            step.status = PlanStatus.COMPLETED
                            step.result = result.output[:500]
                            plan.current_step += 1
                        else:
                            step.status = PlanStatus.FAILED
                            step.error = result.error

        # Max iterations reached
        logger.warning("max_iterations_reached", iterations=max_iterations)

        await self._add_and_persist(
            session,
            Message(
                role="system",
                content=(
                    f"You have reached the maximum of {max_iterations} "
                    "tool-calling iterations. Summarize what you've "
                    "accomplished and what remains."
                ),
            ),
        )

        final_content = ""
        final_usage: TokenUsage | None = None
        final_model = ""
        async for chunk in self.llm.stream_completion(
            messages=self._build_messages(
                session, plan, relevant_facts, vector_results, tool_schemas,
            )
        ):
            if chunk.done:
                final_usage = chunk.usage
                final_model = chunk.model
            elif chunk.content:
                final_content += chunk.content
                yield StreamEvent(type="chunk", content=chunk.content)

        final = LLMResponse(
            content=final_content,
            model=final_model,
            usage=final_usage or TokenUsage(0, 0, 0),
        )
        await self._add_and_persist(
            session,
            Message(
                role="assistant",
                content=final.content,
                model=final.model,
                usage=final.usage,
            ),
        )
        yield StreamEvent(type="done", response=final)
