"""Application lifecycle manager.

Consolidates component initialization, startup, and shutdown
into a single Application class used by ``agent start``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from agent.core.events import EventBus, Events
from agent.core.session import SessionStore
from agent.workspaces.config import ResolvedWorkspace
from agent.workspaces.isolation import WorkspaceIsolation
from agent.workspaces.manager import WorkspaceManager

if TYPE_CHECKING:
    from agent.channels.base import BaseChannel
    from agent.config import AgentConfig
    from agent.core.agent_loop import AgentLoop
    from agent.core.audit import AuditLog
    from agent.core.guardrails import Guardrails
    from agent.core.heartbeat import HeartbeatDaemon
    from agent.core.permissions import PermissionManager
    from agent.core.planner import Planner
    from agent.core.recovery import ErrorRecovery
    from agent.core.scheduler import TaskScheduler
    from agent.llm.provider import LLMProvider
    from agent.memory.database import Database
    from agent.skills.manager import SkillManager
    from agent.tools.executor import ToolExecutor

logger = structlog.get_logger(__name__)


class Application:
    """Full agent application lifecycle.

    Wires all components together and manages startup/shutdown ordering.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.workspace: ResolvedWorkspace | None = None
        self.workspace_isolation: WorkspaceIsolation | None = None

        # Components — populated by initialize()
        self.voice_pipeline: Any = None
        self.database: Database | None = None
        self.event_bus: EventBus | None = None
        self.llm: LLMProvider | None = None
        self.session_store: SessionStore | None = None
        self.guardrails: Guardrails | None = None
        self.permissions: PermissionManager | None = None
        self.audit: AuditLog | None = None
        self.recovery: ErrorRecovery | None = None
        self.tool_executor: ToolExecutor | None = None
        self.planner: Planner | None = None
        self.agent_loop: AgentLoop | None = None
        self.scheduler: TaskScheduler | None = None
        self.heartbeat: HeartbeatDaemon | None = None
        self.skill_manager: SkillManager | None = None
        self.workspace_manager: Any = None
        self.cost_tracker: Any = None
        self.sdk_service: Any = None  # ClaudeSDKService when backend=claude-sdk
        self.app: Any = None  # FastAPI app

        # Phase 4: Memory components
        self.fact_store: Any = None
        self.vector_store: Any = None
        self.soul_loader: Any = None
        self.fact_extractor: Any = None
        self.summarizer: Any = None

        self._channels: list[BaseChannel] = []

    async def initialize(self) -> None:
        """Wire all components in dependency order.

        Creates Database, EventBus, LLM, SessionStore, safety components,
        tools, planner, agent loop, scheduler, heartbeat,
        gateway app, and channels.
        """
        from agent.core.agent_loop import AgentLoop
        from agent.core.audit import AuditLog
        from agent.core.cost_tracker import CostTracker
        from agent.core.guardrails import Guardrails
        from agent.core.heartbeat import HeartbeatDaemon
        from agent.core.permissions import PermissionManager
        from agent.core.planner import Planner
        from agent.core.recovery import ErrorRecovery
        from agent.core.scheduler import TaskScheduler
        from agent.gateway.app import create_app
        from agent.llm.provider import LLMProvider
        from agent.memory.database import Database
        from agent.skills.manager import SkillManager
        from agent.tools.executor import ToolExecutor
        from agent.tools.registry import registry

        # 0. Workspace setup
        ws_manager = WorkspaceManager(self.config)
        ws_manager.ensure_default()
        self.workspace_manager = ws_manager
        try:
            self.workspace = ws_manager.resolve(self.config.workspaces.default)
            self.workspace_isolation = WorkspaceIsolation(self.workspace)
            logger.info(
                "workspace_active",
                workspace=self.workspace.name,
                display_name=self.workspace.display_name,
            )
        except Exception as e:
            logger.debug("workspace_resolve_skipped", error=str(e))

        # 0.5. Database (before everything else)
        try:
            self.database = Database(self.config.memory.db_path)
            await self.database.connect()
        except Exception as e:
            logger.warning("database_init_failed", error=str(e))
            self.database = None

        # 1. Core infrastructure
        self.event_bus = EventBus()
        self.llm = LLMProvider(self.config.models)
        self.session_store = SessionStore(db=self.database)

        # 2. Safety components
        self.guardrails = Guardrails(self.config.tools)
        self.permissions = PermissionManager(self.config.tools)
        self.audit = AuditLog(db=self.database)
        self.recovery = ErrorRecovery()
        self.cost_tracker = CostTracker()

        # 3. Register built-in tools
        import agent.tools.builtins  # noqa: F401

        # 4. Tool executor
        self.tool_executor = ToolExecutor(
            registry=registry,
            config=self.config.tools,
            event_bus=self.event_bus,
            audit=self.audit,
            permissions=self.permissions,
            guardrails=self.guardrails,
        )

        # 5. Planner
        self.planner = Planner(llm=self.llm, config=self.config.agent)

        # 5.5. Phase 4: Memory components
        from agent.memory.extraction import FactExtractor
        from agent.memory.soul import SoulLoader
        from agent.memory.store import FactStore

        if self.database:
            self.fact_store = FactStore(self.database)

            # Wire the fact store into memory tools so they can access it
            from agent.tools.builtins.memory import set_fact_store
            set_fact_store(self.fact_store)

        self.soul_loader = SoulLoader(self.config.memory.soul_path)

        try:
            from agent.memory.vectors import VectorStore

            self.vector_store = VectorStore(
                persist_dir=self.config.memory.markdown_dir + "chroma",
            )
            await self.vector_store.initialize()
        except Exception as e:
            logger.warning("vector_store_init_failed", error=str(e))
            self.vector_store = None

        self.fact_extractor = FactExtractor(
            llm=self.llm,
            fact_store=self.fact_store,
            enabled=self.config.memory.auto_extract and self.fact_store is not None,
        ) if self.fact_store else None

        if self.vector_store:
            from agent.memory.summarizer import ConversationSummarizer

            self.summarizer = ConversationSummarizer(
                llm=self.llm, vector_store=self.vector_store
            )

        # 6. Agent loop
        self.agent_loop = AgentLoop(
            llm=self.llm,
            config=self.config.agent,
            event_bus=self.event_bus,
            tool_executor=self.tool_executor,
            planner=self.planner,
            recovery=self.recovery,
            guardrails=self.guardrails,
            fact_store=self.fact_store,
            vector_store=self.vector_store,
            soul_loader=self.soul_loader,
            fact_extractor=self.fact_extractor,
            summarizer=self.summarizer,
            cost_tracker=self.cost_tracker,
            session_store=self.session_store,
        )

        # 6.5. Claude SDK service (when backend=claude-sdk)
        if self.config.models.backend == "claude-sdk":
            try:
                from agent.llm.claude_sdk import ClaudeSDKService, sdk_available

                if sdk_available():
                    sdk_cfg = self.config.models.claude_sdk
                    self.sdk_service = ClaudeSDKService(
                        working_dir=sdk_cfg.working_dir,
                        max_turns=sdk_cfg.max_turns,
                        permission_mode=sdk_cfg.permission_mode,
                        model=sdk_cfg.model,
                        claude_auth_dir=sdk_cfg.claude_auth_dir,
                        tool_registry=registry,
                        soul_loader=self.soul_loader,
                        fact_store=self.fact_store,
                        vector_store=self.vector_store,
                        fact_extractor=self.fact_extractor,
                        cost_tracker=self.cost_tracker,
                        event_bus=self.event_bus,
                    )
                    ok, msg = await self.sdk_service.check_available()
                    if ok:
                        logger.info("claude_sdk_ready", working_dir=sdk_cfg.working_dir)
                    else:
                        logger.warning("claude_sdk_unavailable", reason=msg)
                        self.sdk_service = None
                else:
                    logger.warning(
                        "claude_sdk_not_installed",
                        hint="pip install claude-agent-sdk",
                    )
            except Exception as e:
                logger.warning("claude_sdk_init_failed", error=str(e))
                self.sdk_service = None

        # 7. Scheduler & heartbeat
        self.scheduler = TaskScheduler(self.event_bus)
        self.heartbeat = HeartbeatDaemon(
            agent_loop=self.agent_loop,
            config=self.config.agent,
            event_bus=self.event_bus,
        )

        # Wire scheduler into tools so the LLM can set reminders
        from agent.tools.builtins.scheduler import set_scheduler
        set_scheduler(self.scheduler)

        # Wire event bus into send_file tool so the LLM can send files
        from agent.tools.builtins.send_file import set_file_send_bus
        set_file_send_bus(self.event_bus)

        # 7.5. Skill manager
        self.skill_manager = SkillManager(
            config=self.config.skills,
            tool_registry=registry,
            event_bus=self.event_bus,
            scheduler=self.scheduler,
        )
        try:
            await self.skill_manager.discover_and_load()
        except Exception as e:
            logger.warning("skill_manager_init_failed", error=str(e))

        # 7.6. Voice pipeline
        try:
            from agent.voice.pipeline import VoicePipeline

            self.voice_pipeline = VoicePipeline(self.config.voice, self.event_bus)
            logger.info(
                "voice_pipeline_ready",
                stt=self.config.voice.stt.provider,
                tts=self.config.voice.tts.provider if self.config.voice.tts.enabled else "disabled",
            )
        except Exception as e:
            logger.warning("voice_pipeline_init_failed", error=str(e))
            self.voice_pipeline = None

        # 8. Gateway app
        self.app = create_app(
            config=self.config,
            agent_loop=self.agent_loop,
            event_bus=self.event_bus,
            audit=self.audit,
            tool_registry=registry,
            heartbeat=self.heartbeat,
            session_store=self.session_store,
            skill_manager=self.skill_manager,
            voice_pipeline=self.voice_pipeline,
            fact_store=self.fact_store,
            vector_store=self.vector_store,
            soul_loader=self.soul_loader,
            cost_tracker=self.cost_tracker,
            workspace_manager=self.workspace_manager,
            sdk_service=self.sdk_service,
        )

        # 9. Telegram channel
        if self.config.channels.telegram.enabled and self.config.channels.telegram.token:
            try:
                from agent.channels.telegram import TelegramChannel

                telegram = TelegramChannel(
                    config=self.config.channels.telegram,
                    event_bus=self.event_bus,
                    session_store=self.session_store,
                    agent_loop=self.agent_loop,
                    heartbeat=self.heartbeat,
                    voice_pipeline=self.voice_pipeline,
                    sdk_service=self.sdk_service,
                    scheduler=self.scheduler,
                    audit_log=self.audit,
                    cost_tracker=self.cost_tracker,
                )
                self._channels.append(telegram)
                self.permissions.set_approval_channel(telegram)
                logger.info("telegram_channel_configured")
            except Exception as e:
                logger.warning("telegram_init_failed", error=str(e))

        # 10. WebChat channel
        if self.config.channels.webchat.enabled:
            try:
                from agent.channels.webchat import WebChatChannel

                webchat = WebChatChannel(
                    config=self.config.channels.webchat,
                    event_bus=self.event_bus,
                    session_store=self.session_store,
                    agent_loop=self.agent_loop,
                )
                self._channels.append(webchat)
                logger.info("webchat_channel_configured")
            except Exception as e:
                logger.warning("webchat_init_failed", error=str(e))

        # 11. Emit started event
        await self.event_bus.emit(Events.AGENT_STARTED, {
            "channels": [ch.name for ch in self._channels],
        })

        logger.info(
            "application_initialized",
            channels=len(self._channels),
        )

    async def start(self) -> None:
        """Start all channels, heartbeat, scheduler, and the gateway server.

        The uvicorn server call is blocking — this method runs until interrupted.
        """
        import uvicorn

        # Start channels
        for channel in self._channels:
            await channel.start()

        # Wire up reminder delivery to channels
        if self.scheduler and self._channels:
            self._setup_reminder_delivery()

        # Start heartbeat & scheduler
        if self.heartbeat:
            await self.heartbeat.start()
        if self.scheduler:
            self.scheduler.start()

        # Start skill watcher for hot-reload
        if self.skill_manager:
            await self.skill_manager.start_watcher()

        # Run uvicorn (blocking)
        server_config = uvicorn.Config(
            self.app,
            host=self.config.gateway.host,
            port=self.config.gateway.port,
            log_level="info",
        )
        server = uvicorn.Server(server_config)
        await server.serve()

    def _setup_reminder_delivery(self) -> None:
        """Wire scheduled task delivery to messaging channels.

        When a reminder fires, the scheduler calls this callback
        to send the reminder message to the user via the appropriate channel.
        """
        from agent.channels.base import OutgoingMessage

        channels_by_name = {ch.name: ch for ch in self._channels}

        async def deliver_reminder(
            description: str,
            channel: str | None,
            user_id: str | None,
        ) -> None:
            # Determine which channel to deliver to
            target_channel = None
            if channel and channel in channels_by_name:
                target_channel = channels_by_name[channel]
            elif self._channels:
                # Default to first available channel
                target_channel = self._channels[0]

            if not target_channel or not user_id:
                logger.warning(
                    "reminder_delivery_skipped",
                    reason="no channel or user_id",
                    channel=channel,
                    user_id=user_id,
                )
                return

            reminder_text = f"⏰ **Reminder:** {description}"
            await target_channel.send_message(
                OutgoingMessage(
                    content=reminder_text,
                    channel_user_id=user_id,
                    parse_mode="Markdown",
                )
            )

        self.scheduler.set_delivery_callback(deliver_reminder)
        logger.info("reminder_delivery_configured", channels=list(channels_by_name.keys()))

    async def shutdown(self) -> None:
        """Gracefully stop all components."""
        if self.heartbeat:
            await self.heartbeat.stop()

        if self.scheduler:
            self.scheduler.stop()

        for channel in self._channels:
            try:
                await channel.stop()
            except Exception as e:
                logger.warning("channel_stop_error", channel=type(channel).__name__, error=str(e))

        if self.skill_manager:
            await self.skill_manager.shutdown()

        # Cleanup browser resources
        try:
            from agent.tools.builtins.browser import cleanup_browser
            await cleanup_browser()
        except Exception:
            pass

        if self.event_bus:
            await self.event_bus.emit(Events.AGENT_STOPPED, {})

        if self.database:
            await self.database.close()

        logger.info("application_shutdown")
