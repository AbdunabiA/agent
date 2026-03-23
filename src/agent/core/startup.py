"""Application lifecycle manager.

Consolidates component initialization, startup, and shutdown
into a single Application class used by ``agent start``.
"""

from __future__ import annotations

import asyncio
import os
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
        self.skill_builder: Any = None
        self.orchestrator: Any = None
        self.proactive: Any = None
        self.trigger_matcher: Any = None
        self.monitor_manager: Any = None
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
        self.controller: Any = None  # ControllerAgent when use_controller=True
        self.app: Any = None  # FastAPI app

        # Phase 4: Memory components
        self.fact_store: Any = None
        self.vector_store: Any = None
        self.soul_loader: Any = None
        self.fact_extractor: Any = None
        self.summarizer: Any = None

        self._channels: list[BaseChannel] = []
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._critical_failures: list[str] = []

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
            logger.warning(
                "workspace_resolve_skipped",
                error=str(e),
                hint="Agent will run without workspace isolation",
            )

        # 0.5. Database (before everything else)
        try:
            self.database = Database(self.config.memory.db_path)
            await self.database.connect()
        except Exception as e:
            logger.error(
                "database_init_failed",
                error=str(e),
                impact="Sessions, facts, audit log, and scheduled tasks will be unavailable",
            )
            self.database = None
            self._critical_failures.append("database")

        # 1. Core infrastructure
        self.event_bus = EventBus()

        # Check if any LiteLLM-compatible API keys exist
        from agent.config import has_litellm_keys

        _llm_available = has_litellm_keys(self.config)
        if _llm_available:
            self.llm = LLMProvider(self.config.models)
        else:
            self.llm = None
            logger.info(
                "litellm_skipped",
                reason="no API keys configured",
                backend=self.config.models.backend,
            )

        # Validate LLM connectivity
        if self.llm:
            try:
                test_ok = await self.llm.test_connection(
                    model=self.config.models.default,
                )
                if not test_ok:
                    logger.warning(
                        "llm_key_validation_failed",
                        hint="API key may be invalid",
                    )
                    self._critical_failures.append("llm_validation")
            except Exception as e:
                logger.warning("llm_key_test_error", error=str(e))

        self.session_store = SessionStore(db=self.database)

        # 2. Safety components
        self.guardrails = Guardrails(self.config.tools)
        self.permissions = PermissionManager(self.config.tools)
        self.audit = AuditLog(db=self.database)
        self.recovery = ErrorRecovery()
        self.cost_tracker = CostTracker()

        # 2.5. Auto-install missing optional dependencies
        from agent.core.deps import ensure_dependencies

        ensure_dependencies()

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

        # 5. Planner (uses LiteLLM or SDK via fallback)
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
                persist_dir=os.path.join(self.config.memory.markdown_dir, "chroma"),
            )
            await self.vector_store.initialize()
        except Exception as e:
            logger.warning("vector_store_init_failed", error=str(e))
            self.vector_store = None

        # Pre-warm embedding model in background to avoid cold-start delay
        if self.vector_store:
            try:
                from agent.memory.embeddings import get_embedding_model

                task = asyncio.create_task(self._warmup_embeddings(get_embedding_model()))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception:
                pass

        # FactExtractor and Summarizer are initially created if LiteLLM
        # is available. If only SDK is available, they're created later
        # in the deferred memory setup (section 6.7) after SDK init.
        self.fact_extractor = (
            FactExtractor(
                llm=self.llm,
                fact_store=self.fact_store,
                enabled=self.config.memory.auto_extract and self.fact_store is not None,
            )
            if self.fact_store and _llm_available and self.llm
            else None
        )

        if self.vector_store and _llm_available and self.llm:
            from agent.memory.summarizer import ConversationSummarizer

            self.summarizer = ConversationSummarizer(llm=self.llm, vector_store=self.vector_store)

        # 6. Agent loop
        # Get platform capabilities for system prompt injection
        platform_capabilities = None
        try:
            from agent.desktop.platform_utils import get_capabilities_summary

            platform_capabilities = get_capabilities_summary()
        except Exception:
            pass

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
            skill_builder_enabled=self.config.skills.builder.enabled,
            orchestration_enabled=self.config.orchestration.enabled,
            platform_capabilities=platform_capabilities,
            use_controller=self.config.orchestration.use_controller,
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
                    self.sdk_service._idle_timeout = sdk_cfg.idle_timeout
                    ok, msg = False, "not checked"
                    for _attempt in range(3):
                        ok, msg = await self.sdk_service.check_available()
                        if ok:
                            break
                        if _attempt < 2:
                            logger.info(
                                "claude_sdk_check_retry",
                                attempt=_attempt + 1,
                                reason=msg,
                            )
                            await asyncio.sleep(2)
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

        # 6.6. Configure universal LLM fallback (LiteLLM → SDK)
        from agent.llm.fallback import configure as configure_llm_fallback

        configure_llm_fallback(llm=self.llm, sdk_service=self.sdk_service)

        # 6.7. Deferred memory setup — create FactExtractor/Summarizer
        # with SDK fallback if they weren't created with LiteLLM above.
        _any_llm = _llm_available or self.sdk_service is not None
        if self.fact_extractor is None and self.fact_store and _any_llm:
            self.fact_extractor = FactExtractor(
                llm=self.llm,
                fact_store=self.fact_store,
                enabled=self.config.memory.auto_extract,
            )
            from agent.tools.builtins.memory import set_fact_store

            set_fact_store(self.fact_store)

        if self.summarizer is None and self.vector_store and _any_llm:
            from agent.memory.summarizer import ConversationSummarizer

            self.summarizer = ConversationSummarizer(llm=self.llm, vector_store=self.vector_store)

        # 7. Scheduler & heartbeat (with persistence and proactive support)
        self.scheduler = TaskScheduler(self.event_bus, database=self.database)

        # Proactive messenger (for heartbeat notifications)
        from agent.core.proactive import ProactiveMessenger

        self.proactive = ProactiveMessenger(event_bus=self.event_bus)

        self.heartbeat = HeartbeatDaemon(
            agent_loop=self.agent_loop,
            config=self.config.agent,
            event_bus=self.event_bus,
            fact_store=self.fact_store,
            scheduler=self.scheduler,
            proactive=self.proactive,
            sdk_service=self.sdk_service,
        )

        # Wire scheduler into tools so the LLM can set reminders
        from agent.tools.builtins.scheduler import set_scheduler

        set_scheduler(self.scheduler)

        # Monitor manager
        from agent.core.monitors import MonitorManager

        self.monitor_manager = MonitorManager(
            event_bus=self.event_bus,
            proactive=self.proactive,
        )
        from agent.tools.builtins.monitor import set_monitor_manager

        set_monitor_manager(self.monitor_manager)

        # Trigger matcher for skill hints
        from agent.skills.triggers import TriggerMatcher

        self.trigger_matcher = TriggerMatcher()

        # Wire event bus into send_file tool so the LLM can send files
        from agent.tools.builtins.send_file import set_file_send_bus

        set_file_send_bus(self.event_bus)

        # Wire event bus + scheduler into telegram posting tools
        from agent.tools.builtins.telegram_post import (
            set_telegram_post_bus,
            set_telegram_post_scheduler,
        )

        set_telegram_post_bus(self.event_bus)
        set_telegram_post_scheduler(self.scheduler)

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

        # 7.6. Register skill triggers
        if self.trigger_matcher and self.skill_manager:
            for skill_info in self.skill_manager.list_skills():
                if skill_info.get("loaded"):
                    loaded = self.skill_manager._loaded.get(skill_info["name"])
                    if loaded and loaded.metadata.triggers:
                        self.trigger_matcher.register_skill(loaded.metadata)

        # 7.7. Skill builder (self-building skills)
        if self.config.skills.builder.enabled and _any_llm:
            try:
                from agent.skills.builder import SkillBuilder
                from agent.tools.builtins.skill_builder import set_skill_builder

                self.skill_builder = SkillBuilder(
                    llm=self.llm,
                    config=self.config.skills.builder,
                    event_bus=self.event_bus,
                    skill_manager=self.skill_manager,
                )
                set_skill_builder(self.skill_builder)
                logger.info("skill_builder_ready")
            except Exception as e:
                logger.warning("skill_builder_init_failed", error=str(e))

        # 7.8. Sub-agent orchestrator
        if self.config.orchestration.enabled:
            try:
                from agent.core.orchestrator import SubAgentOrchestrator
                from agent.core.subagent import AgentTeam

                # Build teams from config + teams/ directory
                from agent.teams.loader import (
                    config_to_team,
                    load_projects_from_directory,
                    load_teams_from_directory,
                    merge_teams,
                )
                from agent.tools.builtins.orchestration import set_orchestrator

                config_teams: list[AgentTeam] = [
                    config_to_team(t) for t in self.config.orchestration.teams
                ]
                file_teams = load_teams_from_directory(
                    self.config.orchestration.teams_directory,
                )
                teams = merge_teams(file_teams, config_teams)

                # Load projects from teams/projects/
                projects = load_projects_from_directory(
                    self.config.orchestration.teams_directory,
                )

                # Create WorkingMemory, AgentTracer, and TaskBoard if database is available
                working_memory = None
                self.tracer = None
                task_board = None
                if self.database:
                    from agent.core.task_board import TaskBoard
                    from agent.core.working_memory import WorkingMemory
                    from agent.observability.tracer import AgentTracer

                    working_memory = WorkingMemory(self.database)
                    self.tracer = AgentTracer(self.database)
                    task_board = TaskBoard(self.database)

                # Create PromptBuilderAgent when SDK and WorkingMemory are available
                prompt_builder = None
                if self.sdk_service is not None:
                    try:
                        from agent.core.prompt_builder_agent import PromptBuilderAgent

                        # RoleRegistry may not exist yet — create a temporary one
                        # for the builder, or reuse the one created below for the
                        # controller.
                        pb_registry = None
                        try:
                            from agent.core.role_registry import RoleRegistry as _PBRoleRegistry

                            pb_registry = _PBRoleRegistry(
                                self.config.orchestration.teams_directory,
                            )
                        except Exception:
                            pass

                        prompt_builder = PromptBuilderAgent(
                            sdk_service=self.sdk_service,
                            working_memory=working_memory,
                            tracer=self.tracer,
                            role_registry=pb_registry,
                        )
                        logger.info("prompt_builder_agent_ready")
                    except Exception as e:
                        logger.warning("prompt_builder_init_failed", error=str(e))

                self.orchestrator = SubAgentOrchestrator(
                    agent_loop=self.agent_loop,
                    config=self.config.orchestration,
                    event_bus=self.event_bus,
                    tool_registry=registry,
                    teams=teams,
                    sdk_service=self.sdk_service,
                    working_memory=working_memory,
                    tracer=self.tracer,
                    task_board=task_board,
                    prompt_builder=prompt_builder,
                )

                # Register projects
                for proj in projects:
                    self.orchestrator.projects[proj.name] = proj

                # Validate project agent references against loaded teams (GAP 18)
                team_names = {t.name for t in teams}
                team_roles: dict[str, set[str]] = {t.name: {r.name for r in t.roles} for t in teams}
                for proj in projects:
                    for stage in proj.stages:
                        for agent_ref in stage.agents:
                            if agent_ref.team not in team_names:
                                logger.warning(
                                    "project_ref_invalid_team",
                                    project=proj.name,
                                    stage=stage.name,
                                    team=agent_ref.team,
                                )
                            elif agent_ref.role not in team_roles.get(agent_ref.team, set()):
                                logger.warning(
                                    "project_ref_invalid_role",
                                    project=proj.name,
                                    stage=stage.name,
                                    team=agent_ref.team,
                                    role=agent_ref.role,
                                )
                        # Validate discussion moderator reference
                        if stage.discussion and stage.discussion.moderator:
                            mod = stage.discussion.moderator
                            if mod.team not in team_names:
                                logger.warning(
                                    "project_ref_invalid_team",
                                    project=proj.name,
                                    stage=stage.name,
                                    team=mod.team,
                                    context="discussion.moderator",
                                )
                            elif mod.role not in team_roles.get(mod.team, set()):
                                logger.warning(
                                    "project_ref_invalid_role",
                                    project=proj.name,
                                    stage=stage.name,
                                    team=mod.team,
                                    role=mod.role,
                                    context="discussion.moderator",
                                )

                # Validate team role tool names against actual registry
                if registry is not None:
                    registered_tools = {td.name for td in registry.list_tools()}
                    for team in teams:
                        for role in team.roles:
                            for tool_name in role.allowed_tools:
                                if tool_name not in registered_tools:
                                    logger.warning(
                                        "team_role_unknown_tool",
                                        team=team.name,
                                        role=role.name,
                                        tool=tool_name,
                                        msg="Tool in allowed_tools not found in registry",
                                    )

                set_orchestrator(self.orchestrator)

                # Wire TaskBoard into collaboration tools
                if task_board is not None:
                    from agent.tools.builtins.collaboration import set_task_board

                    set_task_board(task_board)

                logger.info(
                    "orchestrator_ready",
                    teams=len(teams),
                    projects=len(projects),
                    max_concurrent=self.config.orchestration.max_concurrent_agents,
                )

                # 7.8.1. Controller agent (when use_controller=True)
                if self.config.orchestration.use_controller:
                    try:
                        from agent.core.controller import ControllerAgent
                        from agent.core.role_registry import RoleRegistry
                        from agent.tools.builtins.controller import set_controller

                        role_registry = RoleRegistry(
                            self.config.orchestration.teams_directory,
                        )

                        self.controller = ControllerAgent(
                            orchestrator=self.orchestrator,
                            sdk_service=self.sdk_service,
                            event_bus=self.event_bus,
                            config=self.config.orchestration,
                            role_registry=role_registry,
                        )
                        set_controller(self.controller)

                        # Import controller tools to register them

                        # Hide orchestration tools from the main agent
                        # (but keep them in the global registry so the
                        # controller's ScopedToolRegistry can still see them).
                        from agent.core.orchestrator import ScopedToolRegistry

                        _orchestration_tools = {
                            "spawn_subagent",
                            "spawn_parallel_agents",
                            "spawn_team",
                            "list_agent_teams",
                            "get_subagent_status",
                            "cancel_subagent",
                            "run_project",
                            "list_projects",
                            "consult_agent",
                            "delegate_to_specialist",
                        }
                        scoped = ScopedToolRegistry(
                            parent=registry,
                            denied_tools=_orchestration_tools,
                            exclude_dangerous=False,
                        )
                        self.agent_loop._default_tool_registry = scoped

                        # Also scope the SDK service so the main agent's
                        # SDK path (MCP server + permission checks) hides
                        # orchestration tools — matching the LiteLLM path.
                        if self.sdk_service is not None:
                            self.sdk_service._scoped_tool_registry = scoped

                        logger.info("controller_agent_ready")
                    except Exception as e:
                        logger.warning("controller_init_failed", error=str(e))
                        self.controller = None
            except Exception as e:
                logger.warning("orchestrator_init_failed", error=str(e))

        # 7.9. Project planner service (requires orchestrator + SDK)
        if self.orchestrator and self.sdk_service:
            try:
                from agent.core.project_planner import ProjectPlannerService
                from agent.tools.builtins.planner_tools import set_planner_service

                # Reuse the role_registry from the controller if available
                planner_role_registry = self.controller.role_registry if self.controller else None
                if planner_role_registry is None:
                    try:
                        from agent.core.role_registry import RoleRegistry as _PRoleRegistry

                        planner_role_registry = _PRoleRegistry(
                            self.config.orchestration.teams_directory,
                        )
                    except Exception:
                        pass

                planner_service = ProjectPlannerService(
                    orchestrator=self.orchestrator,
                    sdk_service=self.sdk_service,
                    event_bus=self.event_bus,
                    role_registry=planner_role_registry,
                    working_memory=working_memory,
                )
                set_planner_service(planner_service)

                # Import planner tools to register them

                logger.info("project_planner_service_ready")
            except Exception as e:
                logger.warning("project_planner_init_failed", error=str(e))

        # 8. Voice pipeline
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
            orchestrator=self.orchestrator,
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
                    orchestrator=self.orchestrator,
                    tracer=getattr(self, "tracer", None),
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
        await self.event_bus.emit(
            Events.AGENT_STARTED,
            {
                "channels": [ch.name for ch in self._channels],
            },
        )

        if self._critical_failures:
            logger.warning(
                "application_initialized_with_failures",
                channels=len(self._channels),
                failed_components=self._critical_failures,
            )
        else:
            logger.info(
                "application_initialized",
                channels=len(self._channels),
            )

    @staticmethod
    async def _warmup_embeddings(model: Any) -> None:
        """Pre-load the embedding model so the first vector query is fast."""
        try:
            await model.warmup()
            logger.info("embedding_model_prewarmed")
        except Exception as e:
            logger.debug("embedding_warmup_failed", error=str(e))

    async def start(self) -> None:
        """Start all channels, heartbeat, scheduler, and the gateway server.

        The uvicorn server call is blocking — this method runs until interrupted.
        """
        import uvicorn

        # Start channels
        for channel in self._channels:
            await channel.start()

        # Register channels with proactive messenger
        if self.proactive:
            for channel in self._channels:
                self.proactive.add_channel(channel)

        # Wire up reminder delivery to channels
        if self.scheduler and self._channels:
            self._setup_reminder_delivery()

        # Load persisted scheduled tasks
        if self.scheduler:
            await self.scheduler.load_persisted_tasks()

        # Start heartbeat, scheduler & monitors
        if self.heartbeat:
            await self.heartbeat.start()
        if self.scheduler:
            self.scheduler.start()
        if self.monitor_manager:
            self.monitor_manager.start()

        # Start skill watcher for hot-reload
        if self.skill_manager:
            await self.skill_manager.start_watcher()

        # Start controller agent
        if self.controller:
            await self.controller.start()

        # Start SDK idle reaper
        if self.sdk_service:
            await self.sdk_service.start_reaper()

        # Send startup greeting to all channels
        await self._send_startup_greeting()

        # Run uvicorn (blocking)
        server_config = uvicorn.Config(
            self.app,
            host=self.config.gateway.host,
            port=self.config.gateway.port,
            log_level="info",
        )
        server = uvicorn.Server(server_config)
        await server.serve()

    async def _send_startup_greeting(self) -> None:
        """Send a greeting message through all available channels on startup."""
        from agent.channels.base import OutgoingMessage

        if not self._channels:
            return

        agent_name = self.config.agent.name
        gateway_url = f"http://{self.config.gateway.host}:{self.config.gateway.port}"
        skills_count = len(self.skill_manager.list_skills()) if self.skill_manager else 0

        greeting = (
            f"**{agent_name} is online!** 🟢\n\n"
            f"Gateway: {gateway_url}\n"
            f"Skills loaded: {skills_count}\n"
            f"Channels: {', '.join(ch.name for ch in self._channels)}\n\n"
            f"Send me a message to get started."
        )

        for channel in self._channels:
            try:
                # Telegram: send to all allowed users
                if channel.name == "telegram":
                    allowed = getattr(channel.config, "allowed_users", [])
                    if allowed:
                        for user_id in allowed:
                            await channel.send_message(
                                OutgoingMessage(
                                    content=greeting,
                                    channel_user_id=str(user_id),
                                    parse_mode="Markdown",
                                )
                            )
                    else:
                        # No allowed_users filter — greeting sent on first interaction
                        logger.info(
                            "greeting_skipped_no_users",
                            channel=channel.name,
                            hint="Set allowed_users to receive startup greetings",
                        )
                else:
                    logger.info("greeting_ready", channel=channel.name)
            except Exception as e:
                logger.warning(
                    "startup_greeting_failed",
                    channel=channel.name,
                    error=str(e),
                )

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

            # Check if this is a scheduled post (from schedule_post tool)
            if description.startswith("[scheduled_post:"):
                import json as _json

                try:
                    end = description.index("]", len("[scheduled_post:"))
                    meta = _json.loads(description[len("[scheduled_post:") : end])
                    post_text = description[end + 2 :]  # skip "] "
                    post_data = {
                        "chat_id": meta.get("chat_id", ""),
                        "text": post_text,
                        "photo_path": meta.get("photo_path", ""),
                        "pin": meta.get("pin", False),
                        "parse_mode": meta.get("parse_mode", "Markdown"),
                        "channel": channel or "telegram",
                    }
                    await self.event_bus.emit(Events.CHANNEL_POST, post_data)
                    return
                except (ValueError, _json.JSONDecodeError):
                    pass  # Fall through to normal reminder delivery

            reminder_text = f"⏰ **Reminder:** {description}"
            await target_channel.send_message(
                OutgoingMessage(
                    content=reminder_text,
                    channel_user_id=user_id,
                    parse_mode="Markdown",
                )
            )

        assert self.scheduler is not None
        self.scheduler.set_delivery_callback(deliver_reminder)
        logger.info("reminder_delivery_configured", channels=list(channels_by_name.keys()))

    async def shutdown(self) -> None:
        """Gracefully stop all components."""
        if self.heartbeat:
            await self.heartbeat.stop()

        if self.scheduler:
            self.scheduler.stop()

        if self.monitor_manager:
            self.monitor_manager.stop()

        for channel in self._channels:
            try:
                await channel.stop()
            except Exception as e:
                logger.warning("channel_stop_error", channel=type(channel).__name__, error=str(e))

        if self.skill_manager:
            await self.skill_manager.shutdown()

        # Shut down controller agent
        if self.controller:
            await self.controller.stop()

        # Shut down orchestrator (cancel running sub-agents and async futures)
        if self.orchestrator:
            await self.orchestrator.shutdown()

        # Cancel outstanding background tasks (e.g. embedding warmup)
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # Stop SDK reaper and disconnect persistent clients
        if self.sdk_service:
            try:
                from agent.llm.claude_sdk import ClaudeSDKService

                sdk: ClaudeSDKService = self.sdk_service  # type: ignore[assignment]
                await sdk.stop_reaper()
                for task_id in list(sdk._clients.keys()):
                    await sdk.disconnect_client(task_id)
            except Exception:
                pass

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
