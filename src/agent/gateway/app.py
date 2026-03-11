"""FastAPI application factory for Agent gateway.

Creates and configures the FastAPI app with:
- CORS middleware (from GatewayConfig.cors_origins)
- Auth middleware (Bearer token from GatewayConfig.auth_token)
- Rate limit middleware (60 req/min per IP)
- All component state injected into app.state
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from agent import __version__
from agent.core.session import SessionStore
from agent.gateway.middleware import AuthMiddleware, RateLimitMiddleware
from agent.gateway.routes.api import router as api_router
from agent.gateway.routes.ws import manager as ws_manager
from agent.gateway.routes.ws import router as ws_router
from agent.gateway.routes.ws import setup_event_forwarding

if TYPE_CHECKING:
    from agent.config import AgentConfig
    from agent.core.agent_loop import AgentLoop
    from agent.core.audit import AuditLog
    from agent.core.events import EventBus
    from agent.core.heartbeat import HeartbeatDaemon
    from agent.core.scheduler import TaskScheduler
    from agent.skills.manager import SkillManager
    from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


def create_app(
    config: AgentConfig,
    agent_loop: AgentLoop,
    event_bus: EventBus,
    audit: AuditLog,
    tool_registry: ToolRegistry,
    heartbeat: HeartbeatDaemon | None = None,
    session_store: SessionStore | None = None,
    task_scheduler: TaskScheduler | None = None,
    skill_manager: SkillManager | None = None,
    voice_pipeline: object | None = None,
    fact_store: object | None = None,
    vector_store: object | None = None,
    soul_loader: object | None = None,
    cost_tracker: object | None = None,
    workspace_manager: object | None = None,
    sdk_service: object | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Full agent configuration.
        agent_loop: The agent reasoning loop.
        event_bus: Async event bus.
        audit: Audit log instance.
        tool_registry: Tool registry with registered tools.
        heartbeat: Optional heartbeat daemon.
        session_store: Optional shared session store. Creates a new one if not provided.
        task_scheduler: Optional task scheduler for reminders and cron jobs.
        skill_manager: Optional skill manager for skill lifecycle.
        voice_pipeline: Optional voice pipeline for TTS/STT.
        fact_store: Optional fact store for memory facts.
        vector_store: Optional vector store for semantic search.
        soul_loader: Optional soul.md loader.
        cost_tracker: Optional cost tracker for usage stats.
        workspace_manager: Optional workspace manager.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Agent API",
        version=__version__,
        description="Agent — autonomous AI assistant API",
    )

    # --- CORS middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.gateway.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Rate limit middleware ---
    app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)

    # --- Auth middleware ---
    app.add_middleware(AuthMiddleware, auth_token=config.gateway.auth_token)

    # --- Inject state ---
    app.state.config = config
    app.state.agent_loop = agent_loop
    app.state.event_bus = event_bus
    app.state.audit = audit
    app.state.tool_registry = tool_registry
    app.state.heartbeat = heartbeat
    app.state.session_store = session_store if session_store is not None else SessionStore()
    app.state.task_scheduler = task_scheduler
    app.state.skill_manager = skill_manager
    app.state.voice_pipeline = voice_pipeline
    app.state.fact_store = fact_store
    app.state.vector_store = vector_store
    app.state.soul_loader = soul_loader
    app.state.cost_tracker = cost_tracker
    app.state.workspace_manager = workspace_manager
    app.state.sdk_service = sdk_service

    # --- WebSocket connection manager ---
    app.state.ws_manager = ws_manager

    # --- Include routers ---
    app.include_router(api_router)
    app.include_router(ws_router)

    # --- Bridge EventBus → WebSocket events stream ---
    setup_event_forwarding(event_bus)

    # --- Root redirect ---
    @app.get("/")
    async def root_redirect() -> RedirectResponse:
        """Redirect root to dashboard."""
        return RedirectResponse(url="/dashboard")

    # --- Serve dashboard static files (production build) ---
    dashboard_dist = Path(__file__).parent.parent.parent.parent / "dashboard" / "dist"

    if dashboard_dist.exists() and (dashboard_dist / "index.html").exists():
        # Serve static assets (JS, CSS, images)
        assets_dir = dashboard_dist / "assets"
        if assets_dir.exists():
            app.mount(
                "/dashboard/assets",
                StaticFiles(directory=str(assets_dir)),
                name="dashboard-assets",
            )

        # Serve favicon
        favicon_path = dashboard_dist / "favicon.svg"
        if favicon_path.exists():

            @app.get("/dashboard/favicon.svg")
            async def serve_favicon() -> FileResponse:
                """Serve dashboard favicon."""
                return FileResponse(str(favicon_path), media_type="image/svg+xml")

        @app.get("/dashboard/{path:path}")
        @app.get("/dashboard")
        async def serve_dashboard(path: str = "") -> FileResponse:
            """Serve the dashboard SPA. All routes return index.html for client-side routing."""
            return FileResponse(str(dashboard_dist / "index.html"))

        logger.info("dashboard_mounted", path=str(dashboard_dist))
    else:

        @app.get("/dashboard")
        @app.get("/dashboard/{path:path}")
        async def dashboard_not_found(path: str = "") -> JSONResponse:
            """Dashboard not built — return helpful error."""
            return JSONResponse(
                {"error": "Dashboard not built. Run: cd dashboard && npm run build"},
                status_code=404,
            )

    logger.info(
        "gateway_created",
        host=config.gateway.host,
        port=config.gateway.port,
        auth="enabled" if config.gateway.auth_token else "open",
        cors_origins=config.gateway.cors_origins,
    )

    return app
