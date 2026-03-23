"""REST API endpoints for Agent gateway.

All endpoints are prefixed with /api/v1.
Public: /health
Authenticated: all others (when auth_token is configured).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from agent import __version__
from agent.config import (
    config_to_dict_masked,
    get_editable_config_meta,
    update_config_section,
)
from agent.core.agent_loop import AgentLoop
from agent.core.audit import AuditLog
from agent.core.heartbeat import HeartbeatDaemon
from agent.core.session import SessionStore
from agent.gateway.dependencies import (
    get_agent_loop,
    get_audit_log,
    get_config,
    get_cost_tracker,
    get_fact_store,
    get_heartbeat,
    get_session_store,
    get_skill_manager,
    get_soul_loader,
    get_task_scheduler,
    get_tool_registry,
    get_vector_store,
    get_voice_pipeline,
    get_workspace_manager,
)
from agent.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1")

# Track server start time for uptime
_start_time = time.time()

# Health check cache (result + timestamp)
_health_cache: dict[str, Any] = {}
_health_cache_ttl: float = 10.0  # seconds


# --- Request / Response models ---


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = Field(None, description="Existing session ID to continue")
    channel: str = Field("api", description="Channel identifier")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    response: str
    session_id: str
    model: str
    usage: dict[str, int]


class ControlRequest(BaseModel):
    """Request body for POST /control."""

    action: str = Field(..., description="Action: pause, resume, mute, unmute")


class SessionSummary(BaseModel):
    """Summary of a session for listing."""

    id: str
    channel: str
    message_count: int
    total_tokens: int
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str
    model: str | None = None
    timestamp: str
    tool_calls: list[dict[str, Any]] | None = None


# --- Endpoints ---


@router.get("/health")
async def health(
    request: Request,
    session_store: SessionStore = Depends(get_session_store),  # noqa: B008
) -> dict[str, Any]:
    """Health check — no auth required.

    Includes database connectivity check, disk space, and DB size.
    Results are cached for 10 seconds to avoid excessive checks.
    """
    # Return cached result if fresh enough
    cached_at = _health_cache.get("_cached_at", 0.0)
    if _health_cache and (time.time() - cached_at) < _health_cache_ttl:
        return _health_cache["result"]

    status = "ok"
    db_status = "not_configured"
    db_size_mb: float | None = None

    # Check database connectivity via the session store's underlying DB
    if session_store._db is not None:
        try:
            async with session_store._db.db.execute("SELECT 1") as cursor:
                await cursor.fetchone()
            db_status = "ok"

            # Measure database size
            try:
                async with session_store._db.db.execute(
                    "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        db_size_mb = round(row[0] / (1024 * 1024), 2)
            except Exception:
                pass
        except Exception:
            db_status = "error"
            status = "degraded"

    # Check disk space
    try:
        disk_usage = shutil.disk_usage("/")
        disk_free_gb = round(disk_usage.free / (1024**3), 2)
    except Exception:
        disk_free_gb = -1.0

    if disk_free_gb != -1.0 and disk_free_gb < 1.0:
        status = "degraded"

    # Check LLM availability
    agent_loop = getattr(request.app.state, "agent_loop", None)
    llm_status = "ok" if agent_loop and getattr(agent_loop, "llm", None) else "not_configured"

    result: dict[str, Any] = {
        "status": status,
        "version": __version__,
        "uptime_seconds": int(time.time() - _start_time),
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "llm": llm_status,
        "disk_free_gb": disk_free_gb,
    }
    if db_size_mb is not None:
        result["db_size_mb"] = db_size_mb

    # Cache the result
    _health_cache["result"] = result
    _health_cache["_cached_at"] = time.time()

    return result


@router.get("/status")
async def status(
    session_store: SessionStore = Depends(get_session_store),  # noqa: B008
    heartbeat: HeartbeatDaemon | None = Depends(get_heartbeat),  # noqa: B008
    tool_registry: ToolRegistry = Depends(get_tool_registry),  # noqa: B008
) -> dict[str, Any]:
    """Agent status — sessions, heartbeat, tools."""
    tools = tool_registry.list_tools()

    heartbeat_info: dict[str, Any] = {"enabled": False}
    if heartbeat:
        heartbeat_info = {
            "enabled": heartbeat.is_enabled,
            "last_tick": heartbeat.last_tick.isoformat() if heartbeat.last_tick else None,
        }

    return {
        "status": "running",
        "active_sessions": session_store.active_count,
        "heartbeat": heartbeat_info,
        "tools": {
            "total": len(tools),
            "enabled": sum(1 for t in tools if t.enabled),
        },
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    agent_loop: AgentLoop = Depends(get_agent_loop),  # noqa: B008
    session_store: SessionStore = Depends(get_session_store),  # noqa: B008
) -> ChatResponse:
    """Send a message and get a response."""
    session = await session_store.get_or_create(
        session_id=body.session_id,
        channel=body.channel,
    )

    try:
        response = await agent_loop.process_message(body.message, session)
    except Exception as e:
        logger.error("chat_error", error=str(e), session_id=session.id)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from e

    return ChatResponse(
        response=response.content,
        session_id=session.id,
        model=response.model or "unknown",
        usage={
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
    )


@router.get("/conversations")
async def list_conversations(
    channel: str | None = None,
    limit: int = 50,
    session_store: SessionStore = Depends(get_session_store),  # noqa: B008
) -> list[SessionSummary]:
    """List conversation sessions."""
    sessions = await session_store.list_sessions(channel=channel, limit=limit)
    return [
        SessionSummary(
            id=s.id,
            channel=str(s.metadata.get("channel", "unknown")),
            message_count=s.message_count,
            total_tokens=s.total_tokens,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/conversations/{session_id}/messages")
async def get_messages(
    session_id: str,
    limit: int = 50,
    session_store: SessionStore = Depends(get_session_store),  # noqa: B008
) -> list[MessageOut]:
    """Get messages from a specific conversation."""
    session = await session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session.messages[-limit:]
    return [
        MessageOut(
            role=m.role,
            content=m.content,
            model=m.model,
            timestamp=m.timestamp.isoformat(),
            tool_calls=(
                [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls]
                if m.tool_calls
                else None
            ),
        )
        for m in messages
    ]


@router.get("/audit")
async def get_audit(
    limit: int = 50,
    tool_name: str | None = None,
    status: str | None = None,
    audit: AuditLog = Depends(get_audit_log),  # noqa: B008
) -> list[dict[str, Any]]:
    """Get audit log entries."""
    entries = await audit.get_entries(limit=limit, tool_name=tool_name, status=status)
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "tool_name": e.tool_name,
            "status": e.status,
            "duration_ms": e.duration_ms,
            "trigger": e.trigger,
            "error": e.error,
        }
        for e in entries
    ]


@router.get("/audit/stats")
async def get_audit_stats(
    audit: AuditLog = Depends(get_audit_log),  # noqa: B008
) -> dict[str, Any]:
    """Get audit statistics."""
    return await audit.get_stats()


@router.get("/tools")
async def list_tools(
    tool_registry: ToolRegistry = Depends(get_tool_registry),  # noqa: B008
) -> list[dict[str, Any]]:
    """List registered tools."""
    tools = tool_registry.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "tier": t.tier.value,
            "enabled": t.enabled,
            "parameters": t.parameters,
        }
        for t in tools
    ]


@router.post("/control")
async def control(
    body: ControlRequest,
    heartbeat: HeartbeatDaemon | None = Depends(get_heartbeat),  # noqa: B008
) -> dict[str, str]:
    """Control agent behavior — pause/resume/mute/unmute heartbeat."""
    if body.action not in ("pause", "resume", "mute", "unmute"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {body.action}. Must be: pause, resume, mute, unmute",
        )

    if not heartbeat:
        raise HTTPException(status_code=503, detail="Heartbeat daemon not available")

    if body.action in ("pause", "mute"):
        heartbeat.disable()
        return {"status": "ok", "message": "Heartbeat paused"}
    else:
        heartbeat.enable()
        return {"status": "ok", "message": "Heartbeat resumed"}


@router.get("/config")
async def get_masked_config(
    config: Any = Depends(get_config),  # noqa: B008
) -> dict[str, Any]:
    """Get masked configuration."""
    return config_to_dict_masked(config)


@router.get("/config/editable")
async def get_editable_config(
    config: Any = Depends(get_config),  # noqa: B008
) -> dict[str, Any]:
    """Get config with editability metadata per field.

    Returns sections with fields annotated by type, editable flag,
    current value (secrets masked), and options for enum-like fields.
    """
    return get_editable_config_meta(config)


class ConfigSectionUpdate(BaseModel):
    """Request body for PUT /config/{section}."""

    data: dict[str, Any] = Field(..., description="New values for the config section")


@router.put("/config/{section}")
async def update_config(
    section: str,
    body: ConfigSectionUpdate,
) -> dict[str, Any]:
    """Update a config section.

    Secret fields (API keys, tokens) are ignored — set those via .env.
    Returns the full updated masked config.
    """
    try:
        updated = update_config_section(section, body.data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("config_update_failed", section=section, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update config: {e}") from e

    return {
        "success": True,
        "section": section,
        "config": config_to_dict_masked(updated),
    }


# --- Memory endpoints ---


@router.get("/memory/facts")
async def list_facts(
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    q: str | None = None,
    fact_store: Any = Depends(get_fact_store),  # noqa: B008
) -> dict[str, Any]:
    """List memory facts with optional filtering."""
    if not fact_store:
        return {"facts": [], "total": 0}

    if q:
        facts = await fact_store.search(q, limit=limit, offset=offset)
    elif category:
        facts = await fact_store.get_by_category(category, limit=limit, offset=offset)
    else:
        facts = await fact_store.get_all(limit=limit, offset=offset)

    total = await fact_store.count()

    return {
        "facts": [
            {
                "id": f.id,
                "key": f.key,
                "value": f.value,
                "category": f.category,
                "confidence": f.confidence,
                "source": f.source,
                "created_at": f.created_at.isoformat(),
                "updated_at": f.updated_at.isoformat(),
            }
            for f in facts
        ],
        "total": total,
    }


@router.get("/memory/search")
async def search_memory(
    q: str,
    limit: int = 10,
    vector_store: Any = Depends(get_vector_store),  # noqa: B008
) -> dict[str, Any]:
    """Semantic search across ChromaDB vectors."""
    if not vector_store:
        return {"results": []}

    try:
        results = await vector_store.search(q, limit=limit)
    except Exception as e:
        logger.warning("vector_search_failed", error=str(e))
        return {"results": []}

    return {
        "results": [
            {
                "text": r.text,
                "similarity": round(r.score, 4),
                "metadata": r.metadata,
            }
            for r in results
        ],
    }


@router.delete("/memory/facts/{fact_id}")
async def delete_fact(
    fact_id: str,
    fact_store: Any = Depends(get_fact_store),  # noqa: B008
) -> dict[str, bool]:
    """Delete a specific memory fact by ID."""
    if not fact_store:
        raise HTTPException(status_code=503, detail="Fact store not available")

    deleted = await fact_store.delete_by_id(fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {"success": True}


@router.get("/memory/stats")
async def memory_stats(
    fact_store: Any = Depends(get_fact_store),  # noqa: B008
    vector_store: Any = Depends(get_vector_store),  # noqa: B008
    soul_loader: Any = Depends(get_soul_loader),  # noqa: B008
) -> dict[str, Any]:
    """Get memory system statistics."""
    facts_count = 0
    vectors_count = 0
    soul_loaded = False

    if fact_store:
        with contextlib.suppress(Exception):
            facts_count = await fact_store.count()

    if vector_store:
        with contextlib.suppress(Exception):
            vectors_count = await vector_store.count()

    if soul_loader:
        # Ensure load() has been called so path is resolved
        soul_loader.load()
        soul_loaded = soul_loader.path is not None

    return {
        "facts_count": facts_count,
        "vectors_count": vectors_count,
        "soul_loaded": soul_loaded,
    }


# --- Stats endpoints ---


@router.get("/stats/costs")
async def get_cost_stats(
    period: str = "day",
    cost_tracker: Any = Depends(get_cost_tracker),  # noqa: B008
) -> dict[str, Any]:
    """Get token usage and cost statistics."""
    if not cost_tracker:
        return {
            "total_cost": 0.0,
            "total_tokens": {"input": 0, "output": 0},
            "total_calls": 0,
            "period": period,
            "by_time": [],
            "by_model": [],
            "by_channel": [],
        }

    return cost_tracker.get_stats(period=period)


@router.get("/stats/timeline")
async def get_timeline(
    limit: int = 100,
    after: str | None = None,
    before: str | None = None,
    event_types: str | None = None,
    audit: AuditLog = Depends(get_audit_log),  # noqa: B008
) -> dict[str, Any]:
    """Get timeline of agent events from audit log."""
    entries = await audit.get_entries(limit=limit)

    # Filter by time range
    if after:
        after_dt = datetime.fromisoformat(after)
        entries = [e for e in entries if e.timestamp >= after_dt]
    if before:
        before_dt = datetime.fromisoformat(before)
        entries = [e for e in entries if e.timestamp <= before_dt]

    # Filter by event types
    type_filter: set[str] | None = None
    if event_types:
        type_filter = set(event_types.split(","))

    events: list[dict[str, Any]] = []
    for e in entries:
        event_type = f"tool.{e.status}" if e.tool_name else "system"

        if type_filter and event_type not in type_filter:
            continue

        icon = "wrench"
        if e.status == "error":
            icon = "alert-circle"
        elif e.status == "success":
            icon = "check-circle"
        elif e.status == "denied":
            icon = "x-circle"

        events.append(
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "event": f"tool.{e.status}",
                "description": f"{e.tool_name} → {e.status} ({e.duration_ms}ms)",
                "details": {
                    "tool": e.tool_name,
                    "status": e.status,
                    "duration_ms": e.duration_ms,
                    "trigger": e.trigger,
                    "error": e.error,
                },
                "icon": icon,
            }
        )

    return {"events": events[:limit]}


# --- Soul endpoints ---


class SoulUpdateRequest(BaseModel):
    """Request body for PUT /soul."""

    content: str = Field(..., min_length=1)


@router.get("/soul")
async def get_soul(
    soul_loader: Any = Depends(get_soul_loader),  # noqa: B008
) -> dict[str, Any]:
    """Get soul.md content."""
    if not soul_loader:
        return {
            "content": "",
            "loaded": False,
            "path": "",
            "last_modified": "",
        }

    content = soul_loader.content
    path = str(soul_loader.path) if soul_loader.path else ""
    last_modified = ""
    if soul_loader.path:

        def _get_mtime() -> str:
            soul_p = Path(soul_loader.path)  # type: ignore[arg-type]
            if soul_p.exists():
                return datetime.fromtimestamp(soul_p.stat().st_mtime).isoformat()
            return ""

        last_modified = await asyncio.to_thread(_get_mtime)

    return {
        "content": content,
        "loaded": bool(content),
        "path": path,
        "last_modified": last_modified,
    }


@router.put("/soul")
async def update_soul(
    body: SoulUpdateRequest,
    soul_loader: Any = Depends(get_soul_loader),  # noqa: B008
) -> dict[str, Any]:
    """Update soul.md content."""
    if not soul_loader:
        raise HTTPException(status_code=503, detail="Soul loader not available")

    try:
        if hasattr(soul_loader, "async_update"):
            await soul_loader.async_update(body.content)
        else:
            await asyncio.to_thread(soul_loader.update, body.content)
    except Exception as e:
        logger.error("soul_update_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update soul: {e}") from e

    return {"success": True, "content": body.content}


# --- Tool toggle endpoint ---


class ToolToggleRequest(BaseModel):
    """Request body for PUT /tools/{tool_name}/toggle."""

    enabled: bool


@router.put("/tools/{tool_name}/toggle")
async def toggle_tool(
    tool_name: str,
    body: ToolToggleRequest,
    tool_registry: ToolRegistry = Depends(get_tool_registry),  # noqa: B008
) -> dict[str, Any]:
    """Enable or disable a tool."""
    tool = tool_registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    try:
        if body.enabled:
            tool_registry.enable_tool(tool_name)
        else:
            tool_registry.disable_tool(tool_name)
    except Exception as e:
        logger.error("tool_toggle_failed", tool=tool_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to toggle tool: {e}") from e

    return {"success": True, "name": tool_name, "enabled": body.enabled}


# --- Task endpoints ---


class TaskCreateRequest(BaseModel):
    """Request body for POST /tasks."""

    type: str = Field(..., description="Task type: reminder or cron")
    description: str = Field(..., min_length=1)
    schedule: str = Field(..., description="ISO datetime for reminder, cron expression for cron")
    channel: str | None = Field(None, description="Channel to send notification to")
    user_id: str | None = Field(None, description="User ID to deliver reminder to")


@router.get("/tasks")
async def list_tasks(
    scheduler: Any = Depends(get_task_scheduler),  # noqa: B008
) -> list[dict[str, Any]]:
    """List scheduled tasks."""
    if not scheduler:
        return []

    tasks = scheduler.list_tasks()
    return [
        {
            "id": t.id,
            "description": t.description,
            "type": t.type,
            "schedule": t.schedule,
            "status": t.status,
            "channel": t.channel,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "next_run": t.next_run.isoformat() if t.next_run else None,
            "last_run": t.last_run.isoformat() if t.last_run else None,
        }
        for t in tasks
    ]


@router.post("/tasks")
async def create_task(
    body: TaskCreateRequest,
    scheduler: Any = Depends(get_task_scheduler),  # noqa: B008
) -> dict[str, Any]:
    """Create a new scheduled task (reminder or cron)."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not available")

    if body.type not in ("reminder", "cron"):
        raise HTTPException(status_code=400, detail="Task type must be 'reminder' or 'cron'")

    try:
        if body.type == "reminder":
            run_at = datetime.fromisoformat(body.schedule)
            task = await scheduler.add_reminder(
                description=body.description,
                run_at=run_at,
                channel=body.channel,
                user_id=body.user_id,
            )
        else:
            task = await scheduler.add_cron(
                description=body.description,
                cron_expression=body.schedule,
                channel=body.channel,
                user_id=body.user_id,
            )
    except Exception as e:
        logger.error("task_create_failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Failed to create task: {e}") from e

    return {
        "id": task.id,
        "description": task.description,
        "type": task.type,
        "schedule": task.schedule,
        "status": task.status,
        "channel": task.channel,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "next_run": task.next_run.isoformat() if task.next_run else None,
        "last_run": task.last_run.isoformat() if task.last_run else None,
    }


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    scheduler: Any = Depends(get_task_scheduler),  # noqa: B008
) -> dict[str, bool]:
    """Delete a scheduled task."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Task scheduler not available")

    removed = scheduler.remove_task(task_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return {"success": True}


# --- Skill endpoints ---


@router.get("/skills")
async def list_skills(
    skill_manager: Any = Depends(get_skill_manager),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all discovered and loaded skills."""
    if not skill_manager:
        return []
    return skill_manager.list_skills()


@router.post("/skills/{name}/reload")
async def reload_skill(
    name: str,
    skill_manager: Any = Depends(get_skill_manager),  # noqa: B008
) -> dict[str, Any]:
    """Hot-reload a skill by name."""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="Skill manager not available")

    ok = await skill_manager.reload_skill(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Skill not found or reload failed: {name}")

    return {"success": True, "name": name}


@router.post("/skills/{name}/enable")
async def enable_skill(
    name: str,
    skill_manager: Any = Depends(get_skill_manager),  # noqa: B008
) -> dict[str, Any]:
    """Enable a skill (remove from disabled list and load it)."""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="Skill manager not available")

    if name in skill_manager.config.disabled:
        skill_manager.config.disabled.remove(name)

    # Try to discover and load
    await skill_manager.discover_and_load()
    return {"success": True, "name": name}


@router.post("/skills/{name}/disable")
async def disable_skill(
    name: str,
    skill_manager: Any = Depends(get_skill_manager),  # noqa: B008
) -> dict[str, Any]:
    """Disable a skill (unload and add to disabled list)."""
    if not skill_manager:
        raise HTTPException(status_code=503, detail="Skill manager not available")

    await skill_manager.unload_skill(name)
    if name not in skill_manager.config.disabled:
        skill_manager.config.disabled.append(name)

    return {"success": True, "name": name}


# --- Memory export/import endpoints ---


@router.post("/memory/export")
async def export_memory(
    format: str = "json",
    fact_store: Any = Depends(get_fact_store),  # noqa: B008
    soul_loader: Any = Depends(get_soul_loader),  # noqa: B008
) -> dict[str, Any]:
    """Export memory to JSON."""
    from agent.memory.export import MemoryExporter

    exporter = MemoryExporter(fact_store=fact_store, soul_loader=soul_loader)
    output_path = f"data/memory_export.{format if format == 'json' else 'md'}"

    if format == "markdown":
        await exporter.export_markdown(output_path)
        return {"success": True, "path": output_path}
    else:
        stats = await exporter.export_json(output_path)
        return {"success": True, **stats}


@router.post("/memory/import")
async def import_memory(
    body: dict[str, Any],
    fact_store: Any = Depends(get_fact_store),  # noqa: B008
    soul_loader: Any = Depends(get_soul_loader),  # noqa: B008
) -> dict[str, Any]:
    """Import memory from JSON export data.

    Body should contain the export JSON directly (version, facts, soul).
    """
    from agent.memory.export import MemoryExporter

    if not fact_store:
        raise HTTPException(status_code=503, detail="Fact store not available")

    # Write to temp file and import
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(body, f, ensure_ascii=False)
        temp_path = f.name

    exporter = MemoryExporter(fact_store=fact_store, soul_loader=soul_loader)
    merge = body.get("merge", True)
    stats = await exporter.import_json(temp_path, merge=merge)

    import os

    os.unlink(temp_path)

    return {"success": True, **stats}


# --- Workspace endpoints ---


# --- Voice endpoints ---


class SynthesizeRequest(BaseModel):
    """Request body for POST /voice/synthesize."""

    text: str = Field(..., min_length=1)
    voice: str | None = Field(None, description="Override TTS voice")


@router.get("/voice/voices")
async def list_voices(
    language: str = "",
    voice_pipeline: Any = Depends(get_voice_pipeline),  # noqa: B008
) -> list[dict[str, Any]]:
    """List available TTS voices."""
    if not voice_pipeline:
        return []
    return await voice_pipeline.list_voices(language)


@router.post("/voice/synthesize")
async def synthesize_text(
    body: SynthesizeRequest,
    voice_pipeline: Any = Depends(get_voice_pipeline),  # noqa: B008
) -> Any:
    """Convert text to speech. Returns audio file."""
    if not voice_pipeline:
        raise HTTPException(status_code=503, detail="Voice pipeline not available")

    result = await voice_pipeline.synthesize(body.text)
    if not result:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

    from fastapi.responses import Response

    return Response(
        content=result.audio_data,
        media_type=result.mime_type,
        headers={
            "Content-Disposition": "attachment; filename=speech.ogg",
            "X-Voice": result.voice,
            "X-Duration": str(round(result.duration_seconds, 2)),
        },
    )


@router.post("/voice/transcribe")
async def transcribe_audio(
    voice_pipeline: Any = Depends(get_voice_pipeline),  # noqa: B008
    file: Any = None,
) -> dict[str, Any]:
    """Transcribe audio to text.

    Accepts multipart/form-data with a 'file' field or raw audio body.
    """
    if not voice_pipeline:
        raise HTTPException(status_code=503, detail="Voice pipeline not available")

    if file is None:
        raise HTTPException(status_code=400, detail="No audio file provided")

    audio_data = await file.read()
    mime_type = file.content_type or "audio/ogg"

    result = await voice_pipeline.transcribe(audio_data, mime_type)

    return {
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "duration_seconds": result.duration_seconds,
    }


@router.get("/voice/config")
async def voice_config(
    config: Any = Depends(get_config),  # noqa: B008
) -> dict[str, Any]:
    """Get current voice configuration."""
    return config.voice.model_dump()


class CreateWorkspaceRequest(BaseModel):
    """Request body for POST /workspaces."""

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    display_name: str = ""
    description: str = ""
    clone_from: str | None = None


@router.get("/workspaces")
async def list_workspaces(
    workspace_manager: Any = Depends(get_workspace_manager),  # noqa: B008
    fact_store: Any = Depends(get_fact_store),  # noqa: B008
    vector_store: Any = Depends(get_vector_store),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all workspaces with status and stats."""
    if not workspace_manager:
        return []

    names = workspace_manager.discover()
    results: list[dict[str, Any]] = []

    for name in names:
        try:
            ws = workspace_manager.resolve(name)
            active = workspace_manager.get_active()
            results.append(
                {
                    "name": ws.name,
                    "display_name": ws.display_name,
                    "description": ws.description or "",
                    "is_active": ws.name == active.name,
                    "model": ws.config.default_model,
                    "root_dir": str(ws.root_dir),
                }
            )
        except Exception as e:
            results.append(
                {
                    "name": name,
                    "display_name": name,
                    "description": "",
                    "is_active": False,
                    "error": str(e),
                }
            )

    return results


@router.get("/workspaces/active")
async def get_active_workspace(
    workspace_manager: Any = Depends(get_workspace_manager),  # noqa: B008
) -> dict[str, Any]:
    """Get the currently active workspace."""
    if not workspace_manager:
        raise HTTPException(status_code=503, detail="Workspace manager not available")

    ws = workspace_manager.get_active()
    return {
        "name": ws.name,
        "display_name": ws.display_name,
        "description": ws.description or "",
    }


@router.get("/workspaces/{name}")
async def get_workspace(
    name: str,
    workspace_manager: Any = Depends(get_workspace_manager),  # noqa: B008
) -> dict[str, Any]:
    """Get workspace details."""
    if not workspace_manager:
        raise HTTPException(status_code=503, detail="Workspace manager not available")

    try:
        ws = workspace_manager.resolve(name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {name}") from None

    active = workspace_manager.get_active()
    return {
        "name": ws.name,
        "display_name": ws.display_name,
        "description": ws.description or "",
        "is_active": ws.name == active.name,
        "model": ws.config.default_model,
        "root_dir": str(ws.root_dir),
        "data_dir": str(ws.data_dir),
        "soul_path": str(ws.soul_path),
    }


@router.post("/workspaces")
async def create_workspace(
    body: CreateWorkspaceRequest,
    workspace_manager: Any = Depends(get_workspace_manager),  # noqa: B008
) -> dict[str, Any]:
    """Create a new workspace."""
    if not workspace_manager:
        raise HTTPException(status_code=503, detail="Workspace manager not available")

    try:
        ws = workspace_manager.create(
            name=body.name,
            display_name=body.display_name,
            description=body.description,
            clone_from=body.clone_from,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "name": ws.name,
        "display_name": ws.display_name,
        "description": ws.description or "",
    }


@router.delete("/workspaces/{name}")
async def delete_workspace(
    name: str,
    confirm: bool = False,
    workspace_manager: Any = Depends(get_workspace_manager),  # noqa: B008
) -> dict[str, bool]:
    """Delete a workspace (requires confirm=true query param)."""
    if not workspace_manager:
        raise HTTPException(status_code=503, detail="Workspace manager not available")

    try:
        deleted = workspace_manager.delete(name, confirm=confirm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {name}")

    return {"success": True}


@router.post("/workspaces/{name}/switch")
async def switch_workspace(
    name: str,
    workspace_manager: Any = Depends(get_workspace_manager),  # noqa: B008
) -> dict[str, Any]:
    """Switch active workspace."""
    if not workspace_manager:
        raise HTTPException(status_code=503, detail="Workspace manager not available")

    try:
        ws = workspace_manager.switch(name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "name": ws.name,
        "display_name": ws.display_name,
        "is_active": True,
    }
