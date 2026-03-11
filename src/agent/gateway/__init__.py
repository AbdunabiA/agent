"""API gateway — FastAPI application with REST endpoints.

Phase 3A: Application factory, auth/rate-limit middleware, REST API.
Phase 3B: WebSocket hub, streaming responses.
"""

from agent.gateway.app import create_app

__all__ = ["create_app"]
