"""Gateway middleware — authentication and rate limiting.

AuthMiddleware: Bearer token validation from GatewayConfig.auth_token.
RateLimitMiddleware: Per-IP sliding window rate limiting (60 req/min default).
"""

from __future__ import annotations

import secrets
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from starlette.applications import Starlette

logger = structlog.get_logger(__name__)

# Paths that don't require authentication
PUBLIC_PATHS = {"/", "/api/v1/health", "/docs", "/openapi.json", "/redoc"}

# Path prefixes that don't require authentication
PUBLIC_PREFIXES = ("/dashboard",)


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer token authentication middleware.

    If auth_token is configured, all non-public endpoints require
    a valid Authorization: Bearer <token> header.
    If auth_token is None, all requests are allowed (open mode).
    """

    def __init__(self, app: Starlette, auth_token: str | None = None) -> None:
        super().__init__(app)
        self.auth_token = auth_token
        self._auth_failures: dict[str, list[float]] = {}
        self._request_count: int = 0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check authorization for non-public paths."""
        self._request_count += 1
        now = time.time()

        # Periodically prune all stale auth failure entries
        if self._request_count % 100 == 0:
            self._prune_all_stale(now)

        # No auth configured → open mode
        if not self.auth_token:
            return await call_next(request)

        # Public paths don't need auth
        if request.url.path in PUBLIC_PATHS or request.url.path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        # Check auth failure rate limit
        client_ip = self._get_client_ip(request)
        self._prune_auth_failures(client_ip, now)
        if len(self._auth_failures.get(client_ip, [])) > 5:
            logger.warning("auth_locked_out", ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many authentication failures. Try again later."},
            )

        # Check Authorization header
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("auth_missing", path=request.url.path)
            self._record_auth_failure(client_ip, now)
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        token = auth_header[7:]  # Strip "Bearer "
        if not secrets.compare_digest(token, self.auth_token):
            logger.warning("auth_invalid", path=request.url.path)
            self._record_auth_failure(client_ip, now)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authentication token"},
            )

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _record_auth_failure(self, ip: str, now: float) -> None:
        """Record an authentication failure timestamp for the given IP."""
        if ip not in self._auth_failures:
            self._auth_failures[ip] = []
        self._auth_failures[ip].append(now)

    def _prune_auth_failures(self, ip: str, now: float) -> None:
        """Remove auth failure entries older than 5 minutes."""
        if ip not in self._auth_failures:
            return
        cutoff = now - 300  # 5 minutes
        self._auth_failures[ip] = [ts for ts in self._auth_failures[ip] if ts > cutoff]
        if not self._auth_failures[ip]:
            del self._auth_failures[ip]

    def _prune_all_stale(self, now: float) -> None:
        """Remove all auth failure entries older than lockout window across all IPs."""
        cutoff = now - 300  # 5 minutes (lockout window)
        stale_ips: list[str] = []
        for ip, timestamps in self._auth_failures.items():
            self._auth_failures[ip] = [ts for ts in timestamps if ts > cutoff]
            if not self._auth_failures[ip]:
                stale_ips.append(ip)
        for ip in stale_ips:
            del self._auth_failures[ip]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding window rate limiter.

    Defaults to 60 requests per minute per IP address.
    Returns 429 Too Many Requests when limit is exceeded.
    """

    def __init__(
        self,
        app: Starlette,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_window(self, ip: str, now: float) -> None:
        """Remove expired timestamps from the window."""
        cutoff = now - self.window_seconds
        self._requests[ip] = [ts for ts in self._requests[ip] if ts > cutoff]
        # Remove empty entries to prevent unbounded dict growth
        if not self._requests[ip]:
            del self._requests[ip]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Check rate limit for the client IP."""
        ip = self._get_client_ip(request)
        now = time.time()

        self._cleanup_window(ip, now)

        if len(self._requests[ip]) >= self.max_requests:
            logger.warning("rate_limited", ip=ip, count=len(self._requests[ip]))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        self._requests[ip].append(now)
        return await call_next(request)
