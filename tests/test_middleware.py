"""Tests for gateway middleware — auth and rate limiting."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.gateway.middleware import AuthMiddleware, RateLimitMiddleware


def _make_test_app(auth_token: str | None = None, max_requests: int = 60) -> FastAPI:
    """Create a minimal FastAPI app with middleware for testing."""
    app = FastAPI()

    app.add_middleware(RateLimitMiddleware, max_requests=max_requests, window_seconds=60)
    app.add_middleware(AuthMiddleware, auth_token=auth_token)

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/protected")
    async def protected():
        return {"data": "secret"}

    return app


class TestAuthMiddleware:
    """Tests for bearer token authentication."""

    def test_no_auth_configured_allows_all(self) -> None:
        """Open mode: no auth_token means all requests pass."""
        app = _make_test_app(auth_token=None)
        client = TestClient(app)

        resp = client.get("/api/v1/protected")
        assert resp.status_code == 200

    def test_public_path_no_auth_needed(self) -> None:
        """Health endpoint should not require auth."""
        app = _make_test_app(auth_token="secret-token")
        client = TestClient(app)

        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_missing_auth_header_returns_401(self) -> None:
        """Protected endpoint without auth header should return 401."""
        app = _make_test_app(auth_token="secret-token")
        client = TestClient(app)

        resp = client.get("/api/v1/protected")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_invalid_token_returns_401(self) -> None:
        """Wrong bearer token should return 401."""
        app = _make_test_app(auth_token="secret-token")
        client = TestClient(app)

        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_valid_token_allows_access(self) -> None:
        """Correct bearer token should grant access."""
        app = _make_test_app(auth_token="secret-token")
        client = TestClient(app)

        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == "secret"

    def test_non_bearer_auth_returns_401(self) -> None:
        """Non-Bearer auth scheme should return 401."""
        app = _make_test_app(auth_token="secret-token")
        client = TestClient(app)

        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401


class TestRateLimitMiddleware:
    """Tests for per-IP rate limiting."""

    def test_under_limit_passes(self) -> None:
        """Requests under the limit should succeed."""
        app = _make_test_app(max_requests=10)
        client = TestClient(app)

        for _ in range(10):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_over_limit_returns_429(self) -> None:
        """Exceeding the limit should return 429."""
        app = _make_test_app(max_requests=5)
        client = TestClient(app)

        for _ in range(5):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

        resp = client.get("/api/v1/health")
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["detail"]
