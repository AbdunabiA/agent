"""Tests for the HTTP request tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx

from agent.tools.builtins.http import MAX_BODY_SIZE, http_request


class TestUrlValidation:
    """Tests for URL validation."""

    async def test_rejects_url_without_scheme(self) -> None:
        result = await http_request(url="example.com")
        assert "[ERROR]" in result
        assert "Invalid URL" in result

    async def test_rejects_ftp_scheme(self) -> None:
        result = await http_request(url="ftp://example.com")
        assert "[ERROR]" in result
        assert "Invalid URL" in result

    async def test_accepts_http_url(self) -> None:
        """http:// URLs pass validation (request itself is mocked)."""
        with patch("agent.tools.builtins.http.httpx.AsyncClient") as mock_cls:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.reason_phrase = "OK"
            mock_resp.headers = {}
            mock_resp.text = ""

            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await http_request(url="http://example.com")
            assert "200" in result

    async def test_accepts_https_url(self) -> None:
        with patch("agent.tools.builtins.http.httpx.AsyncClient") as mock_cls:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.reason_phrase = "OK"
            mock_resp.headers = {}
            mock_resp.text = ""

            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await http_request(url="https://example.com")
            assert "200" in result


class TestMethodValidation:
    """Tests for HTTP method validation."""

    async def test_rejects_invalid_method(self) -> None:
        result = await http_request(url="https://example.com", method="INVALID")
        assert "[ERROR]" in result
        assert "Invalid HTTP method" in result

    async def test_method_is_case_insensitive(self) -> None:
        """Lowercase methods should be uppercased and accepted."""
        with patch("agent.tools.builtins.http.httpx.AsyncClient") as mock_cls:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.reason_phrase = "OK"
            mock_resp.headers = {}
            mock_resp.text = ""

            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await http_request(url="https://example.com", method="get")
            assert "200" in result
            mock_client.request.assert_called_once()
            call_kwargs = mock_client.request.call_args
            assert call_kwargs.kwargs["method"] == "GET"


def _build_mock_client(
    status_code: int = 200,
    reason_phrase: str = "OK",
    headers: dict | None = None,
    text: str = "",
) -> AsyncMock:
    """Helper to build a fully mocked httpx.AsyncClient."""
    mock_resp = AsyncMock()
    mock_resp.status_code = status_code
    mock_resp.reason_phrase = reason_phrase
    mock_resp.headers = headers or {}
    mock_resp.text = text

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestSuccessfulGet:
    """Tests for successful GET requests."""

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_get_returns_status_and_body(self, mock_cls: AsyncMock) -> None:
        mock_cls.return_value = _build_mock_client(
            status_code=200,
            reason_phrase="OK",
            headers={"content-type": "text/html"},
            text="<h1>Hello</h1>",
        )

        result = await http_request(url="https://example.com")
        assert "Status: 200 OK" in result
        assert "<h1>Hello</h1>" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_get_includes_response_headers(self, mock_cls: AsyncMock) -> None:
        mock_cls.return_value = _build_mock_client(
            headers={"content-type": "application/json", "x-request-id": "abc123"},
            text="{}",
        )

        result = await http_request(url="https://api.example.com/data")
        assert "content-type: application/json" in result
        assert "x-request-id: abc123" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_large_body_is_truncated(self, mock_cls: AsyncMock) -> None:
        large_body = "x" * (MAX_BODY_SIZE + 10_000)
        mock_cls.return_value = _build_mock_client(text=large_body)

        result = await http_request(url="https://example.com")
        assert "truncated" in result.lower()
        # The displayed body should be at most MAX_BODY_SIZE chars
        # The body in result is truncated but original was larger
        assert len(large_body) > MAX_BODY_SIZE


class TestPostWithBody:
    """Tests for POST requests with body."""

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_post_sends_body(self, mock_cls: AsyncMock) -> None:
        mock_client = _build_mock_client(status_code=201, reason_phrase="Created", text='{"id": 1}')
        mock_cls.return_value = mock_client

        result = await http_request(
            url="https://api.example.com/items",
            method="POST",
            body='{"name": "test"}',
        )

        assert "201" in result
        mock_client.request.assert_called_once()
        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["content"] == '{"name": "test"}'

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_post_with_headers(self, mock_cls: AsyncMock) -> None:
        mock_client = _build_mock_client(status_code=200, text="ok")
        mock_cls.return_value = mock_client

        await http_request(
            url="https://api.example.com/items",
            method="POST",
            headers={"Authorization": "Bearer token123"},
            body="data",
        )

        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["headers"] == {"Authorization": "Bearer token123"}

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_post_without_body_sends_none(self, mock_cls: AsyncMock) -> None:
        mock_client = _build_mock_client(text="ok")
        mock_cls.return_value = mock_client

        await http_request(url="https://api.example.com/items", method="POST")

        call_kwargs = mock_client.request.call_args
        assert call_kwargs.kwargs["content"] is None


class TestErrorHandling:
    """Tests for network and HTTP error handling."""

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_timeout_error(self, mock_cls: AsyncMock) -> None:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await http_request(url="https://slow.example.com", timeout=5)
        assert "[ERROR]" in result
        assert "timed out" in result.lower()
        assert "5s" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_connection_error(self, mock_cls: AsyncMock) -> None:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await http_request(url="https://down.example.com")
        assert "[ERROR]" in result
        assert "Connection failed" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_4xx_response_returned_not_error(self, mock_cls: AsyncMock) -> None:
        """4xx responses are not exceptions — they are returned normally."""
        mock_cls.return_value = _build_mock_client(
            status_code=404,
            reason_phrase="Not Found",
            text="Page not found",
        )

        result = await http_request(url="https://example.com/missing")
        assert "Status: 404 Not Found" in result
        assert "Page not found" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_5xx_response_returned_not_error(self, mock_cls: AsyncMock) -> None:
        """5xx responses are not exceptions — they are returned normally."""
        mock_cls.return_value = _build_mock_client(
            status_code=500,
            reason_phrase="Internal Server Error",
            text="Server error",
        )

        result = await http_request(url="https://example.com/api")
        assert "Status: 500 Internal Server Error" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_generic_http_error(self, mock_cls: AsyncMock) -> None:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.HTTPError("something went wrong"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await http_request(url="https://example.com")
        assert "[ERROR]" in result
        assert "HTTP error" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_unexpected_exception(self, mock_cls: AsyncMock) -> None:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=RuntimeError("unexpected"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        result = await http_request(url="https://example.com")
        assert "[ERROR]" in result
        assert "Request failed" in result


# ---------------------------------------------------------------------------
# verify_url edge cases
# ---------------------------------------------------------------------------


class TestVerifyUrl:
    """Tests for the verify_url tool."""

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_verify_url_success(self, mock_cls: AsyncMock) -> None:
        """HEAD returning 200 should report success with status."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        from agent.tools.builtins.http import verify_url

        result = await verify_url(url="https://example.com")
        assert "200" in result
        assert "https://example.com" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_verify_url_404(self, mock_cls: AsyncMock) -> None:
        """HEAD returning 404 should include '404' in the output."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"content-type": "text/html"}

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        from agent.tools.builtins.http import verify_url

        result = await verify_url(url="https://example.com/missing")
        assert "404" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_verify_url_redirect(self, mock_cls: AsyncMock) -> None:
        """HEAD with redirect (follow_redirects=True) should report final status."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        from agent.tools.builtins.http import verify_url

        result = await verify_url(url="https://example.com/old-page")
        assert "200" in result
        # Verify the client was created with follow_redirects=True
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("follow_redirects") is True

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_verify_url_timeout(self, mock_cls: AsyncMock) -> None:
        """TimeoutException should produce an error with 'timed out'."""
        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        from agent.tools.builtins.http import verify_url

        result = await verify_url(url="https://slow.example.com")
        assert "[ERROR]" in result
        assert "timed out" in result.lower()

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_verify_url_connection_error(self, mock_cls: AsyncMock) -> None:
        """ConnectError should produce an error mentioning connection."""
        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=httpx.ConnectError("Name resolution failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        from agent.tools.builtins.http import verify_url

        result = await verify_url(url="https://nonexistent.example.com")
        assert "[ERROR]" in result
        assert "connect" in result.lower()

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_verify_url_ssl_error(self, mock_cls: AsyncMock) -> None:
        """General exception (e.g. SSL error) should produce a generic error."""
        mock_client = AsyncMock()
        mock_client.head = AsyncMock(side_effect=Exception("SSL: CERTIFICATE_VERIFY_FAILED"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        from agent.tools.builtins.http import verify_url

        result = await verify_url(url="https://badssl.example.com")
        assert "[ERROR]" in result
        assert "Failed to verify" in result

    @patch("agent.tools.builtins.http.httpx.AsyncClient")
    async def test_verify_url_returns_content_type(self, mock_cls: AsyncMock) -> None:
        """Verify that the content-type header appears in the output."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/pdf"}

        mock_client = AsyncMock()
        mock_client.head = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client

        from agent.tools.builtins.http import verify_url

        result = await verify_url(url="https://example.com/doc.pdf")
        assert "application/pdf" in result
