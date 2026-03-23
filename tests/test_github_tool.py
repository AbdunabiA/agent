"""Tests for the GitHub API integration tool."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

from agent.tools.builtins.github import github

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_github_config(
    enabled: bool = True,
    token: str = "ghp_test_token",
    default_owner: str = "testowner",
    default_repo: str = "testrepo",
):
    """Build a mock GitHubConfig."""
    cfg = MagicMock()
    cfg.enabled = enabled
    cfg.token = token
    cfg.default_owner = default_owner
    cfg.default_repo = default_repo
    return cfg


def _build_mock_client(
    status_code: int = 200,
    json_data: dict | list | None = None,
    text: str = "",
) -> AsyncMock:
    """Build a fully mocked httpx.AsyncClient context manager."""
    mock_resp = AsyncMock()
    mock_resp.status_code = status_code
    mock_resp.text = text or json.dumps(json_data or {})
    mock_resp.json = MagicMock(return_value=json_data if json_data is not None else {})

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.put = AsyncMock(return_value=mock_resp)
    mock_client.patch = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


CONFIG_PATCH = "agent.tools.builtins.github._get_github_config"
CLIENT_PATCH = "agent.tools.builtins.github.httpx.AsyncClient"


# ---------------------------------------------------------------------------
# Configuration & validation tests
# ---------------------------------------------------------------------------


class TestGitHubConfigValidation:
    """Tests for config and action validation."""

    @patch(CONFIG_PATCH)
    async def test_disabled_tool_returns_error(self, mock_cfg) -> None:
        mock_cfg.return_value = _mock_github_config(enabled=False)
        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "disabled" in result

    @patch(CONFIG_PATCH)
    async def test_missing_token_returns_error(self, mock_cfg) -> None:
        mock_cfg.return_value = _mock_github_config(token=None)
        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "token" in result.lower()

    @patch(CONFIG_PATCH)
    async def test_missing_token_empty_string_returns_error(self, mock_cfg) -> None:
        mock_cfg.return_value = _mock_github_config(token="")
        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "token" in result.lower()

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_invalid_action_returns_error(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="nonexistent_action")
        assert "[ERROR]" in result
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# list_repos
# ---------------------------------------------------------------------------


class TestListRepos:
    """Tests for list_repos action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_repos_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        repos = [
            {"full_name": "owner/repo1", "private": False, "description": "First repo"},
            {"full_name": "owner/repo2", "private": True, "description": None},
        ]
        mock_client_cls.return_value = _build_mock_client(json_data=repos)

        result = await github(action="list_repos")
        assert "owner/repo1" in result
        assert "public" in result
        assert "owner/repo2" in result
        assert "private" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_repos_empty(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(json_data=[])

        result = await github(action="list_repos")
        assert "No repositories" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_repos_api_error(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(status_code=401, text="Unauthorized")

        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "401" in result


# ---------------------------------------------------------------------------
# create_repo
# ---------------------------------------------------------------------------


class TestCreateRepo:
    """Tests for create_repo action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_repo_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        response_data = {
            "full_name": "testowner/new-repo",
            "html_url": "https://github.com/testowner/new-repo",
        }
        mock_client_cls.return_value = _build_mock_client(status_code=201, json_data=response_data)

        result = await github(action="create_repo", title="new-repo", body="A new repo")
        assert "Repository created" in result
        assert "testowner/new-repo" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_repo_missing_name(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="create_repo")
        assert "[ERROR]" in result
        assert "title" in result.lower()


# ---------------------------------------------------------------------------
# list_issues
# ---------------------------------------------------------------------------


class TestListIssues:
    """Tests for list_issues action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_issues_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        issues = [
            {"number": 1, "title": "Bug report", "labels": [{"name": "bug"}]},
            {"number": 2, "title": "Feature request", "labels": []},
        ]
        mock_client_cls.return_value = _build_mock_client(json_data=issues)

        result = await github(action="list_issues")
        assert "#1" in result
        assert "Bug report" in result
        assert "[bug]" in result
        assert "#2" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_issues_empty(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(json_data=[])

        result = await github(action="list_issues")
        assert "No open issues" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_issues_missing_owner_repo(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config(default_owner=None, default_repo=None)
        result = await github(action="list_issues")
        assert "[ERROR]" in result
        assert "owner" in result


# ---------------------------------------------------------------------------
# create_issue
# ---------------------------------------------------------------------------


class TestCreateIssue:
    """Tests for create_issue action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_issue_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        response_data = {
            "number": 42,
            "title": "New issue",
            "html_url": "https://github.com/testowner/testrepo/issues/42",
        }
        mock_client_cls.return_value = _build_mock_client(status_code=201, json_data=response_data)

        result = await github(
            action="create_issue",
            title="New issue",
            body="Details",
            labels="bug,urgent",
        )
        assert "Issue created" in result
        assert "#42" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_issue_sends_labels(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client = _build_mock_client(
            status_code=201,
            json_data={"number": 1, "title": "T", "html_url": "https://github.com/a/b/issues/1"},
        )
        mock_client_cls.return_value = mock_client

        await github(action="create_issue", title="T", labels="bug, feature")
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["labels"] == ["bug", "feature"]

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_issue_missing_title(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="create_issue")
        assert "[ERROR]" in result


# ---------------------------------------------------------------------------
# close_issue
# ---------------------------------------------------------------------------


class TestCloseIssue:
    """Tests for close_issue action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_close_issue_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(status_code=200)

        result = await github(action="close_issue", number=5)
        assert "closed" in result.lower()
        assert "#5" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_close_issue_missing_number(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="close_issue")
        assert "[ERROR]" in result
        assert "number" in result


# ---------------------------------------------------------------------------
# list_prs
# ---------------------------------------------------------------------------


class TestListPRs:
    """Tests for list_prs action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_prs_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        prs = [
            {
                "number": 10,
                "title": "Add feature",
                "head": {"ref": "feature-branch"},
                "base": {"ref": "main"},
            },
        ]
        mock_client_cls.return_value = _build_mock_client(json_data=prs)

        result = await github(action="list_prs")
        assert "#10" in result
        assert "Add feature" in result
        assert "feature-branch" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_prs_empty(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(json_data=[])

        result = await github(action="list_prs")
        assert "No open pull requests" in result


# ---------------------------------------------------------------------------
# create_pr
# ---------------------------------------------------------------------------


class TestCreatePR:
    """Tests for create_pr action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_pr_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        response_data = {
            "number": 15,
            "title": "New PR",
            "html_url": "https://github.com/testowner/testrepo/pull/15",
        }
        mock_client_cls.return_value = _build_mock_client(status_code=201, json_data=response_data)

        result = await github(action="create_pr", title="New PR", branch="feature")
        assert "PR created" in result
        assert "#15" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_pr_missing_title(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="create_pr", branch="feature")
        assert "[ERROR]" in result


# ---------------------------------------------------------------------------
# get_file
# ---------------------------------------------------------------------------


class TestGetFile:
    """Tests for get_file action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_get_file_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        file_content = "Hello, World!"
        encoded = base64.b64encode(file_content.encode()).decode()
        response_data = {
            "type": "file",
            "content": encoded,
            "size": len(file_content),
        }
        mock_client_cls.return_value = _build_mock_client(json_data=response_data)

        result = await github(action="get_file", path="README.md")
        assert "Hello, World!" in result
        assert "README.md" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_get_file_missing_path(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="get_file")
        assert "[ERROR]" in result
        assert "path" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_get_file_not_a_file(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        response_data = {"type": "dir"}
        mock_client_cls.return_value = _build_mock_client(json_data=response_data)

        result = await github(action="get_file", path="src/")
        assert "[ERROR]" in result
        assert "not a file" in result


# ---------------------------------------------------------------------------
# push_file
# ---------------------------------------------------------------------------


class TestPushFile:
    """Tests for push_file action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_push_file_create_new(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()

        # GET returns 404 (file doesn't exist), PUT returns 201
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        get_resp = AsyncMock()
        get_resp.status_code = 404
        mock_client.get = AsyncMock(return_value=get_resp)

        put_resp = AsyncMock()
        put_resp.status_code = 201
        put_resp.json = MagicMock(
            return_value={
                "commit": {"sha": "abc1234567890"},
            }
        )
        put_resp.text = ""
        mock_client.put = AsyncMock(return_value=put_resp)

        mock_client_cls.return_value = mock_client

        result = await github(
            action="push_file", path="new_file.txt", content="Hello", title="Add file"
        )
        assert "Created" in result
        assert "new_file.txt" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_push_file_update_existing(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        get_resp = AsyncMock()
        get_resp.status_code = 200
        get_resp.json = MagicMock(return_value={"sha": "existingsha123"})
        mock_client.get = AsyncMock(return_value=get_resp)

        put_resp = AsyncMock()
        put_resp.status_code = 200
        put_resp.json = MagicMock(return_value={"commit": {"sha": "newsha456789"}})
        put_resp.text = ""
        mock_client.put = AsyncMock(return_value=put_resp)

        mock_client_cls.return_value = mock_client

        result = await github(
            action="push_file", path="existing.txt", content="Updated", title="Update file"
        )
        assert "Updated" in result

        # Verify SHA was included in the PUT payload
        put_call = mock_client.put.call_args
        payload = put_call.kwargs.get("json") or put_call[1].get("json")
        assert payload["sha"] == "existingsha123"

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_push_file_missing_path(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="push_file", content="data")
        assert "[ERROR]" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_push_file_missing_content(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="push_file", path="file.txt")
        assert "[ERROR]" in result


# ---------------------------------------------------------------------------
# list_actions
# ---------------------------------------------------------------------------


class TestListActions:
    """Tests for list_actions action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_actions_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        response_data = {
            "workflow_runs": [
                {
                    "id": 100,
                    "name": "CI",
                    "status": "completed",
                    "conclusion": "success",
                    "head_branch": "main",
                },
            ],
        }
        mock_client_cls.return_value = _build_mock_client(json_data=response_data)

        result = await github(action="list_actions")
        assert "#100" in result
        assert "CI" in result
        assert "success" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_actions_empty(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(json_data={"workflow_runs": []})

        result = await github(action="list_actions")
        assert "No workflow runs" in result


# ---------------------------------------------------------------------------
# trigger_action
# ---------------------------------------------------------------------------


class TestTriggerAction:
    """Tests for trigger_action action."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_trigger_action_success(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(status_code=204)

        result = await github(action="trigger_action", path="ci.yml", branch="main")
        assert "triggered" in result.lower()
        assert "ci.yml" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_trigger_action_missing_path(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        result = await github(action="trigger_action")
        assert "[ERROR]" in result
        assert "path" in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for network error handling."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_connection_error(self, mock_cfg, mock_client_cls) -> None:
        import httpx

        mock_cfg.return_value = _mock_github_config()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_cls.return_value = mock_client

        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "Connection" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_timeout_error(self, mock_cfg, mock_client_cls) -> None:
        import httpx

        mock_cfg.return_value = _mock_github_config()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client_cls.return_value = mock_client

        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "timed out" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_uses_explicit_owner_repo_over_defaults(self, mock_cfg, mock_client_cls) -> None:
        mock_cfg.return_value = _mock_github_config()
        mock_client = _build_mock_client(json_data=[])
        mock_client_cls.return_value = mock_client

        await github(action="list_issues", owner="custom", repo="myrepo")
        call_args = mock_client.get.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
        assert "custom/myrepo" in url


# ---------------------------------------------------------------------------
# Rate limiting edge cases
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for GitHub API rate limiting behaviour."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_rate_limit_403_returns_readable_error(self, mock_cfg, mock_client_cls) -> None:
        """A 403 with X-RateLimit-Remaining: 0 should return a readable error."""
        mock_cfg.return_value = _mock_github_config()
        mock_resp = AsyncMock()
        mock_resp.status_code = 403
        mock_resp.text = '{"message":"API rate limit exceeded"}'
        mock_resp.json = MagicMock(return_value={"message": "API rate limit exceeded"})
        mock_resp.headers = {"X-RateLimit-Remaining": "0"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "403" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_rate_limit_message_included(self, mock_cfg, mock_client_cls) -> None:
        """The rate-limit error body should be included in the result."""
        mock_cfg.return_value = _mock_github_config()
        mock_resp = AsyncMock()
        mock_resp.status_code = 403
        mock_resp.text = "API rate limit exceeded for user"
        mock_resp.json = MagicMock(return_value={"message": "API rate limit exceeded"})

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await github(action="list_repos")
        assert "rate limit" in result.lower()


# ---------------------------------------------------------------------------
# Pagination and large data edge cases
# ---------------------------------------------------------------------------


class TestPaginationAndLargeData:
    """Tests for empty responses and large payloads."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_repos_handles_empty_json(self, mock_cfg, mock_client_cls) -> None:
        """An empty JSON array should return 'No repositories' message."""
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(json_data=[])

        result = await github(action="list_repos")
        assert "No repositories" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_list_issues_with_many_labels(self, mock_cfg, mock_client_cls) -> None:
        """Issue with 10+ labels should format all of them without crashing."""
        mock_cfg.return_value = _mock_github_config()
        many_labels = [{"name": f"label-{i}"} for i in range(12)]
        issues = [
            {"number": 99, "title": "Multi-label issue", "labels": many_labels},
        ]
        mock_client_cls.return_value = _build_mock_client(json_data=issues)

        result = await github(action="list_issues")
        assert "#99" in result
        assert "Multi-label issue" in result
        # All 12 labels should appear
        for i in range(12):
            assert f"label-{i}" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_get_file_binary_content(self, mock_cfg, mock_client_cls) -> None:
        """Binary content that cannot be decoded as UTF-8 should return a graceful error."""
        mock_cfg.return_value = _mock_github_config()
        # Create bytes that are NOT valid UTF-8
        binary_data = bytes(range(128, 256))
        encoded = base64.b64encode(binary_data).decode("ascii")
        response_data = {
            "type": "file",
            "content": encoded,
            "size": len(binary_data),
        }
        mock_client_cls.return_value = _build_mock_client(json_data=response_data)

        result = await github(action="get_file", path="image.bin")
        # Should either return the content or a decode error — not crash
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Unicode and special characters edge cases
# ---------------------------------------------------------------------------


class TestUnicodeAndSpecialChars:
    """Tests for unicode and special characters in parameters."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_issue_unicode_title(self, mock_cfg, mock_client_cls) -> None:
        """Issue title with emoji and non-ASCII should be sent correctly."""
        mock_cfg.return_value = _mock_github_config()
        mock_client = _build_mock_client(
            status_code=201,
            json_data={
                "number": 50,
                "title": "\U0001f41b Bug: \u00e9\u00e0\u00fc\u00f1\u00f8",
                "html_url": "https://github.com/testowner/testrepo/issues/50",
            },
        )
        mock_client_cls.return_value = mock_client

        result = await github(
            action="create_issue",
            title="\U0001f41b Bug: \u00e9\u00e0\u00fc\u00f1\u00f8",
            body="Description with \u4e16\u754c",
        )
        assert "Issue created" in result
        assert "#50" in result

        # Verify the payload was sent with the unicode title
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "\U0001f41b" in payload["title"]

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_create_issue_labels_with_spaces(self, mock_cfg, mock_client_cls) -> None:
        """Labels like 'bug fix, help wanted' with spaces should be preserved."""
        mock_cfg.return_value = _mock_github_config()
        mock_client = _build_mock_client(
            status_code=201,
            json_data={
                "number": 51,
                "title": "Test",
                "html_url": "https://github.com/testowner/testrepo/issues/51",
            },
        )
        mock_client_cls.return_value = mock_client

        await github(
            action="create_issue",
            title="Test",
            labels="bug fix, help wanted",
        )
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["labels"] == ["bug fix", "help wanted"]

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_push_file_unicode_content(self, mock_cfg, mock_client_cls) -> None:
        """Unicode content should be base64-encoded correctly."""
        mock_cfg.return_value = _mock_github_config()
        unicode_content = "Hello \u4e16\u754c \U0001f600 \u00e9\u00e8\u00ea"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        get_resp = AsyncMock()
        get_resp.status_code = 404
        mock_client.get = AsyncMock(return_value=get_resp)

        put_resp = AsyncMock()
        put_resp.status_code = 201
        put_resp.json = MagicMock(return_value={"commit": {"sha": "abc1234567890"}})
        put_resp.text = ""
        mock_client.put = AsyncMock(return_value=put_resp)
        mock_client_cls.return_value = mock_client

        result = await github(
            action="push_file",
            path="unicode.txt",
            content=unicode_content,
            title="Add unicode file",
        )
        assert "Created" in result

        # Verify base64 encoding is correct
        put_call = mock_client.put.call_args
        payload = put_call.kwargs.get("json") or put_call[1].get("json")
        decoded = base64.b64decode(payload["content"]).decode("utf-8")
        assert decoded == unicode_content


# ---------------------------------------------------------------------------
# Network edge cases
# ---------------------------------------------------------------------------


class TestNetworkEdgeCases:
    """Tests for DNS, JSON parsing, and server error edge cases."""

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_dns_resolution_failure(self, mock_cfg, mock_client_cls) -> None:
        """ConnectError (e.g. DNS failure) should produce a readable error."""
        import httpx

        mock_cfg.return_value = _mock_github_config()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Name or service not known"))
        mock_client_cls.return_value = mock_client

        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "Connection" in result

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_invalid_json_response(self, mock_cfg, mock_client_cls) -> None:
        """Response with invalid JSON body should not cause an unhandled exception."""
        mock_cfg.return_value = _mock_github_config()
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.text = "this is not json {"
        mock_resp.json = MagicMock(side_effect=json.JSONDecodeError("msg", "doc", 0))

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        # The function calls resp.json() which will raise — verify it does not crash unhandled
        result = await github(action="list_repos")
        assert isinstance(result, str)
        # Should contain error or at least not crash
        assert "[ERROR]" in result or len(result) > 0

    @patch(CLIENT_PATCH)
    @patch(CONFIG_PATCH)
    async def test_http_500_server_error(self, mock_cfg, mock_client_cls) -> None:
        """500 server error should return an error message with status code."""
        mock_cfg.return_value = _mock_github_config()
        mock_client_cls.return_value = _build_mock_client(
            status_code=500,
            text="Internal Server Error",
        )

        result = await github(action="list_repos")
        assert "[ERROR]" in result
        assert "500" in result
