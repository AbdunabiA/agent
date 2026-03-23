"""GitHub API integration tool."""

from __future__ import annotations

import base64

import httpx
import structlog

from agent.config import get_config
from agent.tools.registry import ToolTier, tool

logger = structlog.get_logger(__name__)

_API_BASE = "https://api.github.com"


def _get_github_config():
    """Return the GitHub configuration from the global config."""
    config = get_config()
    return config.tools.github


def _headers(token: str) -> dict:
    """Build standard GitHub API headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _resolve_owner_repo(owner: str, repo: str, gh_config) -> tuple[str, str] | str:
    """Resolve owner/repo from arguments or config defaults.

    Returns:
        Tuple of (owner, repo) on success, or an error string.
    """
    owner = owner or (gh_config.default_owner or "")
    repo = repo or (gh_config.default_repo or "")
    if not owner or not repo:
        return "[ERROR] owner and repo are required (set defaults in config or pass explicitly)"
    return owner, repo


@tool(
    name="github",
    description=(
        "Interact with GitHub: manage repos, issues, PRs, files, and actions. "
        "Actions: list_repos, create_repo, list_issues, create_issue, close_issue, "
        "list_prs, create_pr, get_file, push_file, list_actions, trigger_action"
    ),
    tier=ToolTier.MODERATE,
)
async def github(
    action: str,
    owner: str = "",
    repo: str = "",
    title: str = "",
    body: str = "",
    branch: str = "main",
    path: str = "",
    content: str = "",
    number: int = 0,
    labels: str = "",
) -> str:
    """Interact with the GitHub API.

    Args:
        action: The action to perform (list_repos, create_repo, list_issues,
                create_issue, close_issue, list_prs, create_pr, get_file,
                push_file, list_actions, trigger_action).
        owner: Repository owner (uses default from config if empty).
        repo: Repository name (uses default from config if empty).
        title: Title for issues, PRs, commits, or repo name for create_repo.
        body: Body/description text.
        branch: Branch name (default: main).
        path: File path for get_file/push_file, or workflow file for trigger_action.
        content: File content for push_file.
        number: Issue or PR number for close_issue.
        labels: Comma-separated labels for create_issue.

    Returns:
        Formatted string with the result or error message.
    """
    gh_config = _get_github_config()

    if not gh_config.enabled:
        return "[ERROR] GitHub tool is disabled. Enable it in config: tools.github.enabled = true"

    token = gh_config.token
    if not token:
        return (
            "[ERROR] GitHub token not configured. "
            "Set GITHUB_TOKEN in .env or tools.github.token in config"
        )

    hdrs = _headers(token)

    valid_actions = {
        "list_repos",
        "create_repo",
        "list_issues",
        "create_issue",
        "close_issue",
        "list_prs",
        "create_pr",
        "get_file",
        "push_file",
        "list_actions",
        "trigger_action",
    }
    if action not in valid_actions:
        valid = ", ".join(sorted(valid_actions))
        return f"[ERROR] Unknown action: {action}. Valid: {valid}"

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            if action == "list_repos":
                return await _list_repos(client, hdrs)
            elif action == "create_repo":
                return await _create_repo(client, hdrs, title, body)
            elif action == "list_issues":
                return await _list_issues(client, hdrs, owner, repo, gh_config)
            elif action == "create_issue":
                return await _create_issue(
                    client,
                    hdrs,
                    owner,
                    repo,
                    title,
                    body,
                    labels,
                    gh_config,
                )
            elif action == "close_issue":
                return await _close_issue(client, hdrs, owner, repo, number, gh_config)
            elif action == "list_prs":
                return await _list_prs(client, hdrs, owner, repo, gh_config)
            elif action == "create_pr":
                return await _create_pr(client, hdrs, owner, repo, title, body, branch, gh_config)
            elif action == "get_file":
                return await _get_file(client, hdrs, owner, repo, path, branch, gh_config)
            elif action == "push_file":
                return await _push_file(
                    client, hdrs, owner, repo, path, content, title, branch, gh_config
                )
            elif action == "list_actions":
                return await _list_actions(client, hdrs, owner, repo, gh_config)
            elif action == "trigger_action":
                return await _trigger_action(client, hdrs, owner, repo, path, branch, gh_config)
            else:
                return f"[ERROR] Action '{action}' is not implemented"

    except httpx.ConnectError as e:
        return f"[ERROR] Connection to GitHub failed: {e}"
    except httpx.TimeoutException:
        return "[ERROR] GitHub API request timed out"
    except httpx.HTTPError as e:
        return f"[ERROR] HTTP error: {e}"
    except Exception as e:
        logger.error("github_tool_error", action=action, error=str(e))
        return f"[ERROR] GitHub request failed: {e}"


async def _list_repos(client: httpx.AsyncClient, hdrs: dict) -> str:
    """List the authenticated user's repositories."""
    resp = await client.get(f"{_API_BASE}/user/repos?sort=updated&per_page=20", headers=hdrs)
    if resp.status_code != 200:
        return f"[ERROR] GitHub API returned {resp.status_code}: {resp.text}"

    repos = resp.json()
    if not repos:
        return "No repositories found."

    lines = [f"Found {len(repos)} repositories:\n"]
    for r in repos:
        visibility = "private" if r.get("private") else "public"
        desc = r.get("description") or "No description"
        lines.append(f"  - {r['full_name']} ({visibility}) — {desc}")
    return "\n".join(lines)


async def _create_repo(client: httpx.AsyncClient, hdrs: dict, name: str, description: str) -> str:
    """Create a new repository."""
    if not name:
        return "[ERROR] title (repo name) is required for create_repo"

    payload = {"name": name, "description": description, "auto_init": True}
    resp = await client.post(f"{_API_BASE}/user/repos", headers=hdrs, json=payload)
    if resp.status_code not in (200, 201):
        return f"[ERROR] Failed to create repo: {resp.status_code} — {resp.text}"

    data = resp.json()
    return f"Repository created: {data['full_name']}\nURL: {data['html_url']}"


async def _list_issues(
    client: httpx.AsyncClient, hdrs: dict, owner: str, repo: str, gh_config
) -> str:
    """List open issues for a repository."""
    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    resp = await client.get(
        f"{_API_BASE}/repos/{owner}/{repo}/issues?state=open&per_page=20", headers=hdrs
    )
    if resp.status_code != 200:
        return f"[ERROR] GitHub API returned {resp.status_code}: {resp.text}"

    issues = resp.json()
    if not issues:
        return f"No open issues in {owner}/{repo}."

    lines = [f"Open issues in {owner}/{repo} ({len(issues)}):\n"]
    for issue in issues:
        issue_labels = ", ".join(lb["name"] for lb in issue.get("labels", []))
        label_str = f" [{issue_labels}]" if issue_labels else ""
        lines.append(f"  #{issue['number']} {issue['title']}{label_str}")
    return "\n".join(lines)


async def _create_issue(
    client: httpx.AsyncClient,
    hdrs: dict,
    owner: str,
    repo: str,
    title: str,
    body: str,
    labels: str,
    gh_config,
) -> str:
    """Create a new issue."""
    if not title:
        return "[ERROR] title is required for create_issue"

    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    payload: dict = {"title": title, "body": body}
    if labels:
        payload["labels"] = [lb.strip() for lb in labels.split(",") if lb.strip()]

    resp = await client.post(f"{_API_BASE}/repos/{owner}/{repo}/issues", headers=hdrs, json=payload)
    if resp.status_code not in (200, 201):
        return f"[ERROR] Failed to create issue: {resp.status_code} — {resp.text}"

    data = resp.json()
    return f"Issue created: #{data['number']} {data['title']}\nURL: {data['html_url']}"


async def _close_issue(
    client: httpx.AsyncClient, hdrs: dict, owner: str, repo: str, number: int, gh_config
) -> str:
    """Close an issue by number."""
    if not number:
        return "[ERROR] number is required for close_issue"

    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    resp = await client.patch(
        f"{_API_BASE}/repos/{owner}/{repo}/issues/{number}",
        headers=hdrs,
        json={"state": "closed"},
    )
    if resp.status_code != 200:
        return f"[ERROR] Failed to close issue: {resp.status_code} — {resp.text}"

    return f"Issue #{number} closed in {owner}/{repo}."


async def _list_prs(client: httpx.AsyncClient, hdrs: dict, owner: str, repo: str, gh_config) -> str:
    """List open pull requests."""
    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    resp = await client.get(
        f"{_API_BASE}/repos/{owner}/{repo}/pulls?state=open&per_page=20", headers=hdrs
    )
    if resp.status_code != 200:
        return f"[ERROR] GitHub API returned {resp.status_code}: {resp.text}"

    prs = resp.json()
    if not prs:
        return f"No open pull requests in {owner}/{repo}."

    lines = [f"Open PRs in {owner}/{repo} ({len(prs)}):\n"]
    for pr in prs:
        head = pr["head"]["ref"]
        base = pr["base"]["ref"]
        lines.append(f"  #{pr['number']} {pr['title']} ({head} -> {base})")
    return "\n".join(lines)


async def _create_pr(
    client: httpx.AsyncClient,
    hdrs: dict,
    owner: str,
    repo: str,
    title: str,
    body: str,
    branch: str,
    gh_config,
) -> str:
    """Create a pull request."""
    if not title:
        return "[ERROR] title is required for create_pr"

    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    payload = {"title": title, "body": body, "head": branch, "base": "main"}
    resp = await client.post(f"{_API_BASE}/repos/{owner}/{repo}/pulls", headers=hdrs, json=payload)
    if resp.status_code not in (200, 201):
        return f"[ERROR] Failed to create PR: {resp.status_code} — {resp.text}"

    data = resp.json()
    return f"PR created: #{data['number']} {data['title']}\nURL: {data['html_url']}"


async def _get_file(
    client: httpx.AsyncClient,
    hdrs: dict,
    owner: str,
    repo: str,
    path: str,
    branch: str,
    gh_config,
) -> str:
    """Get file contents from a repository."""
    if not path:
        return "[ERROR] path is required for get_file"

    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    resp = await client.get(
        f"{_API_BASE}/repos/{owner}/{repo}/contents/{path}?ref={branch}", headers=hdrs
    )
    if resp.status_code != 200:
        return f"[ERROR] Failed to get file: {resp.status_code} — {resp.text}"

    data = resp.json()
    if data.get("type") != "file":
        return f"[ERROR] Path '{path}' is not a file (type: {data.get('type')})"

    try:
        file_content = base64.b64decode(data["content"]).decode("utf-8")
    except Exception as e:
        return f"[ERROR] Failed to decode file content: {e}"

    size = data.get("size", "unknown")
    return f"File: {path} (branch: {branch}, {size} bytes)\n\n{file_content}"


async def _push_file(
    client: httpx.AsyncClient,
    hdrs: dict,
    owner: str,
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str,
    gh_config,
) -> str:
    """Create or update a file in a repository."""
    if not path:
        return "[ERROR] path is required for push_file"
    if not content:
        return "[ERROR] content is required for push_file"

    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    commit_message = message or f"Update {path}"
    encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")

    payload: dict = {
        "message": commit_message,
        "content": encoded_content,
        "branch": branch,
    }

    # Check if file already exists to get its SHA (required for updates)
    existing = await client.get(
        f"{_API_BASE}/repos/{owner}/{repo}/contents/{path}?ref={branch}", headers=hdrs
    )
    if existing.status_code == 200:
        existing_data = existing.json()
        if isinstance(existing_data, dict) and "sha" in existing_data:
            payload["sha"] = existing_data["sha"]

    resp = await client.put(
        f"{_API_BASE}/repos/{owner}/{repo}/contents/{path}", headers=hdrs, json=payload
    )
    if resp.status_code not in (200, 201):
        return f"[ERROR] Failed to push file: {resp.status_code} — {resp.text}"

    data = resp.json()
    commit_sha = data.get("commit", {}).get("sha", "unknown")
    action_word = "Updated" if "sha" in payload else "Created"
    return f"{action_word} file: {path} (branch: {branch}, commit: {commit_sha[:7]})"


async def _list_actions(
    client: httpx.AsyncClient, hdrs: dict, owner: str, repo: str, gh_config
) -> str:
    """List recent GitHub Actions workflow runs."""
    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    resp = await client.get(
        f"{_API_BASE}/repos/{owner}/{repo}/actions/runs?per_page=10", headers=hdrs
    )
    if resp.status_code != 200:
        return f"[ERROR] GitHub API returned {resp.status_code}: {resp.text}"

    data = resp.json()
    runs = data.get("workflow_runs", [])
    if not runs:
        return f"No workflow runs found in {owner}/{repo}."

    lines = [f"Recent workflow runs in {owner}/{repo} ({len(runs)}):\n"]
    for run in runs:
        status = run.get("status", "unknown")
        conclusion = run.get("conclusion") or "in progress"
        lines.append(
            f"  #{run['id']} {run['name']} — {status}/{conclusion} "
            f"(branch: {run.get('head_branch', 'unknown')})"
        )
    return "\n".join(lines)


async def _trigger_action(
    client: httpx.AsyncClient,
    hdrs: dict,
    owner: str,
    repo: str,
    workflow_file: str,
    branch: str,
    gh_config,
) -> str:
    """Trigger a GitHub Actions workflow dispatch."""
    if not workflow_file:
        return "[ERROR] path (workflow file, e.g. 'ci.yml') is required for trigger_action"

    resolved = _resolve_owner_repo(owner, repo, gh_config)
    if isinstance(resolved, str):
        return resolved
    owner, repo = resolved

    resp = await client.post(
        f"{_API_BASE}/repos/{owner}/{repo}/actions/workflows/{workflow_file}/dispatches",
        headers=hdrs,
        json={"ref": branch},
    )
    if resp.status_code != 204:
        return f"[ERROR] Failed to trigger workflow: {resp.status_code} — {resp.text}"

    return f"Workflow '{workflow_file}' triggered on branch '{branch}' in {owner}/{repo}."
