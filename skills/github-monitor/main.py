"""GitHub Monitor skill — check repos, list issues, get summaries."""

from __future__ import annotations

import os

from agent.skills.base import Skill


class GitHubMonitorSkill(Skill):
    """Monitor GitHub repositories for activity."""

    async def setup(self) -> None:
        """Register GitHub monitoring tools."""
        self.register_tool(
            name="check_repos",
            description="Check the status of a GitHub repository (stars, forks, open issues).",
            function=self._check_repos,
            tier="safe",
        )
        self.register_tool(
            name="list_issues",
            description="List open issues for a GitHub repository. Args: owner, repo, limit.",
            function=self._list_issues,
            tier="safe",
        )
        self.register_tool(
            name="repo_summary",
            description="Get a summary of a GitHub repository. Args: owner, repo.",
            function=self._repo_summary,
            tier="safe",
        )

    def get_system_prompt_extension(self) -> str | None:
        """Inform the LLM about GitHub monitoring capabilities."""
        return (
            "**GitHub Monitor**: You can check GitHub repos, list issues, "
            "and get repo summaries using the github-monitor tools."
        )

    async def _check_repos(self, owner: str, repo: str) -> str:
        """Check a GitHub repository's status."""
        import httpx

        token = os.environ.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=headers,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()

        return (
            f"**{data['full_name']}**\n"
            f"Description: {data.get('description', 'N/A')}\n"
            f"Stars: {data['stargazers_count']} | Forks: {data['forks_count']}\n"
            f"Open issues: {data['open_issues_count']}\n"
            f"Language: {data.get('language', 'N/A')}\n"
            f"Updated: {data['updated_at']}"
        )

    async def _list_issues(self, owner: str, repo: str, limit: int = 10) -> str:
        """List open issues for a repository."""
        import httpx

        token = os.environ.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/issues",
                headers=headers,
                params={"state": "open", "per_page": limit},
                timeout=15.0,
            )
            resp.raise_for_status()
            issues = resp.json()

        if not issues:
            return f"No open issues in {owner}/{repo}."

        lines = [f"**Open issues for {owner}/{repo}** ({len(issues)} shown):\n"]
        for issue in issues:
            labels = ", ".join(lbl["name"] for lbl in issue.get("labels", []))
            label_str = f" [{labels}]" if labels else ""
            lines.append(f"- #{issue['number']}: {issue['title']}{label_str}")
        return "\n".join(lines)

    async def _repo_summary(self, owner: str, repo: str) -> str:
        """Get a summary of a GitHub repository."""
        import httpx

        token = os.environ.get("GITHUB_TOKEN", "")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"

        async with httpx.AsyncClient() as client:
            # Repo info
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=headers,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()

            # Recent commits
            commits_resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/commits",
                headers=headers,
                params={"per_page": 5},
                timeout=15.0,
            )
            commits = commits_resp.json() if commits_resp.status_code == 200 else []

        lines = [
            f"# {data['full_name']}",
            f"**{data.get('description', 'No description')}**\n",
            f"- Language: {data.get('language', 'N/A')}",
            f"- Stars: {data['stargazers_count']} | Forks: {data['forks_count']}",
            f"- Open issues: {data['open_issues_count']}",
            f"- Default branch: {data['default_branch']}",
            f"- Created: {data['created_at'][:10]}",
            f"- Last push: {data.get('pushed_at', 'N/A')[:10] if data.get('pushed_at') else 'N/A'}",
        ]

        if commits:
            lines.append("\n**Recent commits:**")
            for c in commits[:5]:
                msg = c["commit"]["message"].split("\n")[0][:80]
                author = c["commit"]["author"]["name"]
                lines.append(f"- {msg} ({author})")

        return "\n".join(lines)
