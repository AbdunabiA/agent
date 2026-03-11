---
name: github-monitor
description: Monitor GitHub repositories — check repos, list issues, get summaries.
version: "0.1.0"
author: Agent Team
permissions:
  - safe
dependencies:
  - httpx
triggers:
  - github
  - repos
  - issues
---

# GitHub Monitor

Monitor your GitHub repositories for new issues, pull requests, and activity.

## Tools

- **check_repos** — Check the status of watched repositories
- **list_issues** — List open issues for a repository
- **repo_summary** — Get a summary of a repository

## Usage

```
Check my repos on github
List issues for owner/repo
Summarize owner/repo
```
