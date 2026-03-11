# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-03-05

Initial release.

### Added

**Phase 1 — Foundation**
- Typer CLI with `chat`, `start`, `version`, `doctor`, `config show` commands
- YAML + .env configuration with Pydantic validation
- LiteLLM provider with automatic failover (Claude, OpenAI, Gemini, Ollama)
- Interactive terminal chat with Rich UI
- Async event bus (pub/sub)
- Conversation session manager

**Phase 2 — Autonomy**
- Built-in tools: shell, filesystem, python_exec, http, browser, web_search
- Tool registry with `@tool` decorator
- Tool executor with permission tiers (safe/moderate/dangerous)
- Guardrails with blocked command patterns
- Audit log (in-memory + SQLite)
- Heartbeat daemon with HEARTBEAT.md checklist
- Planning engine with multi-step task decomposition
- Error recovery with retry and fallback strategies
- Rollback system for undoing tool actions
- Cost tracker for token usage

**Phase 3 — Channels & Gateway**
- FastAPI gateway with REST + WebSocket endpoints
- Auth middleware (Bearer token), CORS, rate limiting
- Telegram channel (aiogram 3.x) with voice, photo, document support
- Telegram interactive approval for moderate/dangerous tools
- WebChat channel via WebSocket
- Session store with SQLite persistence
- Real-time event streaming via WebSocket

**Phase 4 — Memory**
- SQLite fact store with CRUD operations
- ChromaDB vector store with local embeddings (all-MiniLM-L6-v2)
- soul.md personality loader with file watching
- Automatic fact extraction from conversations
- Conversation summarizer with vector storage
- Memory confidence decay over time
- Context assembler combining facts + vectors + soul
- Memory export/import (JSON + Markdown)

**Phase 5 — Dashboard & Browser**
- Playwright browser automation tool
- DuckDuckGo web search tool
- Dashboard static file serving from gateway

**Phase 6 — Skills & Distribution**
- Skill system: SKILL.md metadata, auto-discovery, hot-reload
- Skill permissions and dependency checking
- Bundled skills: github-monitor, daily-digest, code-reviewer, quick-notes
- Skill CLI: `agent skills list/info/create/enable/disable/reload`
- Memory CLI: `agent memory export/import/stats`
- Comprehensive `agent doctor` with 9 check categories
- Security-focused doctor checks (`agent doctor --security`)
- Dockerfile (multi-stage build) + docker-compose.yml
- GitHub Actions CI/CD (lint, test, type-check, release)
- PyPI-ready package configuration
- Install script
- Full documentation (11 pages)
