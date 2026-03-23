# API Reference

Agent exposes a REST API and WebSocket endpoint via FastAPI.

Base URL: `http://localhost:8765`

Authentication: `Authorization: Bearer <token>` (if `gateway.auth_token` is set). The `/health` endpoint is always public.

## REST Endpoints

### Health & Status

#### `GET /api/v1/health`

No auth required.

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "timestamp": "2026-03-05T12:00:00"
}
```

#### `GET /api/v1/status`

```json
{
  "status": "running",
  "active_sessions": 2,
  "heartbeat": {"enabled": true, "last_tick": "2026-03-05T11:30:00"},
  "tools": {"total": 10, "enabled": 8}
}
```

### Chat

#### `POST /api/v1/chat`

```json
// Request
{
  "message": "Hello, what can you do?",
  "session_id": null,
  "channel": "api"
}

// Response
{
  "response": "I can help with...",
  "session_id": "abc-123",
  "model": "claude-sonnet-4-5-20250929",
  "usage": {"input_tokens": 150, "output_tokens": 50, "total_tokens": 200}
}
```

### Conversations

#### `GET /api/v1/conversations`

Query params: `channel` (optional), `limit` (default 50).

```json
[
  {
    "id": "abc-123",
    "channel": "telegram",
    "message_count": 15,
    "total_tokens": 5000,
    "created_at": "2026-03-05T10:00:00",
    "updated_at": "2026-03-05T11:00:00"
  }
]
```

#### `GET /api/v1/conversations/{session_id}/messages`

Query params: `limit` (default 50).

```json
[
  {
    "role": "user",
    "content": "Hello",
    "model": null,
    "timestamp": "2026-03-05T10:00:00",
    "tool_calls": null
  },
  {
    "role": "assistant",
    "content": "Hi! How can I help?",
    "model": "claude-sonnet-4-5-20250929",
    "timestamp": "2026-03-05T10:00:01",
    "tool_calls": null
  }
]
```

### Tools

#### `GET /api/v1/tools`

```json
[
  {
    "name": "shell",
    "description": "Execute shell commands",
    "tier": "moderate",
    "enabled": true,
    "parameters": {...}
  }
]
```

#### `PUT /api/v1/tools/{tool_name}/toggle`

```json
// Request
{"enabled": false}

// Response
{"success": true, "name": "shell", "enabled": false}
```

### Audit

#### `GET /api/v1/audit`

Query params: `limit` (default 50), `tool_name`, `status`.

```json
[
  {
    "id": "entry-1",
    "timestamp": "2026-03-05T10:00:00",
    "tool_name": "shell",
    "status": "success",
    "duration_ms": 150,
    "trigger": "user",
    "error": null
  }
]
```

#### `GET /api/v1/audit/stats`

```json
{
  "total_calls": 100,
  "success_rate": 0.95,
  "by_tool": {"shell": 50, "read_file": 30},
  "by_status": {"success": 95, "error": 5}
}
```

### Memory

#### `GET /api/v1/memory/facts`

Query params: `limit`, `offset`, `category`, `q`.

```json
{
  "facts": [
    {
      "id": "fact-1",
      "key": "user.name",
      "value": "Abduvohid",
      "category": "user",
      "confidence": 0.95,
      "source": "extracted",
      "created_at": "2026-03-05T10:00:00",
      "updated_at": "2026-03-05T10:00:00"
    }
  ],
  "total": 25
}
```

#### `DELETE /api/v1/memory/facts/{fact_id}`

```json
{"success": true}
```

#### `GET /api/v1/memory/search`

Semantic search. Query params: `q` (required), `limit`.

```json
{
  "results": [
    {
      "text": "User discussed deploying with Docker...",
      "similarity": 0.87,
      "metadata": {"session_id": "abc-123"}
    }
  ]
}
```

#### `GET /api/v1/memory/stats`

```json
{
  "facts_count": 25,
  "vectors_count": 100,
  "soul_loaded": true
}
```

#### `POST /api/v1/memory/export`

Query params: `format` (json or markdown).

#### `POST /api/v1/memory/import`

Body: JSON export data. Query param: `merge` (default true).

### Soul

#### `GET /api/v1/soul`

```json
{
  "content": "# Agent Soul\n...",
  "loaded": true,
  "path": "/app/soul.md",
  "last_modified": "2026-03-05T10:00:00"
}
```

#### `PUT /api/v1/soul`

```json
// Request
{"content": "# Updated Soul\n..."}

// Response
{"success": true, "content": "# Updated Soul\n..."}
```

### Control

#### `POST /api/v1/control`

```json
// Request
{"action": "pause"}  // pause, resume, mute, unmute

// Response
{"status": "ok", "message": "Heartbeat paused"}
```

### Tasks

#### `GET /api/v1/tasks`

```json
[
  {
    "id": "task-1",
    "description": "Check emails",
    "type": "cron",
    "schedule": "0 9 * * *",
    "status": "active",
    "channel": "telegram",
    "created_at": "2026-03-05T10:00:00",
    "next_run": "2026-03-06T09:00:00",
    "last_run": null
  }
]
```

#### `POST /api/v1/tasks`

```json
// Request
{
  "type": "reminder",
  "description": "Review PRs",
  "schedule": "2026-03-05T15:00:00",
  "channel": "telegram"
}
```

#### `DELETE /api/v1/tasks/{task_id}`

```json
{"success": true}
```

### Skills

#### `GET /api/v1/skills`

Returns list of discovered/loaded skills.

#### `POST /api/v1/skills/{name}/reload`

Hot-reload a skill.

#### `POST /api/v1/skills/{name}/enable`

#### `POST /api/v1/skills/{name}/disable`

### Config

#### `GET /api/v1/config`

Returns full config with secrets masked.

### Stats

#### `GET /api/v1/stats/costs`

Query params: `period` (day, week, month).

#### `GET /api/v1/stats/timeline`

Query params: `limit`, `after`, `before`, `event_types`.

## WebSocket

### WebSocket Chat

Endpoint: `ws://localhost:8765/api/v1/ws/chat`

Query params:

| Param | Required | Description |
|-------|----------|-------------|
| `token` | If auth configured | Bearer token for authentication |
| `session_id` | No | Resume an existing session |

**Client → Server messages:**

```json
// Send a chat message
{"type": "message", "content": "Hello"}

// Keepalive ping
{"type": "ping"}

// Voice input (base64-encoded audio)
{"type": "voice.data", "data": "<base64>", "mime": "audio/ogg"}
```

**Server → Client messages:**

```json
// Streaming text chunk
{"type": "chunk", "content": "partial response..."}

// Completion (full response)
{"type": "done", "response": "full response", "session_id": "abc-123"}

// Typing indicator
{"type": "typing", "status": true}

// Error
{"type": "error", "message": "description"}

// Keepalive response
{"type": "pong"}

// Tool execution started
{"type": "tool_start", "tool": "shell_exec", "input": {"command": "ls"}}

// Tool execution result
{"type": "tool_result", "tool": "shell_exec", "status": "success", "duration_ms": 150}
```

**Limits:**

| Limit | Value |
|-------|-------|
| Max concurrent connections | 10 |
| Max message size | 100KB |
| Max messages per minute | 30 per connection |

### WebSocket Events

Endpoint: `ws://localhost:8765/api/v1/ws/events`

Query params:

| Param | Required | Description |
|-------|----------|-------------|
| `token` | If auth configured | Bearer token for authentication |

Receives real-time events from the agent. This is a read-only stream — no client messages expected.

```json
{
  "type": "tool.result",
  "data": {
    "tool": "shell",
    "status": "success",
    "duration_ms": 150
  },
  "timestamp": "2026-03-05T10:00:00"
}
```

Event types: `message.incoming`, `message.outgoing`, `tool.execute`, `tool.result`, `heartbeat.tick`, `heartbeat.action`, `memory.update`, `agent.error`.

### Interactive Docs

FastAPI auto-generates OpenAPI docs at:
- Swagger UI: `http://localhost:8765/docs`
- ReDoc: `http://localhost:8765/redoc`
