# Dashboard

Agent includes a React dashboard served by the FastAPI gateway.

## Access

When running `agent start`, the dashboard is available at:

```
http://localhost:8765/dashboard
```

The dashboard requires the gateway to be running. It communicates via the REST API and WebSocket.

## Building

The dashboard is a React + Vite + TypeScript + Tailwind app in the `dashboard/` directory.

```bash
cd dashboard
npm install
npm run dev     # Development server (port 5173)
npm run build   # Production build (output: dashboard/dist/)
```

The production build is served as static files by the gateway. If `dashboard/dist/` doesn't exist, the `/dashboard` endpoint returns a helpful error.

## Configuration

Set `VITE_API_URL` to point to the gateway if not running on the same host:

```bash
VITE_API_URL=http://your-server:8765 npm run build
```

Default CORS origins in `agent.yaml` include `http://localhost:5173` for development.

```yaml
gateway:
  cors_origins:
    - "http://localhost:5173"
    - "http://your-domain.com"
```

## Authentication

If `gateway.auth_token` is configured, the dashboard must include the token in requests. Set it via the dashboard's login page or environment variable.

## Pages

### Chat
Real-time conversation interface. Supports multiple sessions. Messages stream via WebSocket.

### Tools
View registered tools, their tiers, and enable/disable them.

### Memory
Browse stored facts, search vector memory, view and edit soul.md.

### Audit
Timeline of tool executions with status, duration, and error details.

### Settings
View and modify configuration (secrets masked). Control heartbeat.

## WebSocket Events

The dashboard subscribes to real-time events:

- `message.outgoing` — Agent responses
- `tool.execute` / `tool.result` — Tool activity
- `heartbeat.tick` — Heartbeat ticks
- `agent.error` — Error notifications

Connect at `ws://localhost:8765/api/v1/ws`.
