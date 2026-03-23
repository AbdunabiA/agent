# Security

Agent's security model covers tool permissions, input validation, network protection, and data safety.

## Permission Tiers

Every tool has one of three tiers:

| Tier | Behavior | Examples |
|------|----------|----------|
| **Safe** | Auto-approved | Read files, web search, memory lookup |
| **Moderate** | Configurable (auto or prompt) | Write files, shell commands, HTTP requests |
| **Dangerous** | Always requires confirmation | Python exec, delete files, system config |

In Telegram, moderate/dangerous tools show interactive Approve/Deny buttons.

## Guardrails

### Blocked Command Patterns

The shell tool blocks dangerous patterns:

- `rm -rf /` and variants
- `mkfs`, `dd if=`, `fdisk`
- Fork bombs (`:(){:|:&};:`)
- `chmod 777`, `chmod -R 777`
- `> /dev/sda` and disk writes
- `curl | sh`, `wget | bash` (pipe-to-shell)
- `shutdown`, `reboot`, `halt`
- `iptables -F` (firewall flush)

### Path Validation

Filesystem tools validate paths against the configured root (`tools.filesystem.root`). Path traversal attempts (`../`) are blocked.

### SSRF Protection

The HTTP request tool blocks requests to private IP ranges:
- `127.0.0.0/8` (loopback)
- `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` (private)
- `169.254.0.0/16` (link-local)
- `0.0.0.0`

## Authentication

### Gateway

Set `gateway.auth_token` in config. All API endpoints (except `/health`) require `Authorization: Bearer <token>`.

```yaml
gateway:
  auth_token: "${GATEWAY_TOKEN}"
```

### Telegram

Set `allowed_users` to restrict who can use the bot:

```yaml
channels:
  telegram:
    allowed_users: [123456789]  # Telegram user IDs
```

**Warning**: An empty list allows everyone. `agent doctor` warns about this.

## Rate Limiting

The gateway applies rate limiting: 60 requests per minute per IP address. Exceeding this returns HTTP 429.

## Audit Log

All tool executions are logged with:
- Timestamp
- Tool name
- Input parameters
- Status (success/error/denied/blocked)
- Duration
- Trigger (user/heartbeat/system)

View via `agent audit`, `/audit` chat command, or `GET /api/v1/audit`.

## Resource Limits

- **Max iterations**: 10 per request (configurable)
- **Tool timeout**: 30 seconds default
- **Cost tracking**: Token usage tracked per model/channel

## Data Storage

Agent stores data locally:

| Data | Location | Contains |
|------|----------|----------|
| SQLite DB | `data/agent.db` | Facts, audit log, sessions |
| ChromaDB | `data/memory/chroma/` | Conversation embeddings |
| soul.md | `./soul.md` | Personality definition |

No data is sent to external services except LLM API calls (which include conversation context).

### Secrets

- API keys stored in `.env` (gitignored)
- Config values with `${VAR}` syntax resolved from environment
- `agent config show` masks all secrets in output
- Gateway auth token never logged

## Security Checks

Run security-focused diagnostics:

```bash
agent doctor --security
```

Checks:
- .env file presence
- Gateway auth configuration
- Filesystem root restriction
- Telegram allowlist
- Hardcoded secret scan
- Dependency vulnerability check

## Threat Model

- Designed for single-user local deployment
- Gateway binds to localhost by default (not exposed to the internet)
- If exposed, use a reverse proxy with TLS (nginx, Caddy)
- No multi-tenant isolation (workspaces share the same process)
- No CSRF protection (API-only, no browser forms)

## Authentication Details

| Channel | Mechanism | Details |
|---------|-----------|---------|
| Gateway | Bearer token | `Authorization: Bearer <token>` header or `?token=` query param |
| WebSocket | Query param | `?token=` at connection time |
| Telegram | Allowlist | `allowed_users` restricts access by Telegram user ID |

All tokens are compared using constant-time comparison (`secrets.compare_digest`).

## Rate Limiting Details

| Resource | Limit | Configurable |
|----------|-------|-------------|
| HTTP requests | 60 per minute per IP | `rate_limit_per_minute` |
| Auth failures | 5 in 5 minutes → IP lockout | Yes |
| WebSocket messages | 30 per minute per connection | Yes |
| Concurrent WebSocket connections | 10 | Yes |
| Max message size | 100KB | — |

## Data Protection

- Secrets (passwords, API keys, tokens) are automatically masked in the audit log
- Config secrets are loaded from `.env` (gitignored)
- Config values interpolated via `${VAR_NAME}` syntax from environment
- Gateway auth token is never logged
- `agent config show` masks all secret values in output

## Known Limitations

- **No built-in TLS** — use a reverse proxy for HTTPS
- **No CSRF tokens** — API-only design, no browser form submissions
- **No OAuth2/JWT** — static Bearer token only
- **WebSocket token in URL** — passed via query parameter, visible in logs if proxy is misconfigured
- **No per-user authentication** — single shared token for all API access
