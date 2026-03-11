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
