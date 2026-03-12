# Deployment

## Docker

### Build and Run

```bash
docker build -t agent .
docker run -d \
  --name agent \
  -p 8765:8765 \
  -e ANTHROPIC_API_KEY=sk-... \
  -v $(pwd)/agent.yaml:/app/agent.yaml:ro \
  -v $(pwd)/soul.md:/app/soul.md \
  -v agent-data:/app/data \
  agent
```

### Docker Compose

```bash
# Copy config files
cp agent.yaml.example agent.yaml
cp .env.example .env
# Edit .env with your API keys

# Start
docker compose up -d

# View logs
docker compose logs -f agent

# Stop
docker compose down
```

The `docker-compose.yml` mounts:
- `agent.yaml` (read-only) — configuration
- `.env` (read-only) — secrets
- `soul.md` — personality (writable for API updates)
- `HEARTBEAT.md` — heartbeat checklist
- `skills/` — custom skills
- `agent-data` volume — SQLite DB, ChromaDB, sessions

### Health Check

```bash
curl http://localhost:8765/api/v1/health
```

The container includes a built-in health check (30s interval).

## Systemd

Create `/etc/systemd/system/agent.service`:

```ini
[Unit]
Description=Agent AI Assistant
After=network.target

[Service]
Type=simple
User=agent
WorkingDirectory=/opt/agent
ExecStart=/opt/agent/.venv/bin/agent start
ExecStop=/opt/agent/.venv/bin/agent stop
Restart=unless-stopped
RestartSec=5

# Environment — agent also loads ~/.config/agent/.env automatically
EnvironmentFile=/opt/agent/.env

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/agent/data
ReadWritePaths=/home/agent/.config/agent

[Install]
WantedBy=multi-user.target
```

```bash
# Install
sudo cp agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agent
sudo systemctl start agent

# Check status
sudo systemctl status agent
journalctl -u agent -f
```

## VPS Deployment

### 1. Setup

```bash
# Create user
sudo useradd -m -s /bin/bash agent
sudo su - agent

# Install
python3.12 -m venv .venv
source .venv/bin/activate
pip install agent-ai

# Configure (interactive wizard)
agent init
# Or manually create ~/.config/agent/agent.yaml and ~/.config/agent/.env
```

### 2. Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name agent.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name agent.example.com;

    ssl_certificate /etc/letsencrypt/live/agent.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/agent.example.com/privkey.pem;

    # API and Dashboard
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /api/v1/ws {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

```bash
# Get SSL cert
sudo certbot --nginx -d agent.example.com
```

### 3. Security Checklist

- [ ] Set `gateway.auth_token` in config
- [ ] Set `allowed_users` for Telegram
- [ ] Restrict `tools.filesystem.root` to a specific directory
- [ ] Run as non-root user
- [ ] Use HTTPS (nginx + Let's Encrypt)
- [ ] Keep `.env` permissions restricted (`chmod 600 .env`)
- [ ] Run `agent doctor --security` to verify

## PyPI

Install directly from PyPI:

```bash
# Core (CLI + LLM + basic tools)
pip install agent-ai

# With memory (ChromaDB + embeddings)
pip install agent-ai[memory]

# Everything (all channels, browser, memory)
pip install agent-ai[all]
```

## One-Line Install

```bash
curl -fsSL https://raw.githubusercontent.com/OWNER/agent/main/install.sh | bash
```

Checks Python version, installs via pip, and creates a default config.
