# Stage 1: Build dashboard (if present)
FROM node:20-slim AS dashboard-builder
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json* ./
RUN if [ -f package.json ]; then npm ci; fi
COPY dashboard/ ./
RUN if [ -f package.json ]; then npm run build; else mkdir -p dist; fi

# Stage 2: Build Python package
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[all]"

# Stage 3: Production image
FROM python:3.12-slim AS production
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/agent /usr/local/bin/agent

# Copy application
COPY src/ ./src/
COPY pyproject.toml ./
COPY agent.yaml.example ./agent.yaml
COPY soul.md ./
COPY HEARTBEAT.md ./
COPY skills/ ./skills/

# Copy built dashboard
COPY --from=dashboard-builder /app/dashboard/dist ./dashboard/dist

# Create data directories
RUN mkdir -p data/memory data/backups data/sessions data/uploads data/notes

# Create non-root user
RUN useradd -m -s /bin/bash agent && chown -R agent:agent /app
USER agent

ENV PYTHONUNBUFFERED=1
ENV AGENT_CONFIG=/app/agent.yaml

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8765/api/v1/health || exit 1

EXPOSE 8765

ENTRYPOINT ["agent"]
CMD ["start"]
