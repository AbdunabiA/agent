#!/usr/bin/env bash
# Install Agent - Autonomous AI Assistant
# Usage: curl -fsSL https://raw.githubusercontent.com/OWNER/agent/main/install.sh | bash

set -euo pipefail

MIN_PYTHON="3.12"

echo "Installing Agent..."
echo ""

# Check Python version
if ! command -v python3 &>/dev/null; then
    echo "Error: Python 3 is required. Install from https://python.org"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    echo "OK: Python $PY_VERSION"
else
    echo "Error: Python $MIN_PYTHON+ required (found $PY_VERSION)"
    exit 1
fi

# Install via pip
echo "Installing from PyPI..."
pip install --user agent-ai

# Create default config directory
CONFIG_DIR="$HOME/.config/agent"
mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_DIR/agent.yaml" ]; then
    echo "Creating default config at $CONFIG_DIR/agent.yaml"
    cat > "$CONFIG_DIR/agent.yaml" <<'YAML'
# Agent Configuration — edit as needed
agent:
  name: "Agent"
  max_iterations: 10

models:
  default: "claude-sonnet-4-5-20250929"

logging:
  level: "INFO"
YAML
fi

echo ""
echo "Agent installed successfully!"
echo ""
echo "Quick start:"
echo "  1. Set your API key:  export ANTHROPIC_API_KEY=sk-..."
echo "  2. Start chatting:    agent chat"
echo "  3. Run diagnostics:   agent doctor"
echo "  4. Full agent mode:   agent start"
echo ""
echo "Documentation: https://github.com/OWNER/agent#readme"
