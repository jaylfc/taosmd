#!/usr/bin/env bash
# install-client.sh — install taOSmd as a remote client pointing at a shared server
#
# Usage:
#   ./scripts/install-client.sh [SERVER_URL]
#
# If SERVER_URL is not passed as an argument the script will prompt for it.
# Example:
#   ./scripts/install-client.sh http://pi.local:7900
#   ./scripts/install-client.sh https://my-device.tailscale.ts.net:7900
#
# What this does:
#   1. Install/upgrade the taosmd Python package.
#   2. Set the remote server URL in ~/.taosmd/config.json.
#   3. Install the taosmd-a2a Claude skill into ~/.claude/skills/.
#   4. Verify the server is reachable via GET /health.

set -euo pipefail

echo "=== taOSmd client install ==="

# --- Step 1: install taosmd ---------------------------------------------------
echo ""
echo "Step 1: Installing taosmd..."
pip install --quiet --upgrade taosmd
echo "  taosmd $(taosmd --version 2>/dev/null || python -c "import taosmd; print(taosmd.__version__)") installed."

# --- Step 2: configure the remote server URL ---------------------------------
echo ""
echo "Step 2: Configuring remote server URL..."

SERVER_URL="${1:-}"
if [ -z "$SERVER_URL" ]; then
  read -rp "  Enter the remote taOSmd server URL (e.g. http://pi.local:7900): " SERVER_URL
fi

if [ -z "$SERVER_URL" ]; then
  echo "error: server URL is required." >&2
  exit 1
fi

taosmd config set-server "$SERVER_URL"
echo "  Remote server URL set: $SERVER_URL"

# --- Step 3: install the Claude skill ----------------------------------------
echo ""
echo "Step 3: Installing taosmd-a2a skill..."
taosmd install-skill || taosmd install-skill --force
echo "  Skill installed."

# --- Step 4: health check -----------------------------------------------------
echo ""
echo "Step 4: Checking server health..."
if command -v curl >/dev/null 2>&1; then
  HEALTH=$(curl -s --max-time 10 "${SERVER_URL}/health" 2>/dev/null || true)
elif command -v wget >/dev/null 2>&1; then
  HEALTH=$(wget -qO- --timeout=10 "${SERVER_URL}/health" 2>/dev/null || true)
else
  # Fallback: use Python urllib
  HEALTH=$(python -c "
import urllib.request, json
try:
    with urllib.request.urlopen('${SERVER_URL}/health', timeout=10) as r:
        print(r.read().decode())
except Exception as e:
    print('{\"error\": \"' + str(e) + '\"}')
" 2>/dev/null || true)
fi

if echo "$HEALTH" | grep -q '"status": *"ok"'; then
  echo "  Server is healthy: $HEALTH"
else
  echo "  Warning: server health check failed or server is unreachable."
  echo "  Response: $HEALTH"
  echo "  Check that the server is running and the URL is correct."
  exit 1
fi

echo ""
echo "=== taOSmd client setup complete ==="
echo "  Server : $SERVER_URL"
echo "  Run 'taosmd config show' to confirm settings."
echo "  Run 'taosmd a2a-poll --channel CHANNEL' to poll the bus."
