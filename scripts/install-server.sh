#!/usr/bin/env bash
# install-server.sh — install taOSmd and start it as a persistent background service
#
# Usage:
#   ./scripts/install-server.sh [--host HOST] [--port PORT]
#
# Defaults: host=0.0.0.0, port=7900
#
# What this does:
#   1. Install/upgrade the taosmd Python package.
#   2. Install and start the taosmd serve background service on 0.0.0.0:7900
#      (or the specified host/port) via taosmd serve --install-service.
#   3. Verify the service is reachable on localhost.
#   4. Print Tailscale guidance and token reminder.

set -euo pipefail

HOST="0.0.0.0"
PORT="7900"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "=== taOSmd server install ==="
echo "  Host: $HOST   Port: $PORT"

# --- Step 1: install taosmd --------------------------------------------------
echo ""
echo "Step 1: Installing taosmd..."
pip install --quiet --upgrade taosmd
echo "  taosmd $(taosmd --version 2>/dev/null || python -c "import taosmd; print(taosmd.__version__)") installed."

# --- Step 2: install and start the background service -----------------------
echo ""
echo "Step 2: Installing background service (systemd / LaunchAgent)..."
taosmd serve --install-service --host "$HOST" --port "$PORT" --serve-data-dir ~/.taosmd
echo "  Service installed."

# --- Step 3: health check ----------------------------------------------------
echo ""
echo "Step 3: Checking server health on localhost:$PORT..."
sleep 2  # give the service a moment to start

if command -v curl >/dev/null 2>&1; then
  HEALTH=$(curl -s --max-time 10 "http://127.0.0.1:${PORT}/health" 2>/dev/null || true)
else
  HEALTH=$(python -c "
import urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=10) as r:
        print(r.read().decode())
except Exception as e:
    print('{\"error\": \"' + str(e) + '\"}')
" 2>/dev/null || true)
fi

if echo "$HEALTH" | grep -q '"status": *"ok"'; then
  echo "  Server is healthy: $HEALTH"
else
  echo "  Warning: health check did not confirm 'ok'. Response: $HEALTH"
  echo "  Check 'taosmd serve --service-status' for details."
fi

# --- Step 4: guidance --------------------------------------------------------
echo ""
echo "=== Server installation complete ==="
echo ""
echo "The server is bound to $HOST:$PORT."
echo ""
echo "--- Tailscale / remote access guidance ---"
echo ""
echo "To reach this server from other machines over Tailscale:"
echo "  1. Install Tailscale on both machines: https://tailscale.com/download"
echo "  2. Run 'tailscale up' on this machine and on the client machines."
echo "  3. Find this machine's Tailscale IP or MagicDNS name:"
echo "       tailscale ip -4"
echo "       tailscale status"
echo "  4. On each client run:"
echo "       taosmd config set-server http://<TAILSCALE_IP_OR_HOSTNAME>:$PORT"
echo "     or run the client install script:"
echo "       ./scripts/install-client.sh http://<TAILSCALE_IP_OR_HOSTNAME>:$PORT"
echo ""
echo "--- Token auth (optional but recommended on shared networks) ---"
echo ""
echo "To require a bearer token:"
echo "  On the SERVER:"
echo "    taosmd config set-token <your-secret-token>"
echo "    taosmd serve --uninstall-service  # stop"
echo "    taosmd serve --install-service --host $HOST --port $PORT  # restart"
echo "  On each CLIENT:"
echo "    taosmd config set-token <your-secret-token>"
echo "  (Or set TAOSMD_TOKEN=<token> in both environments.)"
echo ""
echo "Never hardcode the token in scripts or commit it to version control."
