# install-server.ps1 — install taOSmd and start it as a persistent background service
#
# Usage:
#   .\scripts\install-server.ps1 [-Host 0.0.0.0] [-Port 7900]
#
# What this does:
#   1. Install/upgrade the taosmd Python package.
#   2. Install the taosmd serve background service (Windows service via taosmd --install-service).
#   3. Verify the service is reachable on localhost.
#   4. Print Tailscale guidance and token reminder.

param (
    [string]$ServerHost = "0.0.0.0",
    [int]$Port = 7900
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== taOSmd server install ===" -ForegroundColor Cyan
Write-Host "  Host: $ServerHost   Port: $Port"

# --- Step 1: install taosmd --------------------------------------------------
Write-Host ""
Write-Host "Step 1: Installing taosmd..."
pip install --quiet --upgrade taosmd
Write-Host "  taosmd installed."

# --- Step 2: install the background service ----------------------------------
Write-Host ""
Write-Host "Step 2: Installing background service..."
$dataDir = Join-Path $env:USERPROFILE ".taosmd"
taosmd serve --install-service --host $ServerHost --port $Port --serve-data-dir $dataDir
Write-Host "  Service installed."

# --- Step 3: health check ----------------------------------------------------
Write-Host ""
Write-Host "Step 3: Checking server health on localhost:$Port..."
Start-Sleep -Seconds 3  # give the service a moment to start

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 10 -Method Get
    if ($health.status -eq "ok") {
        Write-Host "  Server is healthy: version $($health.version)" -ForegroundColor Green
    } else {
        Write-Warning "  Unexpected health response: $($health | ConvertTo-Json)"
    }
} catch {
    Write-Warning "  Health check failed: $_"
    Write-Warning "  Run 'taosmd serve --service-status' for details."
}

# --- Step 4: guidance --------------------------------------------------------
Write-Host ""
Write-Host "=== Server installation complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "The server is bound to ${ServerHost}:${Port}."
Write-Host ""
Write-Host "--- Tailscale / remote access guidance ---"
Write-Host ""
Write-Host "To reach this server from other machines over Tailscale:"
Write-Host "  1. Install Tailscale: https://tailscale.com/download"
Write-Host "  2. Run 'tailscale up' on both machines."
Write-Host "  3. Find this machine's Tailscale IP: tailscale ip -4"
Write-Host "  4. On each client:"
Write-Host "       taosmd config set-server http://<TAILSCALE_IP>:$Port"
Write-Host "     or run the client install script:"
Write-Host "       .\scripts\install-client.ps1 -ServerUrl http://<TAILSCALE_IP>:$Port"
Write-Host ""
Write-Host "--- Token auth (optional but recommended on shared networks) ---"
Write-Host ""
Write-Host "To require a bearer token:"
Write-Host "  On the SERVER:"
Write-Host "    taosmd config set-token <your-secret-token>"
Write-Host "    taosmd serve --uninstall-service"
Write-Host "    taosmd serve --install-service --host $ServerHost --port $Port"
Write-Host "  On each CLIENT:"
Write-Host "    taosmd config set-token <your-secret-token>"
Write-Host "  (Or set TAOSMD_TOKEN=<token> in the environment.)"
Write-Host ""
Write-Host "Never hardcode the token in scripts or commit it to version control."
