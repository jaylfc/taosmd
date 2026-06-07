# install-client.ps1 — install taOSmd as a remote client pointing at a shared server
#
# Usage:
#   .\scripts\install-client.ps1 [-ServerUrl <url>]
#
# Example:
#   .\scripts\install-client.ps1 -ServerUrl http://pi.local:7900
#
# What this does:
#   1. Install/upgrade the taosmd Python package.
#   2. Set the remote server URL in %USERPROFILE%\.taosmd\config.json.
#   3. Install the taosmd-a2a Claude skill into %USERPROFILE%\.claude\skills\.
#   4. Verify the server is reachable via GET /health.

param (
    [string]$ServerUrl = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== taOSmd client install ===" -ForegroundColor Cyan

# --- Step 1: install taosmd --------------------------------------------------
Write-Host ""
Write-Host "Step 1: Installing taosmd..."
pip install --quiet --upgrade taosmd
Write-Host "  taosmd installed."

# --- Step 2: configure the remote server URL ---------------------------------
Write-Host ""
Write-Host "Step 2: Configuring remote server URL..."

if (-not $ServerUrl) {
    $ServerUrl = Read-Host "  Enter the remote taOSmd server URL (e.g. http://pi.local:7900)"
}

if (-not $ServerUrl) {
    Write-Error "error: server URL is required."
    exit 1
}

taosmd config set-server $ServerUrl
Write-Host "  Remote server URL set: $ServerUrl"

# --- Step 3: install the Claude skill ----------------------------------------
Write-Host ""
Write-Host "Step 3: Installing taosmd-a2a skill..."
try {
    taosmd install-skill
} catch {
    taosmd install-skill --force
}
Write-Host "  Skill installed."

# --- Step 4: health check ----------------------------------------------------
Write-Host ""
Write-Host "Step 4: Checking server health..."
try {
    $response = Invoke-RestMethod -Uri "$ServerUrl/health" -TimeoutSec 10 -Method Get
    if ($response.status -eq "ok") {
        Write-Host "  Server is healthy: version $($response.version)"
    } else {
        Write-Warning "  Unexpected health response: $($response | ConvertTo-Json)"
        exit 1
    }
} catch {
    Write-Warning "  Warning: server health check failed: $_"
    Write-Warning "  Check that the server is running and the URL is correct."
    exit 1
}

Write-Host ""
Write-Host "=== taOSmd client setup complete ===" -ForegroundColor Green
Write-Host "  Server : $ServerUrl"
Write-Host "  Run 'taosmd config show' to confirm settings."
Write-Host "  Run 'taosmd a2a-poll --channel CHANNEL' to poll the bus."
