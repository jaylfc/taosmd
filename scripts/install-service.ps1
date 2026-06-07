<#
.SYNOPSIS
    Install or uninstall the taosmd serve background service on Windows.

.DESCRIPTION
    Registers (or removes) a Windows Scheduled Task that runs
    "python -m taosmd serve" at logon with automatic restart on failure.
    The task runs under the current user account and does not require
    elevation unless the task already exists with different permissions.

.PARAMETER Host
    Bind address passed to "taosmd serve" (default: 127.0.0.1).

.PARAMETER Port
    Bind port passed to "taosmd serve" (default: 7833).

.PARAMETER DataDir
    Data directory passed to "taosmd serve --serve-data-dir".
    If omitted, the service inherits $env:TAOSMD_DATA_DIR or ~/.taosmd.

.PARAMETER Uninstall
    Remove the Scheduled Task instead of creating it.

.EXAMPLE
    .\install-service.ps1
    .\install-service.ps1 -Port 8000 -Host 0.0.0.0
    .\install-service.ps1 -DataDir C:\Users\jay\.taosmd
    .\install-service.ps1 -Uninstall

.NOTES
    Logs are written to %LOCALAPPDATA%\taosmd\logs\.
    To view them: Get-Content "$env:LOCALAPPDATA\taosmd\logs\taosmd-serve.log" -Wait
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 7833,
    [string]$DataDir = "",
    [switch]$Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TaskName = "taosmd-serve"

if ($Uninstall) {
    Write-Host "Uninstalling taosmd Scheduled Task..."
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -eq $existing) {
        Write-Host "Task '$TaskName' not found — nothing to remove."
    } else {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Task '$TaskName' removed."
    }
    exit 0
}

# Resolve the Python executable on PATH.
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
}
if (-not $PythonExe) {
    Write-Error ("python / python3 not found on PATH. " +
                 "Install Python 3.10+ and make sure it is on your PATH.")
    exit 1
}

# Build argument list.
$Arguments = @("-m", "taosmd", "serve", "--host", $BindHost, "--port", "$Port")
if ($DataDir -ne "") {
    $Arguments += @("--serve-data-dir", $DataDir)
}

# Log directory.
$LogDir = Join-Path $env:LOCALAPPDATA "taosmd\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogOut = Join-Path $LogDir "taosmd-serve.log"

# Build the scheduled task.
# Trigger: at logon for the current user.
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Action: run python with the serve arguments; redirect stdout+stderr to log
# via cmd /c so the log file captures output without a visible console window.
$CmdArgs = "/c `"$PythonExe`" " + ($Arguments -join " ") + " >> `"$LogOut`" 2>&1"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $CmdArgs

# Settings: restart on failure (up to 3 times, 1 min apart), hidden.
$Settings = New-ScheduledTaskSettingsSet `
    -Hidden `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)  # no time limit

# Principal: interactive logon for the current user (no elevation needed for
# localhost-only default bind).
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

if ($PSCmdlet.ShouldProcess($TaskName, "Register Scheduled Task")) {
    # Remove any existing instance first so re-registration is idempotent.
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Replaced existing task '$TaskName'."
    }

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Trigger $Trigger `
        -Action $Action `
        -Settings $Settings `
        -Principal $Principal | Out-Null

    # Start it now so you don't have to log out and back in.
    Start-ScheduledTask -TaskName $TaskName

    Write-Host ""
    Write-Host "taosmd Scheduled Task registered and started."
    Write-Host "  Task name : $TaskName"
    Write-Host "  Log file  : $LogOut"
    Write-Host "  Status    : schtasks /Query /TN $TaskName"
    Write-Host "  Uninstall : .\install-service.ps1 -Uninstall"
}
