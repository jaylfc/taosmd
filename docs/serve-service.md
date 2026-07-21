# Running taosmd serve as a persistent background service

`taosmd serve` starts the local HTTP/REST memory API and inspection UI on
`http://127.0.0.1:7900` by default.  Running it in the foreground works fine
for development, but for a live-always dashboard you want the process to
survive logouts, restart after a crash, and start automatically at login.

The `--install-service` flag handles this for you, using the native
process-supervision mechanism on each platform.

---

## Linux (systemd user unit)

No root or `sudo` required. This is a **user** unit that lives in
`~/.config/systemd/user/`.

### Install

```bash
taosmd serve --install-service
# With a custom port or data directory:
taosmd serve --host 127.0.0.1 --port 7900 --install-service
taosmd serve --serve-data-dir /mnt/data/taosmd --install-service
```

This writes `~/.config/systemd/user/taosmd.service`, runs
`systemctl --user daemon-reload`, then `systemctl --user enable --now taosmd.service`.

### Status

```bash
taosmd serve --service-status
# Or directly:
systemctl --user status taosmd.service
```

### Logs

```bash
journalctl --user -u taosmd.service -f
```

### Uninstall

```bash
taosmd serve --uninstall-service
```

Runs `systemctl --user disable --now taosmd.service` and removes the unit file.

### Change host / port / data-dir

Uninstall and reinstall with the new flags:

```bash
taosmd serve --uninstall-service
taosmd serve --port 8080 --install-service
```

---

## macOS (launchd LaunchAgent)

The service is registered as a LaunchAgent in `~/Library/LaunchAgents/` so it
starts at login and restarts if it exits.

### Install

```bash
taosmd serve --install-service
# With a custom port or data directory:
taosmd serve --port 7900 --install-service
taosmd serve --serve-data-dir ~/.taosmd --install-service
```

This writes `~/Library/LaunchAgents/com.taosmd.serve.plist` and loads it with
`launchctl load -w`.

### Status

```bash
taosmd serve --service-status
# Or directly:
launchctl list | grep com.taosmd.serve
```

### Logs

Stdout and stderr are written to `~/Library/Logs/taosmd/`:

```bash
tail -f ~/Library/Logs/taosmd/taosmd-serve.log
tail -f ~/Library/Logs/taosmd/taosmd-serve-error.log
```

### Uninstall

```bash
taosmd serve --uninstall-service
```

Runs `launchctl unload -w` and removes the plist file.

### Change host / port / data-dir

```bash
taosmd serve --uninstall-service
taosmd serve --port 8080 --install-service
```

---

## Windows (Scheduled Task via PowerShell)

Python does not register Windows Scheduled Tasks directly.  Use the
PowerShell script shipped at `scripts/install-service.ps1`.

### Install

Open an elevated PowerShell prompt in the taosmd repo directory:

```powershell
.\scripts\install-service.ps1
# With a custom port or data directory:
.\scripts\install-service.ps1 -Port 8080
.\scripts\install-service.ps1 -DataDir C:\Users\you\.taosmd
```

The script registers a Scheduled Task named `taosmd-serve` that runs at logon
and restarts automatically on failure.  It starts the task immediately so you
do not need to log out.

The CLI will print these instructions when you run `--install-service` on
Windows rather than attempting the registration itself.

### Status

```powershell
schtasks /Query /TN taosmd-serve
```

### Logs

```powershell
Get-Content "$env:LOCALAPPDATA\taosmd\logs\taosmd-serve.log" -Wait
```

### Uninstall

```powershell
.\scripts\install-service.ps1 -Uninstall
```

### Change host / port / data-dir

Uninstall and reinstall with updated parameters:

```powershell
.\scripts\install-service.ps1 -Uninstall
.\scripts\install-service.ps1 -Port 8080
```

---

## Security note

`taosmd serve` binds to `127.0.0.1` by default, which means only processes on
the same machine can reach the API. Authentication is **off by default**, but
a bearer token is available: when `server_token` is set in the server config
(via `taosmd config set-token`) or the `TAOSMD_TOKEN` environment variable is
set, every data and A2A endpoint requires an `Authorization: Bearer <token>`
header and returns `401` otherwise. The `/health`, `/version`, `/`, and `/ui`
endpoints always stay public so monitoring probes, capability probes, and the
browser dashboard keep working.

On a trusted private network (a home LAN, a Tailscale network), the network
boundary is typically sufficient as the access control. For any public-facing or
multi-tenant setup, enable the bearer token.

If you bind to `0.0.0.0` (to expose the API on the LAN), you are responsible
for network-level access controls. Consider a firewall rule or a reverse proxy,
and enable the bearer token for defence in depth. The same caveat applies
whether you run `taosmd serve` in the foreground or as a background service.

---

## Checking the port is live

```bash
curl http://127.0.0.1:7900/health
# Expected: {"status": "ok", "version": "...", "capabilities": [...]}
```

To check which build is running and what it actually supports (useful for
spotting a box left on a stale build):

```bash
curl http://127.0.0.1:7900/version
# {"version": "...", "commit": "...", "commit_source": "git",
#  "built_at": "...", "built_at_source": "install",
#  "capabilities": ["a2a.v1", "collections.v1", ...]}
```

`capabilities` holds stable contract identifiers derived from what the running
build implements. Test membership (`"collections.v1" in capabilities`) rather
than probing a route's status code: unknown non-API paths serve the dashboard
SPA, so a `200` there says nothing about what the server supports. See the
"Version and capability discovery" section of the README for the `.vN` contract.

The read-only inspection UI is at `http://127.0.0.1:7900/` in your browser.
