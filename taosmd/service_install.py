"""Process-supervision helpers for ``taosmd serve`` as a persistent service.

Supports three platforms:

* **Linux**: a systemd *user* unit (``~/.config/systemd/user/taosmd.service``),
  no root required. The unit runs ``python -m taosmd serve`` with the
  host/port/data-dir baked in, restarts automatically on failure, and starts
  at login via ``WantedBy=default.target``.

* **macOS**: a launchd LaunchAgent plist
  (``~/Library/LaunchAgents/com.taosmd.serve.plist``). Loaded via
  ``launchctl`` so it starts at login and keeps the process alive.

* **Windows**: the CLI prints instructions pointing to
  ``scripts/install-service.ps1``, which registers a Scheduled Task with
  restart-on-failure via PowerShell. The Python side does not call
  ``schtasks``/``sc.exe`` directly, because doing so reliably (correct quoting,
  service account, restart policy) is simpler from a .ps1 module than from a
  subprocess shim.

All subprocess calls use explicit argument lists (never ``shell=True``).
Missing system tools (``systemctl``, ``launchctl``) produce a clear error
message instead of an unhandled exception.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── constants ────────────────────────────────────────────────────────────────

_SYSTEMD_UNIT_NAME = "taosmd.service"
_LAUNCHD_PLIST_LABEL = "com.taosmd.serve"
_LAUNCHD_PLIST_NAME = f"{_LAUNCHD_PLIST_LABEL}.plist"

# ── systemd (Linux) ──────────────────────────────────────────────────────────

_SYSTEMD_UNIT_TEMPLATE = """\
[Unit]
Description=taOSmd memory HTTP API
After=network.target

[Service]
ExecStart={exec_start}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def render_systemd_unit(
    python_exe: str,
    host: str,
    port: int,
    data_dir: Optional[str] = None,
) -> str:
    """Return the text of a systemd user unit for ``taosmd serve``.

    The invocation bakes in *host*, *port*, and *data_dir* (when provided)
    so the service starts with the exact same parameters the user specified
    at install time.

    Parameters
    ----------
    python_exe:
        Absolute path to the Python interpreter to use (normally
        ``sys.executable``).
    host:
        Bind address (e.g. ``127.0.0.1``).
    port:
        Bind port (e.g. ``7900``).
    data_dir:
        Optional data directory.  When ``None`` the service inherits
        ``$TAOSMD_DATA_DIR`` or the default ``~/.taosmd``, the same
        resolution the foreground server would apply.
    """
    args = [python_exe, "-m", "taosmd", "serve", "--host", host, "--port", str(port)]
    if data_dir is not None:
        args += ["--serve-data-dir", data_dir]
    exec_start = " ".join(_shell_quote(a) for a in args)
    return _SYSTEMD_UNIT_TEMPLATE.format(exec_start=exec_start)


def install_systemd(
    python_exe: str,
    host: str,
    port: int,
    data_dir: Optional[str] = None,
) -> int:
    """Write the systemd user unit and enable+start the service.

    Returns 0 on success, non-zero on error.
    """
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / _SYSTEMD_UNIT_NAME

    unit_text = render_systemd_unit(python_exe, host, port, data_dir)
    unit_path.write_text(unit_text, encoding="utf-8")
    print(f"Wrote unit file: {unit_path}")

    rc = _run_systemctl(["daemon-reload"])
    if rc != 0:
        return rc
    rc = _run_systemctl(["enable", "--now", _SYSTEMD_UNIT_NAME])
    if rc != 0:
        return rc
    print(
        f"taosmd service enabled and started.\n"
        f"  Check status: taosmd serve --service-status\n"
        f"  View logs:    journalctl --user -u {_SYSTEMD_UNIT_NAME} -f"
    )
    return 0


def uninstall_systemd() -> int:
    """Disable+stop the systemd user unit and remove the unit file.

    Returns 0 on success, non-zero on error.
    """
    rc = _run_systemctl(["disable", "--now", _SYSTEMD_UNIT_NAME])
    # rc may be non-zero if the unit was never enabled; tolerate that.
    unit_path = Path.home() / ".config" / "systemd" / "user" / _SYSTEMD_UNIT_NAME
    if unit_path.exists():
        unit_path.unlink()
        print(f"Removed unit file: {unit_path}")
        _run_systemctl(["daemon-reload"])
    else:
        print("Unit file not found, nothing to remove.")
    print("taosmd service stopped and uninstalled.")
    return 0


def status_systemd() -> int:
    """Print the systemd user unit status.

    Returns the exit code from ``systemctl --user status``.
    """
    return _run_systemctl(["status", _SYSTEMD_UNIT_NAME])


def _run_systemctl(args: list[str]) -> int:
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        print(
            "error: systemctl not found. "
            "Ensure systemd is running and systemctl is on your PATH.",
            file=sys.stderr,
        )
        return 1
    cmd = [systemctl, "--user"] + args
    result = subprocess.run(cmd)  # noqa: S603 -- explicit arg list, no shell=True
    return result.returncode


# ── launchd (macOS) ──────────────────────────────────────────────────────────

_LAUNCHD_PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
{program_arguments}
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{log_out}</string>

    <key>StandardErrorPath</key>
    <string>{log_err}</string>
</dict>
</plist>
"""


def render_launchd_plist(
    python_exe: str,
    host: str,
    port: int,
    data_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> str:
    """Return the text of a launchd LaunchAgent plist for ``taosmd serve``.

    Parameters
    ----------
    python_exe:
        Absolute path to the Python interpreter (normally ``sys.executable``).
    host:
        Bind address.
    port:
        Bind port.
    data_dir:
        Optional data directory.
    log_dir:
        Directory for stdout/stderr logs.  Defaults to
        ``~/Library/Logs/taosmd``.
    """
    args = [python_exe, "-m", "taosmd", "serve", "--host", host, "--port", str(port)]
    if data_dir is not None:
        args += ["--serve-data-dir", data_dir]

    pa_lines = "\n".join(
        f"        <string>{_xml_escape(a)}</string>" for a in args
    )

    if log_dir is None:
        log_dir = str(Path.home() / "Library" / "Logs" / "taosmd")

    log_out = os.path.join(log_dir, "taosmd-serve.log")
    log_err = os.path.join(log_dir, "taosmd-serve-error.log")

    return _LAUNCHD_PLIST_TEMPLATE.format(
        label=_LAUNCHD_PLIST_LABEL,
        program_arguments=pa_lines,
        log_out=log_out,
        log_err=log_err,
    )


def install_launchd(
    python_exe: str,
    host: str,
    port: int,
    data_dir: Optional[str] = None,
) -> int:
    """Write the launchd plist and load+start the LaunchAgent.

    Returns 0 on success, non-zero on error.
    """
    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    plist_path = agents_dir / _LAUNCHD_PLIST_NAME

    log_dir = Path.home() / "Library" / "Logs" / "taosmd"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_text = render_launchd_plist(
        python_exe, host, port, data_dir, log_dir=str(log_dir)
    )
    plist_path.write_text(plist_text, encoding="utf-8")
    print(f"Wrote plist: {plist_path}")

    # Unload any previous instance silently before reloading.
    _run_launchctl(["unload", str(plist_path)], check=False)

    rc = _run_launchctl(["load", "-w", str(plist_path)])
    if rc != 0:
        return rc
    print(
        f"taosmd LaunchAgent loaded.\n"
        f"  Logs: {log_dir}/taosmd-serve.log\n"
        f"  Check status: taosmd serve --service-status"
    )
    return 0


def uninstall_launchd() -> int:
    """Unload the launchd LaunchAgent and remove the plist file.

    Returns 0 on success, non-zero on error.
    """
    plist_path = Path.home() / "Library" / "LaunchAgents" / _LAUNCHD_PLIST_NAME
    if plist_path.exists():
        _run_launchctl(["unload", "-w", str(plist_path)], check=False)
        plist_path.unlink()
        print(f"Removed plist: {plist_path}")
    else:
        print("Plist not found, nothing to remove.")
    print("taosmd LaunchAgent unloaded and uninstalled.")
    return 0


def status_launchd() -> int:
    """Print launchd service status by grepping ``launchctl list``.

    Returns 0 if the label is listed (running), 1 otherwise.
    """
    launchctl = shutil.which("launchctl")
    if launchctl is None:
        print(
            "error: launchctl not found. Are you on macOS?",
            file=sys.stderr,
        )
        return 1
    result = subprocess.run(  # noqa: S603
        [launchctl, "list"],
        capture_output=True,
        text=True,
    )
    found = any(
        _LAUNCHD_PLIST_LABEL in line for line in result.stdout.splitlines()
    )
    if found:
        matching = [l for l in result.stdout.splitlines() if _LAUNCHD_PLIST_LABEL in l]  # noqa: E741
        print("\n".join(matching))
        return 0
    print(f"{_LAUNCHD_PLIST_LABEL}: not running")
    return 1


def _run_launchctl(args: list[str], *, check: bool = True) -> int:
    launchctl = shutil.which("launchctl")
    if launchctl is None:
        print(
            "error: launchctl not found. Are you on macOS?",
            file=sys.stderr,
        )
        return 1
    result = subprocess.run([launchctl] + args)  # noqa: S603
    if check:
        return result.returncode
    return 0


# ── Windows ──────────────────────────────────────────────────────────────────

_WINDOWS_PS1_PATH = "scripts/install-service.ps1"

_WINDOWS_INSTALL_HINT = """\
Windows service installation is handled via PowerShell.

Run the following in an elevated PowerShell prompt:

    .\\{ps1}

To uninstall:

    .\\{ps1} -Uninstall

The script registers a Scheduled Task that runs
  python -m taosmd serve
at logon with automatic restart on failure.

See {ps1} for the full script and customisation options.
"""


def install_windows(host: str, port: int, data_dir: Optional[str] = None) -> int:
    """Print installation instructions for Windows and exit without error.

    Python does not attempt to register the Scheduled Task directly.
    """
    print(_WINDOWS_INSTALL_HINT.format(ps1=_WINDOWS_PS1_PATH))
    return 0


def uninstall_windows() -> int:
    """Print uninstall instructions for Windows."""
    print(
        f"To uninstall the taosmd Scheduled Task on Windows, run:\n\n"
        f"    .\\{_WINDOWS_PS1_PATH} -Uninstall\n"
    )
    return 0


def status_windows() -> int:
    """Print status instructions for Windows."""
    print(
        "To check the taosmd Scheduled Task status on Windows, run:\n\n"
        "    schtasks /Query /TN taosmd-serve\n"
    )
    return 0


# ── platform dispatch ─────────────────────────────────────────────────────────

def install_service(
    host: str,
    port: int,
    data_dir: Optional[str] = None,
    python_exe: Optional[str] = None,
) -> int:
    """Install and start the ``taosmd serve`` background service.

    Delegates to the platform-appropriate implementation:

    * ``linux``: systemd user unit
    * ``darwin``: launchd LaunchAgent
    * ``win32``: prints instructions for the PowerShell script

    Parameters
    ----------
    host:
        Bind address to bake into the service definition.
    port:
        Bind port.
    data_dir:
        Optional data directory.  When ``None`` the service inherits the
        same default resolution as the foreground ``taosmd serve`` command
        (``$TAOSMD_DATA_DIR`` or ``~/.taosmd``).
    python_exe:
        Python interpreter path.  Defaults to ``sys.executable``.
    """
    exe = python_exe or sys.executable
    platform = sys.platform
    if platform.startswith("linux"):
        return install_systemd(exe, host, port, data_dir)
    if platform == "darwin":
        return install_launchd(exe, host, port, data_dir)
    if platform == "win32":
        return install_windows(host, port, data_dir)
    print(
        f"error: unsupported platform {platform!r}. "
        "Install manually or open an issue at https://github.com/jaylfc/taosmd.",
        file=sys.stderr,
    )
    return 1


def uninstall_service() -> int:
    """Stop and remove the ``taosmd serve`` background service."""
    platform = sys.platform
    if platform.startswith("linux"):
        return uninstall_systemd()
    if platform == "darwin":
        return uninstall_launchd()
    if platform == "win32":
        return uninstall_windows()
    print(
        f"error: unsupported platform {platform!r}.",
        file=sys.stderr,
    )
    return 1


def service_status() -> int:
    """Print the running status of the ``taosmd serve`` background service."""
    platform = sys.platform
    if platform.startswith("linux"):
        return status_systemd()
    if platform == "darwin":
        return status_launchd()
    if platform == "win32":
        return status_windows()
    print(
        f"error: unsupported platform {platform!r}.",
        file=sys.stderr,
    )
    return 1


# ── internal helpers ─────────────────────────────────────────────────────────

def _shell_quote(s: str) -> str:
    """Minimal POSIX-safe shell quoting for unit-file ``ExecStart`` lines.

    Single-quotes the argument and escapes any embedded single-quotes.
    This avoids a ``shlex`` import while producing the same result for the
    paths and hostnames we encounter here.
    """
    escaped = s.replace("'", "'\\''")
    return f"'{escaped}'"


def _xml_escape(s: str) -> str:
    """Escape the five XML special characters for plist string values."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
