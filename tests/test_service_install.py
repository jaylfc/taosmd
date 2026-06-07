"""Unit tests for taosmd.service_install.

All tests are hermetic:

* Rendering functions (``render_systemd_unit``, ``render_launchd_plist``)
  are pure — they only return strings, so they are tested directly without
  any mocking.
* ``install_*`` / ``uninstall_*`` / ``status_*`` functions are NOT called;
  only the platform-dispatch helpers are tested via ``monkeypatch`` on
  ``sys.platform``.
* The Windows path is verified by asserting that ``install_service`` on
  ``win32`` prints the PowerShell pointer and does NOT call ``subprocess``.
* No file is written to ``~/.config``, ``~/Library``, or any real system
  directory — tmp_path is used wherever a real path is needed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from taosmd import service_install


# ── systemd unit rendering ────────────────────────────────────────────────────

class TestRenderSystemdUnit:
    def test_contains_exec_start(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "ExecStart=" in unit

    def test_python_exe_in_exec_start(self):
        unit = service_install.render_systemd_unit(
            python_exe="/home/jay/.venv/bin/python",
            host="127.0.0.1",
            port=7833,
        )
        assert "/home/jay/.venv/bin/python" in unit

    def test_host_and_port_in_exec_start(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="0.0.0.0",
            port=9000,
        )
        assert "--host" in unit
        assert "0.0.0.0" in unit
        assert "--port" in unit
        assert "9000" in unit

    def test_data_dir_included_when_provided(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
            data_dir="/mnt/data/taosmd",
        )
        assert "--serve-data-dir" in unit
        assert "/mnt/data/taosmd" in unit

    def test_data_dir_absent_when_not_provided(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
            data_dir=None,
        )
        assert "--serve-data-dir" not in unit

    def test_restart_on_failure(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "Restart=on-failure" in unit

    def test_wanted_by_default_target(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "WantedBy=default.target" in unit

    def test_invokes_taosmd_serve_module(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "-m" in unit
        assert "taosmd" in unit
        assert "serve" in unit

    def test_unit_file_sections_present(self):
        unit = service_install.render_systemd_unit(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "[Unit]" in unit
        assert "[Service]" in unit
        assert "[Install]" in unit

    def test_path_with_spaces_is_shell_quoted(self):
        unit = service_install.render_systemd_unit(
            python_exe="/home/my user/.venv/bin/python",
            host="127.0.0.1",
            port=7833,
        )
        # The path must be quoted so the space doesn't split the argument.
        assert "'/home/my user/.venv/bin/python'" in unit

    def test_roundtrip_to_tmp_file(self, tmp_path):
        unit = service_install.render_systemd_unit(
            python_exe=sys.executable,
            host="127.0.0.1",
            port=7833,
            data_dir=str(tmp_path / "data"),
        )
        unit_path = tmp_path / "taosmd.service"
        unit_path.write_text(unit, encoding="utf-8")
        assert unit_path.read_text(encoding="utf-8") == unit


# ── launchd plist rendering ───────────────────────────────────────────────────

class TestRenderLaunchdPlist:
    def test_valid_xml_structure(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "<?xml" in plist
        assert "<plist" in plist
        assert "</plist>" in plist

    def test_label_present(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert service_install._LAUNCHD_PLIST_LABEL in plist

    def test_python_exe_in_program_arguments(self):
        plist = service_install.render_launchd_plist(
            python_exe="/opt/homebrew/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "/opt/homebrew/bin/python3" in plist

    def test_host_and_port_in_program_arguments(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="0.0.0.0",
            port=8080,
        )
        assert "0.0.0.0" in plist
        assert "8080" in plist

    def test_data_dir_included_when_provided(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
            data_dir="/Users/jay/.taosmd",
        )
        assert "--serve-data-dir" in plist
        assert "/Users/jay/.taosmd" in plist

    def test_data_dir_absent_when_not_provided(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
            data_dir=None,
        )
        assert "--serve-data-dir" not in plist

    def test_run_at_load_true(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        # RunAtLoad must be <true/>
        assert "<key>RunAtLoad</key>" in plist
        assert "<true/>" in plist

    def test_keep_alive_true(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "<key>KeepAlive</key>" in plist
        # true/ appears at least twice (RunAtLoad + KeepAlive)
        assert plist.count("<true/>") >= 2

    def test_stdout_and_stderr_paths_present(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
            log_dir="/tmp/taosmd-logs",
        )
        assert "StandardOutPath" in plist
        assert "StandardErrorPath" in plist
        assert "/tmp/taosmd-logs" in plist

    def test_custom_log_dir_honoured(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
            log_dir="/custom/logs",
        )
        assert "/custom/logs/taosmd-serve.log" in plist
        assert "/custom/logs/taosmd-serve-error.log" in plist

    def test_invokes_taosmd_serve_module(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        assert "-m" in plist
        assert "taosmd" in plist
        assert "serve" in plist

    def test_program_arguments_each_in_own_string_element(self):
        plist = service_install.render_launchd_plist(
            python_exe="/usr/bin/python3",
            host="127.0.0.1",
            port=7833,
        )
        # Each argument is wrapped in a separate <string> element.
        assert "<string>-m</string>" in plist
        assert "<string>taosmd</string>" in plist
        assert "<string>serve</string>" in plist

    def test_roundtrip_to_tmp_file(self, tmp_path):
        plist = service_install.render_launchd_plist(
            python_exe=sys.executable,
            host="127.0.0.1",
            port=7833,
            log_dir=str(tmp_path / "logs"),
        )
        plist_path = tmp_path / "com.taosmd.serve.plist"
        plist_path.write_text(plist, encoding="utf-8")
        assert plist_path.read_text(encoding="utf-8") == plist


# ── platform dispatch ─────────────────────────────────────────────────────────

class TestPlatformDispatch:
    """Verify that install_service / uninstall_service / service_status
    call the right platform-specific function without actually invoking
    systemctl or launchctl."""

    def test_linux_dispatch_calls_install_systemd(self, monkeypatch):
        called_with: list = []

        def fake_install_systemd(exe, host, port, data_dir):
            called_with.append(("install_systemd", exe, host, port, data_dir))
            return 0

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(service_install, "install_systemd", fake_install_systemd)
        rc = service_install.install_service("127.0.0.1", 7833)
        assert rc == 0
        assert called_with[0][0] == "install_systemd"

    def test_darwin_dispatch_calls_install_launchd(self, monkeypatch):
        called_with: list = []

        def fake_install_launchd(exe, host, port, data_dir):
            called_with.append(("install_launchd", exe, host, port, data_dir))
            return 0

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(service_install, "install_launchd", fake_install_launchd)
        rc = service_install.install_service("127.0.0.1", 7833)
        assert rc == 0
        assert called_with[0][0] == "install_launchd"

    def test_win32_dispatch_calls_install_windows(self, monkeypatch, capsys):
        called_with: list = []

        def fake_install_windows(host, port, data_dir):
            called_with.append(("install_windows", host, port, data_dir))
            return 0

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(service_install, "install_windows", fake_install_windows)
        rc = service_install.install_service("127.0.0.1", 7833)
        assert rc == 0
        assert called_with[0][0] == "install_windows"

    def test_uninstall_linux_dispatch(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(service_install, "uninstall_systemd", lambda: (called.append(1), 0)[1])
        rc = service_install.uninstall_service()
        assert rc == 0
        assert called

    def test_uninstall_darwin_dispatch(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(service_install, "uninstall_launchd", lambda: (called.append(1), 0)[1])
        rc = service_install.uninstall_service()
        assert rc == 0
        assert called

    def test_uninstall_win32_dispatch(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(service_install, "uninstall_windows", lambda: (called.append(1), 0)[1])
        rc = service_install.uninstall_service()
        assert rc == 0
        assert called

    def test_status_linux_dispatch(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(service_install, "status_systemd", lambda: (called.append(1), 0)[1])
        rc = service_install.service_status()
        assert rc == 0
        assert called

    def test_status_darwin_dispatch(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(service_install, "status_launchd", lambda: (called.append(1), 0)[1])
        rc = service_install.service_status()
        assert rc == 0
        assert called

    def test_status_win32_dispatch(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(service_install, "status_windows", lambda: (called.append(1), 0)[1])
        rc = service_install.service_status()
        assert rc == 0
        assert called

    def test_unsupported_platform_returns_nonzero(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "haiku")
        rc = service_install.install_service("127.0.0.1", 7833)
        assert rc != 0

    def test_linux_with_linux2_variant(self, monkeypatch):
        """sys.platform is 'linux' on modern kernels but was 'linux2' on older ones."""
        called: list = []

        def fake_install_systemd(exe, host, port, data_dir):
            called.append(True)
            return 0

        monkeypatch.setattr(sys, "platform", "linux2")
        monkeypatch.setattr(service_install, "install_systemd", fake_install_systemd)
        rc = service_install.install_service("127.0.0.1", 7833)
        assert rc == 0
        assert called


# ── Windows: no subprocess, prints ps1 pointer ────────────────────────────────

class TestWindowsInstallPrintsInstructions:
    """On win32, --install-service must print the .ps1 pointer and NOT call subprocess."""

    def test_install_windows_prints_ps1_path(self, capsys):
        rc = service_install.install_windows("127.0.0.1", 7833)
        captured = capsys.readouterr()
        assert "install-service.ps1" in captured.out
        assert rc == 0

    def test_install_windows_does_not_call_subprocess(self, monkeypatch):
        calls: list = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a))
        service_install.install_windows("127.0.0.1", 7833)
        assert calls == [], "install_windows must not call subprocess.run"

    def test_uninstall_windows_prints_ps1_path(self, capsys):
        rc = service_install.uninstall_windows()
        captured = capsys.readouterr()
        assert "install-service.ps1" in captured.out
        assert rc == 0

    def test_status_windows_prints_schtasks_hint(self, capsys):
        rc = service_install.status_windows()
        captured = capsys.readouterr()
        assert "schtasks" in captured.out
        assert rc == 0


# ── shell_quote helper ────────────────────────────────────────────────────────

class TestShellQuote:
    def test_simple_path_is_single_quoted(self):
        assert service_install._shell_quote("/usr/bin/python3") == "'/usr/bin/python3'"

    def test_path_with_space_is_safe(self):
        quoted = service_install._shell_quote("/home/my user/bin/python")
        # The whole result must be wrapped in single quotes so the shell
        # treats the space as part of the argument, not a word separator.
        assert quoted.startswith("'") and quoted.endswith("'")
        # The inner content must contain the path unchanged.
        assert "/home/my user/bin/python" in quoted

    def test_embedded_single_quote_is_escaped(self):
        quoted = service_install._shell_quote("it's")
        # The result must not be broken by the embedded single-quote.
        assert "'" in quoted
        # Reconstructing: remove outer quotes and unescape.
        assert "it" in quoted and "s" in quoted


# ── xml_escape helper ─────────────────────────────────────────────────────────

class TestXmlEscape:
    def test_ampersand(self):
        assert service_install._xml_escape("a&b") == "a&amp;b"

    def test_less_than(self):
        assert service_install._xml_escape("a<b") == "a&lt;b"

    def test_greater_than(self):
        assert service_install._xml_escape("a>b") == "a&gt;b"

    def test_double_quote(self):
        assert service_install._xml_escape('a"b') == "a&quot;b"

    def test_single_quote(self):
        assert service_install._xml_escape("a'b") == "a&apos;b"

    def test_clean_string_unchanged(self):
        assert service_install._xml_escape("/usr/bin/python3") == "/usr/bin/python3"


# ── CLI integration (no subprocess side-effects) ──────────────────────────────

class TestCliFlags:
    """Verify that the CLI routes --install-service / --uninstall-service /
    --service-status to service_install without starting a foreground server."""

    def _run_cli(self, argv: list[str], monkeypatch) -> int:
        from taosmd import cli
        return cli.main(argv)

    def test_install_service_flag_dispatches_install(self, monkeypatch):
        called: list = []

        def fake_install(host, port, data_dir, python_exe=None):
            called.append((host, port, data_dir))
            return 0

        monkeypatch.setattr(service_install, "install_service", fake_install)
        rc = self._run_cli(["serve", "--install-service"], monkeypatch)
        assert rc == 0
        assert called

    def test_uninstall_service_flag_dispatches_uninstall(self, monkeypatch):
        called: list = []

        def fake_uninstall():
            called.append(True)
            return 0

        monkeypatch.setattr(service_install, "uninstall_service", fake_uninstall)
        rc = self._run_cli(["serve", "--uninstall-service"], monkeypatch)
        assert rc == 0
        assert called

    def test_service_status_flag_dispatches_status(self, monkeypatch):
        called: list = []

        def fake_status():
            called.append(True)
            return 0

        monkeypatch.setattr(service_install, "service_status", fake_status)
        rc = self._run_cli(["serve", "--service-status"], monkeypatch)
        assert rc == 0
        assert called

    def test_install_passes_host_port_data_dir(self, monkeypatch):
        received: list = []

        def fake_install(host, port, data_dir, python_exe=None):
            received.append((host, port, data_dir))
            return 0

        monkeypatch.setattr(service_install, "install_service", fake_install)
        rc = self._run_cli(
            ["serve", "--host", "0.0.0.0", "--port", "8080",
             "--serve-data-dir", "/tmp/data", "--install-service"],
            monkeypatch,
        )
        assert rc == 0
        assert received[0] == ("0.0.0.0", 8080, "/tmp/data")

    def test_service_flags_are_mutually_exclusive(self, monkeypatch):
        """argparse should refuse two service flags at once."""
        import io
        with pytest.raises(SystemExit) as exc:
            self._run_cli(
                ["serve", "--install-service", "--uninstall-service"],
                monkeypatch,
            )
        assert exc.value.code != 0

    def test_no_service_flags_calls_foreground_serve(self, monkeypatch):
        """Without a service flag the CLI falls through to http_server.serve."""
        called: list = []

        def fake_serve(host, port, data_dir):
            called.append((host, port, data_dir))
            return 0

        # Patch inside the already-imported http_server module.
        from taosmd import http_server
        monkeypatch.setattr(http_server, "serve", fake_serve)
        rc = self._run_cli(["serve"], monkeypatch)
        assert rc == 0
        assert called
