"""Tests for _install_cron in taosmd.auto_setup."""

from __future__ import annotations

import os
import shlex


def _make_fake_subprocess_run(monkeypatch, data_dir: str):
    """Replace subprocess.run with a fake that captures crontab -l / crontab - calls.

    Returns a dict with captured values so assertions can check them.
    """

    captured = {"crontab_l_stdout": None, "crontab_minus_input": None}

    def fake_run(*args, **kwargs):
        if not args:
            class _FakeResult:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode
            return _FakeResult(stdout="", stderr="", returncode=0)

        cmd_list = args[0]

        # crontab -l call: subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if isinstance(cmd_list, list) and len(cmd_list) >= 2 and cmd_list[1] == "-l":
            # Return empty stdout so the "already installed" check is skipped
            captured["crontab_l_stdout"] = ""
            class _FakeResult:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode
            return _FakeResult(stdout="\n", stderr="", returncode=0)

        # crontab - call: subprocess.run(["crontab", "-"], input=new_cron, capture_output=True, text=True)
        if isinstance(cmd_list, list) and len(cmd_list) >= 2 and cmd_list[0] == "crontab" and cmd_list[1] == "-":
            captured["crontab_minus_input"] = kwargs.get("input", None)
            class _FakeResult:
                def __init__(self, stdout, stderr, returncode):
                    self.stdout = stdout
                    self.stderr = stderr
                    self.returncode = returncode
            return _FakeResult(stdout="\n", stderr="", returncode=0)

        class _FakeResult:
            def __init__(self, stdout, stderr, returncode):
                self.stdout = stdout
                self.stderr = stderr
                self.returncode = returncode
        return _FakeResult(stdout="", stderr="", returncode=0)

    import taosmd.auto_setup as _mod
    monkeypatch.setattr(_mod.subprocess, "run", fake_run)
    return captured


class TestInstallCron:
    """Verification that _install_cron builds the cron command correctly."""

    def test_cron_line_has_redirect_not_devnull(self, tmp_path, monkeypatch):
        """The cron line must contain '>> <quoted_log> 2>&1' and NOT contain '/dev/null'."""
        data_dir = str(tmp_path)
        captured = _make_fake_subprocess_run(monkeypatch, data_dir)

        from taosmd.auto_setup import _install_cron
        _install_cron(data_dir)

        stdin_input = captured["crontab_minus_input"]
        assert stdin_input is not None, "crontab - input was not captured"

        # The cron line must have >> with the quoted log path and 2>&1
        assert ">> " in stdin_input, (
            f"Expected '>> ' in captured cron line, got: {stdin_input!r}"
        )

        # Must NOT contain /dev/null
        assert "/dev/null" not in stdin_input, (
            f"Cron line should not contain '/dev/null', got: {stdin_input!r}"
        )

        # The log path must be shlex.quoted (quoted with single or double quotes)
        q_logfile = shlex.quote(os.path.join(data_dir, "logs", "librarian-cron.log"))
        # The quoted path should appear in the line
        assert q_logfile in stdin_input, (
            f"Expected quoted log path '{q_logfile}' in cron line, got: {stdin_input!r}"
        )

    def test_logs_dir_created(self, tmp_path, monkeypatch):
        """<tmp>/logs must exist after _install_cron calls os.makedirs."""
        data_dir = str(tmp_path)
        # Ensure logs subdir does not exist yet
        logs_dir = os.path.join(data_dir, "logs")
        assert not os.path.isdir(logs_dir), "logs dir should not exist before test"

        _make_fake_subprocess_run(monkeypatch, data_dir)

        from taosmd.auto_setup import _install_cron
        _install_cron(data_dir)

        assert os.path.isdir(logs_dir), f"Expected {logs_dir} to be created by os.makedirs"

    def test_space_in_data_dir_is_quoted(self, monkeypatch, tmp_path):
        """If data_dir contains a space, the log path in the cron line must be shlex.quoted."""
        # Create a data_dir with a space in it
        data_dir = str(tmp_path / "my data dir")
        os.makedirs(data_dir, exist_ok=True)

        captured = _make_fake_subprocess_run(monkeypatch, data_dir)

        from taosmd.auto_setup import _install_cron
        _install_cron(data_dir)

        stdin_input = captured["crontab_minus_input"]
        assert stdin_input is not None

        # The shlex.quote of the log path must appear in the line
        q_logfile = shlex.quote(os.path.join(data_dir, "logs", "librarian-cron.log"))
        assert q_logfile in stdin_input, (
            f"Expected quoted log path '{q_logfile}' in cron line, got: {stdin_input!r}"
        )
        # /dev/null must NOT be present
        assert "/dev/null" not in stdin_input
