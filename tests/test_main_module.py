"""Regression test: ``python -m taosmd`` must be runnable.

Service supervisors (systemd/launchd/Windows) installed by
``taosmd serve --install-service`` invoke ``python -m taosmd serve``. Without a
``taosmd/__main__.py`` that fails with "No module named taosmd.__main__", which
silently breaks the background service. This guards that invocation path.
"""

import subprocess
import sys


def test_python_dash_m_taosmd_runs():
    result = subprocess.run(
        [sys.executable, "-m", "taosmd", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "serve" in result.stdout


def test_python_dash_m_taosmd_serve_help():
    result = subprocess.run(
        [sys.executable, "-m", "taosmd", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--install-service" in result.stdout
