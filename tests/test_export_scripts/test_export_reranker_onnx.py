"""Behavioural tests for scripts/export_reranker_onnx.sh and its ps1 counterpart.

Executes the real bash script against a stub `uv` on PATH and asserts the
export venv's own optimum-cli binary is invoked exactly once with the expected
model and task. Also contains a static ps1 test that verifies $LASTEXITCODE
guards cover every uv/optimum-cli call and that uv.exe never appears.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SH_SCRIPT = REPO_ROOT / "scripts" / "export_reranker_onnx.sh"
PS1_SCRIPT = REPO_ROOT / "scripts" / "export_reranker_onnx.ps1"


def _write_stub_uv(tmp_path: Path) -> Path:
    """Create a fake `uv` that implements the subset the script needs.

    * `uv venv --seed <dir>` -> creates <dir>/bin/python
    * `uv pip install ...` -> creates <dir>/bin/optimum-cli, a bash script that
      appends its argv to <tmp>/uv.log
    * `uv run ...` -> appends 'UV_RUN_CALLED' to <tmp>/uv.log and exits 2
    """
    stub = tmp_path / "bin"
    stub.mkdir()
    uv_log = tmp_path / "uv.log"
    uv_log.write_text("")

    uv = stub / "uv"
    uv.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
LOG="{uv_log}"
mkdir -p "$(dirname "$LOG")"
if [ "$1" = "venv" ] && [ "$2" = "--seed" ]; then
    mkdir -p "$3/bin"
    touch "$3/bin/python"
    chmod +x "$3/bin/python"
elif [ "$1" = "pip" ] && [ "$2" = "install" ]; then
    shift 2
    python_path=""
    while [ $# -gt 0 ]; do
        if [ "$1" = "--python" ]; then
            shift
            python_path="$1"
        else
            shift
        fi
    done
    venv_bin="$(dirname "$python_path")"
    mkdir -p "$venv_bin"
    cat > "$venv_bin/optimum-cli" <<CLI
#!/usr/bin/env bash
echo "\\$@" >> "$LOG"
exit 0
CLI
    chmod +x "$venv_bin/optimum-cli"
elif [ "$1" = "run" ]; then
    echo "UV_RUN_CALLED" >> "$LOG"
    exit 2
else
    echo "unexpected uv call: $*" >&2
    exit 1
fi
""")
    uv.chmod(0o755)
    return stub


class TestExportRerankerOnnxScript:
    """Execute the real bash script with a stub uv on PATH."""

    def test_export_reranker_onnx_uses_venv_optimum_cli(self, tmp_path):
        """Green run: rc==0, exactly one 'export onnx' line, no UV_RUN_CALLED."""
        stub_dir = _write_stub_uv(tmp_path)
        dest = tmp_path / "export-out"
        dest.mkdir()
        env = os.environ.copy()
        env["PATH"] = f"{stub_dir}:{env.get('PATH', '')}"

        result = subprocess.run(
            ["bash", str(SH_SCRIPT), str(dest)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"script exited {result.returncode}:\n{result.stdout}\n{result.stderr}"
        )

        log = (tmp_path / "uv.log").read_text()
        lines = [l for l in log.splitlines() if l.startswith("export onnx")]
        assert len(lines) == 1, f"expected exactly one 'export onnx' line, got: {log!r}"
        assert "BAAI/bge-reranker-v2-m3" in lines[0]
        assert "text-classification" in lines[0]
        assert "UV_RUN_CALLED" not in log


class TestExportRerankerOnnxPs1:
    """Static checks on the PowerShell script."""

    def test_last_exit_code_checks_cover_all_uv_and_optimum_cli_calls(self):
        """$LASTEXITCODE checks >= uv/optimum-cli invocations, no uv.exe."""
        content = PS1_SCRIPT.read_text()
        lines = content.splitlines()
        last_exit_checks = content.count("$LASTEXITCODE")

        invocation_lines = [
            ln for ln in lines
            if (
                ln.strip().startswith("uv ")
                or ("& $optimumCliPath" in ln and "export onnx" in ln)
            )
            and not ln.strip().startswith("#")
        ]
        total_invocations = len(invocation_lines)
        assert last_exit_checks >= total_invocations, (
            f"$LASTEXITCODE checks ({last_exit_checks}) < uv/optimum-cli invocations "
            f"({total_invocations})"
        )
        assert "uv.exe" not in content, "ps1 must not reference uv.exe"
