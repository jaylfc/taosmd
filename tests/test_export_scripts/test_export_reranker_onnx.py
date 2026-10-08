def test_export_reranker_onnx_scripts_use_export_venv_optimum_cli(tmp_path, monkeypatch):
    """Test that export scripts invoke optimum-cli from the export venv, not via uv run."""
    import os
    from pathlib import Path
    
    # Create temporary directory for our test
    test_dir = tmp_path / "test_export"
    test_dir.mkdir()
    
    # Create fake uv executable that just records when it's called
    fake_uv = test_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env echo fake-uv-called\n")
    fake_uv.chmod(0o755)
    
    # Create fake export venv directory structure
    fake_venv = test_dir / "fake-venv"
    fake_venv_bin = fake_venv / "bin"
    fake_venv_bin.mkdir(parents=True)
    
    # Create fake optimum-cli that records its argv
    fake_optimum_cli = fake_venv_bin / "optimum-cli"
    argv_file = test_dir / "optimum_cli_argv.txt"
    fake_optimum_cli.write_text(f"""#!/usr/bin/env bash
echo "$@" > {argv_file}
exit 0
""")
    fake_optimum_cli.chmod(0o755)
    
    # Set up PATH to include our fake uv first
    monkeypatch.setenv("PATH", f"{test_dir}:{os.environ.get('PATH', '')}")
    
    # Read the actual scripts from the repository
    actual_sh = Path("/tmp/exec-tsk-fv4wpx/scripts/export_reranker_onnx.sh").read_text()
    actual_ps1 = Path("/tmp/exec-tsk-fv4wpx/scripts/export_reranker_onnx.ps1").read_text()
    
    # Assert that the bash script does NOT contain "uv run" without --no-project
    # (it shouldn't contain uv run at all in our implementation)
    assert "uv run" not in actual_sh or "--no-project" in actual_sh, \
        "bash script should not use 'uv run' without --no-project"
    
    # Assert that the bash script invokes optimum-cli from ${EXPORT_VENV}/bin/
    assert '"$EXPORT_VENV/bin/optimum-cli"' in actual_sh, \
        "bash script should invoke optimum-cli from export venv's bin directory"
    
    # Assert that the PowerShell script does NOT contain "uv run" without --no-project
    assert "uv run" not in actual_ps1 or "--no-project" in actual_ps1, \
        "PowerShell script should not use 'uv run' without --no-project"
    
    # Assert that the PowerShell script invokes optimum-cli from $ExportVenv/Scripts/
    assert '$optimumCliPath = Join-Path $ExportVenv "Scripts\\optimum-cli.exe"' in actual_ps1, \
        "PowerShell script should set optimumCliPath to export venv's Scripts directory"
    assert '& $optimumCliPath export onnx' in actual_ps1, \
        "PowerShell script should invoke optimum-cli via the optimumCliPath variable"
    
    # Additional check: make sure neither script uses bare "pip" from the venv
    # They should use uv pip install --python <venv python> or similar
    # In our bash script, we use "uv pip install --python \"$EXPORT_VENV/bin/python\""
    assert 'uv pip install --python "$EXPORT_VENV/bin/python"' in actual_sh, \
        "bash script should use uv pip with export venv's python"
    
    # In our PowerShell script, we use "uv pip install --python \"$ExportVenv\\Scripts\\python.exe\""
    assert 'uv pip install --python "$ExportVenv\\Scripts\\python.exe"' in actual_ps1, \
        "PowerShell script should use uv pip with export venv's python"