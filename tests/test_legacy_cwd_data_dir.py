"""Legacy CWD ./data store discovery (Card D).

The v0.4.0 CLI wrote all stores to $CWD/data/. The current CLI resolves
the default data dir to ~/.taosmd (or TAOSMD_DATA_DIR). When a user runs
the CLI from a directory that still contains a ./data tree from a 0.4.0
install, we must warn about every legacy store found and tell the user
the exact migration command. Never auto-move.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from taosmd import cli
from taosmd.migrations import DB_FILES


def _reset_warnings(monkeypatch):
    """Reset both legacy warning guards."""
    from taosmd import agents
    monkeypatch.setattr(agents, "_legacy_registry_warned", False)
    monkeypatch.setattr(cli, "_legacy_data_dir_warned", False)


def test_legacy_cwd_data_dir_warns_for_vector_memory_and_knowledge_graph(monkeypatch, tmp_path, caplog):
    """CWD with ./data/vector-memory.db and ./data/knowledge-graph.db warns."""
    _reset_warnings(monkeypatch)

    # Point canonical data dir elsewhere
    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    # Create legacy ./data with two DB_FILES entries
    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "vector-memory.db").write_text("")  # empty file is enough to detect
    (legacy / "knowledge-graph.db").write_text("")

    # Run a CLI command that uses the default data dir
    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        rc = cli.main(["agent", "list"])

    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING about legacy ./data stores"

    msg = warnings[0]
    # Should name both found stores
    assert "vector-memory.db" in msg
    assert "knowledge-graph.db" in msg
    # Should show the exact migration command
    assert "taosmd migrate --import-legacy ./data" in msg
    # Legacy files must not be moved
    assert (legacy / "vector-memory.db").exists()
    assert (legacy / "knowledge-graph.db").exists()


def test_legacy_cwd_data_dir_warns_for_archive_and_agent_memory(monkeypatch, tmp_path, caplog):
    """CWD with ./data/archive/ and ./data/agent-memory/ warns."""
    _reset_warnings(monkeypatch)

    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "archive").mkdir()
    (legacy / "agent-memory").mkdir()

    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        rc = cli.main(["agent", "list"])

    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING about legacy ./data stores"

    msg = warnings[0]
    assert "archive/" in msg
    assert "agent-memory/" in msg
    assert "taosmd migrate --import-legacy ./data" in msg


def test_legacy_cwd_data_dir_warns_for_agents_json(monkeypatch, tmp_path, caplog):
    """CWD with ./data/agents.json warns (existing behavior extended)."""
    _reset_warnings(monkeypatch)

    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "agents.json").write_text('{"agents": []}')

    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        rc = cli.main(["agent", "list"])

    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING about legacy ./data/agents.json"

    msg = warnings[0]
    assert "agents.json" in msg
    assert "taosmd migrate --import-legacy ./data" in msg


def test_legacy_cwd_data_dir_no_warn_when_same_as_resolved(monkeypatch, tmp_path, caplog):
    """No warning when ./data IS the resolved data dir (explicit --data-dir ./data)."""
    _reset_warnings(monkeypatch)

    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "vector-memory.db").write_text("")

    # Explicit data dir = ./data, so no warning
    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        rc = cli.main(["--data-dir", str(legacy), "agent", "list"])

    assert rc == 0
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, "explicit --data-dir ./data must not warn"


def test_legacy_cwd_data_dir_no_warn_when_empty(monkeypatch, tmp_path, caplog):
    """No warning when ./data exists but is empty."""
    _reset_warnings(monkeypatch)

    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    legacy = tmp_path / "data"
    legacy.mkdir()

    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        rc = cli.main(["agent", "list"])

    assert rc == 0
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, "empty ./data must not warn"


def test_legacy_cwd_data_dir_single_warning_per_process(monkeypatch, tmp_path, caplog):
    """Only one warning per process even across multiple CLI invocations."""
    _reset_warnings(monkeypatch)

    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "vector-memory.db").write_text("")

    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        cli.main(["agent", "list"])
        cli.main(["agent", "list"])
        cli.main(["agent", "list"])

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"


def test_legacy_cwd_data_dir_lists_all_db_files(monkeypatch, tmp_path, caplog):
    """Warning lists every DB_FILES entry found in ./data."""
    _reset_warnings(monkeypatch)

    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    legacy = tmp_path / "data"
    legacy.mkdir()
    # Create a subset of DB_FILES
    for fname in ("vector-memory.db", "knowledge-graph.db", "claims.db", "archive-index.db"):
        (legacy / fname).write_text("")

    with caplog.at_level(logging.WARNING, logger="taosmd.cli"):
        rc = cli.main(["agent", "list"])

    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    msg = warnings[0]
    for fname in ("vector-memory.db", "knowledge-graph.db", "claims.db", "archive-index.db"):
        assert fname in msg, f"missing {fname} in warning: {msg}"
    assert "taosmd migrate --import-legacy ./data" in msg
