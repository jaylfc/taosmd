"""Agent registry data-dir resolution.

The registry must live in the SAME data dir as every other store. Before this
fix, AgentRegistry defaulted to CWD-relative "data" while the service layer
resolved None to TAOSMD_DATA_DIR / ~/.taosmd, so auto-registration on
ingest/search and per-agent generator profiles landed in a split-brain
$CWD/data/agents.json that nothing else read.

Pinned here:
  - default (None) resolves through the canonical resolver (env respected)
  - explicit data_dir args are honored verbatim (tests pass tmp dirs)
  - a legacy ./data/agents.json triggers a WARNING telling the user to move
    it, and is never auto-moved
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from taosmd import agents
from taosmd import cli
from taosmd.agents import AgentRegistry


def test_default_resolves_env_data_dir(monkeypatch, tmp_path):
    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    reg = AgentRegistry()
    assert reg.registry_path == real / "agents.json"

    reg.register_agent("alpha")
    assert (real / "agents.json").exists()
    assert not (tmp_path / "data").exists(), "CWD-relative ./data must not be created"


def test_default_resolves_home_when_no_env(monkeypatch):
    monkeypatch.delenv("TAOSMD_DATA_DIR", raising=False)
    reg = AgentRegistry()  # construction only; no writes against the real home
    assert reg.registry_path == Path(os.path.expanduser("~/.taosmd")) / "agents.json"


def test_explicit_data_dir_honored(monkeypatch, tmp_path):
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path / "elsewhere"))
    reg = AgentRegistry(tmp_path / "explicit")
    assert reg.registry_path == tmp_path / "explicit" / "agents.json"

    reg.register_agent("beta")
    assert (tmp_path / "explicit" / "agents.json").exists()
    assert not (tmp_path / "elsewhere").exists()


def test_module_wrappers_resolve_canonically(monkeypatch, tmp_path):
    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    agents.ensure_agent("gamma")
    assert (real / "agents.json").exists()
    assert agents.agent_exists("gamma")
    assert not (tmp_path / "data").exists()


def test_legacy_data_dir_warning_fires_even_when_canonical_exists(monkeypatch, tmp_path, caplog):
    """The stranded legacy file matters whether or not the canonical registry exists."""
    monkeypatch.setattr(agents, "_legacy_registry_warned", False)
    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "agents.json").write_text('{"agents": []}')
    # Canonical registry ALSO exists: the warning must still fire (a
    # missing-canonical-only check would go permanently silent seconds
    # after the first write while the stranded file lives on).
    real.mkdir()
    (real / "agents.json").write_text('{"agents": []}')

    with caplog.at_level(logging.WARNING, logger="taosmd.agents"):
        AgentRegistry()

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("agents.json" in m for m in warnings), (
        "expected a WARNING pointing at the legacy ./data/agents.json"
    )
    # Never auto-moved: both files untouched.
    assert (legacy / "agents.json").read_text() == '{"agents": []}'
    assert (real / "agents.json").read_text() == '{"agents": []}'


def test_legacy_data_dir_warning_once_per_process(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(agents, "_legacy_registry_warned", False)
    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "agents.json").write_text('{"agents": []}')

    with caplog.at_level(logging.WARNING, logger="taosmd.agents"):
        AgentRegistry()
        # Repeated constructions and module-wrapper calls (each of which
        # builds a fresh registry) must not re-warn within this process.
        AgentRegistry().list_agents()
        agents.agent_exists("nobody")

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f"expected exactly one legacy warning per process, got {len(warnings)}"
    )
    # Legacy file untouched.
    assert (legacy / "agents.json").exists()


def test_no_legacy_warning_for_explicit_data_dir(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(agents, "_legacy_registry_warned", False)
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "data"
    legacy.mkdir()
    (legacy / "agents.json").write_text('{"agents": []}')

    reg = AgentRegistry(tmp_path / "explicit")
    with caplog.at_level(logging.WARNING, logger="taosmd.agents"):
        reg.list_agents()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "an explicit data_dir (tests, non-default installs) must not warn"
    )


def test_cli_default_data_dir_resolves_canonically(monkeypatch, tmp_path, capsys):
    """`taosmd agent list` without --data-dir reads the canonical dir, not ./data."""
    real = tmp_path / "real-data"
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(real))
    monkeypatch.chdir(tmp_path)

    AgentRegistry(real).register_agent("delta")

    rc = cli.main(["agent", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "delta" in out
    assert not (tmp_path / "data").exists()


def test_cli_explicit_data_dir_still_honored(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(tmp_path / "elsewhere"))
    explicit = tmp_path / "explicit"
    AgentRegistry(explicit).register_agent("epsilon")

    rc = cli.main(["--data-dir", str(explicit), "agent", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "epsilon" in out
