"""Red-first tests for the upstream usage-window rollover measurement.

Each test encodes a requirement violated by the original PR #288
(revision card tsk-rcnct6).  The tests inspect source code and committed
evidence to confirm the measurement is real, not simulated, and that
every blocker from the review is addressed.

Blockers addressed:
  1. Does not measure the quantity (used a MockAPI instead of the real API)
  2. Evidence not in the PR and not reproducible (stored in /tmp)
  3. Measurement dated in the future
  4. Range "10.0s-10.0s" (identical endpoints from a mock)
  5. Mutates a file outside this repo
  6. Source contradicts itself (TO MEASURE above a claim it was performed)
  Required: scratch scripts must not live at the repo root
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "benchmarks" / "measure_upstream_propagation.py"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
EVIDENCE_PATH = RESULTS_DIR / "tsk-rcnct6_upstream_propagation.json"


def _load_module():
    """Load benchmarks/measure_upstream_propagation.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "measure_upstream_propagation", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source():
    return SCRIPT_PATH.read_text()


# --- BLOCKER 1: it does not measure the quantity (used a mock API) ---

def test_no_mock_api_class():
    """The harness must call the real Anthropic API, never substitute a mock."""
    src = _source()
    assert "class MockAPI" not in src, "MockAPI class present: not a real measurement"


def test_no_window_flips_dict():
    """No pre-baked window_flips data that replaces real API responses."""
    src = _source()
    assert "window_flips" not in src


def test_uses_real_api_endpoint():
    """The harness must target the real Anthropic usage endpoint."""
    src = _source()
    assert "api.anthropic.com" in src
    assert "oauth/usage" in src


def test_fetch_usage_uses_httpx():
    """fetch_usage must issue a real HTTP request via httpx."""
    src = _source()
    assert "def fetch_usage" in src
    assert "httpx" in src


def test_no_simulation_keywords():
    """The harness must not describe itself as a simulation."""
    src_lower = _source().lower()
    for forbidden in ("simulate", "simulator", "simulation"):
        assert forbidden not in src_lower, f"forbidden word '{forbidden}' in source"


# --- BLOCKER 2: evidence not in the PR and not reproducible ---

def test_evidence_file_exists_in_repo():
    """Evidence must be committed in benchmarks/results/, not in /tmp."""
    assert EVIDENCE_PATH.is_file(), "evidence file must be committed to the repo"


def test_no_tmp_result_paths():
    """No hardcoded /tmp paths for results or evidence."""
    src = _source()
    assert "/tmp/" not in src, "hardcoded /tmp path found in measurement script"


# --- BLOCKER 3: measurement dated in the future ---

def test_evidence_not_future_dated():
    """Evidence must not be dated after today."""
    today = datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
    data = json.loads(EVIDENCE_PATH.read_text())
    recorded = data.get("recorded_at", "")
    assert recorded, "evidence must carry a recorded_at timestamp"
    assert recorded[:10] <= today, f"evidence dated in the future: {recorded}"


# --- BLOCKER 4: range 10.0s-10.0s ---

def test_evidence_no_fake_constant_range():
    """Evidence must not claim a fake 10.0s measurement."""
    blob = json.dumps(json.loads(EVIDENCE_PATH.read_text()))
    assert "10.0s-10.0s" not in blob
    assert '"average_delay": 10.0' not in blob


def test_evidence_status_is_unmeasured():
    """With no live credentials, status must remain UNMEASURED."""
    data = json.loads(EVIDENCE_PATH.read_text())
    assert data["status"] == "UNMEASURED"


# --- BLOCKER 5: mutates a file outside this repo ---

def test_no_external_file_mutation():
    """No code that rewrites files outside the repository."""
    src = _source()
    assert "apply_change" not in src, "script generates external mutation: apply_change"
    assert "sed -i" not in src
    assert "cp ~/.taos-team" not in src
    assert "resume_arm_time.py.backup" not in src


def test_no_hardcoded_external_paths():
    """No hardcoded paths to ~/.taos-team or /home/jay."""
    src = _source()
    assert "/home/jay/.taos-team" not in src, "hardcoded external path"
    assert "/home/jay/.claude/.credentials.json" not in src


# --- BLOCKER 6: file contradicts itself ---

def test_no_contradictory_measurement_claims():
    """Source must not claim the quantity was measured when it was not."""
    src = _source()
    assert "10.0s on average" not in src
    assert "Measured propagation is" not in src


# --- REQUIRED: scratch scripts at repo root ---

def test_no_scripts_at_root():
    """Measurement and helper scripts must not live at the repository root."""
    assert not (REPO_ROOT / "measure_upstream_propagation.py").exists()
    assert not (REPO_ROOT / "analyze_and_update.py").exists()
    assert not (REPO_ROOT / "test_api_access.py").exists()
    assert not (REPO_ROOT / "test_measurement.py").exists()


# --- functional: fetch_usage returns parsed JSON from the real endpoint ---

def test_fetch_usage_returns_parsed_json():
    """fetch_usage must return the parsed JSON body, not canned data."""
    mod = _load_module()
    payload = {
        "five_hour": {"utilization": 0.12, "resets_at": "2026-08-17T08:00:00Z"},
        "seven_day": {"utilization": 0.03, "resets_at": "2026-08-18T02:00:00Z"},
    }
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=payload)
    )
    with httpx.Client(transport=transport) as client:
        result = mod.fetch_usage("token", client=client)
    assert result["five_hour"]["utilization"] == 0.12


# --- functional: measure_at_boundary polls and detects the flip ---

def test_measure_at_boundary_detects_flip():
    """measure_at_boundary must poll until utilization changes."""
    mod = _load_module()
    reset_at = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

    state = {"call": 0}

    def fake_fetch(token, client=None):
        state["call"] += 1
        if state["call"] <= 1:
            return {
                "five_hour": {"utilization": 0.80, "resets_at": "2020-01-01T08:00:00Z"}
            }
        return {
            "five_hour": {"utilization": 0.20, "resets_at": "2020-01-01T13:00:00Z"}
        }

    result = mod.measure_at_boundary(
        reset_at, "token", fetch_fn=fake_fetch, sleep_fn=lambda s: None
    )
    assert result["status"] == "MEASURED"
    assert result["propagation_seconds"] is not None
    assert state["call"] == 2


def test_measure_at_boundary_no_flip_times_out():
    """When the window never flips, status is NO_FLIP_DETECTED."""
    mod = _load_module()
    reset_at = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

    def no_flip(token, client=None):
        return {
            "five_hour": {"utilization": 0.80, "resets_at": "2020-01-01T08:00:00Z"}
        }

    result = mod.measure_at_boundary(
        reset_at, "token",
        fetch_fn=no_flip, sleep_fn=lambda s: None,
        max_wait=0.1,
    )
    assert result["status"] == "NO_FLIP_DETECTED"
