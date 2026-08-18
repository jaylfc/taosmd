"""Red-first tests for the upstream usage-window rollover measurement.

The 14 source-text grep tests verify the credit items from PR #325
(real API, no mock, evidence committed in-repo, not future-dated,
no fake range, no external mutation, no hardcoded paths, no
contradictory claims, no scripts at root).

The functional tests verify the four remaining blockers from
revision tsk-3te4pi, using disagreement controls (ARM A, ARM B) for
the two behavioural blockers:

  BLOCKER 1: measure_at_boundary must key the flip on resets_at
             (window identity), not utilization.
  BLOCKER 2: The polling budget must start when the boundary arrives,
             and the pre-boundary baseline must be sampled before the
             boundary.
  BLOCKER 3: The evidence file's procedure must match what the code
             actually implements.
  BLOCKER 4: Flip-detection tests must use one fixture per signal,
             so a false positive (utilization shift without rollover)
             and a false negative (rollover on an idle window) are
             both caught.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import time
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


# --- BLOCKER 1 (credit): real API, not a mock ---

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


# --- BLOCKER 2 (credit): evidence committed in-repo, not /tmp ---

def test_evidence_file_exists_in_repo():
    """Evidence must be committed in benchmarks/results/, not in /tmp."""
    assert EVIDENCE_PATH.is_file(), "evidence file must be committed to the repo"


def test_no_tmp_result_paths():
    """No hardcoded /tmp paths for results or evidence."""
    src = _source()
    assert "/tmp/" not in src, "hardcoded /tmp path found in measurement script"


# --- BLOCKER 3 (credit): evidence not future-dated ---

def test_evidence_not_future_dated():
    """Evidence must not be dated after today."""
    today = datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
    data = json.loads(EVIDENCE_PATH.read_text())
    recorded = data.get("recorded_at", "")
    assert recorded, "evidence must carry a recorded_at timestamp"
    assert recorded[:10] <= today, f"evidence dated in the future: {recorded}"


# --- BLOCKER 4 (credit): no fake constant range ---

def test_evidence_no_fake_constant_range():
    """Evidence must not claim a fake 10.0s measurement."""
    blob = json.dumps(json.loads(EVIDENCE_PATH.read_text()))
    assert "10.0s-10.0s" not in blob
    assert '"average_delay": 10.0' not in blob


def test_evidence_status_is_unmeasured():
    """With no live credentials, status must remain UNMEASURED."""
    data = json.loads(EVIDENCE_PATH.read_text())
    assert data["status"] == "UNMEASURED"


# --- BLOCKER 5 (credit): no external-file mutation ---

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


# --- BLOCKER 6 (credit): no contradictory claims ---

def test_no_contradictory_measurement_claims():
    """Source must not claim the quantity was measured when it was not."""
    src = _source()
    assert "10.0s on average" not in src
    assert "Measured propagation is" not in src


# --- REQUIRED (credit): no scratch scripts at repo root ---

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


# --- BLOCKER 1 (tsk-3te4pi): key flip on resets_at, not utilization ---
# Disagreement control: ARM A (consumption without rollover) and ARM B
# (rollover with flat utilization) must give opposite verdicts.

def test_arm_a_consumption_without_rollover_not_measured():
    """ARM A: resets_at pinned, utilization changes. No rollover occurred.

    The old code keyed on utilization and would return MEASURED here
    (a false positive).  The corrected code keys on resets_at and
    returns NO_FLIP_DETECTED.
    """
    mod = _load_module()
    reset_at = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

    state = {"call": 0}

    def fake_fetch(token, client=None):
        state["call"] += 1
        if state["call"] <= 1:
            return {
                "five_hour": {"utilization": 0.20, "resets_at": "2020-01-01T13:00:00Z"}
            }
        return {
            "five_hour": {"utilization": 0.22, "resets_at": "2020-01-01T13:00:00Z"}
        }

    result = mod.measure_at_boundary(
        reset_at, "token", fetch_fn=fake_fetch, sleep_fn=lambda s: None, max_wait=0.1
    )
    assert result["status"] == "NO_FLIP_DETECTED", (
        "consumption without rollover must not be reported as a flip"
    )


def test_arm_b_rollover_with_flat_utilization_is_measured():
    """ARM B: resets_at changes, utilization flat. Rollover occurred.

    The old code keyed on utilization and would return NO_FLIP_DETECTED
    here (a false negative).  The corrected code keys on resets_at and
    returns MEASURED.

    Also verifies BLOCKER 2: the pre-boundary baseline (pre_resets_at,
    pre_reset_utilization) is captured before the boundary.
    """
    mod = _load_module()
    reset_at = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

    state = {"call": 0}

    def fake_fetch(token, client=None):
        state["call"] += 1
        if state["call"] <= 1:
            return {
                "five_hour": {"utilization": 0.0, "resets_at": "2020-01-01T08:00:00Z"}
            }
        return {
            "five_hour": {"utilization": 0.0, "resets_at": "2020-01-01T13:00:00Z"}
        }

    result = mod.measure_at_boundary(
        reset_at, "token", fetch_fn=fake_fetch, sleep_fn=lambda s: None, max_wait=0.1
    )
    assert result["status"] == "MEASURED", (
        "rollover with flat utilization must be detected"
    )
    assert result["propagation_seconds"] is not None
    assert result["pre_resets_at"] == "2020-01-01T08:00:00Z"
    assert result["post_resets_at"] == "2020-01-01T13:00:00Z"
    assert result["pre_reset_utilization"] == 0.0
    # flips detected on identity, not utilization
    assert result["pre_resets_at"] != result["post_resets_at"]
    assert result["pre_reset_utilization"] == result["post_reset_utilization"]


# --- BLOCKER 2 (tsk-3te4pi): budget starts after boundary, baseline before ---

def test_polling_budget_starts_after_boundary():
    """The polling budget must start when the boundary arrives.

    Old code set t0 before the pre-boundary wait, so a boundary that is
    farther in the future than max_wait exhausts the budget and never
    polls.  New code sets t0 only after the boundary, so polling proceeds.
    """
    mod = _load_module()
    reset_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=1.5)

    state = {"call": 0}

    def fake_fetch(token, client=None):
        state["call"] += 1
        return {
            "five_hour": {"utilization": 0.80, "resets_at": "2020-01-01T08:00:00Z"}
        }

    result = mod.measure_at_boundary(
        reset_at, "token",
        fetch_fn=fake_fetch, sleep_fn=time.sleep,
        poll_interval=0.3, max_wait=1.0,
    )
    assert state["call"] >= 2, (
        f"expected at least one pre-boundary + one post-boundary fetch, got {state['call']}"
    )
    assert result["status"] == "NO_FLIP_DETECTED"


def test_pre_reset_baseline_in_no_flip_result():
    """When no flip occurs, pre_resets_at and pre_util are still present."""
    mod = _load_module()
    reset_at = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)

    def no_flip(token, client=None):
        return {"five_hour": {"utilization": 0.80, "resets_at": "2020-01-01T08:00:00Z"}}

    result = mod.measure_at_boundary(
        reset_at, "token",
        fetch_fn=no_flip, sleep_fn=lambda s: None,
        max_wait=0.1,
    )
    assert result["status"] == "NO_FLIP_DETECTED"
    assert result["pre_resets_at"] == "2020-01-01T08:00:00Z"
    assert result["pre_reset_utilization"] == 0.80


# --- BLOCKER 4 (tsk-3te4pi): no-flip timeout still works ---

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


# --- BLOCKER 3 (tsk-3te4pi): evidence procedure matches code ---

def test_evidence_procedure_matches_code():
    """The evidence file's procedure must match what the code implements.

    The evidence says the flip is keyed on resets_at (window identity).
    The code must do the same: compare pre_resets_at against the current
    resets_at, never against utilization alone.
    """
    data = json.loads(EVIDENCE_PATH.read_text())
    procedure = data.get("procedure", "")

    assert "resets_at" in procedure, (
        "evidence procedure must reference resets_at as the flip signal"
    )

    src = _source()
    assert "pre_resets_at" in src, "code must capture pre-boundary resets_at"
    assert "cur_resets_at" in src, "code must compare against current resets_at"
    assert "post_resets_at" in src, "code must record the post-flip resets_at"
