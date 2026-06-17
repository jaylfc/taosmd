# tests/test_profiles.py
from taosmd import profiles


def test_every_profile_references_only_real_switch_ids():
    valid = set(profiles.SWITCHES)
    for prof in profiles.PROFILES.values():
        for sid in prof.overrides:
            assert sid in valid, f"profile {prof.id} references unknown switch {sid}"


def test_minimal_profile_enables_no_consent_required_switch():
    minimal = profiles.PROFILES["minimal"]
    for sw in profiles.SWITCHES.values():
        if sw.requires_consent:
            assert minimal.overrides.get(sw.id, False) is False, (
                f"Minimal must not enable consent-required switch {sw.id}"
            )


def test_switch_categories_are_valid():
    allowed = {"hardware", "quality", "integrity"}
    for sw in profiles.SWITCHES.values():
        assert sw.category in allowed


def test_accessors_round_trip():
    assert profiles.get_switch("rerank").id == "rerank"
    assert profiles.get_switch("nope") is None
    assert profiles.get_profile("quality").id == "quality"
    assert {s.id for s in profiles.list_switches()} == set(profiles.SWITCHES)
    assert {p.id for p in profiles.list_profiles()} == set(profiles.PROFILES)
