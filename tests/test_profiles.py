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


def test_recommend_profile_by_tier():
    assert profiles.recommend_profile("cpu") == "minimal"
    assert profiles.recommend_profile("pi-npu") == "minimal"
    assert profiles.recommend_profile("gpu-4gb") == "quality"
    assert profiles.recommend_profile("gpu-8gb") == "quality"
    assert profiles.recommend_profile("gpu-12gb") == "quality"


def test_recommend_profile_needs_override_to_integrity():
    # A stated audit/compliance need leans Integrity regardless of tier.
    assert profiles.recommend_profile("gpu-12gb", needs="we need an audit trail") == "integrity"
    assert profiles.recommend_profile("cpu", needs="compliance and provenance") == "integrity"


def test_resolve_config_minimal_enables_no_consent_switch():
    cfg = profiles.resolve_config("minimal", consented_switches=[])
    # arctic_embed is a free default (no consent), so it is present; the
    # consent-required keys are absent.
    assert cfg.get("vector_memory.embed_model") == "arctic-embed-s"
    assert "retrieval.reranker" not in cfg
    assert "answer.self_verify" not in cfg
    # The recall gate writes the live key the runtime reads, and Minimal opts
    # out explicitly (the runtime default is on, so "off" must be written).
    assert cfg["controls.prefer_verified"] == "off"
    assert "claims.prefer_verified" not in cfg  # the old dead key is gone


def test_resolve_config_integrity_enables_recall_gate():
    cfg = profiles.resolve_config("integrity", consented_switches=["rerank", "self_verify"])
    # Integrity turns the gate on with the value the runtime recall gate reads.
    assert cfg["controls.prefer_verified"] == "prefer_verified"


def test_resolve_config_consent_required_switch_needs_explicit_consent():
    # Quality wants rerank+self_verify, but without consent they stay OFF.
    cfg_no = profiles.resolve_config("quality", consented_switches=[])
    assert "retrieval.reranker" not in cfg_no
    assert "answer.self_verify" not in cfg_no
    # With consent they are written with their on_value.
    cfg_yes = profiles.resolve_config("quality", consented_switches=["rerank", "self_verify"])
    assert cfg_yes["retrieval.reranker"] == "bge-v2-m3"
    assert cfg_yes["answer.self_verify"] is True


def test_resolve_config_unknown_profile_raises():
    import pytest
    with pytest.raises(ValueError):
        profiles.resolve_config("nope", consented_switches=[])


def test_profiles_schema_lists_switches_and_profiles():
    schema = profiles.profiles_schema()
    assert {s["id"] for s in schema["switches"]} == set(profiles.SWITCHES)
    assert {p["id"] for p in schema["profiles"]} == set(profiles.PROFILES)
    # each switch row carries the dashboard-facing fields
    row = next(s for s in schema["switches"] if s["id"] == "self_verify")
    assert row["requires_consent"] is True
    assert row["cost"] and row["help"]
