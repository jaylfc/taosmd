from taosmd import controls


def test_generator_profile_control_present():
    c = controls.CONTROLS["generator_profile"]
    assert c.type == "choice"
    assert "balanced" in c.choices and "factual-recall" in c.choices
    assert c.default == "balanced"


def test_generator_profile_validates():
    assert controls.validate_control("generator_profile", "factual-recall") == "factual-recall"


def test_generator_profile_rejects_unknown():
    import pytest
    with pytest.raises(Exception):
        controls.validate_control("generator_profile", "nope")


def test_generator_profile_in_schema():
    schema = controls.controls_schema()
    ids = [c["id"] for c in schema] if isinstance(schema, list) else list(schema)
    assert "generator_profile" in ids or "generator_profile" in str(schema)
