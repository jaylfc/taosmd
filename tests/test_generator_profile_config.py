import pytest

from taosmd import config


def test_generator_profile_roundtrip(tmp_path):
    assert config.get_generator_profile(data_dir=tmp_path) is None
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    assert config.get_generator_profile(data_dir=tmp_path) == "factual-recall"
    config.set_generator_profile("", clear=True, data_dir=tmp_path)
    assert config.get_generator_profile(data_dir=tmp_path) is None


def test_set_generator_profile_rejects_blank_and_nonstr(tmp_path):
    # clear=False with a blank or non-string id must raise and persist nothing.
    for bad in ("", "   ", 123, None):
        with pytest.raises(ValueError):
            config.set_generator_profile(bad, data_dir=tmp_path)
    assert config.get_generator_profile(data_dir=tmp_path) is None
