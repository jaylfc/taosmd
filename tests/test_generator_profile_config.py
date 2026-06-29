from taosmd import config


def test_generator_profile_roundtrip(tmp_path):
    assert config.get_generator_profile(data_dir=tmp_path) is None
    config.set_generator_profile("factual-recall", data_dir=tmp_path)
    assert config.get_generator_profile(data_dir=tmp_path) == "factual-recall"
    config.set_generator_profile("", clear=True, data_dir=tmp_path)
    assert config.get_generator_profile(data_dir=tmp_path) is None
