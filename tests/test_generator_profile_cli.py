from taosmd import cli


def test_cli_list_and_set(tmp_path, capsys):
    rc = cli._generator_profile_list(data_dir=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "balanced" in out and "factual-recall" in out

    rc = cli._generator_profile_set("factual-recall", agent=None, data_dir=tmp_path)
    assert rc == 0
    from taosmd import config
    assert config.get_generator_profile(data_dir=tmp_path) == "factual-recall"


def test_cli_set_rejects_unknown(tmp_path):
    rc = cli._generator_profile_set("nope", agent=None, data_dir=tmp_path)
    assert rc != 0
