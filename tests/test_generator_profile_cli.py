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


def test_cli_set_per_agent(tmp_path):
    from taosmd import agents
    agents.AgentRegistry(tmp_path).register_agent("alice")
    rc = cli._generator_profile_set("factual-recall", agent="alice", data_dir=tmp_path)
    assert rc == 0
    assert agents.get_agent_generator_profile("alice", data_dir=tmp_path) == "factual-recall"


def test_cli_set_unknown_agent_returns_nonzero(tmp_path, capsys):
    rc = cli._generator_profile_set("balanced", agent="ghost", data_dir=tmp_path)
    assert rc != 0
    err = capsys.readouterr().err
    assert "ghost" in err
