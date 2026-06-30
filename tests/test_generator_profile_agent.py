import pytest

from taosmd import agents


def test_agent_generator_profile_roundtrip(tmp_path):
    # register_agent (module-level) does not take data_dir; use the registry
    # directly against tmp_path, exactly like tests/test_agents.py does.
    agents.AgentRegistry(tmp_path).register_agent("alice")
    assert agents.get_agent_generator_profile("alice", data_dir=tmp_path) is None
    agents.set_agent_generator_profile("alice", "factual-recall", data_dir=tmp_path)
    assert agents.get_agent_generator_profile("alice", data_dir=tmp_path) == "factual-recall"
    agents.set_agent_generator_profile("alice", None, data_dir=tmp_path)
    assert agents.get_agent_generator_profile("alice", data_dir=tmp_path) is None


def test_set_agent_generator_profile_unknown_agent_raises(tmp_path):
    agents.AgentRegistry(tmp_path).register_agent("real")
    with pytest.raises(agents.AgentNotFoundError):
        agents.set_agent_generator_profile("ghost", "factual-recall", data_dir=tmp_path)


def test_get_agent_generator_profile_unknown_agent_raises(tmp_path):
    agents.AgentRegistry(tmp_path).register_agent("real")
    with pytest.raises(agents.AgentNotFoundError):
        agents.get_agent_generator_profile("ghost", data_dir=tmp_path)


def test_set_agent_generator_profile_empty_string_clears(tmp_path):
    agents.AgentRegistry(tmp_path).register_agent("bob")
    agents.set_agent_generator_profile("bob", "factual-recall", data_dir=tmp_path)
    agents.set_agent_generator_profile("bob", "", data_dir=tmp_path)
    assert agents.get_agent_generator_profile("bob", data_dir=tmp_path) is None
