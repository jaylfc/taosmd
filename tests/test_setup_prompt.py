# tests/test_setup_prompt.py
from taosmd import setup_prompt


GPU12 = {"host": {"cpu": {"arch": "x86_64", "cores": 16}, "ram_mb": 64000,
                  "npu": {"type": "none"},
                  "gpu": {"type": "cuda", "name": "RTX 3060", "vram_mb": 12288}}}
CPU = {"host": {"cpu": {"arch": "arm64", "cores": 8}, "ram_mb": 8000,
                "npu": {"type": "none"}, "gpu": {"type": "none", "vram_mb": 0}}}


def test_prompt_is_deterministic_for_fixed_device_info():
    a = setup_prompt.render_setup_prompt(GPU12)
    b = setup_prompt.render_setup_prompt(GPU12)
    assert a == b


def test_gpu_prompt_names_tier_recommended_profile_and_asklist():
    text = setup_prompt.render_setup_prompt(GPU12)
    assert "gpu-12gb" in text
    # capable hardware -> Quality recommended
    assert "Quality" in text
    # the consent-required switches appear as an ask-list with their cost
    assert "rerank" in text.lower()
    assert "self-verification" in text.lower() or "self_verify" in text
    # a free (non-consent) switch is reported, not asked
    assert "arctic" in text.lower()


def test_cpu_prompt_recommends_minimal():
    text = setup_prompt.render_setup_prompt(CPU)
    assert "cpu" in text
    assert "Minimal" in text


def test_stated_need_leans_integrity():
    text = setup_prompt.render_setup_prompt(GPU12, needs="we need an audit trail")
    assert "Integrity" in text


def test_probe_miss_degrades_to_cpu_minimal():
    # An empty / malformed device_info must not raise; it degrades.
    text = setup_prompt.render_setup_prompt({})
    assert "cpu" in text
    assert "Minimal" in text
