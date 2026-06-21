"""The README's Configuration and controls section must document every control.

A drift guard: if someone adds a control to taosmd.controls but forgets to
document it (id, default, pros/cons, cost) in the README, this fails.
"""
from __future__ import annotations

from pathlib import Path

from taosmd import controls as C

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_documents_every_control():
    text = README.read_text(encoding="utf-8")
    assert "## Configuration and controls" in text
    for cid in C.CONTROLS:
        assert f"`{cid}`" in text, f"control {cid} is not documented in the README"


def test_readme_documents_every_preset():
    text = README.read_text(encoding="utf-8")
    for p in C.PRESETS.values():
        assert p["label"] in text, f"preset {p['label']} is not documented in the README"
