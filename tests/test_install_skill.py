"""Tests for the versioned ``taosmd install-skill`` path.

``taosmd install-skill`` compares the packaged skill ``version`` (read from
the SKILL.md frontmatter) against the installed copy and records a content
hash in a ``.taosmd-skill-manifest.json`` at install time. The behaviours
under test:

* a stale install with a newer package upgrades by default (non-silent, the
  file actually changes);
* identical copies stay quiet and exit 0;
* a locally-edited copy is never clobbered without ``--force``.
"""
from __future__ import annotations

import hashlib
import json

from taosmd.cli import _run_install_skill


MANIFEST_NAME = ".taosmd-skill-manifest.json"


def _write_skill(dest_dir, version, body="skill body"):
    """Write a SKILL.md with the given frontmatter version into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    text = f"---\nname: taosmd-a2a\nversion: {version}\n---\n{body}\n"
    (dest_dir / "SKILL.md").write_text(text)
    return text


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_install_writes_skill_and_manifest(tmp_path):
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"

    rc = _run_install_skill(src, dest, force=False)
    assert rc == 0
    assert (dest / "SKILL.md").exists()
    assert (dest / MANIFEST_NAME).exists()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.0.0"
    assert data["skill_md_sha256"] == _sha(dest / "SKILL.md")


def test_identical_copy_is_quiet_and_exits_zero(tmp_path, capsys):
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    _run_install_skill(src, dest, force=False)  # fresh install
    capsys.readouterr()  # clear

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0
    assert "up to date" in out.out
    assert out.err == ""


def test_stale_clean_copy_upgrades_by_default(tmp_path, capsys):
    """A stale install whose content still matches the recorded hash upgrades."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)  # install v1.0.0
    before = (dest / "SKILL.md").read_text()

    # Packaged skill evolves: newer version + new content.
    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body\nnew a2a-watch / a2a-bridge lines")

    rc = _run_install_skill(src_v2, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0
    assert "upgrad" in out.out.lower()  # non-silent
    after = (dest / "SKILL.md").read_text()
    assert after != before  # non-zero-change: the file actually advanced
    assert "1.1.0" in after
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"
    assert data["skill_md_sha256"] == _sha(dest / "SKILL.md")


def test_locally_edited_copy_not_clobbered_without_force(tmp_path, capsys):
    """Newer package + a user edit is refused without --force (zero-loss)."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)  # install v1.0.0 baseline

    # Packaged skill evolves to 1.1, AND the user edits their copy in parallel.
    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body\nnew lines")

    edited = (dest / "SKILL.md").read_text() + "\n# local edit by user\n"
    (dest / "SKILL.md").write_text(edited)

    rc = _run_install_skill(src_v2, dest, force=False)
    out = capsys.readouterr()
    assert rc != 0
    assert "local edits" in out.err
    assert "--force" in out.err
    assert (dest / "SKILL.md").read_text() == edited  # not clobbered

    rc2 = _run_install_skill(src_v2, dest, force=True)
    new_text = (dest / "SKILL.md").read_text()
    assert rc2 == 0
    assert new_text != edited
    assert "1.1.0" in new_text
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"
    assert data["skill_md_sha256"] == _sha(dest / "SKILL.md")


def test_pre_versioning_stale_copy_refuses_without_force(tmp_path, capsys):
    """A copy with no version field and no manifest is unknown provenance."""
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "SKILL.md").write_text(
        "---\nname: taosmd-a2a\n---\nold body, no version\n"
    )

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc != 0
    assert "local edits" in out.err
    assert "unknown (pre-versioning)" in out.err
    # Not overwritten, no manifest written on a refused run.
    assert (dest / "SKILL.md").read_text() == "---\nname: taosmd-a2a\n---\nold body, no version\n"
    assert not (dest / MANIFEST_NAME).exists()

    # --force upgrades it and records a manifest.
    rc2 = _run_install_skill(src, dest, force=True)
    capsys.readouterr()  # consume
    assert rc2 == 0
    assert "1.0.0" in (dest / "SKILL.md").read_text()
    assert (dest / MANIFEST_NAME).exists()
