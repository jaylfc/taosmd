"""Tests for the versioned ``taosmd install-skill`` path.

``taosmd install-skill`` compares the packaged skill ``version`` (read from
the SKILL.md frontmatter) against the installed copy and records a content
hash in a ``.taosmd-skill-manifest.json`` at install time. The behaviours
under test:

* a stale install with a newer package upgrades by default (non-silent, the
  file actually changes);
* identical copies stay quiet and exit 0;
* a locally-edited copy is never clobbered without ``--force``;
* versions are ordered numerically, so an older package is refused rather
  than installed under the word "upgraded";
* an unreadable manifest degrades to the no-manifest path instead of raising;
* a non-empty directory at the manifest path is removed so the install
  completes cleanly on both arms;
* a failed manifest write leaves SKILL.md unchanged (no half-apply).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from taosmd.cli import _parse_skill_version, _run_install_skill, _version_tuple


MANIFEST_NAME = ".taosmd-skill-manifest.json"


def _write_skill(dest_dir, version, body="skill body"):
    """Write a SKILL.md with the given frontmatter version into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    text = f"---\nname: taosmd-a2a\nversion: {version}\n---\n{body}\n"
    (dest_dir / "SKILL.md").write_text(text)
    return text


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corrupt_manifest(dest_dir):
    """Leave the manifest present but unparseable, as a half-written file would."""
    (dest_dir / MANIFEST_NAME).write_text("{ truncated")


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

    # User edits the installed copy.
    local_edit = "# LOCAL EDIT BY USER\n"
    skill_md = dest / "SKILL.md"
    skill_md.write_text(skill_md.read_text() + local_edit)

    # Packaged skill evolves: newer version + new content.
    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body\nnew a2a-watch / a2a-bridge lines")

    rc = _run_install_skill(src_v2, dest, force=False)
    out = capsys.readouterr()
    assert rc == 1
    assert "local edits" in out.err
    assert local_edit in (dest / "SKILL.md").read_text()
    # Manifest must remain the original one.
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.0.0"


def test_force_overwrites_local_edits(tmp_path, capsys):
    """--force clobbers local edits and records the new manifest."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    local_edit = "# LOCAL EDIT BY USER\n"
    skill_md = dest / "SKILL.md"
    skill_md.write_text(skill_md.read_text() + local_edit)

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "new body")

    rc = _run_install_skill(src_v2, dest, force=True)
    out = capsys.readouterr()
    assert rc == 0
    assert "overwriting local edits" in out.out
    assert local_edit not in (dest / "SKILL.md").read_text()
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


def test_downgrade_refused_without_force(tmp_path, capsys):
    """An older packaged copy is refused unless --force is set."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.1.0", "newer body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    src_v0 = tmp_path / "pkg-v0"
    _write_skill(src_v0, "1.0.0", "older body")

    rc = _run_install_skill(src_v0, dest, force=False)
    out = capsys.readouterr()
    assert rc == 1
    assert "older" in out.err.lower()
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


def test_force_allows_downgrade(tmp_path, capsys):
    """--force installs an older copy and records the downgrade."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.1.0", "newer body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    src_v0 = tmp_path / "pkg-v0"
    _write_skill(src_v0, "1.0.0", "older body")

    rc = _run_install_skill(src_v0, dest, force=True)
    out = capsys.readouterr()
    assert rc == 0
    assert "downgraded" in out.out.lower()
    assert "1.0.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.0.0"


def test_corrupt_manifest_does_not_raise_without_force(tmp_path, capsys):
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    _run_install_skill(src, dest, force=False)
    _corrupt_manifest(dest)
    capsys.readouterr()

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "up to date" in out.out


def test_corrupt_manifest_does_not_raise_with_force(tmp_path, capsys):
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)
    _corrupt_manifest(dest)

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "new body")
    capsys.readouterr()

    rc = _run_install_skill(src_v2, dest, force=True)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["skill_md_sha256"] == _sha(dest / "SKILL.md")


def test_corrupt_manifest_falls_back_to_the_content_comparison(tmp_path, capsys):
    """Degrading means the no-manifest path, not 'assume the copy is clean'."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    edited = (dest / "SKILL.md").read_text() + "\n# local edit by user\n"
    (dest / "SKILL.md").write_text(edited)
    _corrupt_manifest(dest)

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body")
    capsys.readouterr()

    rc = _run_install_skill(src_v2, dest, force=False)
    out = capsys.readouterr()
    assert rc != 0
    assert "local edits" in out.err
    assert (dest / "SKILL.md").read_text() == edited


def test_manifest_that_is_not_an_object_degrades(tmp_path, capsys):
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    _run_install_skill(src, dest, force=False)
    (dest / MANIFEST_NAME).write_text("[]\n")  # valid JSON, wrong shape
    capsys.readouterr()

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "up to date" in out.out


def test_unreadable_manifest_directory_degrades(tmp_path, capsys):
    """A directory at the manifest path degrades to the no-manifest path."""
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    _run_install_skill(src, dest, force=False)
    manifest = dest / MANIFEST_NAME
    manifest.unlink()
    manifest.mkdir()  # a directory where a file belongs
    capsys.readouterr()

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "up to date" in out.out


def test_parse_skill_version_empty_version_line_does_not_consume_next_line(tmp_path):
    """An empty 'version:' line must not consume the next line as the version."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: taosmd-a2a\n"
        "version:\n"
        "description: some description\n"
        "---\n"
        "Body text.\n"
    )
    assert _parse_skill_version(skill_md) is None


# --- Manifest write path must not raise on both arms ------------------------


def test_empty_manifest_directory_positive_control(tmp_path):
    """Positive control: empty manifest directory is removed and replaced by a file."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    manifest_dir = dest / MANIFEST_NAME
    manifest_dir.unlink()
    manifest_dir.mkdir()  # empty directory

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body")

    rc = _run_install_skill(src_v2, dest, force=False)
    assert rc == 0
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    assert (dest / MANIFEST_NAME).is_file()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


def test_non_empty_manifest_directory_non_force_arm(tmp_path):
    """Non-force arm with a genuinely non-empty manifest directory."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    manifest_dir = dest / MANIFEST_NAME
    manifest_dir.unlink()
    manifest_dir.mkdir()
    (manifest_dir / "occupant.txt").write_text("obstruction")

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body")

    rc = _run_install_skill(src_v2, dest, force=False)
    assert rc == 0
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    assert (dest / MANIFEST_NAME).is_file()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


def test_non_empty_manifest_directory_force_arm(tmp_path):
    """Force arm with a genuinely non-empty manifest directory."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    manifest_dir = dest / MANIFEST_NAME
    manifest_dir.unlink()
    manifest_dir.mkdir()
    (manifest_dir / "occupant.txt").write_text("obstruction")

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "new body")

    rc = _run_install_skill(src_v2, dest, force=True)
    assert rc == 0
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    assert (dest / MANIFEST_NAME).is_file()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


def test_failed_manifest_write_does_not_advance_skill_md(tmp_path):
    """If the manifest cannot be written, SKILL.md must be left unchanged."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)
    original_skill_md = (dest / "SKILL.md").read_bytes()

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "new body")

    # Simulate a manifest write failure by patching _write_skill_manifest.
    def boom(*args, **kwargs):
        raise OSError("simulated manifest write failure")

    with patch("taosmd.cli._write_skill_manifest", side_effect=boom):
        rc = _run_install_skill(src_v2, dest, force=False)
    assert rc == 1
    # SKILL.md must be unchanged.
    assert (dest / "SKILL.md").read_bytes() == original_skill_md
    # Manifest must still be the original one.
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.0.0"


def test_read_only_manifest_non_force_arm(tmp_path):
    """Non-force arm with a stale clean copy and read-only manifest."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    manifest = dest / MANIFEST_NAME
    manifest.chmod(0o444)

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body")

    rc = _run_install_skill(src_v2, dest, force=False)
    assert rc == 0
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


def test_read_only_manifest_force_arm(tmp_path):
    """Force arm with a stale copy and read-only manifest."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    manifest = dest / MANIFEST_NAME
    manifest.chmod(0o444)

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "new body")

    rc = _run_install_skill(src_v2, dest, force=True)
    assert rc == 0
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


# --- D1: install-client.sh must not auto-force on refusal -------------------


class TestInstallClientScript:
    """End-to-end probe for scripts/install-client.sh refusal handling."""

    def test_refusal_does_not_auto_force(self, tmp_path, monkeypatch):
        """When taosmd install-skill refuses, install-client.sh must not force."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        fake_taosmd = bin_dir / "taosmd"
        fake_taosmd.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "install-skill" ]; then\n'
            "    if [ \"$2\" = \"--force\" ]; then\n"
            "        DEST=\"$HOME/.claude/skills/taosmd-a2a\"\n"
            "        mkdir -p \"$DEST\"\n"
            "        echo \"packaged skill\" > \"$DEST/SKILL.md\"\n"
            "        echo '{\"skill\":\"taosmd-a2a\",\"version\":\"1.0.0\"}' > \"$DEST/.taosmd-skill-manifest.json\"\n"
            "    else\n"
            "        echo \"error: local edits\" >&2\n"
            "        echo \"  taosmd install-skill --force\" >&2\n"
            "        exit 1\n"
            "    fi\n"
            "elif [ \"$1\" = \"config\" ]; then\n"
            "    exit 0\n"
            "elif [ \"$1\" = \"--version\" ]; then\n"
            "    echo \"1.0.0\"\n"
            "else\n"
            "    exit 0\n"
            "fi\n"
        )
        fake_taosmd.chmod(0o755)

        fake_pip = bin_dir / "pip"
        fake_pip.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_pip.chmod(0o755)

        fake_curl = bin_dir / "curl"
        fake_curl.write_text('#!/usr/bin/env bash\necho \'{"status": "ok"}\'')
        fake_curl.chmod(0o755)

        skill_dir = fake_home / ".claude" / "skills" / "taosmd-a2a"
        skill_dir.mkdir(parents=True)
        local_edit_marker = "# LOCAL EDIT BY USER\n"
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: taosmd-a2a\nversion: 1.0.0\n---\n{local_edit_marker}"
        )

        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv(
            "PATH", str(bin_dir) + ":" + os.environ.get("PATH", "")
        )

        script_path = Path(__file__).parent.parent / "scripts" / "install-client.sh"
        subprocess.run(
            ["bash", str(script_path), "http://localhost:7900"],
            capture_output=True,
            text=True,
        )

        skill_md = (skill_dir / "SKILL.md").read_text()
        assert local_edit_marker in skill_md, (
            "Local edit was clobbered by auto-forced reinstall"
        )
