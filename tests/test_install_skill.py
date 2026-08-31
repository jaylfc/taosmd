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
* an unreadable manifest degrades to the no-manifest path instead of raising.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

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


def _corrupt_manifest(dest_dir):
    """Leave the manifest present but unparseable, as a half-written file would."""
    (dest_dir / MANIFEST_NAME).write_text("{ truncated")


# --- an unreadable manifest must degrade, not raise -----------------------
#
# The manifest is a file this tool writes itself, so a truncated or
# hand-edited copy is an ordinary state to find on disk. If reading it raises,
# install-skill dies before it ever reaches the ``force`` branch, which makes
# the --force escape hatch its own error message advertises unreachable. Both
# arms are tested separately because a guard can be live on one and blind on
# the other.


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
    # The run also repairs the manifest it could not read.
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["skill_md_sha256"] == _sha(dest / "SKILL.md")


def test_corrupt_manifest_falls_back_to_the_content_comparison(tmp_path, capsys):
    """Degrading means the no-manifest path, not "assume the copy is clean"."""
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


def test_unreadable_manifest_degrades(tmp_path, capsys):
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    _run_install_skill(src, dest, force=False)
    manifest = dest / MANIFEST_NAME
    manifest.unlink()
    manifest.mkdir()  # a directory where a file belongs: read_text raises OSError
    capsys.readouterr()

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "up to date" in out.out


# --- version ordering -----------------------------------------------------


def test_version_tuple_parses_only_plain_numeric_versions():
    assert _version_tuple("1.10.0") == (1, 10, 0)
    assert _version_tuple("2") == (2,)
    assert _version_tuple("0.9.0") < _version_tuple("0.10.0")
    assert _version_tuple("1.0.0-rc1") is None
    assert _version_tuple("") is None
    assert _version_tuple(None) is None
    assert _version_tuple("1.\u00b2") is None  # a non-ascii digit int() would reject


def test_older_package_is_refused_without_force(tmp_path, capsys):
    src_new = tmp_path / "pkg-1.1.0"
    _write_skill(src_new, "1.1.0", "newer body")
    dest = tmp_path / "dest"
    _run_install_skill(src_new, dest, force=False)
    installed = (dest / "SKILL.md").read_text()

    src_old = tmp_path / "pkg-1.0.0"
    _write_skill(src_old, "1.0.0", "older body")
    capsys.readouterr()

    rc = _run_install_skill(src_old, dest, force=False)
    out = capsys.readouterr()
    assert rc != 0
    assert "older than the installed copy" in out.err
    assert "--force" in out.err
    assert "upgrad" not in (out.out + out.err).lower()
    assert (dest / "SKILL.md").read_text() == installed


def test_forced_downgrade_is_never_called_an_upgrade(tmp_path, capsys):
    src_new = tmp_path / "pkg-1.1.0"
    _write_skill(src_new, "1.1.0", "newer body")
    dest = tmp_path / "dest"
    _run_install_skill(src_new, dest, force=False)

    src_old = tmp_path / "pkg-1.0.0"
    _write_skill(src_old, "1.0.0", "older body")
    capsys.readouterr()

    rc = _run_install_skill(src_old, dest, force=True)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "downgraded" in out.out
    assert "upgrad" not in out.out.lower()
    assert "1.0.0" in (dest / "SKILL.md").read_text()


def test_multi_digit_versions_upgrade_in_numeric_order(tmp_path, capsys):
    """0.9.0 -> 0.10.0 is an upgrade. A string compare would call it a downgrade."""
    src_9 = tmp_path / "pkg-0.9.0"
    _write_skill(src_9, "0.9.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_9, dest, force=False)

    src_10 = tmp_path / "pkg-0.10.0"
    _write_skill(src_10, "0.10.0", "new body")
    capsys.readouterr()

    rc = _run_install_skill(src_10, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "upgraded" in out.out
    assert "0.10.0" in (dest / "SKILL.md").read_text()


def test_multi_digit_downgrade_is_refused(tmp_path, capsys):
    """0.10.0 -> 0.9.0 is a downgrade. A string compare would call it an upgrade."""
    src_10 = tmp_path / "pkg-0.10.0"
    _write_skill(src_10, "0.10.0", "new body")
    dest = tmp_path / "dest"
    _run_install_skill(src_10, dest, force=False)
    installed = (dest / "SKILL.md").read_text()

    src_9 = tmp_path / "pkg-0.9.0"
    _write_skill(src_9, "0.9.0", "old body")
    capsys.readouterr()

    rc = _run_install_skill(src_9, dest, force=False)
    out = capsys.readouterr()
    assert rc != 0
    assert "older than the installed copy" in out.err
    assert (dest / "SKILL.md").read_text() == installed


def test_unorderable_versions_are_reported_as_replaced(tmp_path, capsys):
    """With no ordering available the packaged copy still wins, but honestly."""
    src_rc = tmp_path / "pkg-rc"
    _write_skill(src_rc, "1.0.0-rc1", "rc body")
    dest = tmp_path / "dest"
    _run_install_skill(src_rc, dest, force=False)

    src_final = tmp_path / "pkg-final"
    _write_skill(src_final, "1.0.0", "final body")
    capsys.readouterr()

    rc = _run_install_skill(src_final, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "replaced" in out.out
    assert "upgrad" not in out.out.lower()
    assert "final body" in (dest / "SKILL.md").read_text()


# --- the migration path every installed copy in the field takes -----------


def test_pre_versioning_copy_differing_only_by_the_version_line_migrates(
    tmp_path, capsys
):
    """The one upgrade every existing user gets: no manifest, no version line.

    The packaged SKILL.md differs from the installed one by exactly the added
    ``version:`` line, so the whole migration rests on the content comparison
    ignoring that line. Without that filter this run refuses as "local edits".
    """
    common = (
        "---\n"
        "name: taosmd-a2a\n"
        "description: Set up agent-to-agent comms via the taOSmd A2A bus.\n"
        "user-invocable: true\n"
        "---\n"
        "\n"
        "# taosmd-a2a\n"
        "\n"
        "Body text that did not change in this release.\n"
    )
    packaged_text = common.replace(
        "name: taosmd-a2a\n", "name: taosmd-a2a\nversion: 1.0.0\n"
    )

    src = tmp_path / "pkg"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text(packaged_text)

    dest = tmp_path / "dest"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text(common)
    assert not (dest / MANIFEST_NAME).exists()

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert out.err == ""
    assert (dest / "SKILL.md").read_text() == packaged_text
    assert "unknown (pre-versioning)" in out.out
    # No ordering is possible against a version-less copy, so it is a replace.
    assert "replaced" in out.out
    assert "upgrad" not in out.out.lower()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.0.0"
    assert data["skill_md_sha256"] == _sha(dest / "SKILL.md")


def test_same_version_with_local_edits_is_refused(tmp_path, capsys):
    """Equal versions are not enough to call a copy up to date."""
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    _run_install_skill(src, dest, force=False)

    edited = (dest / "SKILL.md").read_text() + "\n# local edit by user\n"
    (dest / "SKILL.md").write_text(edited)
    capsys.readouterr()

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc != 0
    assert "local edits" in out.err
    assert "up to date" not in out.out
    assert (dest / "SKILL.md").read_text() == edited


def test_manifest_without_a_hash_is_treated_as_clean(tmp_path, capsys):
    """A manifest that records no hash cannot accuse the user of an edit."""
    src = tmp_path / "pkg"
    _write_skill(src, "1.0.0", "packaged body")
    dest = tmp_path / "dest"
    _run_install_skill(src, dest, force=False)
    (dest / MANIFEST_NAME).write_text(
        json.dumps({"skill": "taosmd-a2a", "version": "1.0.0"}) + "\n"
    )
    capsys.readouterr()

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "up to date" in out.out
    assert out.err == ""


def test_packaged_skill_without_a_version_records_the_zero_fallback(
    tmp_path, capsys
):
    src = tmp_path / "pkg"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: taosmd-a2a\n---\npackaged body\n")
    dest = tmp_path / "dest"

    rc = _run_install_skill(src, dest, force=False)
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "v0.0.0" in out.out
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "0.0.0"


# --- frontmatter boundaries ----------------------------------------------


def test_parse_skill_version_ignores_a_file_without_frontmatter(tmp_path):
    """A ``version:`` line in prose is not a frontmatter version."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "# taosmd-a2a\n"
        "\n"
        "version: 9.9.9 appears here in prose, not in frontmatter\n"
        "\n"
        "---\n"
        "\n"
        "More body text.\n"
    )
    assert _parse_skill_version(skill_md) is None


def test_parse_skill_version_ignores_an_unclosed_frontmatter_block(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\nname: taosmd-a2a\nversion: 9.9.9\n")
    assert _parse_skill_version(skill_md) is None


def test_parse_skill_version_ignores_a_version_line_below_the_frontmatter(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: taosmd-a2a\n---\n\nversion: 9.9.9\n\nBody text.\n"
    )
    assert _parse_skill_version(skill_md) is None


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


# --- D2: manifest write path must not raise on both arms --------------------


def test_directory_in_place_manifest_non_force_arm(tmp_path):
    """Non-force arm with a stale clean copy and directory-in-place manifest."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    manifest_dir = dest / MANIFEST_NAME
    manifest_dir.unlink()
    manifest_dir.mkdir()

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "old body")

    rc = _run_install_skill(src_v2, dest, force=False)
    assert rc == 0
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


def test_directory_in_place_manifest_force_arm(tmp_path):
    """Force arm with a stale copy and directory-in-place manifest."""
    src_v1 = tmp_path / "pkg-v1"
    _write_skill(src_v1, "1.0.0", "old body")
    dest = tmp_path / "dest"
    _run_install_skill(src_v1, dest, force=False)

    manifest_dir = dest / MANIFEST_NAME
    manifest_dir.unlink()
    manifest_dir.mkdir()

    src_v2 = tmp_path / "pkg-v2"
    _write_skill(src_v2, "1.1.0", "new body")

    rc = _run_install_skill(src_v2, dest, force=True)
    assert rc == 0
    assert "1.1.0" in (dest / "SKILL.md").read_text()
    data = json.loads((dest / MANIFEST_NAME).read_text())
    assert data["version"] == "1.1.0"


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
