import json
import os
from pathlib import Path

import pytest

from taosmd import config
from taosmd.cli import main
from taosmd import data_relocate as dr


@pytest.fixture
def populated_data_dir(tmp_path, monkeypatch):
    d = tmp_path / "old"
    d.mkdir()
    monkeypatch.setenv("TAOSMD_DATA_DIR", str(d))
    monkeypatch.delenv("TAOSMD_ONNX_PATH", raising=False)
    from taosmd import api as taosmd_api
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})
    import asyncio
    asyncio.run(taosmd_api._ensure_stores(str(d)))
    return str(d)


def test_status_low_space(populated_data_dir, monkeypatch):
    # patch free space to be tiny
    monkeypatch.setattr(dr, "_free_space", lambda p: 100)
    monkeypatch.setattr(dr, "_total_space", lambda p: 1000)
    res = dr.status(data_dir=populated_data_dir)
    assert res["low_space"] is True


def test_relocate_happy_path(populated_data_dir, tmp_path):
    new = tmp_path / "new"
    old = populated_data_dir
    # compute sha before
    res = dr.relocate(str(new), data_dir=old)
    assert res["exit_code"] == 0
    assert res["status"] == "moved"
    assert Path(old, dr._MARKER_NAME).exists()
    # config points new
    assert config._resolve_data_dir() == str(new.resolve()) or config._resolve_data_dir() == str(new)


def test_target_too_small(populated_data_dir, tmp_path, monkeypatch):
    new = tmp_path / "new"
    old = populated_data_dir
    monkeypatch.setattr(dr, "_free_space", lambda p: 0)
    res = dr.relocate(str(new), data_dir=old)
    assert res["exit_code"] == 2
    assert not Path(new).exists()


def test_corrupted_copy_exit_4(populated_data_dir, tmp_path, monkeypatch):
    new = tmp_path / "new"
    old = populated_data_dir
    import hashlib
    def snap(d):
        out = {}
        for root, _dirs, files in os.walk(d):
            for fn in files:
                p = Path(root) / fn
                out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out
    before = snap(old)
    monkeypatch.setattr(dr, "_verify", lambda *a, **k: (False, "corrupted"))
    res = dr.relocate(str(new), data_dir=old)
    assert res["exit_code"] == 4
    assert not Path(new).exists()
    # old still configured
    assert config._resolve_data_dir() == old or os.environ.get("TAOSMD_DATA_DIR") == old
    # old dir byte-identical
    assert snap(old) == before


def test_old_survives_until_confirm(populated_data_dir, tmp_path):
    new = tmp_path / "new"
    old = populated_data_dir
    res = dr.relocate(str(new), data_dir=old)
    assert res["exit_code"] == 0
    assert Path(old).exists()
    # confirm removes
    os.environ["TAOSMD_DATA_DIR"] = str(new)
    # ensure resolve follows
    res2 = dr.relocate(confirm_remove_old=True)
    assert res2["exit_code"] == 0
    assert not Path(old).exists()


def test_cli_status_json(populated_data_dir, capsys):
    rc = main(["data", "status", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    d = json.loads(out)
    assert "low_space" in d


def test_cli_relocate_exit_codes(populated_data_dir, tmp_path, monkeypatch, capsys):
    new = tmp_path / "new"
    orig_free = dr._free_space
    monkeypatch.setattr(dr, "_free_space", lambda p: 0)
    rc = main(["data", "relocate", str(new)])
    assert rc == 2
    monkeypatch.setattr(dr, "_free_space", orig_free)
    # happy
    rc = main(["data", "relocate", str(new)])
    assert rc == 0


def test_sigkill_mid_copy_rerun(populated_data_dir, tmp_path):
    import subprocess
    import sys
    import os
    new = tmp_path / "new"
    old = populated_data_dir
    script = tmp_path / "killer.py"
    script.write_text(
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, " + repr(str(Path.cwd())) + ")\n"
        "os.environ['TAOSMD_DATA_DIR'] = " + repr(old) + "\n"
        "from taosmd import data_relocate as dr\n"
        "def killer(old_dir, new_dir):\n"
        "    Path(new_dir).mkdir(parents=True, exist_ok=True)\n"
        "    (Path(new_dir) / 'partial.bin').write_bytes(b'x' * 100)\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "dr._copy_tree = killer\n"
        "dr.relocate(" + repr(str(new)) + ", data_dir=" + repr(old) + ")\n"
    )
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert proc.returncode != 0
    # re-run should succeed
    os.environ["TAOSMD_DATA_DIR"] = old
    from taosmd import data_relocate as dr2
    res = dr2.relocate(str(new), data_dir=old)
    assert res["exit_code"] == 0
    assert Path(new).exists()


def test_mutation_skip_verify_allows_corrupt(populated_data_dir, tmp_path, monkeypatch):
    """Mutation: skipping the verify step lets a corrupted copy slip through."""
    new = tmp_path / "new"
    old = populated_data_dir

    # Use real copy then corrupt a database file in the target.
    orig_copy = dr._copy_tree
    def patched_copy(old_dir, new_dir):
        orig_copy(old_dir, new_dir)
        for p in Path(new_dir).rglob("*.db"):
            with p.open("ab") as f:
                f.write(b"corrupt")
            break
    monkeypatch.setattr(dr, "_copy_tree", patched_copy)
    # skip verify entirely
    monkeypatch.setattr(dr, "_verify", lambda *a, **k: (True, None))
    res = dr.relocate(str(new), data_dir=old)
    # With verify skipped, the corrupted copy is (incorrectly) accepted.
    assert res["exit_code"] == 0
