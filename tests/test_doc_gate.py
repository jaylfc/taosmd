"""Tests for scripts/check_doc_gate.py.

Covers the two defects this port closes:
  HOLE 1 -- a modification to a protected doc that deletes a required section
            must fail (the old gate saw only A/D, and no rule listed a doc
            path, so a doc-only change passed trivially).
  HOLE 2 -- a rule satisfied by a doc touch is content-asserted, so a
            one-character edit cannot mask a gutted section.

Plus the tsk-t2lsre fix-forward set (5 items from card tsk-t2lsre):
  1. rename/C bypass -- R/C expands to D + A so structural rules fire
  2. changelog.d convention -- changelog.d/*.md added to require_doc
  3. taosmd/ Layer A scope -- taosmd/ tokens checked, data-dir paths excluded
  4. conflict-marker invariant -- changed files grepped for unresolved markers

The red proofs in this file mirror the evidence blocks required by the merge
gate (DOC-GATE FAIL: ... / exit 1).
"""
from __future__ import annotations

import subprocess

import scripts.check_doc_gate as dg
from scripts.check_doc_gate import (
    _glob_match,
    _is_test_path,
    _match_any,
    _parse_name_status,
    _missing_headings,
    check_required_headings,
    evaluate_rules,
    extract_path_tokens,
    main,
)

A2A_DOC = """# A2A Comms

## What this does

taOSmd bus lets agents share a channel.

### HTTP endpoints

POST /api/send
"""

A2A_DOC_GUTTED = """# A2A Comms

## What this does

taOSmd bus lets agents share a channel.

No endpoints section anymore.
"""

PR_DOC = """# PR verification on taOSmd

## The split

Two stages.

## The zero-token layer

Three checks.
"""

# Mirrors docs/doc-gate.toml but kept minimal and self-contained.
CFG = """
[gate]
trailer = "Docs-Reviewed:"

[invariants]
referenced_paths_scan = []

[invariants.required_headings]
"taosmd/docs/a2a-comms.md" = ["# A2A Comms", "## What this does", "### HTTP endpoints"]
"docs/pr-verification.md" = ["# PR verification on taOSmd", "## The split", "## The zero-token layer"]

[[rules]]
name = "changelog"
when_changed = ["taosmd/**"]
require_doc = ["CHANGELOG.md", "changelog.d/*.md"]
hint = "changelog required"

[[rules]]
name = "a2a-handlers"
when_changed = ["taosmd/http_server.py", "taosmd/service.py", "taosmd/docs/a2a-comms.md"]
on_modify = true
require_doc = ["taosmd/docs/a2a-comms.md"]
hint = "a2a handlers need docs"

[[rules]]
name = "contributor-surface"
when_changed = [".github/workflows/**", "pyproject.toml", "docs/pr-verification.md"]
on_modify = true
require_doc = ["docs/pr-verification.md"]
hint = "contributor surface needs doc"
"""


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )


def _init_repo(tmp_path):
    """A temp git repo with the protected docs, a tiny taosmd package, and the
    doc-gate config. Committed once as the base so diff-gate --staged / --base
    both have something to compare against."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "taosmd").mkdir()
    (repo / "taosmd" / "docs").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "taosmd" / "http_server.py").write_text("def handler():\n    pass\n")
    (repo / "taosmd" / "service.py").write_text("def svc():\n    pass\n")
    (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC)
    (repo / "docs" / "pr-verification.md").write_text(PR_DOC)
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n")
    (repo / "docs" / "doc-gate.toml").write_text(CFG)
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    return repo


def _load_cfg(repo):
    import tomllib
    with open(repo / "docs" / "doc-gate.toml", "rb") as f:
        return tomllib.load(f)


# ----------------------------------------------------------------------
# pure-function units
# ----------------------------------------------------------------------

class TestGlobMatch:
    def test_star_stays_in_segment(self):
        assert _glob_match("taosmd/foo.py", "taosmd/*.py")
        assert not _glob_match("taosmd/sub/foo.py", "taosmd/*.py")

    def test_double_star_matches_nested_and_parent(self):
        assert _glob_match("taosmd/docs/x.md", "taosmd/**")
        assert _glob_match("taosmd", "taosmd/**")
        assert not _glob_match("other/taosmd/x", "taosmd/**")

    def test_question_mark_single_char(self):
        assert _glob_match("a/x.py", "a/?*.py")
        assert not _glob_match("a/xy.py", "a/?")


class TestMatchAny:
    def test_matches_one_of_many(self):
        assert _match_any("taosmd/service.py", ["taosmd/http_server.py", "taosmd/service.py"])

    def test_no_match(self):
        assert not _match_any("CHANGELOG.md", ["taosmd/docs/a2a-comms.md"])


class TestExtractPathTokens:
    def test_strips_trailing_punct(self):
        tok = extract_path_tokens("see docs/foo.md, thanks")
        assert tok == ["docs/foo.md"]

    def test_ignores_globs_and_placeholders(self):
        tok = extract_path_tokens("see docs/*.md and <scripts/run.sh>")
        assert tok == []

    def test_ignores_embedded_prefix(self):
        # "tinyagentos/" inside a deploy path is not a repo path to assert.
        tok = extract_path_tokens("copy to /home/u/tinyagentos/data/x")
        assert tok == []

    def test_includes_taosmd_module(self):
        tok = extract_path_tokens("see taosmd/http_server.py for details")
        assert tok == ["taosmd/http_server.py"]

    def test_excludes_data_dir_paths(self):
        text = (
            "runtime paths: ~/.taosmd/config.json, taosmd/archive, "
            "taosmd/archive-index.db, taosmd/knowledge-graph.db, "
            "taosmd/vector-memory.db, taosmd/project.toml"
        )
        tok = extract_path_tokens(text)
        assert "taosmd/config.json" not in tok
        assert "taosmd/archive" not in tok
        assert "taosmd/archive-index.db" not in tok
        assert "taosmd/knowledge-graph.db" not in tok
        assert "taosmd/vector-memory.db" not in tok
        assert "taosmd/project.toml" not in tok

    def test_excludes_generated_build_info(self):
        tok = extract_path_tokens("see taosmd/_build_info.py")
        assert "taosmd/_build_info.py" not in tok

    def test_taosmd_in_other_prefix_not_matched(self):
        tok = extract_path_tokens("see foo/taosmd/x.py")
        assert tok == []


class TestIsTestPath:
    def test_pytest_module(self):
        assert _is_test_path("tests/test_foo.py")
        assert _is_test_path("taosmd/tests/test_bar.py")

    def test_frontend_spec(self):
        assert _is_test_path("dashboard/src/App.test.tsx")

    def test_real_code(self):
        assert not _is_test_path("taosmd/http_server.py")


class TestParseNameStatus:
    def test_add_unchanged(self):
        assert _parse_name_status("A\tnewfile.py") == [("A", "newfile.py")]

    def test_delete_unchanged(self):
        assert _parse_name_status("D\tdeleted.py") == [("D", "deleted.py")]

    def test_modify_unchanged(self):
        assert _parse_name_status("M\tmodified.py") == [("M", "modified.py")]

    def test_rename_expands_to_d_and_a(self):
        result = _parse_name_status("R100\ttaosmd/http_server.py\ttaosmd/http_server_renamed.py")
        assert ("D", "taosmd/http_server.py") in result
        assert ("A", "taosmd/http_server_renamed.py") in result
        assert len(result) == 2

    def test_copy_expands_to_d_and_a(self):
        result = _parse_name_status("C100\ttaosmd/http_server.py\ttaosmd/http_server_copy.py")
        assert ("D", "taosmd/http_server.py") in result
        assert ("A", "taosmd/http_server_copy.py") in result

    def test_regular_statuses_unchanged(self):
        result = _parse_name_status("A\tnewfile.py\nM\tmodified.py\nD\tdeleted.py")
        assert result == [("A", "newfile.py"), ("M", "modified.py"), ("D", "deleted.py")]


class TestMissingHeadings:
    def test_all_present(self, tmp_path):
        doc = tmp_path / "d.md"
        doc.write_text("# Title\n\n## Body\n\n### HTTP endpoints\n")
        assert _missing_headings(tmp_path, "d.md", ["# Title", "## Body"]) == []

    def test_one_deleted(self, tmp_path):
        doc = tmp_path / "d.md"
        doc.write_text("# Title\n\n## Body\n\n")
        assert _missing_headings(tmp_path, "d.md", ["# Title", "### HTTP endpoints"]) == [
            "### HTTP endpoints"
        ]

    def test_missing_file_is_all_required(self, tmp_path):
        assert _missing_headings(tmp_path, "d.md", ["# Title"]) == ["# Title"]

    def test_no_required_returns_empty(self, tmp_path):
        assert _missing_headings(tmp_path, "d.md", []) == []


# ----------------------------------------------------------------------
# evaluate_rules units (constructed changesets, no git)
# ----------------------------------------------------------------------

class TestEvaluateRules:
    def test_default_rule_ignores_modification(self, tmp_path):
        # changelog has no on_modify: a plain M under taosmd/ does NOT fire it.
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        fails = evaluate_rules([("M", "taosmd/retrieval.py")], [], cfg, repo)
        assert fails == []

    def test_changelog_fires_on_add(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        fails = evaluate_rules([("A", "taosmd/retrieval.py")], [], cfg, repo)
        assert len(fails) == 1
        assert "changelog" in fails[0]

    def test_a2a_handlers_fires_on_modify_no_doc(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        fails = evaluate_rules([("M", "taosmd/http_server.py")], [], cfg, repo)
        assert len(fails) == 1
        assert "a2a-handlers" in fails[0]
        assert "required section" not in fails[0]

    def test_a2a_handlers_satisfied_by_intact_doc(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        changed = [("M", "taosmd/http_server.py"), ("M", "taosmd/docs/a2a-comms.md")]
        assert evaluate_rules(changed, [], cfg, repo) == []

    def test_gutted_doc_fails_even_when_touched(self, tmp_path):
        # HOLE 2 red: doc is touched (path-satisfied) but a required section
        # was deleted -> content assertion must still fail.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC_GUTTED)
        cfg = _load_cfg(repo)
        changed = [("M", "taosmd/http_server.py"), ("M", "taosmd/docs/a2a-comms.md")]
        fails = evaluate_rules(changed, [], cfg, repo)
        assert any("a2a-handlers" in f and "required section" in f for f in fails)

    def test_one_char_doc_touch_does_not_mask_gutting(self, tmp_path):
        # A token edit to the gutted doc must not satisfy the rule.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC_GUTTED + "\n")
        cfg = _load_cfg(repo)
        changed = [("M", "taosmd/http_server.py"), ("M", "taosmd/docs/a2a-comms.md")]
        fails = evaluate_rules(changed, [], cfg, repo)
        assert any("required section" in f for f in fails)

    def test_trailer_waives_doc_change(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        changed = [("M", "taosmd/http_server.py")]
        assert evaluate_rules(changed, ["Docs-Reviewed: no doc needed"], cfg, repo) == []

    def test_trailer_does_not_mask_a_gutted_doc_in_layer_a(self, tmp_path):
        # The trailer bypass is preserved (Layer B), but Layer A still catches
        # content loss regardless of any waiver.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC_GUTTED)
        cfg = _load_cfg(repo)
        changed = [("M", "taosmd/http_server.py"), ("M", "taosmd/docs/a2a-comms.md")]
        assert evaluate_rules(changed, ["Docs-Reviewed: waive"], cfg, repo) == []
        assert check_required_headings(repo, cfg)

    def test_doc_only_gutting_fires_a2a_rule(self, tmp_path):
        # HOLE 1 red: ONLY the protected doc changes (a section deleted),
        # nothing else. A rule with on_modify + doc in when_changed must fire
        # and the content check must fail.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC_GUTTED)
        cfg = _load_cfg(repo)
        changed = [("M", "taosmd/docs/a2a-comms.md")]
        fails = evaluate_rules(changed, [], cfg, repo)
        assert any("a2a-handlers" in f and "required section" in f for f in fails)

    def test_add_delete_triggers_without_on_modify(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        # An A of a non-test file under taosmd/** fires changelog (no on_modify).
        fails = evaluate_rules([("A", "taosmd/newmod.py")], [], cfg, repo)
        assert any("changelog" in f for f in fails)

    def test_test_files_never_satisfy_or_trigger(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        # Adding a test file under a triggering glob must not fire a rule.
        assert evaluate_rules([("A", "taosmd/test_newmod.py")], [], cfg, repo) == []
        assert evaluate_rules([("A", "taosmd/newmod.py")], [], cfg, repo) != []

    def test_rename_out_of_guarded_surface_fires_rules(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        # Rename out of taosmd/ is emitted as D (old) + A (new) by _parse_name_status.
        fails = evaluate_rules(
            [("D", "taosmd/http_server.py"), ("A", "taosmd/http_server_renamed.py")],
            [],
            cfg,
            repo,
        )
        assert any("changelog" in f for f in fails)
        assert any("a2a-handlers" in f for f in fails)

    def test_benign_rename_outside_guarded_surface_is_clean(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        fails = evaluate_rules([("D", "README.md"), ("A", "README_NEW.md")], [], cfg, repo)
        assert fails == []

    def test_changelog_d_fragment_satisfies_changelog_rule(self, tmp_path):
        repo = _init_repo(tmp_path)
        cfg = _load_cfg(repo)
        changed = [("A", "taosmd/retrieval.py"), ("M", "changelog.d/tsk-xxx-add-thing.md")]
        assert evaluate_rules(changed, [], cfg, repo) == []


# ----------------------------------------------------------------------
# end-to-end via main(): invariants (Layer A) and diff-gate --staged (Layer B)
# ----------------------------------------------------------------------

class TestMainIntegration:
    def test_invariants_clean_on_intact_docs(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path)
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", str(repo / "docs" / "doc-gate.toml"), "invariants"])
        assert rc == 0
        assert "doc-gate: clean" in capsys.readouterr().out

    def test_invariants_fails_when_section_deleted(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC_GUTTED)
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "invariants"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "DOC-GATE FAIL" in out
        assert "required section" in out

    def test_diff_gate_staged_red1_rule_triggered_doc_untouched(self, tmp_path, monkeypatch, capsys):
        # RED 1: handler modified, doc untouched -> a2a-handlers fails.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "http_server.py").write_text("def handler():\n    pass\n# touched\n")
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 1
        assert "a2a-handlers" in capsys.readouterr().out

    def test_diff_gate_staged_red2_doc_touched_section_deleted(self, tmp_path, monkeypatch, capsys):
        # RED 2 / HOLE 1: doc present and touched, required section deleted.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "http_server.py").write_text("def handler():\n    pass\n# touched\n")
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC_GUTTED)
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "DOC-GATE FAIL" in out
        assert "required section" in out
        assert "a2a-handlers" in out

    def test_diff_gate_staged_doc_only_gutting_fails(self, tmp_path, monkeypatch, capsys):
        # HOLE 1 red: only the doc changes, section deleted, nothing else.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC_GUTTED)
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 1
        assert "DOC-GATE FAIL" in capsys.readouterr().out

    def test_diff_gate_staged_green(self, tmp_path, monkeypatch, capsys):
        # GREEN: handler modified AND doc touched intact -> satisfied.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "http_server.py").write_text("def handler():\n    pass\n# touched\n")
        (repo / "taosmd" / "docs" / "a2a-comms.md").write_text(A2A_DOC + "\n<!-- added line -->\n")
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 0
        assert "doc-gate: clean" in capsys.readouterr().out

    def test_diff_gate_base_red1(self, tmp_path, monkeypatch, capsys):
        # Same red as --staged but via a real --base comparison (CI path).
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "http_server.py").write_text("def handler():\n    pass\n# touched\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "feat: touch handler\n\nno doc change")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--base", "HEAD~1"])
        # No docs touched and no trailer on the commit message -> fail.
        assert rc == 1
        assert "a2a-handlers" in capsys.readouterr().out

    def test_diff_gate_base_green_with_trailer(self, tmp_path, monkeypatch, capsys):
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "http_server.py").write_text("def handler():\n    pass\n# touched\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "feat: touch handler\n\nDocs-Reviewed: internal refactor, no API change")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--base", "HEAD~1"])
        assert rc == 0

    def test_diff_gate_staged_rename_out_of_guarded_surface_fails(self, tmp_path, monkeypatch, capsys):
        # PROVE IT RED: rename taosmd/http_server.py -> old path is D, new path
        # is A, so both changelog and a2a-handlers must fire.
        repo = _init_repo(tmp_path)
        _git(repo, "mv", "taosmd/http_server.py", "taosmd/http_server_renamed.py")
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "changelog" in out
        assert "a2a-handlers" in out

    def test_diff_gate_staged_benign_rename_outside_taosmd_is_clean(self, tmp_path, monkeypatch, capsys):
        # PROVE IT GREEN: a rename outside taosmd/ does not trip any rule.
        repo = _init_repo(tmp_path)
        (repo / "docs" / "extra.md").write_text("# Extra\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add extra")
        _git(repo, "mv", "docs/extra.md", "docs/extra_renamed.md")
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 0
        assert "doc-gate: clean" in capsys.readouterr().out

    def test_diff_gate_staged_conflict_markers_fail(self, tmp_path, monkeypatch, capsys):
        # PROVE IT RED: a file with conflict markers fails the gate.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "http_server.py").write_text(
            "def handler():\n    pass\n<<<<<<< HEAD\n# conflict\n=======\n# other\n>>>>>>> branch\n"
        )
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 1
        assert "conflict-marker" in capsys.readouterr().out

    def test_diff_gate_staged_no_conflict_markers_green(self, tmp_path, monkeypatch, capsys):
        # PROVE IT GREEN: clean files pass.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "retrieval.py").write_text("def retrieve():\n    pass\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add retrieval")
        (repo / "taosmd" / "retrieval.py").write_text("def retrieve():\n    pass\n# touched\n")
        _git(repo, "add", ".")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--staged"])
        assert rc == 0
        assert "doc-gate: clean" in capsys.readouterr().out

    def test_diff_gate_base_conflict_markers_fail(self, tmp_path, monkeypatch, capsys):
        # PROVE IT RED via --base path (CI flow): markers committed then
        # diffed against HEAD~1 still fire.
        repo = _init_repo(tmp_path)
        (repo / "taosmd" / "http_server.py").write_text(
            "def handler():\n    pass\n<<<<<<< HEAD\n# conflict\n=======\n# other\n>>>>>>> branch\n"
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "add conflict markers")
        cfg = str(repo / "docs" / "doc-gate.toml")
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", cfg, "diff-gate", "--base", "HEAD~1"])
        assert rc == 1
        assert "conflict-marker" in capsys.readouterr().out

    def test_invariants_taosmd_module_reference_now_fails(self, tmp_path, monkeypatch, capsys):
        # PROVE IT RED: an invented taosmd/ module reference now fails invariants.
        repo = _init_repo(tmp_path)
        (repo / "docs" / "test.md").write_text("see taosmd/NOPE-NOT-REAL.py\n")
        cfg = repo / "docs" / "doc-gate-test.toml"
        cfg.write_text(
            "[gate]\ntrailer = \"Docs-Reviewed:\"\n\n[invariants]\nreferenced_paths_scan = [\"docs/test.md\"]\n"
        )
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", str(cfg), "invariants"])
        assert rc == 1
        assert "taosmd/NOPE-NOT-REAL.py" in capsys.readouterr().out

    def test_invariants_taosmd_data_dir_paths_excluded(self, tmp_path, monkeypatch, capsys):
        # PROVE IT GREEN: data-dir paths and generated files are excluded.
        repo = _init_repo(tmp_path)
        (repo / "docs" / "test.md").write_text(
            "runtime paths: ~/.taosmd/config.json, taosmd/archive, taosmd/_build_info.py\n"
        )
        cfg = repo / "docs" / "doc-gate-test.toml"
        cfg.write_text(
            "[gate]\ntrailer = \"Docs-Reviewed:\"\n\n[invariants]\nreferenced_paths_scan = [\"docs/test.md\"]\n"
        )
        monkeypatch.setattr(dg, "REPO_ROOT", repo)
        rc = main(["--config", str(cfg), "invariants"])
        assert rc == 0
        assert "doc-gate: clean" in capsys.readouterr().out
