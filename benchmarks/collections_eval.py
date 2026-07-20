#!/usr/bin/env python3
"""Phase 1 collections eval: file-level Recall@5 over the repo's own docs/.

Pre-registered per the design spec (section 5): index the taosmd repo's
docs/ folder into a temporary collection and answer the 20-question set in
``benchmarks/data/collections_eval_questions.json`` (written before the
index was built, gold labels are file paths). A question scores 1 when any
of the top-5 results' ``file_path`` metadata matches a gold file. No LLM
judge anywhere in the loop.

Modes
-----
- ``semantic``: the real path; requires a local ONNX embedding model. This
  is the number that counts against the section-5 kill bar; run it on the
  bench host where the ONNX embedder is installed.
- ``lexical``: the fallback smoke for hosts with no local ONNX model. The
  embedder is stubbed with a deterministic hash vector purely so rows land
  in the vector store; retrieval then uses the engine's BM25-only mode
  (``mode="bm25"``), which never touches the stubbed vectors. This
  exercises the full production walk/chunk/ingest/grant/scope path with a
  lexical ranker, and validates the walker + container plumbing
  independently of the embedder (the spec's stated purpose for the Phase 1
  eval).
- ``auto`` (default): semantic when an ONNX model is present, else lexical.

Usage::

    python3 benchmarks/collections_eval.py [--mode auto|semantic|lexical]
        [--docs-dir docs] [--k 5] [--questions benchmarks/data/...json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

AGENT = "eval-agent"


async def _hash_embed(text: str, task: str = "search_document") -> list[float]:
    """Deterministic stub (same as the test suite's). BM25 mode ignores it."""
    h = hash(text) & 0xFFFFFFFF
    return [((h >> (i * 4)) & 0xFF) / 255.0 for i in range(8)]


async def run(docs_dir: Path, questions_path: Path, mode: str, k: int) -> int:
    from taosmd import api, config
    from taosmd.collections import CollectionStore, ingest_folder

    spec = json.loads(questions_path.read_text())
    questions = spec["questions"]

    with tempfile.TemporaryDirectory(prefix="taosmd-collections-eval-") as tmp:
        data_dir = str(Path(tmp) / "data")
        Path(data_dir).mkdir()
        config.set_collections_allowed_roots([str(docs_dir)], data_dir=data_dir)

        stores = await api._ensure_stores(data_dir)
        if mode == "auto":
            has_onnx = api._resolve_onnx_path(data_dir) is not None
            mode = "semantic" if has_onnx else "lexical"
            if mode == "lexical":
                print("auto: no local ONNX embedding model found; using the "
                      "lexical (BM25) fallback. Run the semantic arm on the "
                      "bench host for the number that counts.")
        if mode == "lexical":
            stores["vector"].embed = _hash_embed  # rows must land; BM25 ignores vectors
        search_opts = {"mode": "bm25"} if mode == "lexical" else {}

        store = CollectionStore(data_dir)
        col = store.create(name="repo-docs", kind="docs", source_path=str(docs_dir))
        store.grant(col["id"], AGENT)

        t0 = time.time()
        stats = await ingest_folder(col["id"], data_dir=data_dir)
        dt = time.time() - t0
        print(f"indexed {stats['files_indexed']} files / "
              f"{stats['chunks_ingested']} chunks in {dt:.1f}s "
              f"(skipped: unclaimed={stats['skipped_unclaimed']} "
              f"ignored={stats['skipped_ignored']} binary={stats['skipped_binary']})")
        if stats.get("degraded"):
            print("ERROR: embedder unavailable; aborting", file=sys.stderr)
            return 2

        hits_n = 0
        for q in questions:
            results = await api.search(
                q["question"], agent=AGENT, limit=k,
                collections=[col["id"]], collections_only=True,
                data_dir=data_dir, **search_opts,
            )
            got_files = [h["metadata"].get("file_path") for h in results]
            hit = any(f in q["answer_files"] for f in got_files)
            hits_n += hit
            mark = "HIT " if hit else "MISS"
            print(f"  [{mark}] {q['question'][:70]:<70} -> {got_files[:3]}")

        recall = hits_n / len(questions)
        print(f"\nmode={mode}  file-level Recall@{k}: {hits_n}/{len(questions)} = {recall:.3f}")
        if mode == "lexical":
            print("note: lexical fallback smoke. The pre-registered kill bar "
                  "(Recall@5 >= 0.8) is judged on the semantic arm with the "
                  "ONNX embedder on the bench host.")
        store.close()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs-dir", default=str(REPO_ROOT / "docs"))
    p.add_argument("--questions",
                   default=str(REPO_ROOT / "benchmarks" / "data" / "collections_eval_questions.json"))
    p.add_argument("--mode", choices=["auto", "semantic", "lexical"], default="auto")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    return asyncio.run(run(Path(args.docs_dir).resolve(),
                           Path(args.questions), args.mode, args.k))


if __name__ == "__main__":
    raise SystemExit(main())
