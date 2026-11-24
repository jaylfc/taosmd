"""Shared pieces of the runners' ``--context product`` mode.

In product mode a runner ingests the haystack through the normal product
ingest path and builds the reader's context only with
``taosmd.answer_context``. Nothing here cuts the context by characters; the
token budget inside answer_context is the only limit.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import sys
import tempfile

from taosmd import api as _api

CONTEXT_MODES = ("legacy", "product")

#: Agent name the product-mode haystack is ingested under.
PRODUCT_AGENT = "bench_eval"

#: The guidance text from answer_context is the instruction.
PRODUCT_PROMPT = """{guidance}

{context}

Question: {question}

Answer:"""


def resolve_context_mode(args) -> str:
    """The context mode: --context, else TAOSMD_CONTEXT_MODE, else legacy."""
    mode = getattr(args, "context", None) or os.environ.get("TAOSMD_CONTEXT_MODE") or "legacy"
    if mode not in CONTEXT_MODES:
        print(
            f"  ERROR: context mode {mode!r} is not valid. Valid values: "
            f"{', '.join(CONTEXT_MODES)}",
            file=sys.stderr,
        )
        sys.exit(2)
    return mode


def refuse(problems: list[str]) -> None:
    """Print every problem and exit 2 when there are any."""
    if problems:
        for msg in problems:
            print(f"  ERROR: {msg}", file=sys.stderr)
        sys.exit(2)


def reader_ctx_problem(reader_ctx_tokens) -> list[str]:
    if not reader_ctx_tokens or reader_ctx_tokens <= 0:
        return ["--reader-ctx-tokens (a positive token count) is required with --context product"]
    return []


def new_product_dir() -> str:
    """A fresh data dir for one haystack, seeded with the product config.

    ``TAOSMD_PRODUCT_CONFIG`` may name a ``config.json`` (as written by setup)
    to copy in, so the embedder and store options are the install's, not the
    bare defaults.
    """
    path = tempfile.mkdtemp(prefix="bench-product-")
    cfg = os.environ.get("TAOSMD_PRODUCT_CONFIG")
    if cfg:
        shutil.copyfile(cfg, os.path.join(path, "config.json"))
    return path


async def release_product_stores(data_dir: str) -> None:
    """Close and forget the cached product stores, then delete the data dir."""
    resolved = _api._resolve_data_dir(data_dir)
    stores = _api._stores_cache.pop(resolved, None) or {}
    for store in stores.values():
        close = getattr(store, "close", None)
        if close is None:
            continue
        res = close()
        if inspect.isawaitable(res):
            await res
    shutil.rmtree(data_dir, ignore_errors=True)


def context_sha256(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def _compact(blocks: list[dict]) -> list[dict]:
    return [
        {"session_id": b["session_id"], "span": b.get("span"), "tokens": b.get("tokens"),
         "turns": [t["turn_index"] for t in b["turns"]]}
        for b in blocks
    ]


def product_meta(result: dict) -> dict:
    """What a runner writes per question for an answer_context result."""
    return {
        "included": _compact(result["included"]),
        "dropped": _compact(result["dropped"]),
        "context_sha256": context_sha256(result["context"]),
        "tokens_used": result["tokens_used"],
        "sessions_dropped": result["sessions_dropped"],
        "sessions_included": result["sessions_included"],
        "hits_total": result["hits_total"],
        "budget_available": result["budget_available"],
        "expanded": result["expanded"],
    }


def count_in(blocks: list[dict], wanted: set[tuple]) -> int:
    """How many wanted ``(session_id, turn_index)`` pairs appear in ``blocks``
    (the ``included`` or ``dropped`` list of an answer_context result)."""
    have = {(t["session_id"], t["turn_index"]) for b in blocks for t in b["turns"]}
    return len(wanted & have)


def check_ingest(result: dict, what: str) -> None:
    """Fail loudly when ingest could not embed everything.

    ``api.ingest`` archives first and reports an embedder failure instead of
    raising, so a run on a host without a working embedder would otherwise
    score an empty store as wrong answers.
    """
    if result.get("degraded"):
        raise RuntimeError(
            f"{what}: {result.get('vector_failures', '?')} turns were archived but not embedded "
            "(embedder unavailable); the store is not searchable. Fix the embedder and rerun."
        )


async def product_environment(data_dir: str, agent: str = PRODUCT_AGENT) -> dict:
    """What the product actually resolved for this data dir, for the results.

    The embedder, the recipe, and whether the reranker the recipe asks for is
    present. A missing reranker makes search degrade silently, so a runner
    refuses to continue on it (see :func:`require_healthy`).
    """
    from taosmd import recipes as _recipes  # noqa: PLC0415
    from taosmd import config as _config  # noqa: PLC0415

    stores = await _api._ensure_stores(data_dir)
    recipe = _recipes.resolve_recipe(agent, data_dir=data_dir)
    rc = _api._apply_runtime_overrides(recipe.retrieval, _config.get_runtime_overrides(data_dir))
    wanted = rc.get("reranker")
    available = True
    if wanted == "bge-v2-m3":
        from taosmd.cross_encoder import CrossEncoderReranker  # noqa: PLC0415
        available = bool(CrossEncoderReranker(data_dir=stores["data_dir"]).available)
    return {
        "recipe": recipe.id,
        "embedder": stores["vector"]._embedder_identity(),
        "reranker": wanted,
        "reranker_available": available,
        "fusion": rc.get("fusion"),
        "strategy": rc.get("strategy"),
        "candidate_top_k": rc.get("candidate_top_k"),
    }


def require_healthy(env: dict) -> None:
    """Exit 1 if the recipe's reranker is missing: search would run without it."""
    if not env["reranker_available"]:
        print(
            f"  ERROR: recipe {env['recipe']!r} uses the {env['reranker']!r} reranker but it is "
            "not installed, so search would silently run without it and the result would not "
            "describe the product. Install it or pick a recipe without it.",
            file=sys.stderr,
        )
        sys.exit(1)
