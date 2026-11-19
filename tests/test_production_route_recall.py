"""Production route recall test: exercises the REAL POST /search route for every
recipe and strategy, using the same code path agents call.

This test guards against regressions where a recipe's strategy (e.g. fast/minimal)
skips the vector store for queries where the intent primary is not 'qmd', which
happened on lite-pi (card tsk-rmebxw). It runs the full HTTP stack:
make_server -> POST /ingest -> apply_recipe -> POST /search with limit=5.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import urllib.error
import urllib.request

import pytest

from taosmd import api as taosmd_api
from taosmd import http_server
from taosmd import recipes as recipes_mod
from taosmd.intent_classifier import get_search_strategy


# ---------------------------------------------------------------------------
# Deterministic bag-of-words hash embedder (384 dims, L2-normalised)
# Lower-cased word tokens, minus a fixed stopword list, hashed into 384 dims.
# Word overlap drives similarity so semantic matching works without a model.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "among", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "what", "when",
    "where", "who", "whom", "which", "whose", "why", "how", "my", "your", "his",
    "her", "its", "our", "their", "this", "that", "these", "those", "i", "you",
    "he", "she", "it", "we", "they", "me", "him", "us", "them", "am", "shall",
}


def _bow_embed(text: str, dim: int = 384) -> list[float]:
    """Bag-of-words hash embedding: deterministic, no model download needed."""
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    vec = [0.0] * dim
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _patch_embedder(stores: dict) -> None:
    """Override vector store's embed() with the deterministic BoW embedder."""
    vmem = stores["vector"]

    async def _fake_embed(text: str, task: str = "search_document") -> list[float]:
        return _bow_embed(text)

    vmem.embed = _fake_embed  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Test server context (mirrors test_http_server.py)
# ---------------------------------------------------------------------------


class _ServerCtx:
    def __init__(self, url, httpd, stores, data_dir):
        self.url = url
        self.httpd = httpd
        self.stores = stores
        self.data_dir = data_dir

    def run(self, coro):
        return self.httpd.service_loop.run(coro)


@pytest.fixture
def live_server_ctx(tmp_path, monkeypatch):
    """Start HTTP server on ephemeral port with isolated data dir + BoW embedder."""
    data_dir = tmp_path / "taosmd-data"
    data_dir.mkdir()
    monkeypatch.setattr(taosmd_api, "_stores_cache", {})

    httpd = http_server.make_server("127.0.0.1", 0, data_dir=str(data_dir))

    stores = httpd.service_loop.run(taosmd_api._ensure_stores(str(data_dir)))
    _patch_embedder(stores)

    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield _ServerCtx(f"http://{host}:{port}", httpd, stores, str(data_dir))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        for s in list(taosmd_api._stores_cache.values()):
            for store in (s.get("archive"), s.get("vector"), s.get("kg")):
                if store and hasattr(store, "close"):
                    try:
                        httpd.service_loop.run(store.close())
                    except Exception:
                        pass
        httpd.service_loop.close()


@pytest.fixture
def live_server(live_server_ctx):
    return live_server_ctx.url


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(url: str, payload) -> tuple[int, dict]:
    data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return _send(req)


def _send(req) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


# ---------------------------------------------------------------------------
# Test data: 5 facts + 3 query forms each
# ---------------------------------------------------------------------------

FACTS = [
    "My NAS backup runs every Sunday at 02:30 and goes to the shed NAS.",
    "The deploy check phrase is heliotrope-lantern-42.",
    "Jay prefers dark mode in every app.",
    "The taOS bus box is an Orange Pi 5 Plus with 16 GB of RAM.",
    "Our staging database password rotates on the first Monday of each month.",
]

# (fact_index, query_form, query_text, fact_label)
# query_form: "question" | "keyword" | "paraphrase"
# fact_label: short descriptor for test ids (e.g. "nas", "deploy", "darkmode", "busbox", "staging")
QUERIES = [
    # NAS backup
    (0, "question", "When does my NAS backup run?", "nas"),
    (0, "keyword", "NAS backup schedule", "nas"),
    (0, "paraphrase", "which day is the NAS backup", "nas"),
    # Deploy phrase
    (1, "question", "What is the deploy check phrase?", "deploy"),
    (1, "keyword", "deploy check phrase", "deploy"),
    (1, "paraphrase", "deploy phrase", "deploy"),
    # Jay dark mode
    (2, "question", "What does Jay prefer for app themes?", "darkmode"),
    (2, "keyword", "Jay dark mode", "darkmode"),
    (2, "paraphrase", "theme preference Jay", "darkmode"),
    # Bus box hardware
    (3, "question", "What hardware is the taOS bus box?", "busbox"),
    (3, "keyword", "bus box hardware", "busbox"),
    (3, "paraphrase", "what board runs the bus", "busbox"),
    # Staging password rotation
    (4, "question", "When does the staging database password rotate?", "staging"),
    (4, "keyword", "staging password rotation", "staging"),
    (4, "paraphrase", "password change schedule staging", "staging"),
]


# ---------------------------------------------------------------------------
# Parametrization: every recipe x every fact x every query form
# ---------------------------------------------------------------------------


def _build_test_cases():
    """Build list of test cases with pytest.param for xfail marking.

    Xfail condition: queries on fast/minimal strategies where the intent
    classifier routes to a primary source other than 'qmd' (vector store).
    This matches the tsk-rmebxw bug: fast/minimal only queries the intent
    primary source, so factual/preference/recent/relational/timeline queries
    (primary != qmd) skip the vector store.
    """
    all_recipes = recipes_mod.list_recipes()
    cases = []
    for recipe in all_recipes:
        strategy = recipe.retrieval.get("strategy", "thorough")
        for fact_idx, fact_text in enumerate(FACTS):
            for q_fact_idx, query_form, query_text, fact_label in QUERIES:
                if q_fact_idx == fact_idx:
                    # Determine if this case should be xfail
                    marks = []
                    if strategy in ("fast", "minimal"):
                        intent_primary = get_search_strategy(query_text).get("primary")
                        if intent_primary != "qmd":
                            marks.append(pytest.mark.xfail(strict=True, reason="tsk-rmebxw"))
                    cases.append(
                        pytest.param(
                            recipe.id,
                            strategy,
                            fact_idx,
                            query_form,
                            query_text,
                            fact_text,
                            fact_label,
                            marks=marks,
                        )
                    )
    return cases


TEST_CASES = _build_test_cases()


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recipe_id,strategy,fact_idx,query_form,query_text,fact_text,fact_label",
    TEST_CASES,
    ids=[f"[{case.values[0]}-{case.values[3]}-{case.values[6]}]" for case in TEST_CASES],
)
def test_production_route_recall(
    live_server_ctx, recipe_id, strategy, fact_idx, query_form, query_text, fact_text, fact_label
):
    """For each recipe, seed facts via /ingest, apply recipe, search via /search.

    Asserts the seeded fact text appears in the returned hits (limit=5).
    Fast/minimal strategy cases with primary != qmd are xfail (tsk-rmebxw).
    """
    url = live_server_ctx.url
    agent = "recall-ci"
    data_dir = live_server_ctx.data_dir

    # Ensure agent exists (synchronous, operates on shared data_dir)
    from taosmd.agents import ensure_agent as _ensure_agent  # noqa: PLC0415
    _ensure_agent(agent, data_dir=data_dir)

    # Apply the recipe to the agent (uses real config/agent API)
    recipes_mod.apply_recipe(agent, recipe_id, data_dir=data_dir)

    # Seed all 5 facts through the REAL POST /ingest route
    for fact in FACTS:
        status, body = _post(f"{url}/ingest", {"text": fact, "agent": agent})
        assert status == 200, f"ingest failed: {body}"

    # Search with the query through the REAL POST /search route
    status, body = _post(
        f"{url}/search", {"query": query_text, "agent": agent, "limit": 5}
    )
    assert status == 200, f"search failed: {body}"

    hits = body.get("hits", [])
    hit_texts = [h.get("text", "") for h in hits]

    # Check if the seeded fact is in the returned hits
    found = any(fact_text in hit for hit in hit_texts)

    assert found, (
        f"Fact not found in hits for recipe={recipe_id} strategy={strategy} "
        f"query_form={query_form} query={query_text!r}\n"
        f"Expected to find: {fact_text!r}\n"
        f"Hits: {hit_texts}"
    )


# ---------------------------------------------------------------------------
# Meta-assertion: number of recipes tested >= registry size
# ---------------------------------------------------------------------------


def test_recipe_coverage_meta():
    """Ensure the test iterates all recipes in the registry."""
    all_recipes = recipes_mod.list_recipes()
    tested_recipe_ids = {case.values[0] for case in TEST_CASES}
    registry_ids = {r.id for r in all_recipes}

    # Every recipe in registry must appear in test cases
    assert tested_recipe_ids == registry_ids, (
        f"Test misses recipes: {registry_ids - tested_recipe_ids}"
    )
