#!/usr/bin/env python3
"""BEAM feasibility probe: how heavy is it to ingest one BEAM conversation on our tier?

Pulls ONE conversation from Mohammadta/BEAM via the HF datasets-server rows API
(no full-split download), embeds its chat with the local ONNX embedder, and
reports the cost so we can judge whether a full BEAM-1M run is realistic on a
low-end tier before committing to it.

Usage: python3 benchmarks/beam_feasibility_probe.py [SPLIT]   # SPLIT = 100K|500K|1M
"""
import asyncio
import json
import os
import resource
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taosmd.vector_memory import VectorMemory

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "100K"
ONNX = os.environ.get(
    "TAOSMD_ONNX_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "minilm-onnx"),
)


def fetch_one_conversation(split):
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset=Mohammadta/BEAM&config=default&split={split}&offset=0&length=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "taosmd-beam-probe/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["rows"][0]["row"]


def chat_to_turns(chat):
    """Normalise the chat field (list of turns) into (role, text) strings."""
    turns = []
    for t in chat:
        if isinstance(t, dict):
            text = t.get("content") or t.get("text") or t.get("message") or ""
            role = t.get("role") or t.get("speaker") or "user"
        else:
            text, role = str(t), "user"
        if text:
            turns.append((role, text))
    return turns


async def main():
    print(f"== BEAM feasibility probe: split={SPLIT}, embedder={os.path.basename(ONNX)} ==")
    t0 = time.time()
    row = fetch_one_conversation(SPLIT)
    chat = row.get("chat", [])
    turns = chat_to_turns(chat)
    chars = sum(len(x) for _, x in turns)
    est_tokens = chars // 4
    n_questions = len(row.get("user_questions", []) or [])
    print(f"  fetched in {time.time()-t0:.1f}s | turns={len(turns)} | chars={chars} | ~tokens={est_tokens} | questions={n_questions}")

    import tempfile
    import httpx
    vmem = VectorMemory(
        db_path=os.path.join(tempfile.mkdtemp(), "v.db"),
        embed_mode=os.environ.get("TAOSMD_EMBED_MODE", "onnx"),
        onnx_path=ONNX,
    )
    async with httpx.AsyncClient(timeout=30) as c:
        await vmem.init(http_client=c)
        # Chunk the chat the way the runners do: ~100-word chunks, embed each.
        t1 = time.time()
        nchunks = 0
        for role, text in turns:
            words = text.split()
            for s in range(0, len(words), 80):
                chunk = " ".join(words[s:s + 100])
                if chunk.strip():
                    await vmem.add(chunk, metadata={"role": role})
                    nchunks += 1
        embed_s = time.time() - t1
        # One retrieval to confirm it works end to end.
        t2 = time.time()
        hits = await vmem.search("what did the user say", limit=5)
        search_s = time.time() - t2
        await vmem.close()

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    print(f"  ingest: {nchunks} chunks embedded in {embed_s:.1f}s ({nchunks/max(embed_s,0.01):.0f} chunks/s) | search {search_s*1000:.0f}ms ({len(hits)} hits)")
    print(f"  peak RSS: {rss_mb:.0f} MB")
    print("  --- extrapolation (one conversation; a full BEAM run is 100 of these) ---")
    print(f"    this split full ingest (x100 convs): ~{embed_s*100/60:.1f} min embed")
    if SPLIT == "100K":
        print(f"    1M tier is ~10x heavier per conversation; 10M ~100x")
    print(f"  VERDICT INPUTS: tokens/conv={est_tokens}, embed_s/conv={embed_s:.1f}, RSS={rss_mb:.0f}MB")


if __name__ == "__main__":
    asyncio.run(main())
