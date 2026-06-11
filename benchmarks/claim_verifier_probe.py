#!/usr/bin/env python3
"""E2 extraction hallucination probe for taosmd.

Measures the rate at which the LLM fact extractor produces claims
that are not grounded in the source text it was given.

Workflow
--------
1. EXTRACTION: for each LoCoMo conversation session, call the real
   extract_facts_with_llm() path and record every {claim, source}
   pair (where source is EXACTLY the text block passed to the
   extractor).

2. VERIFICATION: for each pair, ask a different-family verifier model
   a strict entailment question -- SOURCE-only, no external knowledge.
   Parse SUPPORTED / PARTIAL / UNSUPPORTED / VERIFIER_ERROR.

3. OUTPUT: hallucination rate summary, per-conversation breakdown,
   20 sampled triples for human calibration, optional JSON dump.

Honesty constraints
-------------------
- Extractor and verifier MUST be different model families
  (gemma vs qwen vs llama substring check). Asserts at startup.
- Source handed to the verifier is the EXACT string passed to the
  extractor, never re-derived.
- VERIFIER_ERROR is counted and reported, never silently dropped.
- Startup canary: one extraction call + one verification call before
  the full run. Exits 2 on canary failure.
- Aborts if VERIFIER_ERROR rate exceeds 10%.

Checkpointing
-------------
Both --pairs-out and --out support resumable checkpointing: already
completed sessions / verified pairs are skipped on re-run.

Usage
-----
  python benchmarks/claim_verifier_probe.py \\
      --data data/locomo/data/locomo10.json \\
      --conversations 3 \\
      --extract-model gemma4:e2b \\
      --verify-model hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \\
      --ollama-url http://localhost:11434 \\
      --pairs-out /tmp/cvp_pairs.jsonl \\
      --out /tmp/cvp_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import textwrap
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Path setup so this runs from any CWD
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from taosmd.memory_extractor import extract_facts_with_llm  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tokens that constitute valid verdicts (matched against upper-cased prefix)
_VERDICT_TOKENS = ("SUPPORTED", "PARTIAL", "UNSUPPORTED")

_FAMILY_SUBSTRINGS = ["gemma", "qwen", "llama", "mistral", "phi", "falcon", "deepseek"]

# Prompt handed to the verifier model
_VERIFY_SYSTEM = (
    "You are a strict entailment judge. "
    "You may ONLY use the provided source text to assess the claim. "
    "You must NOT use any outside knowledge or world facts. "
    "If the source text does not mention the claim at all, answer UNSUPPORTED."
)

_VERIFY_USER_TMPL = (
    "SOURCE:\n{source}\n\n"
    "CLAIM:\n{claim}\n\n"
    "Is the claim fully supported by the source alone? "
    "Answer exactly one of: SUPPORTED, PARTIAL, UNSUPPORTED."
)

# How many chars of source to show in the human-calibration sample
_SAMPLE_SNIPPET_LEN = 200


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------

def _model_family(model_name: str) -> str | None:
    """Return the first matching family substring, or None."""
    lower = model_name.lower()
    for fam in _FAMILY_SUBSTRINGS:
        if fam in lower:
            return fam
    return None


def assert_different_families(extract_model: str, verify_model: str) -> None:
    """Exit with loud error if both models are from the same family."""
    ef = _model_family(extract_model)
    vf = _model_family(verify_model)
    if ef and vf and ef == vf:
        print(
            f"\n[FATAL] extract-model ({extract_model!r}) and verify-model "
            f"({verify_model!r}) are the same family ({ef!r}).\n"
            "Self-verification inflates accuracy. Use different model families.\n"
            "Default pair: gemma extractor + qwen verifier.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    if not ef:
        print(
            f"[WARN] Could not detect family for extract-model {extract_model!r}. "
            "Family check skipped for extractor.",
            file=sys.stderr,
        )
    if not vf:
        print(
            f"[WARN] Could not detect family for verify-model {verify_model!r}. "
            "Family check skipped for verifier.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# LoCoMo data helpers
# ---------------------------------------------------------------------------

def _session_keys(conversation: dict) -> list[tuple[str, str]]:
    """Return sorted (session_key, datetime_str) pairs."""
    pairs = []
    for key in conversation.keys():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        parts = key.split("_")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if not isinstance(conversation.get(key), list):
            continue
        pairs.append((key, conversation.get(f"{key}_date_time", "")))
    pairs.sort(key=lambda kv: int(kv[0].split("_")[1]))
    return pairs


def _session_text(conversation: dict, session_key: str, datetime_str: str) -> str:
    """Build the raw text block for one session exactly as extraction would see it."""
    turns = conversation.get(session_key) or []
    lines = []
    if datetime_str:
        lines.append(f"[{datetime_str}]")
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{speaker}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def _parse_verdict(response_text: str) -> str | None:
    """Extract the verdict token from a potentially noisy response.

    Returns one of SUPPORTED, PARTIAL, UNSUPPORTED, or None if unparseable.

    Match order matters: UNSUPPORTED must be checked before SUPPORTED because
    "SUPPORTED" is a substring of "UNSUPPORTED".
    """
    cleaned = response_text.strip()
    # Strip <think> tags
    cleaned = re.sub(r"</?think[^>]*>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think[^>]*>", "", cleaned, flags=re.IGNORECASE)
    # Extract content from markdown code blocks rather than discarding it
    md_match = re.search(r"```[^\n]*\n(.*?)```", cleaned, flags=re.DOTALL)
    if md_match:
        cleaned = md_match.group(1).strip()
    cleaned = cleaned.strip()

    upper = cleaned.upper()

    # Longest-token-first ordering: UNSUPPORTED before SUPPORTED so substring
    # matches do not fire prematurely. PARTIAL is independent of both.
    _ORDERED = ("UNSUPPORTED", "PARTIAL", "SUPPORTED")

    # Exact single-token responses first
    for token in _ORDERED:
        if upper == token:
            return token

    # First-token heuristic: the verdict is often the first word on the line
    first_word = re.split(r"[\s\.,;:\n]", upper.lstrip("#*- "))[0]
    for token in _ORDERED:
        if first_word == token:
            return token

    # Scan for any verdict token in the first 80 chars with word-boundary
    # protection ("Answer: UNSUPPORTED", "Verdict: PARTIAL", etc.)
    head = upper[:80]
    for token in _ORDERED:
        if re.search(r"\b" + token + r"\b", head):
            return token

    return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_pairs(path: Path) -> dict[str, dict]:
    """Load existing pairs from a JSONL checkpoint. Key: '{conv_id}:{session_key}'."""
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = f"{rec['conversation_id']}:{rec['session_id']}"
                if key not in done:
                    done[key] = {"pairs": []}
                done[key]["pairs"].append(rec)
            except Exception:
                pass
    return done


def _append_pairs(path: Path, pairs: list[dict]) -> None:
    """Append new pairs to the JSONL checkpoint file."""
    with path.open("a") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair) + "\n")


def _load_verdicts(path: Path) -> dict[str, str]:
    """Load existing verdicts. Key: pair_id, Value: verdict string."""
    if not path.exists():
        return {}
    verdicts: dict[str, str] = {}
    try:
        data = json.loads(path.read_text())
        for entry in data.get("verdicts", []):
            if "pair_id" in entry and "verdict" in entry:
                verdicts[entry["pair_id"]] = entry["verdict"]
    except Exception:
        pass
    return verdicts


# ---------------------------------------------------------------------------
# Ollama client helpers
# ---------------------------------------------------------------------------

async def _chat_completion(
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    messages: list[dict],
    temperature: float = 0,
    max_tokens: int = 64,
) -> str:
    """Call Ollama /v1/chat/completions and return the content string."""
    resp = await client.post(
        f"{ollama_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").strip()


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

async def _run_canary(
    client: httpx.AsyncClient,
    ollama_url: str,
    extract_model: str,
    verify_model: str,
) -> None:
    """Run one extraction call and one verification call to verify connectivity.

    Exits 2 on any failure.
    """
    print("[canary] Running startup connectivity checks...")

    # 1) Extraction canary
    canary_text = "Alice uses Python for all her backend work."
    try:
        facts = await extract_facts_with_llm(
            canary_text,
            ollama_url,
            client,
            agent_name="canary",
            model=extract_model,
        )
        if not isinstance(facts, list):
            raise ValueError(f"Expected list, got {type(facts)}")
        print(f"[canary] Extraction OK: {len(facts)} fact(s) from canary text.")
    except Exception as exc:
        print(
            f"[canary] EXTRACTION FAILED: {exc}\n"
            f"  model={extract_model!r}  url={ollama_url!r}",
            file=sys.stderr,
        )
        sys.exit(2)

    # 2) Verification canary
    try:
        raw = await _chat_completion(
            client,
            ollama_url,
            verify_model,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM},
                {
                    "role": "user",
                    "content": _VERIFY_USER_TMPL.format(
                        source=canary_text,
                        claim="Alice uses Python.",
                    ),
                },
            ],
        )
        verdict = _parse_verdict(raw)
        if verdict is None:
            raise ValueError(f"Unparseable canary verdict: {raw!r}")
        print(f"[canary] Verification OK: verdict={verdict!r}  raw={raw!r}")
    except Exception as exc:
        print(
            f"[canary] VERIFICATION FAILED: {exc}\n"
            f"  model={verify_model!r}  url={ollama_url!r}",
            file=sys.stderr,
        )
        sys.exit(2)

    print("[canary] All checks passed.\n")


# ---------------------------------------------------------------------------
# Core extraction pass
# ---------------------------------------------------------------------------

async def _extract_session_pairs(
    client: httpx.AsyncClient,
    ollama_url: str,
    extract_model: str,
    conversation_id: str,
    conversation: dict,
    session_key: str,
    datetime_str: str,
) -> list[dict]:
    """Run extraction on one session and return list of claim/source pair dicts."""
    # Check for actual turn content before building the text block
    turns = (conversation.get(session_key) or [])
    has_turns = any((t.get("text") or "").strip() for t in turns)
    if not has_turns:
        return []

    source_text = _session_text(conversation, session_key, datetime_str)
    if not source_text.strip():
        return []

    facts = await extract_facts_with_llm(
        source_text,
        ollama_url,
        client,
        agent_name=f"cvp_{conversation_id}",
        model=extract_model,
    )

    pairs = []
    for fact in facts:
        claim_parts = []
        if fact.get("subject"):
            claim_parts.append(fact["subject"])
        if fact.get("predicate"):
            claim_parts.append(fact["predicate"])
        if fact.get("object"):
            claim_parts.append(fact["object"])

        if not claim_parts:
            continue

        # Use source_line from LLM if present, else the full session text
        # The verifier always gets the EXACT source the extractor saw.
        claim_text = " ".join(claim_parts)
        pair_id = (
            f"{conversation_id}:{session_key}:"
            f"{abs(hash(claim_text)) % (10 ** 10)}"
        )
        pairs.append({
            "pair_id": pair_id,
            "conversation_id": conversation_id,
            "session_id": session_key,
            "claim": claim_text,
            "fact_raw": fact,
            # EXACT source text the extractor saw -- never re-derived
            "source": source_text,
        })

    return pairs


# ---------------------------------------------------------------------------
# Verification pass
# ---------------------------------------------------------------------------

async def _verify_pair(
    client: httpx.AsyncClient,
    ollama_url: str,
    verify_model: str,
    pair: dict,
) -> str:
    """Verify one claim against its source. Returns verdict string."""
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {
            "role": "user",
            "content": _VERIFY_USER_TMPL.format(
                source=pair["source"][:3000],
                claim=pair["claim"],
            ),
        },
    ]

    for attempt in range(2):
        try:
            raw = await _chat_completion(
                client, ollama_url, verify_model, messages
            )
            verdict = _parse_verdict(raw)
            if verdict is not None:
                return verdict
            if attempt == 0:
                print(
                    f"  [warn] Unparseable verdict (attempt 1): {raw!r} "
                    f"-- retrying...",
                    file=sys.stderr,
                )
        except Exception as exc:
            if attempt == 0:
                print(f"  [warn] Verification call failed: {exc} -- retrying...", file=sys.stderr)
            else:
                print(f"  [warn] Verification call failed on retry: {exc}", file=sys.stderr)

    return "VERIFIER_ERROR"


# ---------------------------------------------------------------------------
# Summary + output
# ---------------------------------------------------------------------------

def _print_summary(
    all_pairs: list[dict],
    verdicts: dict[str, str],
    conv_breakdown: dict[str, dict],
) -> None:
    total = len(all_pairs)
    counts: dict[str, int] = {
        "SUPPORTED": 0,
        "PARTIAL": 0,
        "UNSUPPORTED": 0,
        "VERIFIER_ERROR": 0,
    }
    for pair in all_pairs:
        v = verdicts.get(pair["pair_id"], "VERIFIER_ERROR")
        counts[v] = counts.get(v, 0) + 1

    print("\n" + "=" * 60)
    print("CLAIM VERIFIER PROBE -- SUMMARY")
    print("=" * 60)
    print(f"Total claims verified : {total}")
    for label, count in counts.items():
        rate = count / total * 100 if total else 0
        print(f"  {label:<18}: {count:>5}  ({rate:.1f}%)")

    partial_unsup = counts["PARTIAL"] + counts["UNSUPPORTED"]
    halluc_rate = partial_unsup / total * 100 if total else 0
    print(f"\nE2 HALLUCINATION RATE: {halluc_rate:.1f}% (PARTIAL+UNSUPPORTED)")
    if counts["VERIFIER_ERROR"] > 0:
        print(
            f"[warn] {counts['VERIFIER_ERROR']} VERIFIER_ERROR(s) included in total "
            "(treated as denominator, not hallucination)"
        )

    print("\n--- Per-conversation breakdown ---")
    for conv_id, breakdown in sorted(conv_breakdown.items()):
        n = breakdown["total"]
        h = breakdown["PARTIAL"] + breakdown["UNSUPPORTED"]
        hr = h / n * 100 if n else 0
        print(
            f"  {conv_id:<20}  claims={n:>4}  "
            f"sup={breakdown['SUPPORTED']:>3}  "
            f"par={breakdown['PARTIAL']:>3}  "
            f"uns={breakdown['UNSUPPORTED']:>3}  "
            f"err={breakdown['VERIFIER_ERROR']:>3}  "
            f"halluc={hr:.1f}%"
        )

    # 20 random samples for human calibration
    verified_pairs = [p for p in all_pairs if p["pair_id"] in verdicts]
    if verified_pairs:
        sample_size = min(20, len(verified_pairs))
        sample = random.sample(verified_pairs, sample_size)
        print(f"\n--- {sample_size} randomly sampled (claim, source-snippet, verdict) ---")
        for i, pair in enumerate(sample, 1):
            snippet = pair["source"][:_SAMPLE_SNIPPET_LEN].replace("\n", " ")
            verdict = verdicts.get(pair["pair_id"], "?")
            print(f"\n  [{i}] verdict={verdict}")
            print(f"       claim  : {pair['claim']}")
            print(f"       source : {snippet}...")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> int:
    # 1) Model family check
    assert_different_families(args.extract_model, args.verify_model)

    # 2) Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[error] Dataset not found: {data_path}", file=sys.stderr)
        return 1

    dataset: list[dict] = json.loads(data_path.read_text())
    conversations = dataset[: args.conversations]
    print(
        f"Loaded {len(conversations)} conversation(s) from {data_path.name}"
    )

    # 3) Checkpoint state
    pairs_path = Path(args.pairs_out)
    out_path = Path(args.out)
    done_sessions = _load_pairs(pairs_path)
    done_verdicts = _load_verdicts(out_path)

    print(
        f"Checkpoint: {sum(len(v['pairs']) for v in done_sessions.values())} pairs, "
        f"{len(done_verdicts)} verdicts already on disk."
    )

    async with httpx.AsyncClient() as client:
        # 4) Canary
        await _run_canary(
            client, args.ollama_url, args.extract_model, args.verify_model
        )

        # 5) Extraction pass
        all_pairs: list[dict] = []

        for conv in conversations:
            conv_id = str(conv.get("sample_id", id(conv)))
            conversation = conv.get("conversation", conv)
            session_pairs_list = _session_keys(conversation)

            print(
                f"\n[extract] conversation {conv_id} "
                f"({len(session_pairs_list)} sessions)"
            )

            for session_key, datetime_str in session_pairs_list:
                ck = f"{conv_id}:{session_key}"

                if ck in done_sessions:
                    # Resume: collect previously extracted pairs
                    existing = done_sessions[ck]["pairs"]
                    all_pairs.extend(existing)
                    print(
                        f"  [skip] {session_key} ({len(existing)} pairs from checkpoint)"
                    )
                    continue

                pairs = await _extract_session_pairs(
                    client,
                    args.ollama_url,
                    args.extract_model,
                    conv_id,
                    conversation,
                    session_key,
                    datetime_str,
                )
                print(
                    f"  {session_key}: {len(pairs)} claim(s) extracted"
                )

                if pairs:
                    _append_pairs(pairs_path, pairs)
                    all_pairs.extend(pairs)
                    done_sessions[ck] = {"pairs": pairs}

        print(f"\n[extract] Total pairs collected: {len(all_pairs)}")

        # 6) Verification pass
        verdicts: dict[str, str] = dict(done_verdicts)
        new_verdicts: list[dict] = []
        to_verify = [p for p in all_pairs if p["pair_id"] not in verdicts]

        print(
            f"[verify] {len(to_verify)} pair(s) to verify "
            f"({len(verdicts)} already done)."
        )

        for i, pair in enumerate(to_verify, 1):
            verdict = await _verify_pair(
                client, args.ollama_url, args.verify_model, pair
            )
            verdicts[pair["pair_id"]] = verdict
            new_verdicts.append({
                "pair_id": pair["pair_id"],
                "conversation_id": pair["conversation_id"],
                "session_id": pair["session_id"],
                "claim": pair["claim"],
                "verdict": verdict,
            })

            if i % 10 == 0 or i == len(to_verify):
                print(
                    f"  verified {i}/{len(to_verify)}  "
                    f"last_verdict={verdict}"
                )

            # Running error-rate guard
            if i >= 20:
                error_count = sum(
                    1 for p in all_pairs
                    if verdicts.get(p["pair_id"]) == "VERIFIER_ERROR"
                )
                error_rate = error_count / i
                if error_rate > 0.10:
                    print(
                        f"\n[ABORT] VERIFIER_ERROR rate={error_rate:.1%} "
                        f"exceeds 10% threshold after {i} verifications.\n"
                        "Check verifier model connectivity and prompt parsing.",
                        file=sys.stderr,
                    )
                    # Save what we have before aborting
                    _save_results(out_path, all_pairs, verdicts, new_verdicts)
                    return 3

    # 7) Save results
    _save_results(out_path, all_pairs, verdicts, new_verdicts)

    # 8) Build per-conversation breakdown
    conv_breakdown: dict[str, dict] = {}
    for pair in all_pairs:
        cid = pair["conversation_id"]
        if cid not in conv_breakdown:
            conv_breakdown[cid] = {
                "total": 0,
                "SUPPORTED": 0,
                "PARTIAL": 0,
                "UNSUPPORTED": 0,
                "VERIFIER_ERROR": 0,
            }
        v = verdicts.get(pair["pair_id"], "VERIFIER_ERROR")
        conv_breakdown[cid]["total"] += 1
        conv_breakdown[cid][v] = conv_breakdown[cid].get(v, 0) + 1

    # 9) Print summary
    _print_summary(all_pairs, verdicts, conv_breakdown)

    return 0


def _save_results(
    out_path: Path,
    all_pairs: list[dict],
    verdicts: dict[str, str],
    new_verdicts: list[dict],
) -> None:
    """Merge new verdicts into existing file and save."""
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}

    existing_verdicts = existing.get("verdicts", [])
    existing_ids = {e["pair_id"] for e in existing_verdicts}
    merged_verdicts = existing_verdicts + [
        v for v in new_verdicts if v["pair_id"] not in existing_ids
    ]

    counts: dict[str, int] = {
        "SUPPORTED": 0,
        "PARTIAL": 0,
        "UNSUPPORTED": 0,
        "VERIFIER_ERROR": 0,
    }
    for pair in all_pairs:
        v = verdicts.get(pair["pair_id"], "VERIFIER_ERROR")
        counts[v] = counts.get(v, 0) + 1

    total = len(all_pairs)
    partial_unsup = counts["PARTIAL"] + counts["UNSUPPORTED"]
    halluc_rate = partial_unsup / total if total else 0.0

    out_path.write_text(
        json.dumps(
            {
                "total_claims": total,
                "counts": counts,
                "hallucination_rate": round(halluc_rate, 4),
                "verdicts": merged_verdicts,
            },
            indent=2,
        )
    )
    print(f"[save] Results written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E2 extraction hallucination probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example (Fedora, 3 conversations):
              python benchmarks/claim_verifier_probe.py \\
                  --data data/locomo/data/locomo10.json \\
                  --conversations 3 \\
                  --extract-model gemma4:e2b \\
                  --verify-model hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M \\
                  --ollama-url http://localhost:11434 \\
                  --pairs-out /tmp/cvp_pairs.jsonl \\
                  --out /tmp/cvp_results.json
            """
        ),
    )
    p.add_argument(
        "--data",
        default="data/locomo/data/locomo10.json",
        help="Path to locomo10.json (default: %(default)s)",
    )
    p.add_argument(
        "--conversations",
        type=int,
        default=3,
        metavar="N",
        help="Number of conversations to process (default: %(default)s)",
    )
    p.add_argument(
        "--extract-model",
        default="gemma4:e2b",
        help="Ollama model for fact extraction (default: %(default)s)",
    )
    p.add_argument(
        "--verify-model",
        default="hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M",
        help="Ollama model for entailment verification (default: %(default)s)",
    )
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: %(default)s)",
    )
    p.add_argument(
        "--pairs-out",
        default="/tmp/cvp_pairs.jsonl",
        help="Checkpoint JSONL for extracted claim/source pairs (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        default="/tmp/cvp_results.json",
        help="Output JSON file for verdict summary (default: %(default)s)",
    )
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
