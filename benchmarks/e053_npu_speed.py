#!/usr/bin/env python3
"""E-053 NPU speed/memory harness for Orange Pi RK3588.

Three subcommands: prepare, run, score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# =============================================================================
# Shared constants
# =============================================================================

QWEN_TOKENIZER_REPO = "Qwen/Qwen3.5-2B"
QWEN_TOKENIZER_REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
GRANITE_TOKENIZER_REPO = "ibm-granite/granite-4.2-3b"
GRANITE_TOKENIZER_REV = "4e9c275e232f10211ec88644d106a1f6a1b60e06"

LENGTH_TOKENS = (128, 512, 2048)
LOOP_MAX_QWEN_TOKENS = 300
LOOP_MAX_PROMPTS = 20
LOOP_MIN_WORDS = 20
LOOP_CAP_TOKENS = 512

ARM_TOKENIZER_MAP = {
    "G": "granite",
    "QH": "qwen",
    "Q": "qwen",
}

KILL_PREFILL_RATIO = 0.8
KILL_RSS_RATIO = 1.5
KILL_CAP_HITS_MAX = 2


# =============================================================================
# Prepare subcommand
# =============================================================================

def _load_tokenizers():
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    qwen_tok = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_REPO, revision=QWEN_TOKENIZER_REV)
    granite_tok = AutoTokenizer.from_pretrained(GRANITE_TOKENIZER_REPO, revision=GRANITE_TOKENIZER_REV)
    return qwen_tok, granite_tok


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_and_decode(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_prepare(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    out_path = Path(args.out)

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print("Dataset must be a non-empty JSON list", file=sys.stderr)
        return 1

    first_q = data[0]
    haystack = first_q.get("haystack_sessions", [])
    if not isinstance(haystack, list):
        print("haystack_sessions must be a list", file=sys.stderr)
        return 1

    flat_parts: list[str] = []
    for session in haystack:
        if isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict):
                    role = turn.get("role", "")
                    content = turn.get("content", "")
                    if role and content:
                        flat_parts.append(f"{role}: {content}")

    flat_text = "\n".join(flat_parts)

    qwen_tok, granite_tok = _load_tokenizers()

    dataset_sha256 = _sha256_file(dataset_path)

    prompts = []
    header = {
        "dataset_sha256": dataset_sha256,
        "qwen_tokenizer_repo": QWEN_TOKENIZER_REPO,
        "qwen_tokenizer_revision": QWEN_TOKENIZER_REV,
        "granite_tokenizer_repo": GRANITE_TOKENIZER_REPO,
        "granite_tokenizer_revision": GRANITE_TOKENIZER_REV,
    }

    for L in LENGTH_TOKENS:
        truncated = _truncate_and_decode(qwen_tok, flat_text, L)
        tokens_qwen = _count_tokens(qwen_tok, truncated)
        tokens_granite = _count_tokens(granite_tok, truncated)
        prompts.append({
            "kind": "length",
            "length": L,
            "text": truncated,
            "tokens_qwen": tokens_qwen,
            "tokens_granite": tokens_granite,
        })

    from taosmd.prompts import extraction_prompt

    user_turns = []
    for session in haystack:
        if isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict) and turn.get("role") == "user":
                    content = turn.get("content", "")
                    word_count = len(content.split())
                    if word_count >= LOOP_MIN_WORDS:
                        user_turns.append(content)
                        if len(user_turns) >= LOOP_MAX_PROMPTS:
                            break
        if len(user_turns) >= LOOP_MAX_PROMPTS:
            break

    for i, content in enumerate(user_turns):
        truncated = _truncate_and_decode(qwen_tok, content, LOOP_MAX_QWEN_TOKENS)
        wrapped = extraction_prompt(truncated)
        tokens_qwen = _count_tokens(qwen_tok, wrapped)
        tokens_granite = _count_tokens(granite_tok, wrapped)
        prompts.append({
            "kind": "loop",
            "index": i,
            "text": wrapped,
            "tokens_qwen": tokens_qwen,
            "tokens_granite": tokens_granite,
        })

    output = {"header": header, "prompts": prompts}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"Prepared {len(prompts)} prompts -> {out_path}")
    return 0


# =============================================================================
# Run subcommand (stdlib only)
# =============================================================================

def _read_proc_status(pid: int) -> dict[str, Any]:
    path = f"/proc/{pid}/status"
    try:
        with open(path, "r") as f:
            data = f.read()
    except Exception as e:
        return {"error": str(e)}

    result = {}
    for line in data.splitlines():
        if line.startswith("VmRSS:"):
            result["VmRSS_kB"] = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            result["VmHWM_kB"] = int(line.split()[1])
    return result


def _read_cpu_governors() -> list[str] | None:
    governors = set()
    base = "/sys/devices/system/cpu"
    try:
        for cpu_dir in os.listdir(base):
            if cpu_dir.startswith("cpu"):
                gov_path = os.path.join(base, cpu_dir, "cpufreq", "scaling_governor")
                try:
                    with open(gov_path, "r") as f:
                        gov = f.read().strip()
                        if gov:
                            governors.add(gov)
                except Exception:
                    pass
        return sorted(governors) if governors else None
    except Exception:
        return None


def _read_npu_freqs() -> list[int] | None:
    freqs = []
    base = "/sys/class/devfreq"
    try:
        for dev in os.listdir(base):
            if "npu" in dev.lower():
                freq_path = os.path.join(base, dev, "cur_freq")
                try:
                    with open(freq_path, "r") as f:
                        val = int(f.read().strip())
                        freqs.append(val)
                except Exception:
                    pass
        return freqs if freqs else None
    except Exception:
        return None


def _get_host_facts() -> dict[str, Any]:
    return {
        "cpu_governors": _read_cpu_governors(),
        "npu_freqs_hz": _read_npu_freqs(),
        "uname": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def _post_json(url: str, payload: dict, timeout: float = 600.0) -> urllib.request.urlopen:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _parse_ndjson_stream(resp, prompt_tokens_qwen: int, prompt_tokens_granite: int, arm: str) -> dict[str, Any]:
    first_text_time = None
    output_parts = []
    done_chunk = None
    start_time = time.perf_counter()

    for line in resp:
        line = line.decode("utf-8").strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = None
        if "message" in chunk and isinstance(chunk["message"], dict):
            text = chunk["message"].get("content")
        if text is None:
            text = chunk.get("response")

        if text and first_text_time is None:
            first_text_time = time.perf_counter()

        if text:
            output_parts.append(text)

        if chunk.get("done") is True:
            done_chunk = chunk
            break

    end_time = time.perf_counter()

    if done_chunk is None:
        return {"error": "No done chunk received"}

    output_text = "".join(output_parts)

    total_duration = done_chunk.get("total_duration")
    load_duration = done_chunk.get("load_duration")
    prompt_eval_count = done_chunk.get("prompt_eval_count")
    prompt_eval_duration = done_chunk.get("prompt_eval_duration")
    eval_count = done_chunk.get("eval_count")
    eval_duration = done_chunk.get("eval_duration")
    done_reason = done_chunk.get("done_reason")

    ttft_s = None
    prefill_tps = None
    decode_tps = None
    server_prefill_fabricated = False

    if first_text_time is not None:
        ttft_s = first_text_time - start_time

    prompt_tokens_arm = prompt_tokens_granite if arm == "G" else prompt_tokens_qwen

    if prompt_eval_count is not None and prompt_eval_count > 0 and ttft_s is not None and ttft_s > 0:
        prefill_tps = prompt_eval_count / ttft_s
    elif ttft_s is not None and ttft_s > 0:
        prefill_tps = prompt_tokens_arm / ttft_s

    if first_text_time is not None and eval_count is not None and eval_count > 1:
        decode_wall = end_time - first_text_time
        if decode_wall > 0:
            decode_tps = (eval_count - 1) / decode_wall

    if (prompt_eval_duration is not None and total_duration is not None
            and isinstance(prompt_eval_duration, int) and isinstance(total_duration, int)):
        expected = int(0.1 * total_duration)
        if abs(prompt_eval_duration - expected) <= 1_000_000:
            server_prefill_fabricated = True

    return {
        "ttft_s": ttft_s,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "output_text": output_text,
        "eval_count": eval_count,
        "total_duration_ns": total_duration,
        "load_duration_ns": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_ns": prompt_eval_duration,
        "eval_duration_ns": eval_duration,
        "done_reason": done_reason,
        "server_prefill_fabricated": server_prefill_fabricated,
    }


def cmd_run(args: argparse.Namespace) -> int:
    prompts_path = Path(args.prompts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with prompts_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get("prompts", [])

    arm = args.arm
    model_name = args.model
    worker_pid = args.worker_pid
    url = args.url.rstrip("/") + "/api/generate"
    repeats = args.repeats
    rest_seconds = args.rest_seconds
    ctx_limit = args.ctx_limit

    mem_before = _read_proc_status(worker_pid)

    raw_path = out_dir / f"e053-{arm}-raw.jsonl"
    host_path = out_dir / f"e053-{arm}-host.json"

    host_facts = _get_host_facts()

    with raw_path.open("w", encoding="utf-8") as raw_f:
        length_prompts = [p for p in prompts if p["kind"] == "length"]
        loop_prompts = [p for p in prompts if p["kind"] == "loop"]

        for repeat in range(1, repeats + 1):
            for p in length_prompts:
                L = p["length"]
                prompt_text = p["text"]
                tokens_qwen = p["tokens_qwen"]
                tokens_granite = p["tokens_granite"]
                prompt_tokens_arm = tokens_granite if arm == "G" else tokens_qwen
                max_new = 128

                if ctx_limit is not None and prompt_tokens_arm + max_new > ctx_limit:
                    row = {
                        "arm": arm,
                        "kind": "length",
                        "length": L,
                        "repeat": repeat,
                        "skipped": "context",
                        "utc_timestamp": time.time(),
                    }
                    raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    time.sleep(rest_seconds)
                    continue

                payload = {
                    "model": model_name,
                    "prompt": prompt_text,
                    "stream": True,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "max_new_tokens": max_new,
                    },
                }

                try:
                    with _post_json(url, payload) as resp:
                        result = _parse_ndjson_stream(resp, tokens_qwen, tokens_granite, arm)
                except Exception as e:
                    result = {"error": str(e)}

                row = {
                    "arm": arm,
                    "kind": "length",
                    "length": L,
                    "repeat": repeat,
                    "utc_timestamp": time.time(),
                }
                row.update(result)
                raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_f.flush()

                time.sleep(rest_seconds)

            for p in loop_prompts:
                idx = p["index"]
                prompt_text = p["text"]
                tokens_qwen = p["tokens_qwen"]
                tokens_granite = p["tokens_granite"]
                prompt_tokens_arm = tokens_granite if arm == "G" else tokens_qwen
                max_new = 512

                if ctx_limit is not None and prompt_tokens_arm + max_new > ctx_limit:
                    row = {
                        "arm": arm,
                        "kind": "loop",
                        "index": idx,
                        "repeat": repeat,
                        "skipped": "context",
                        "utc_timestamp": time.time(),
                    }
                    raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    time.sleep(rest_seconds)
                    continue

                payload = {
                    "model": model_name,
                    "prompt": prompt_text,
                    "stream": True,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "max_new_tokens": max_new,
                    },
                }

                try:
                    with _post_json(url, payload) as resp:
                        result = _parse_ndjson_stream(resp, tokens_qwen, tokens_granite, arm)
                except Exception as e:
                    result = {"error": str(e)}

                row = {
                    "arm": arm,
                    "kind": "loop",
                    "index": idx,
                    "repeat": repeat,
                    "utc_timestamp": time.time(),
                }
                row.update(result)
                raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_f.flush()

                time.sleep(rest_seconds)

    mem_after = _read_proc_status(worker_pid)

    host_record = {
        "arm": arm,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "host": host_facts,
    }
    with host_path.open("w", encoding="utf-8") as f:
        json.dump(host_record, f, ensure_ascii=False, indent=2)

    print(f"Run complete: {raw_path}, {host_path}")
    return 0


# =============================================================================
# Score subcommand
# =============================================================================

def _load_raw_lines(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_host(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def cmd_score(args: argparse.Namespace) -> int:
    prompts_path = Path(args.prompts)
    dir_path = Path(args.dir)
    out_path = Path(args.out)

    with prompts_path.open("r", encoding="utf-8") as f:
        json.load(f)

    arms = ["G", "QH", "Q"]
    summary = {"arms": {}, "kill_checks": {}}

    for arm in arms:
        raw_path = dir_path / f"e053-{arm}-raw.jsonl"
        host_path = dir_path / f"e053-{arm}-host.json"

        if not raw_path.exists():
            summary["arms"][arm] = {"error": "raw file not found"}
            continue

        rows = _load_raw_lines(raw_path)
        host = _load_host(host_path) if host_path.exists() else {}

        arm_summary = {"length": {}, "loop": {}, "host": host}
        skipped_count = 0
        error_count = 0

        for L in LENGTH_TOKENS:
            length_rows = [r for r in rows if r.get("kind") == "length" and r.get("length") == L]
            valid = [r for r in length_rows if "error" not in r and "skipped" not in r]
            skipped = [r for r in length_rows if r.get("skipped") == "context"]
            errors = [r for r in length_rows if "error" in r]
            skipped_count += len(skipped)
            error_count += len(errors)

            if valid:
                ttfts = [r["ttft_s"] for r in valid if r.get("ttft_s") is not None]
                prefill = [r["prefill_tps"] for r in valid if r.get("prefill_tps") is not None]
                decode = [r["decode_tps"] for r in valid if r.get("decode_tps") is not None]

                arm_summary["length"][L] = {
                    "repeats": len(valid),
                    "skipped": len(skipped),
                    "errors": len(errors),
                    "ttft_s": {
                        "median": _median(ttfts),
                        "min": min(ttfts) if ttfts else None,
                        "max": max(ttfts) if ttfts else None,
                    },
                    "prefill_tps": {
                        "median": _median(prefill),
                        "min": min(prefill) if prefill else None,
                        "max": max(prefill) if prefill else None,
                    },
                    "decode_tps": {
                        "median": _median(decode),
                        "min": min(decode) if decode else None,
                        "max": max(decode) if decode else None,
                    },
                }
            else:
                arm_summary["length"][L] = {
                    "repeats": 0,
                    "skipped": len(skipped),
                    "errors": len(errors),
                }

        loop_rows = [r for r in rows if r.get("kind") == "loop"]
        valid_loop = [r for r in loop_rows if "error" not in r and "skipped" not in r]
        skipped_loop = [r for r in loop_rows if r.get("skipped") == "context"]
        errors_loop = [r for r in loop_rows if "error" in r]
        skipped_count += len(skipped_loop)
        error_count += len(errors_loop)

        cap_hits = 0
        cap_unknown = 0

        for r in valid_loop:
            eval_count = r.get("eval_count")

            if eval_count is not None and eval_count >= LOOP_CAP_TOKENS:
                cap_hits += 1
            elif eval_count is None:
                cap_unknown += 1

        arm_summary["loop"] = {
            "total": len(loop_rows),
            "valid": len(valid_loop),
            "skipped": len(skipped_loop),
            "errors": len(errors_loop),
            "cap_hits": cap_hits,
            "cap_unknown": cap_unknown,
        }

        arm_summary["skipped_total"] = skipped_count
        arm_summary["errors_total"] = error_count
        summary["arms"][arm] = arm_summary

    g_prefill_512 = summary["arms"].get("G", {}).get("length", {}).get(512, {}).get("prefill_tps", {}).get("median")
    g_prefill_2048 = summary["arms"].get("G", {}).get("length", {}).get(2048, {}).get("prefill_tps", {}).get("median")
    qh_prefill_512 = summary["arms"].get("QH", {}).get("length", {}).get(512, {}).get("prefill_tps", {}).get("median")
    qh_prefill_2048 = summary["arms"].get("QH", {}).get("length", {}).get(2048, {}).get("prefill_tps", {}).get("median")

    g_vmhwm = summary["arms"].get("G", {}).get("host", {}).get("mem_after", {}).get("VmHWM_kB")
    qh_vmhwm = summary["arms"].get("QH", {}).get("host", {}).get("mem_after", {}).get("VmHWM_kB")

    g_cap_hits = summary["arms"].get("G", {}).get("loop", {}).get("cap_hits", 0)
    g_cap_unknown = summary["arms"].get("G", {}).get("loop", {}).get("cap_unknown", 0)

    checks = {}

    if g_prefill_512 is not None and qh_prefill_512 is not None and qh_prefill_512 > 0:
        ratio = g_prefill_512 / qh_prefill_512
        status = "PASS" if ratio >= KILL_PREFILL_RATIO else "FAIL"
    else:
        ratio = None
        status = "MISSING"
    checks["prefill_512"] = {
        "check": "G median prefill_tps at 512 >= 0.8 x QH",
        "g_value": g_prefill_512,
        "qh_value": qh_prefill_512,
        "ratio": ratio,
        "threshold": KILL_PREFILL_RATIO,
        "status": status,
    }
    print(f"Kill check 1 (prefill 512): {status} -- G={g_prefill_512}, QH={qh_prefill_512}, ratio={ratio}")

    if g_prefill_2048 is not None and qh_prefill_2048 is not None and qh_prefill_2048 > 0:
        ratio = g_prefill_2048 / qh_prefill_2048
        status = "PASS" if ratio >= KILL_PREFILL_RATIO else "FAIL"
    else:
        ratio = None
        status = "MISSING"
    checks["prefill_2048"] = {
        "check": "G median prefill_tps at 2048 >= 0.8 x QH",
        "g_value": g_prefill_2048,
        "qh_value": qh_prefill_2048,
        "ratio": ratio,
        "threshold": KILL_PREFILL_RATIO,
        "status": status,
    }
    print(f"Kill check 2 (prefill 2048): {status} -- G={g_prefill_2048}, QH={qh_prefill_2048}, ratio={ratio}")

    if g_vmhwm is not None and qh_vmhwm is not None and qh_vmhwm > 0:
        ratio = g_vmhwm / qh_vmhwm
        status = "PASS" if ratio <= KILL_RSS_RATIO else "FAIL"
    else:
        ratio = None
        status = "MISSING"
    checks["rss"] = {
        "check": "G VmHWM <= 1.5 x QH",
        "g_value": g_vmhwm,
        "qh_value": qh_vmhwm,
        "ratio": ratio,
        "threshold": KILL_RSS_RATIO,
        "status": status,
    }
    print(f"Kill check 3 (RSS): {status} -- G={g_vmhwm}, QH={qh_vmhwm}, ratio={ratio}")

    if g_cap_unknown > 0:
        status = "MISSING"
    elif g_cap_hits is not None:
        status = "PASS" if g_cap_hits <= KILL_CAP_HITS_MAX else "FAIL"
    else:
        status = "MISSING"
    checks["cap_hits"] = {
        "check": "G cap hits <= 2 of 20",
        "g_value": g_cap_hits,
        "unknown": g_cap_unknown,
        "threshold": KILL_CAP_HITS_MAX,
        "status": status,
    }
    print(f"Kill check 4 (cap hits): {status} -- G={g_cap_hits}, unknown={g_cap_unknown}")

    summary["kill_checks"] = checks

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Score written to {out_path}")
    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(prog="e053_npu_speed")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--dataset", required=True)
    p_prepare.add_argument("--out", required=True)
    p_prepare.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run")
    p_run.add_argument("--prompts", required=True)
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--arm", required=True, choices=["G", "QH", "Q"])
    p_run.add_argument("--worker-pid", required=True, type=int)
    p_run.add_argument("--url", required=True)
    p_run.add_argument("--repeats", required=True, type=int)
    p_run.add_argument("--rest-seconds", required=True, type=int)
    p_run.add_argument("--out-dir", required=True)
    p_run.add_argument("--ctx-limit", type=int, default=None)
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score")
    p_score.add_argument("--prompts", required=True)
    p_score.add_argument("--dir", required=True)
    p_score.add_argument("--out", required=True)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())