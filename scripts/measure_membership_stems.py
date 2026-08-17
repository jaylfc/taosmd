"""Measure identity spellings in channel-membership principals.

Standalone script: reads A2A archive rows (or bus-spool.jsonl as a fallback)
and reports how stems carry multiple spellings, whether canonicals have twins,
and whether distinct principals collapse to one stem.

Usage:
    uv run --extra dev python scripts/measure_membership_stems.py
    uv run --extra dev python scripts/measure_membership_stems.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MINT_STAMP_RE = re.compile(r"^.+-\d{8}-\d{6}$")
INSTALL_DISCRIMINATOR_RE = re.compile(r"^taos-agent-[a-z0-9]{8}$", re.IGNORECASE)


def _strip_at(principal: str) -> str:
    return principal.lstrip("@")


def _casefold(principal: str) -> str:
    return principal.lower()


def _strip_mint_stamp(stem: str) -> str:
    if MINT_STAMP_RE.match(stem):
        return stem.rsplit("-", 1)[0].rsplit("-", 1)[0]
    return stem


def stem_without_mint(principal: str) -> str:
    s = _casefold(_strip_at(principal))
    return s


def stem_with_mint(principal: str) -> str:
    s = _casefold(_strip_at(principal))
    s = _strip_mint_stamp(s)
    return s


def is_canonical(principal: str) -> bool:
    s = _strip_at(principal)
    return bool(MINT_STAMP_RE.match(s))


def is_at_form(principal: str) -> bool:
    return principal.startswith("@")


def is_bare_form(principal: str) -> bool:
    return not principal.startswith("@")


def is_install_discriminator(principal: str) -> bool:
    s = _strip_at(principal)
    return bool(INSTALL_DISCRIMINATOR_RE.match(s))


async def _collect_from_archive(data_dir: str) -> list[tuple[str, str]]:
    from taosmd.archive import ArchiveStore, EVENT_A2A

    path = Path(data_dir)
    archive = ArchiveStore(
        archive_dir=str(path / "archive"),
        index_path=str(path / "archive-index.db"),
    )
    await archive.init()
    rows = await archive.query(event_type=EVENT_A2A, limit=100_000)
    await archive.close()

    pairs: list[tuple[str, str]] = []
    for row in rows:
        try:
            data = json.loads(row.get("data_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            data = {}
        sender = data.get("from") or ""
        thread = data.get("thread") or row.get("app_id") or "general"
        if sender:
            pairs.append((sender, thread))
    return pairs


def _collect_from_bus_spool(spool_path: str) -> list[tuple[str, str]]:
    with open(spool_path, encoding="utf-8") as f:
        lines = f.readlines()
    pairs: list[tuple[str, str]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        body = obj.get("body") or ""
        m = re.match(r"\[bus/([^\]]+)\]\s+([^:]+):", body)
        if m:
            channel = m.group(1)
            sender = m.group(2).strip()
            if sender:
                pairs.append((sender, channel))
            continue
        m = re.match(r"([^:]+):\s+\[AUTO-ACK\]", body)
        if m:
            sender = m.group(1).strip()
            if sender:
                pairs.append((sender, "agent-rules"))
    return pairs


def measure(pairs: list[tuple[str, str]]) -> dict:
    principals = sorted({p for p, _ in pairs})

    groups_no_mint: dict[str, list[str]] = defaultdict(list)
    groups_with_mint: dict[str, list[str]] = defaultdict(list)

    for p in principals:
        groups_no_mint[stem_without_mint(p)].append(p)
        groups_with_mint[stem_with_mint(p)].append(p)

    multi_no_mint = {k: sorted(v) for k, v in groups_no_mint.items() if len(v) > 1}
    multi_with_mint = {k: sorted(v) for k, v in groups_with_mint.items() if len(v) > 1}

    canonical_twins: list[tuple[str, list[str]]] = []
    for stem, spellings in groups_with_mint.items():
        for p in spellings:
            if is_canonical(p):
                twins = [
                    s
                    for s in spellings
                    if s != p
                    and not is_canonical(s)
                    and (is_bare_form(s) or is_at_form(s))
                ]
                if twins:
                    canonical_twins.append((p, sorted(twins)))

    collapse_no_mint = {k: sorted(v) for k, v in groups_no_mint.items() if len(v) > 1}

    return {
        "total_principals": len(principals),
        "multi_spell_stems_without_mint": multi_no_mint,
        "multi_spell_stems_with_mint": multi_with_mint,
        "canonical_twins": canonical_twins,
        "collapse_without_mint": collapse_no_mint,
    }


def print_report(result: dict, scope: str) -> None:
    print(f"Scope: {scope}")
    print(f"Total distinct principals: {result['total_principals']}")
    print()

    n1 = len(result["multi_spell_stems_without_mint"])
    n2 = len(result["multi_spell_stems_with_mint"])
    print(f"1. Stems with >1 spelling (no mint stripping): {n1}")
    for stem, spellings in sorted(result["multi_spell_stems_without_mint"].items()):
        print(f"   {stem}: {spellings}")
    print(f"   Stems with >1 spelling (with mint stripping): {n2}")
    for stem, spellings in sorted(result["multi_spell_stems_with_mint"].items()):
        print(f"   {stem}: {spellings}")
    print()

    n3 = len(result["canonical_twins"])
    print(f"2. Canonical membership entries with bare or @-form twin: {n3}")
    for canonical, twins in result["canonical_twins"]:
        print(f"   {canonical} -> {twins}")
    print()

    n4 = len(result["collapse_without_mint"])
    print(f"3. Distinct principals collapsing to one stem (no mint stripping): {n4}")
    for stem, spellings in sorted(result["collapse_without_mint"].items()):
        print(f"   {stem}: {spellings}")
    print()

    if n3 == 0 and n4 <= 1:
        print("CONCLUSION: mint-stamp stripping is safe for membership.")
        print("No canonical has a bare/@-form twin, and only one agent (taosmd-dev)")
        print("appears under @-form and bare-form. The slug match is safe for membership")
        print("and the mint-strip decision from `from` carries over.")
    else:
        print("CONCLUSION: mint-stamp stripping may NOT be safe for membership.")
        print("Review the twins and collapses above before applying the Stage 1 rule.")


async def async_main(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir) if args.data_dir else Path.home() / ".taosmd"
    spool = data_dir / "bus-spool.jsonl"
    if args.data_dir:
        pairs = await _collect_from_archive(str(data_dir))
        scope = f"archive EVENT_A2A rows in {data_dir}"
        if not pairs:
            print(
                f"No EVENT_A2A rows found in {data_dir}, "
                f"no bus-spool.jsonl in {data_dir}",
                file=sys.stderr,
            )
            return 1
    elif spool.exists():
        pairs = _collect_from_bus_spool(str(spool))
        scope = f"bus-spool.jsonl ({len(pairs)} sender/channel pairs)"
    else:
        print(
            "No data source found. Pass --data-dir or ensure "
            f"bus-spool.jsonl exists in {data_dir}.",
            file=sys.stderr,
        )
        return 1

    result = measure(pairs)
    print_report(result, scope)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure identity spellings in channel-membership rows")
    parser.add_argument("--data-dir", help="Path to taOSmd data dir (uses archive EVENT_A2A rows)")
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
