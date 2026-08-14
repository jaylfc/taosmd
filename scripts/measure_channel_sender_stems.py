"""Measure distinct from-values per channel - identity spellings in A2A channel principals.

Standalone script: reads A2A archive rows via ``service.a2a_channels``
and reports how stems carry multiple spellings, whether canonicals have twins,
and whether distinct principals collapse to one stem. Channel dimension is
preserved throughout. Measures distinct ``from`` values per channel, not
channel membership (which does not exist in this system).

Usage:
    uv run --extra dev python scripts/measure_channel_sender_stems.py --data-dir /path/to/data
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


async def _collect_from_archive(data_dir: str) -> list[tuple[str, str]]:
    from taosmd.service import a2a_channels

    channels = await a2a_channels(data_dir=data_dir)
    pairs: list[tuple[str, str]] = []
    for ch in channels:
        for sender in ch["members"]:
            pairs.append((sender, ch["channel"]))
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
                twins = [s for s in spellings if s != p and (is_bare_form(s) or is_at_form(s))]
                if twins:
                    canonical_twins.append((p, sorted(twins)))

    by_channel: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for p, ch in pairs:
        by_channel[ch].append((p, ch))

    per_channel: dict[str, dict] = {}
    for ch in sorted(by_channel):
        ch_principals = sorted({p for p, _ in by_channel[ch]})
        ch_groups_no_mint: dict[str, list[str]] = defaultdict(list)
        ch_groups_with_mint: dict[str, list[str]] = defaultdict(list)
        for p in ch_principals:
            ch_groups_no_mint[stem_without_mint(p)].append(p)
            ch_groups_with_mint[stem_with_mint(p)].append(p)
        ch_multi_no_mint = {k: sorted(v) for k, v in ch_groups_no_mint.items() if len(v) > 1}
        ch_multi_with_mint = {k: sorted(v) for k, v in ch_groups_with_mint.items() if len(v) > 1}
        ch_canonical_twins: list[tuple[str, list[str]]] = []
        for stem, spellings in ch_groups_with_mint.items():
            for p in spellings:
                if is_canonical(p):
                    twins = [s for s in spellings if s != p and (is_bare_form(s) or is_at_form(s))]
                    if twins:
                        ch_canonical_twins.append((p, sorted(twins)))
        per_channel[ch] = {
            "total_principals": len(ch_principals),
            "multi_spell_stems_without_mint": ch_multi_no_mint,
            "multi_spell_stems_with_mint": ch_multi_with_mint,
            "canonical_twins": ch_canonical_twins,
        }

    return {
        "total_principals": len(principals),
        "multi_spell_stems_without_mint": multi_no_mint,
        "multi_spell_stems_with_mint": multi_with_mint,
        "canonical_twins": canonical_twins,
        "per_channel": per_channel,
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

    if n3 == 0:
        print(f"CONCLUSION [scope: {scope}]: mint-stamp stripping is safe for membership.")
        print("No canonical has a bare/@-form twin. The mint-stamp rule is safe.")
    else:
        print(f"CONCLUSION [scope: {scope}]: mint-stamp stripping is NOT safe for membership.")
        print("Review the twins above before applying the Stage 1 rule.")
    print()


async def async_main(args: argparse.Namespace) -> int:
    if not args.data_dir:
        print(
            "--data-dir is required. Pass a path to a taOSmd data dir containing "
            "EVENT_A2A rows.",
            file=sys.stderr,
        )
        return 1

    pairs = await _collect_from_archive(args.data_dir)
    scope = f"archive EVENT_A2A rows in {args.data_dir}"
    if not pairs:
        print(
            f"No EVENT_A2A rows found in {args.data_dir}",
            file=sys.stderr,
        )
        return 1

    if len({p for p, _ in pairs}) < 3:
        print(
            f"INSUFFICIENT DATA: principal count below threshold in {args.data_dir}",
            file=sys.stderr,
        )
        return 1

    result = measure(pairs)
    print_report(result, scope)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure distinct from-values per channel")
    parser.add_argument("--data-dir", required=True, help="Path to taOSmd data dir")
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())