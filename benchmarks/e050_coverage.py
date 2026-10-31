#!/usr/bin/env python3
"""E-050 judge-free evidence-coverage check for the 2000-char extraction cut."""

import argparse
import json
import os


CUT = 2000  # mirrors text[:2000] in taosmd/prompts.py extraction_prompt


def coverage(dataset: list[dict], question_ids: set[str] | None = None) -> dict:
    evidence_turns = 0
    within_cut = 0
    answer_before_cut = 0
    answer_after_cut = 0
    answer_not_verbatim = 0
    question_ids_with_answer_after_cut = []

    for q in dataset:
        qid = q.get("question_id")
        if question_ids is not None and qid not in question_ids:
            continue

        answer = q.get("answer", "")
        q_answer_after_cut = False

        haystack = q.get("haystack_sessions", []) or []
        for session in haystack:
            if not isinstance(session, list):
                continue
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                if not turn.get("has_answer"):
                    continue
                content = str(turn.get("content", "") or "")
                evidence_turns += 1
                if len(content) <= CUT:
                    within_cut += 1
                    continue
                # Long turn - search case-insensitively for answer
                answer_str = str(answer) if answer is not None else ""
                if not answer_str:
                    answer_not_verbatim += 1
                    continue
                content_lower = content.lower()
                answer_lower = answer_str.lower()
                idx = content_lower.find(answer_lower)
                if idx == -1:
                    answer_not_verbatim += 1
                    continue
                # Found - check if it ends within cut
                end_pos = idx + len(answer_str)
                if end_pos <= CUT:
                    answer_before_cut += 1
                else:
                    answer_after_cut += 1
                    q_answer_after_cut = True

        if q_answer_after_cut and qid is not None:
            question_ids_with_answer_after_cut.append(qid)

    if evidence_turns == 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = (within_cut + answer_before_cut) / evidence_turns

    return {
        "evidence_turns": evidence_turns,
        "within_cut": within_cut,
        "answer_before_cut": answer_before_cut,
        "answer_after_cut": answer_after_cut,
        "answer_not_verbatim": answer_not_verbatim,
        "coverage": coverage_ratio,
        "question_ids_with_answer_after_cut": question_ids_with_answer_after_cut,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "data", "longmemeval_oracle.json"),
    )
    parser.add_argument("--ids-file", help="optional, one question_id per line", default=None)
    parser.add_argument("--out", help="optional JSON path", default=None)
    args = parser.parse_args(argv)

    with open(args.data, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    question_ids = None
    if args.ids_file:
        question_ids = set()
        with open(args.ids_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    question_ids.add(line)

    result = coverage(dataset, question_ids=question_ids)
    result_json = json.dumps(result)
    print(result_json)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result_json)


if __name__ == "__main__":
    main()
