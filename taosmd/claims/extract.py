"""Turn archived text into claims with provenance.

Wraps the existing regex fact extractor; the one thing this adds is the link
back to the archive span the fact came from, which the verifier and the gate
both need. LLM-based extractors can be added later behind the same shape.

The extractor returns {subject, predicate, object} dicts; the claim text is
their natural-language join (predicate underscores become spaces).
"""
from __future__ import annotations

from taosmd.memory_extractor import extract_facts_from_text


def claims_from_text(text: str, archive_span_id: int) -> list[dict]:
    """Return claim dicts {text, archive_span_ids, source_extractor} for *text*.

    Each fact extracted from *text* is tagged with the archive row id it was
    drawn from, so a later verify-pass can fetch exactly that source span.
    """
    if not text or not text.strip():
        return []
    out = []
    for fact in extract_facts_from_text(text):
        subject = (fact.get("subject") or "").strip()
        predicate = (fact.get("predicate") or "").replace("_", " ").strip()
        obj = (fact.get("object") or "").strip()
        ftext = " ".join(p for p in (subject, predicate, obj) if p).strip()
        if ftext:
            out.append({
                "text": ftext,
                "archive_span_ids": [archive_span_id],
                "source_extractor": "regex",
            })
    return out
