"""Turn archived text into claims with provenance.

Wraps the existing regex fact extractor; the one thing this adds is the link
back to the archive span the fact came from, which the verifier and the gate
both need. LLM-based extractors can be added later behind the same shape.

The extractor returns {subject, predicate, object} dicts; the claim text is
their natural-language join (predicate underscores become spaces).
"""
from __future__ import annotations

import hashlib

from taosmd import memory_extractor
from taosmd.memory_extractor import extract_facts_from_text

REGEX_EXTRACTOR_VERSION = 1


def regex_pattern_source() -> str:
    """Stable string over the regex extraction behaviour for hashing/versioning."""
    parts = []
    for pattern, predicate in memory_extractor.RELATIONSHIP_PATTERNS:
        src = pattern.pattern if hasattr(pattern, "pattern") else str(pattern)
        parts.append(f"{src}\t{predicate}")
    # Regex extract_facts_from_text also applies meta/split/cleanup logic; include the
    # stable markers for the filters it consults (lines ~113-150 in memory_extractor).
    parts.append("SKIP_WORDS=" + ",".join(sorted(memory_extractor.SKIP_WORDS)))
    parts.append("BAD_OBJ_FIRST=" + ",".join(sorted(memory_extractor._BAD_OBJ_FIRST)))
    parts.append("BAD_OBJ_LAST=" + ",".join(sorted(memory_extractor._BAD_OBJ_LAST)))
    parts.append("VAGUE_HEAD=" + ",".join(sorted(memory_extractor._VAGUE_HEAD)))
    parts.append("CLAUSE_INITIAL=" + ",".join(sorted(memory_extractor._CLAUSE_INITIAL)))
    parts.append("META_TOKENS=" + ",".join(memory_extractor._META_TOKENS))
    parts.append("META_PHRASES=" + ",".join(memory_extractor._META_PHRASES))
    return "\n".join(parts)


REGEX_PATTERN_HASH = hashlib.sha256(regex_pattern_source().encode()).hexdigest()


def regex_extractor_id() -> str:
    return f"regex@{REGEX_EXTRACTOR_VERSION}"


def llm_extractor_id(model: str, prompt_template: str) -> str:
    return f"llm:{model}:{hashlib.sha256(prompt_template.encode()).hexdigest()[:12]}"


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
                "source_extractor": regex_extractor_id(),
            })
    return out
