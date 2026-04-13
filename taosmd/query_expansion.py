"""Query Expansion (taOSmd).

Expands user queries into multiple reformulations before search,
improving recall by bridging vocabulary gaps. Two modes:

1. Fast regex-based entity extraction (free, ~1ms)
2. LLM-powered full expansion with reformulations + temporal
   concretization (requires Ollama/rkllama, ~2s)
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Temporal references to resolve into date ranges
TEMPORAL_REFS = {
    "today": lambda: (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0), timedelta(days=1)),
    "yesterday": lambda: (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0) - timedelta(days=1), timedelta(days=1)),
    "this week": lambda: (datetime.now(timezone.utc) - timedelta(days=datetime.now(timezone.utc).weekday()), timedelta(days=7)),
    "last week": lambda: (datetime.now(timezone.utc) - timedelta(days=datetime.now(timezone.utc).weekday() + 7), timedelta(days=7)),
    "this month": lambda: (datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0), timedelta(days=30)),
    "last month": lambda: (datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0) - timedelta(days=30), timedelta(days=30)),
    "recently": lambda: (datetime.now(timezone.utc) - timedelta(days=7), timedelta(days=7)),
}


def extract_entities_regex(query: str) -> list[str]:
    """Fast entity extraction using capitalization and pattern heuristics.

    Finds: proper nouns, quoted strings, CamelCase, ACRONYMS, file paths.
    """
    entities: list[str] = []
    seen = set()

    # Quoted strings
    for match in re.finditer(r'"([^"]+)"', query):
        val = match.group(1).strip()
        if val.lower() not in seen:
            entities.append(val)
            seen.add(val.lower())

    # Capitalised multi-word names (e.g., "Orange Pi", "Jay Smith")
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
        val = match.group(1).strip()
        if val.lower() not in seen:
            entities.append(val)
            seen.add(val.lower())

    # Single capitalised words (not at sentence start, not common words)
    common_caps = {"I", "The", "What", "When", "Where", "How", "Why", "Who",
                   "Which", "Can", "Do", "Does", "Did", "Is", "Are", "Was",
                   "Were", "Has", "Have", "Had", "Will", "Would", "Could",
                   "Should", "May", "Might", "A", "An", "In", "On", "At",
                   "To", "For", "Of", "And", "Or", "But", "Not", "If", "My"}
    words = query.split()
    for i, word in enumerate(words):
        clean = word.strip("?.,!:;'\"()")
        if (clean and clean[0].isupper() and clean not in common_caps
                and i > 0 and clean.lower() not in seen):
            entities.append(clean)
            seen.add(clean.lower())

    # CamelCase identifiers
    for match in re.finditer(r"\b([a-z]+(?:[A-Z][a-z]+)+|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", query):
        val = match.group(1)
        if val.lower() not in seen:
            entities.append(val)
            seen.add(val.lower())

    # ACRONYMS (2+ uppercase letters)
    for match in re.finditer(r"\b([A-Z]{2,})\b", query):
        val = match.group(1)
        if val not in common_caps and val.lower() not in seen:
            entities.append(val)
            seen.add(val.lower())

    # File paths
    for match in re.finditer(r"(?:~/|/|\./)[\w\-./]+", query):
        val = match.group(0)
        if val.lower() not in seen:
            entities.append(val)
            seen.add(val.lower())

    return entities


def resolve_temporal_refs(query: str) -> dict | None:
    """Resolve temporal references in query to date ranges.

    Returns {start: float, end: float, ref: str} or None.
    """
    query_lower = query.lower()
    for ref, resolver in TEMPORAL_REFS.items():
        if ref in query_lower:
            start_dt, duration = resolver()
            end_dt = start_dt + duration
            return {
                "start": start_dt.timestamp(),
                "end": end_dt.timestamp(),
                "ref": ref,
            }
    return None


def expand_query_fast(query: str) -> dict:
    """Fast query expansion using regex only (no LLM).

    Returns {
        original: str,
        entities: list[str],
        temporal: dict | None,
        keywords: list[str],
    }
    """
    entities = extract_entities_regex(query)
    temporal = resolve_temporal_refs(query)

    # Extract meaningful keywords (lowercase, 3+ chars, not stop words)
    stop = {"the", "what", "how", "did", "does", "was", "were", "are", "is",
            "my", "your", "for", "and", "but", "not", "with", "this", "that",
            "from", "have", "has", "had", "been", "can", "will", "would",
            "when", "where", "which", "who", "whom", "many", "much", "long",
            "about", "tell", "remember", "know", "think", "said", "told"}
    keywords = [w.lower().strip("?.,!:;'\"()") for w in query.split()
                if len(w) > 2 and w.lower().strip("?.,!:;'\"()") not in stop]

    return {
        "original": query,
        "entities": entities,
        "temporal": temporal,
        "keywords": list(dict.fromkeys(keywords)),  # dedupe, preserve order
    }


async def expand_query_llm(
    query: str,
    llm_url: str = "http://localhost:11434",
    model: str = "qwen3:4b",
) -> dict:
    """Full LLM-powered query expansion.

    Generates 3-5 reformulations, extracts entities, resolves temporal refs.
    Falls back to fast expansion on LLM failure.
    """
    fast = expand_query_fast(query)

    try:
        import httpx

        prompt = f"""Given this search query, generate 3-5 alternative phrasings that might help find the answer in a memory system. Also extract any named entities.

Query: {query}

Respond in this exact format:
REFORMULATIONS:
1. <rephrasing 1>
2. <rephrasing 2>
3. <rephrasing 3>
ENTITIES:
- <entity 1>
- <entity 2>"""

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{llm_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.3, "num_predict": 200}},
            )
            if resp.status_code != 200:
                return fast

            text = resp.json().get("response", "")

            # Parse reformulations
            reformulations = []
            in_reformulations = False
            for line in text.split("\n"):
                line = line.strip()
                if "REFORMULATIONS:" in line.upper():
                    in_reformulations = True
                    continue
                if "ENTITIES:" in line.upper():
                    in_reformulations = False
                    continue
                if in_reformulations and line:
                    # Strip numbering
                    cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
                    if cleaned and cleaned != query:
                        reformulations.append(cleaned)

            # Parse LLM entities and merge with regex entities
            llm_entities = []
            in_entities = False
            for line in text.split("\n"):
                line = line.strip()
                if "ENTITIES:" in line.upper():
                    in_entities = True
                    continue
                if in_entities and line.startswith("-"):
                    entity = line.lstrip("- ").strip()
                    if entity:
                        llm_entities.append(entity)

            # Merge
            all_entities = list(dict.fromkeys(fast["entities"] + llm_entities))

            return {
                "original": query,
                "reformulations": reformulations[:5],
                "entities": all_entities,
                "temporal": fast["temporal"],
                "keywords": fast["keywords"],
                "method": "llm",
            }

    except Exception as e:
        logger.debug("LLM query expansion failed: %s", e)
        return fast
