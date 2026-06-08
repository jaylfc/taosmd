"""EMem-style EDU extraction + LLM filter (vector-only variant).

Port of arXiv:2511.17208 EMem (no graph). Two functions:

  extract_session_edus(session_text, speaker_names, ...)
      One LLM call per session. Decomposes turns into atomic, self-contained
      Elementary Discourse Units (EDUs) with turn attributions.

  filter_edus(query, candidate_edus, ...)
      One LLM call per query. Selects relevant EDUs from a candidate list
      using EMem's "be maximally inclusive" prompt.

Prompts adapted from /tmp/memresearch/emem/src/emem/prompts/templates/
(conversation_edu_extraction_v1.py, edu_filter_locomo_v1.py).

Both call Ollama's /api/chat with format=json so the open-source generators
return parseable JSON without us building an explicit JSON schema for every
model. JSON-parse failure surfaces as an exception; the caller decides
whether to fallback (the LoCoMo runner does turn-level fallback at ingest
and an empty-filter fallback at retrieval).
"""

from __future__ import annotations

import json
from typing import Any

import httpx


_EXTRACTION_SYSTEM = (
    "Given a conversation session between speakers with numbered turns, your task is to decompose it into Elementary Discourse Units (EDUs) "
    "- short spans of text that are minimal yet complete in meaning. Each EDU should express a single fact, event, or "
    "proposition and be atomic (not easily divisible further while still making sense). "
    "It is important that you preserve all information from the conversation - no detail should be lost in the extraction process.\n"
    "Requirements for Conversation EDUs with Turn Attribution:\n"
    "1. Each EDU should be a self-contained unit of meaning that can be understood independently. It should not depend on any other EDU for understanding, although it may relate to it\n"
    "2. Avoid pronouns or ambiguous references - use specific names and details, and consistently use the most informative name for each entity in all EDUs\n"
    "3. The extracted EDUs must include all the information conveyed in the current conversation session. The extracted EDUs should collectively capture everything discussed\n"
    "4. For each EDU, you must provide the source_turn_ids field containing a list of turn ID integers from which the EDU was extracted or referenced (e.g., [1], [3, 5], etc.)\n"
    "5. EDUs can span multiple turns if they represent the same factual unit - in such cases, include all relevant turn IDs\n"
    "6. Focus on extracting facts, events, and substantive information rather than conversational pleasantries\n"
    "7. Infer and add complete temporal context where needed for clarity\n"
    "8. Pay attention to capturing all details, facts, decisions, concerns, and substantive information from all speakers\n"
    'Return JSON of the form {"edus": [{"edu_text": "...", "source_turn_ids": [n, ...]}, ...]}.'
)


_EXTRACTION_ONESHOT_INPUT = """Session conversation:
Date: 2:30 pm on 15 March, 2024

Turn 1:
Alice: Hey Bob! How was your trip to Tokyo?

Turn 2:
Bob: It was amazing! I spent 5 days there for the Global AI Innovation Symposium 2024. The conference at Tokyo University was incredible.

Turn 3:
Alice: That sounds exciting! What was the main focus?

Turn 4:
Bob: They had sessions on large language models and robotics. I presented our recent work on multimodal learning.

Turn 5:
Alice: How did it go?

Turn 6:
Bob: Really well! We got great feedback and Dr. Yamamoto from Sony AI wants to collaborate on our next project.

Turn 7:
Alice: That's fantastic! When are you planning to start that collaboration?

Turn 8:
Bob: We're aiming for next month. He said they have a $2 million budget for joint research.

Speaker names: Alice, Bob"""


_EXTRACTION_ONESHOT_OUTPUT = json.dumps({
    "edus": [
        {"edu_text": "Bob traveled to Tokyo for 5 days to attend the Global AI Innovation Symposium 2024 in March 2024", "source_turn_ids": [2]},
        {"edu_text": "The Global AI Innovation Symposium 2024 was held at Tokyo University in Tokyo", "source_turn_ids": [2]},
        {"edu_text": "The Global AI Innovation Symposium 2024 included sessions on large language models and robotics", "source_turn_ids": [4]},
        {"edu_text": "Bob presented his team's recent work on multimodal learning at the Global AI Innovation Symposium 2024", "source_turn_ids": [4]},
        {"edu_text": "Bob's presentation on multimodal learning at the Global AI Innovation Symposium 2024 received great feedback", "source_turn_ids": [6]},
        {"edu_text": "Dr. Yamamoto from Sony AI expressed interest in collaborating with Bob on a joint research project", "source_turn_ids": [6]},
        {"edu_text": "Bob and Dr. Yamamoto from Sony AI plan to start the Bob-Sony AI collaboration project in April 2024", "source_turn_ids": [8]},
        {"edu_text": "Dr. Yamamoto from Sony AI has a $2 million budget for the Bob-Sony AI collaboration project", "source_turn_ids": [8]},
    ]
})


_FILTER_SYSTEM = (
    "You are a conversational memory retrieval assistant helping to filter candidate memory units (EDUs - Elementary Discourse Units) for answering a user's query.\n\n"
    "Your task is to select EDUs that are relevant to answering the query. Be MAXIMALLY INCLUSIVE - err on the side of keeping too many rather than too few:\n\n"
    "**CRITICAL RULES:**\n"
    "1. **Keep EDUs with embedded information**: Even if an EDU contains extra context or discusses multiple topics, KEEP IT if ANY part is relevant to the query. Don't discard EDUs just because they're long or contain additional information.\n"
    "2. **All temporal references**: Keep EVERY EDU that mentions dates, times, durations, or events within the query's timeframe, even if buried in longer descriptions.\n"
    "3. **All quantitative information**: Keep EVERY EDU containing numbers, counts, amounts, or measurements related to the query domain.\n"
    "4. **Direct matches**: Keep ANY EDU that directly mentions entities, actions, or topics from the query.\n"
    "5. **Multi-hop reasoning**: Include EDUs that form chains of information.\n"
    "6. **Historical context**: Keep EDUs about past activities, purchases, visits, or experiences that relate to the query domain.\n"
    "7. **Prefer false positives over false negatives**: It's MUCH better to include an irrelevant EDU than to miss a relevant one. When uncertain, ALWAYS include it.\n"
    "8. **Copy exactly**: Selected EDUs must be copied exactly as they appear in the candidate list.\n\n"
    "Remember: Your job is NOT to find the perfect answer, but to keep ALL EDUs that MIGHT help answer the query. Be generous!\n\n"
    'Return JSON of the form {"selected_edus": ["edu text 1", "edu text 2", ...]}.'
)


_FILTER_ONESHOT_QUERY = "How many loyalty points does Sarah currently have at her favorite bookstore?"

_FILTER_ONESHOT_CANDIDATES = [
    "On June 10, 2023, Sarah purchased three books at Barnes & Noble and earned 45 loyalty points, bringing the total to 320 points.",
    "Sarah enjoys reading mystery novels and has been collecting books from her favorite authors.",
    "On May 15, 2023, Sarah signed up for the Barnes & Noble membership program and received 50 welcome bonus points.",
    "Sarah is considering purchasing a new bookshelf to organize her growing book collection.",
    "Barnes & Noble offers 1 point for every dollar spent on books and other items.",
    "Sarah's favorite bookstore is Barnes & Noble, where she shops regularly for books and gifts.",
    "Sarah redeemed 75 points for a discount on June 12, 2023, leaving 245 points in her account.",
    "On June 14, 2023, Sarah purchased a journal at Barnes & Noble for $25 and earned 25 points, bringing the total to 270 points.",
]

_FILTER_ONESHOT_OUTPUT = json.dumps({
    "selected_edus": [
        "On May 15, 2023, Sarah signed up for the Barnes & Noble membership program and received 50 welcome bonus points.",
        "On June 10, 2023, Sarah purchased three books at Barnes & Noble and earned 45 loyalty points, bringing the total to 320 points.",
        "Barnes & Noble offers 1 point for every dollar spent on books and other items.",
        "Sarah's favorite bookstore is Barnes & Noble, where she shops regularly for books and gifts.",
        "Sarah redeemed 75 points for a discount on June 12, 2023, leaving 245 points in her account.",
        "On June 14, 2023, Sarah purchased a journal at Barnes & Noble for $25 and earned 25 points, bringing the total to 270 points.",
    ]
})


def format_session_for_extraction(turns: list[dict], session_date: str) -> str:
    """Format LoCoMo session turns into the EMem extraction prompt input.

    Returns a numbered-turn block exactly matching the one-shot example. Turn
    IDs are 1-indexed because the prompt's exemplar uses 1-indexed turns.
    """
    lines = [f"Date: {session_date}", ""]
    for i, turn in enumerate(turns, start=1):
        lines.append(f"Turn {i}:")
        lines.append(f"{turn.get('speaker', '')}: {turn.get('text', '')}")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


async def _ollama_chat_json(
    client: httpx.AsyncClient, ollama_url: str, model: str,
    messages: list[dict], *, timeout: float = 120.0, num_predict: int = 4096,
) -> str:
    """POST to /api/chat with format=json. Returns the assistant content string."""
    resp = await client.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message") or {}).get("content", "")


async def extract_session_edus(
    session_text: str, speaker_names: list[str], *,
    model: str, ollama_url: str, http_client: httpx.AsyncClient,
    timeout: float = 240.0,
) -> list[dict]:
    """Extract EDUs from one session. Returns list of {edu_text, source_turn_ids}.

    Raises on transport / JSON-parse failure; caller decides fallback. (The
    LoCoMo runner falls back to a per-turn EDU when this raises.)
    """
    user_msg = f"Session conversation:\n{session_text}\n\nSpeaker names: {', '.join(speaker_names)}"
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": _EXTRACTION_ONESHOT_INPUT},
        {"role": "assistant", "content": _EXTRACTION_ONESHOT_OUTPUT},
        {"role": "user", "content": user_msg},
    ]

    raw = await _ollama_chat_json(
        http_client, ollama_url, model, messages,
        timeout=timeout, num_predict=8192,
    )
    parsed = json.loads(raw)
    edus: list[dict] = []
    for entry in parsed.get("edus", []) or []:
        text = (entry.get("edu_text") or "").strip()
        if not text:
            continue
        turn_ids_raw = entry.get("source_turn_ids") or []
        turn_ids: list[int] = []
        for tid in turn_ids_raw:
            try:
                turn_ids.append(int(tid))
            except (TypeError, ValueError):
                continue
        edus.append({"edu_text": text, "source_turn_ids": turn_ids})
    return edus


async def filter_edus(
    query: str, candidate_edus: list[str], *,
    model: str, ollama_url: str, http_client: httpx.AsyncClient,
    timeout: float = 120.0,
) -> list[str]:
    """Filter candidate EDUs to those relevant to the query. Exact-text match.

    Returns a subset of `candidate_edus` (preserving original order). Empty
    list on parse / transport failure rather than raising; at retrieval time
    we'd rather pass the unfiltered candidates through than crash the whole QA.
    """
    if not candidate_edus:
        return []

    user_msg = (
        f"Query: {query}\n\n"
        f"Candidate EDUs:\n{json.dumps(candidate_edus, indent=2)}"
    )
    oneshot_user = (
        f"Query: {_FILTER_ONESHOT_QUERY}\n\n"
        f"Candidate EDUs:\n{json.dumps(_FILTER_ONESHOT_CANDIDATES, indent=2)}"
    )
    messages = [
        {"role": "system", "content": _FILTER_SYSTEM},
        {"role": "user", "content": oneshot_user},
        {"role": "assistant", "content": _FILTER_ONESHOT_OUTPUT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = await _ollama_chat_json(
            http_client, ollama_url, model, messages,
            timeout=timeout, num_predict=4096,
        )
        parsed = json.loads(raw)
    except (httpx.HTTPError, json.JSONDecodeError):
        return []

    selected = parsed.get("selected_edus", []) or []
    if not selected:
        return []

    # Fuzzy-match selected -> candidate. Smaller open-source models paraphrase
    # the EDU when writing JSON (truncate, drop punctuation, slight reword)
    # often enough that exact-string lookup silently drops most candidates.
    # HippoRAG's reference uses difflib.get_close_matches with cutoff=0.0;
    # we use 0.6 to keep unrelated EDUs out on pure noise. Each selected
    # entry resolves to at most one candidate; candidates are emitted only
    # once even if the model lists them twice.
    import difflib
    keep_in_order: list[str] = []
    seen_idx: set[int] = set()
    for edu in selected:
        if not isinstance(edu, str) or not edu.strip():
            continue
        matches = difflib.get_close_matches(edu, candidate_edus, n=1, cutoff=0.6)
        if not matches:
            continue
        idx = candidate_edus.index(matches[0])
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        keep_in_order.append(candidate_edus[idx])
    return keep_in_order
