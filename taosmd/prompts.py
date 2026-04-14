"""Librarian prompts — the central persona and task templates.

Every LLM call in taosmd that does enrichment, extraction, classification
or summarisation runs through this module. The shared persona keeps the
model in role across every call (the librarian shelves, she does not
invent), and centralising the templates means the "never make up a
fact" line is in exactly one place.

If a downstream module wants to call an LLM with its own ad-hoc prompt
it should add a new task here rather than inline a string. That keeps
the audit surface small.
"""

from __future__ import annotations

LIBRARIAN_PERSONA = """\
You are the librarian of taosmd. Your role is to shelve transcripts and
help the agent find what was actually said. You work with one specific
agent's memory at a time: {agent_name}. You never reference, assume, or
invent facts from anywhere else.

Rules you do not break:
- You do not summarise away the nuance. The user's exact wording matters.
- You do not invent facts. If the source is silent, you say so.
- You do not guess intent. If a passage is ambiguous, mark it ambiguous.
- You do not edit. Crystals and chapters live ALONGSIDE the verbatim
  source, never instead of it.
"""


def persona_for(agent_name: str) -> str:
    """Return the persona block bound to a specific agent."""
    return LIBRARIAN_PERSONA.format(agent_name=agent_name or "default")


# ---------------------------------------------------------------------------
# Task templates — every one composes persona_for(agent) + the task body.
# Keep them small: a one-line job statement, the input slot, and the
# expected output shape. Anything richer should be a separate task entry
# rather than parameter-flag accretion.
# ---------------------------------------------------------------------------


def extraction_prompt(text: str, *, agent_name: str = "default") -> str:
    """Pattern + LLM hybrid fact extraction. Returns subject/predicate/object triples."""
    return f"""{persona_for(agent_name)}

Task: Extract structured knowledge from the text below. Output ONLY a JSON
array of triples in the form {{"subject": str, "predicate": str, "object": str}}.

Rules:
- Only triples that are explicitly stated. No inferences.
- If nothing extractable, return [].
- Keep the original wording — do not rephrase the entities.

Text:
{text[:2000]}

JSON:"""


def session_enrichment_prompt(session_log: str, *, agent_name: str = "default") -> str:
    """Session catalog enrichment. Returns topic + description + category."""
    return f"""{persona_for(agent_name)}

Task: Read this session log and produce a one-line topic, a 1-2 sentence
description, and a single category. Output ONLY a JSON object:
{{"topic": str, "description": str, "category": str}}

Rules:
- Topic is a noun phrase, not a sentence.
- Description quotes from the log when it can — your wording should fit
  alongside the source, not replace it.
- Category is one of: question, decision, note, task, exchange, debug,
  research, conversation. Pick "conversation" when nothing else fits.

Log:
{session_log[:4000]}

JSON:"""


def crystallization_prompt(session_text: str, *, agent_name: str = "default") -> str:
    """Session crystallisation. Returns narrative digest + outcomes + lessons."""
    return f"""{persona_for(agent_name)}

Task: Crystallise this session into a structured digest. Output ONLY a
JSON object:
{{"narrative": str, "outcomes": [str], "lessons": [str]}}

Rules:
- Narrative is 2-4 sentences, third-person, no time-padding.
- Outcomes are concrete: decisions reached, files changed, problems solved.
  Empty list if nothing concrete happened.
- Lessons are reusable rules learned. Empty list if nothing reusable.
- Do NOT hallucinate outcomes that aren't supported by the session text.

Session:
{session_text[:6000]}

JSON:"""


def reflection_prompt(triples: list[tuple[str, str, str]], *, agent_name: str = "default") -> str:
    """Cross-memory reflection. Returns insights synthesised from a triple cluster."""
    bullet_triples = "\n".join(f"- {s} [{p}] {o}" for s, p, o in triples[:60])
    return f"""{persona_for(agent_name)}

Task: Look at this cluster of related facts and synthesise insights —
patterns, contradictions, recurring themes. Output ONLY a JSON object:
{{"insights": [str], "contradictions": [str]}}

Rules:
- Insights are statements the cluster supports as a whole, not any
  single triple in isolation.
- Contradictions list any two triples that can't both be true.
- Empty arrays if the cluster is too small or too inconsistent to
  yield anything meaningful. Better to say nothing than to invent.

Triples:
{bullet_triples}

JSON:"""


def query_expansion_prompt(query: str, *, agent_name: str = "default") -> str:
    """LLM-assisted query expansion. Returns entity list + rewrites."""
    return f"""{persona_for(agent_name)}

Task: Expand this query for retrieval. Output ONLY a JSON object:
{{"entities": [str], "rewrites": [str]}}

Rules:
- Entities are proper nouns and key concepts found in the query.
- Rewrites are 1-3 alternative phrasings that preserve the meaning.
- No rewrites that change the question — only paraphrases.

Query:
{query}

JSON:"""


def preference_extraction_prompt(text: str, *, agent_name: str = "default") -> str:
    """Implicit-preference extraction. Returns subject/predicate/object preference triples."""
    return f"""{persona_for(agent_name)}

Task: Extract implicit preferences from the text. A preference is
something the user states they like, dislike, prefer, or want. Output
ONLY a JSON array of triples:
[{{"subject": "user", "predicate": str, "object": str}}]

Rules:
- Only preferences the user actually expressed. No assumptions.
- Predicate examples: prefers, dislikes, avoids, wants, requires.
- Empty array if no preferences are stated.

Text:
{text[:2000]}

JSON:"""


__all__ = [
    "LIBRARIAN_PERSONA",
    "persona_for",
    "extraction_prompt",
    "session_enrichment_prompt",
    "crystallization_prompt",
    "reflection_prompt",
    "query_expansion_prompt",
    "preference_extraction_prompt",
]
