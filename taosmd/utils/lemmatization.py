"""BM25 lemmatization for consistent keyword matching.

Adapted from mem0ai/mem0 (mem0/utils/lemmatization.py, Apache 2.0). Uses
spaCy's lemmatizer to handle:

- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also keeps original -ing forms alongside lemmas to handle noun/verb
ambiguity (meeting vs meet) where spaCy's context-dependent
lemmatization produces inconsistent results across short turns.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_NLP = None
_NLP_TRIED = False


def _get_nlp() -> Optional[object]:
    """Lazy-load spaCy's en_core_web_sm pipeline once.

    Returns None if spaCy isn't installed or the model isn't available;
    callers should fall back gracefully.
    """
    global _NLP, _NLP_TRIED
    if _NLP_TRIED:
        return _NLP
    _NLP_TRIED = True
    try:
        import spacy

        # Disable everything except tagger + lemmatizer for speed.
        _NLP = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner", "textcat", "attribute_ruler"],
        )
    except Exception as e:
        logger.debug("spaCy/en_core_web_sm unavailable for lemmatization: %s", e)
        _NLP = None
    return _NLP


def lemmatize_for_bm25(text: str) -> str:
    """Return space-joined lemmas for BM25 matching.

    Falls back to the original text if spaCy is unavailable.
    """
    nlp = _get_nlp()
    if nlp is None:
        return text

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also keep the original if it ends in -ing and differs from
        # the lemma (handles meeting/meet noun/verb ambiguity).
        if (
            token.text.endswith("ing")
            and token.text != lemma
            and token.text.isalnum()
        ):
            tokens.append(token.text)

    return " ".join(tokens) if tokens else text


__all__ = ["lemmatize_for_bm25"]
