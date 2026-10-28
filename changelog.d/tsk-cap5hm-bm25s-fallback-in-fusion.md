### Fixed
- VectorMemory.search with fusion="bm25_rrf"/"bm25_lemma_rrf"/"mem0_additive" now correctly falls back to pure-Python BM25 when bm25s is not installed, instead of silently dropping BM25 and using pure cosine ranking. The fallback logs a warning matching search_bm25.
- binary_quant and late_interaction modes no longer raise false "requires numpy" errors when bm25s is missing; the outer ImportError handler now re-raises non-numpy import errors.
- bm25_lemma_rrf and mem0_additive now properly raise ImportError when the lemmatization module is unavailable, instead of silently falling back to cosine.
- Pure-Python BM25 tie-breaking now matches bm25s (zero-score docs sorted by descending index) so RRF fusion produces identical rankings.
