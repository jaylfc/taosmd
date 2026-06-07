"""Lightweight vector memory store using QMD embeddings (taOSmd).

Stores text passages with their embeddings for semantic search.
Uses QMD's /embed endpoint for on-device NPU-accelerated embedding.
Vectors stored in SQLite for persistence — no external vector DB needed.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import sqlite3
import time
from pathlib import Path

from . import _db

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS vector_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vm_created ON vector_memory(created_at DESC);
"""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def pack_sign_bits(embedding: list[float]) -> str:
    """Pack an embedding into a base64 string of sign bits (1 bit/dim).

    Each dimension becomes a single bit (1 if >= 0, else 0), so a 768-dim
    float32 vector (3072 bytes) stores as 96 bytes — the 32x footprint of the
    binary-quant path. Decode + score with :func:`hamming_similarity`.
    """
    import numpy as np

    bits = np.packbits(np.asarray(embedding, dtype=np.float32) >= 0.0)
    return base64.b64encode(bits.tobytes()).decode("ascii")


class VectorMemory:
    """SQLite-backed vector store with pluggable embeddings.

    Supports:
    - QMD NPU embeddings (Qwen3-Embed-0.6B on RK3588)
    - sentence-transformers CPU embeddings (all-MiniLM-L6-v2 — same as MemPalace)
    """

    def __init__(
        self,
        db_path: str | Path = "data/vector-memory.db",
        qmd_url: str = "http://localhost:7832",
        embed_mode: str = "qmd",  # "qmd", "local", or "onnx"
        local_model: str = "all-MiniLM-L6-v2",
        onnx_path: str = "",
        binary_quant: bool = False,
    ):
        self._db_path = str(db_path)
        self._qmd_url = qmd_url
        self._embed_mode = embed_mode
        self._local_model_name = local_model
        self._onnx_path = onnx_path
        # Score retrieval by sign-bit Hamming similarity instead of full-precision
        # cosine. Off by default — opt-in footprint/speed option for memory- or
        # CPU-constrained (SBC) deployments: vectors are stored as packed sign
        # bits (1 bit/dim, 32x smaller) and scored by XOR+popcount. Recall-neutral
        # on LoCoMo-1540 (see docs/benchmarks.md). Must be fixed for a store's
        # lifetime — a binary-quant store persists bit blobs, not float vectors,
        # so you cannot flip an existing DB between modes.
        self.binary_quant = binary_quant
        self._conn: sqlite3.Connection | None = None
        self._http = None
        self._local_model = None
        self._onnx_session = None
        self._onnx_tokenizer = None

    async def init(self, http_client=None) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _db.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._http = http_client

        # Load local embedding model if requested
        if self._embed_mode == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(self._local_model_name)
                logger.info("Loaded local embedding model: %s", self._local_model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to QMD")
                self._embed_mode = "qmd"
        elif self._embed_mode == "onnx":
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                model_dir = self._onnx_path or "models/minilm-onnx"
                self._onnx_session = ort.InferenceSession(
                    f"{model_dir}/model.onnx",
                    providers=["CPUExecutionProvider"],
                )
                self._onnx_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                logger.info("Loaded ONNX embedding model from %s", model_dir)
            except Exception as e:
                logger.warning("ONNX model failed to load: %s, falling back to QMD", e)
                self._embed_mode = "qmd"

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def embed(self, text: str, task: str = "search_document") -> list[float]:
        """Get embedding vector.

        Args:
            text: Text to embed.
            task: Task prefix for models that support it (e.g., nomic-embed).
                  "search_document" for indexing, "search_query" for queries.
                  Ignored by models that don't use prefixes (MiniLM).
        """
        if self._embed_mode == "onnx" and self._onnx_session is not None:
            return self._embed_onnx(text, task)
        if self._embed_mode == "local" and self._local_model is not None:
            return self._embed_local(text)
        return await self._embed_qmd(text)

    def _embed_onnx(self, text: str, task: str = "search_document") -> list[float]:
        """Embed using ONNX Runtime (fast CPU inference, no PyTorch)."""
        import numpy as np
        try:
            # Add task prefix for models that use it (nomic-embed)
            # Detected by checking if the model config mentions "nomic" or has task_type support
            embed_text = text[:512]
            if self._onnx_path and "nomic" in str(self._onnx_path).lower():
                embed_text = f"{task}: {embed_text}"

            inputs = self._onnx_tokenizer(embed_text, return_tensors="np", padding=True, truncation=True)
            feed = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            # Add token_type_ids if the model expects it
            if any(inp.name == "token_type_ids" for inp in self._onnx_session.get_inputs()):
                feed["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

            outputs = self._onnx_session.run(None, feed)

            # Check if model provides sentence_embedding directly
            output_names = [o.name for o in self._onnx_session.get_outputs()]
            if "sentence_embedding" in output_names:
                idx = output_names.index("sentence_embedding")
                emb = outputs[idx][0]  # (384,)
            else:
                # Manual mean pooling
                token_embeddings = outputs[0]
                mask = inputs["attention_mask"][..., np.newaxis].astype(np.float32)
                pooled = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)
                emb = pooled[0]

            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.tolist()
        except Exception as e:
            logger.debug("ONNX embedding failed: %s", e)
            return []

    def _embed_local(self, text: str) -> list[float]:
        """Embed using local sentence-transformers model (CPU)."""
        try:
            emb = self._local_model.encode(text[:512], convert_to_numpy=True)
            return emb.tolist()
        except Exception as e:
            logger.debug("Local embedding failed: %s", e)
            return []

    async def _embed_qmd(self, text: str) -> list[float]:
        """Embed using QMD NPU."""
        if not self._http:
            return []
        try:
            resp = await self._http.post(
                f"{self._qmd_url}/embed",
                json={"text": text[:512]},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("embedding", [])
        except Exception as e:
            logger.debug("QMD embedding failed: %s", e)
        return []

    async def add(self, text: str, metadata: dict | None = None) -> int:
        """Add a text passage with its embedding. Returns row ID."""
        # Redact secrets before storage
        from .secret_filter import redact_secrets
        text, _ = redact_secrets(text)

        embedding = await self.embed(text, task="search_document")
        if not embedding:
            return -1

        # In binary-quant mode store only the packed sign bits (1 bit/dim, 32x
        # smaller) — the full-precision floats are intentionally discarded, that
        # footprint saving is the point of the mode. Cosine mode stores the
        # float vector as JSON as before.
        embedding_text = pack_sign_bits(embedding) if self.binary_quant else json.dumps(embedding)

        now = time.time()
        cursor = self._conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, ?, ?)",
            (text, embedding_text, json.dumps(metadata or {}), now),
        )
        self._conn.commit()
        return cursor.lastrowid

    async def search(
        self,
        query: str,
        limit: int = 5,
        hybrid: bool = True,
        fusion: str = "boost",
    ) -> list[dict]:
        """Semantic search with optional hybrid fusion.

        Fusion modes:
          "rrf"            — RRF over (semantic ranks, keyword-presence-
                              heuristic ranks). Legacy "hybrid" path.
          "bm25_rrf"       — RRF over (semantic ranks, proper BM25 ranks).
                              Industrial BM25 (IDF + TF saturation + length
                              norm via bm25s) on raw text.
          "bm25_lemma_rrf" — Same as bm25_rrf but BM25 indexes and queries
                              run on spaCy-lemmatised text (handles
                              meetings/meeting, attending/attend etc.).
                              ~+50 ms/query for spaCy on top of BM25.
                              Falls back to bm25_rrf if spaCy unavailable.
          "mem0_additive"  — Mem0-style additive scoring instead of RRF.
                              (semantic + sigmoid-normalised BM25 lemma) / 2.
                              Adaptive sigmoid params from query length.
                              Keeps signal magnitudes calibrated rather than
                              flattening to ranks.
          "boost"          — Legacy additive keyword boost.
          "none"           — Pure semantic cosine similarity.

        When hybrid=False, fusion mode is ignored and pure semantic is used.
        """
        query_emb = await self.embed(query, task="search_query")
        if not query_emb:
            return []

        # Extract meaningful keywords from query (3+ chars, not stop words)
        stop = {"the", "what", "how", "did", "does", "was", "were", "are", "is",
                "my", "your", "for", "and", "but", "not", "with", "this", "that",
                "from", "have", "has", "had", "been", "can", "will", "would",
                "when", "where", "which", "who", "whom", "many", "much", "long"}
        keywords = [w.lower().strip("?.,!") for w in query.split() if len(w) > 2 and w.lower() not in stop]

        # Load all embeddings and compute similarity using numpy batch operations
        rows = self._conn.execute("SELECT id, text, embedding, metadata_json, created_at FROM vector_memory").fetchall()
        if not rows:
            return []

        try:
            import numpy as np

            # Parse all stored embeddings. In binary-quant mode the column holds
            # base64 packed sign bits; in cosine mode it holds a JSON float list.
            ids = []
            texts = []
            metas = []
            created = []
            emb_list = []
            for row in rows:
                try:
                    raw = row["embedding"]
                    emb = base64.b64decode(raw) if self.binary_quant else json.loads(raw)
                    if emb:
                        ids.append(row["id"])
                        texts.append(row["text"])
                        metas.append(row["metadata_json"])
                        created.append(row["created_at"])
                        emb_list.append(emb)
                except (json.JSONDecodeError, TypeError, ValueError, base64.binascii.Error):
                    continue

            if not emb_list:
                return []

            if self.binary_quant:
                # Hamming similarity over packed sign bits. similarity =
                # 1 - popcount(query_bits XOR doc_bits) / dim, which is bit-for-bit
                # identical to the sign-agreement score ((eb·qb)/dim + 1)/2 the
                # recall-neutral LoCoMo-1540 result was measured on — packing only
                # changes storage/speed, not ranking (see docs/benchmarks.md).
                dim = len(query_emb)
                query_bits = np.packbits(np.asarray(query_emb, dtype=np.float32) >= 0.0)
                doc_bits = np.frombuffer(b"".join(emb_list), dtype=np.uint8).reshape(len(emb_list), -1)
                xor = np.bitwise_xor(doc_bits, query_bits[None, :])
                hamming = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)
                similarities = 1.0 - hamming / dim
            else:
                # Batch cosine similarity with numpy
                query_vec = np.array(query_emb, dtype=np.float32)
                emb_matrix = np.array(emb_list, dtype=np.float32)
                # Normalise
                query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
                emb_norms = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
                # Dot product = cosine similarity (both normalised)
                similarities = emb_norms @ query_norm

            if hybrid and keywords and fusion in ("rrf", "bm25_rrf", "bm25_lemma_rrf", "mem0_additive"):
                rrf_k = 60
                n = len(ids)
                semantic_ranks = np.argsort(np.argsort(-similarities))  # rank 0 = best

                if fusion in ("bm25_rrf", "bm25_lemma_rrf", "mem0_additive"):
                    # Proper BM25. bm25s tokenises by lowercasing + splitting
                    # on non-word characters; we skip stemming because spaCy
                    # lemmatisation (when fusion in {bm25_lemma_rrf,
                    # mem0_additive}) does the same job better.
                    import bm25s
                    if fusion in ("bm25_lemma_rrf", "mem0_additive"):
                        from taosmd.utils.lemmatization import lemmatize_for_bm25
                        bm25_texts = [lemmatize_for_bm25(t) for t in texts]
                        bm25_query = lemmatize_for_bm25(query)
                    else:
                        bm25_texts = texts
                        bm25_query = query
                    corpus_tokens = bm25s.tokenize(bm25_texts, stopwords=None, stemmer=None)
                    query_tokens = bm25s.tokenize([bm25_query], stopwords=None, stemmer=None)
                    retriever = bm25s.BM25(k1=1.5, b=0.75)
                    retriever.index(corpus_tokens)
                    bm25_indices, bm25_raw = retriever.retrieve(query_tokens, k=n)
                    # ranks for RRF
                    keyword_ranks = np.empty(n, dtype=np.int64)
                    for rank, idx in enumerate(bm25_indices[0]):
                        keyword_ranks[idx] = rank
                    # raw bm25 score per index, for additive scoring
                    bm25_raw_per_idx = np.zeros(n, dtype=np.float32)
                    for rank, idx in enumerate(bm25_indices[0]):
                        bm25_raw_per_idx[idx] = float(bm25_raw[0, rank])
                else:
                    # Legacy substring-presence heuristic (fusion == "rrf").
                    keyword_scores = np.zeros(n, dtype=np.float32)
                    for i, text in enumerate(texts):
                        text_lower = text.lower()
                        keyword_scores[i] = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
                    keyword_ranks = np.argsort(np.argsort(-keyword_scores))
                    bm25_raw_per_idx = None

                if fusion == "mem0_additive":
                    # Mem0-style additive scoring: sigmoid-normalise BM25 to
                    # [0, 1], add to semantic similarity, divide by 2 for
                    # max_possible=2 (no entity boost in this base impl).
                    # Threshold-gates on semantic >= 0.0 (i.e. always; we
                    # over-fetch and let the reranker handle low-similarity).
                    from taosmd.utils.scoring import get_bm25_params, normalize_bm25
                    midpoint, steepness = get_bm25_params(query, lemmatized=lemmatize_for_bm25(query) if fusion in ("bm25_lemma_rrf", "mem0_additive") else None)
                    bm25_norm = np.array(
                        [normalize_bm25(float(s), midpoint, steepness) for s in bm25_raw_per_idx],
                        dtype=np.float32,
                    )
                    additive_scores = (similarities + bm25_norm) / 2.0
                    top_indices = np.argsort(additive_scores)[::-1][:limit]
                    return [
                        {
                            "id": ids[i],
                            "text": texts[i],
                            "similarity": round(float(similarities[i]), 4),
                            "rrf_score": round(float(additive_scores[i]), 6),
                            "metadata": json.loads(metas[i]),
                            "created_at": created[i],
                        }
                        for i in top_indices
                    ]

                # RRF fusion (default for rrf, bm25_rrf, bm25_lemma_rrf)
                rrf_scores = (1.0 / (rrf_k + semantic_ranks)) + (1.0 / (rrf_k + keyword_ranks))
                top_indices = np.argsort(rrf_scores)[::-1][:limit]
                return [
                    {
                        "id": ids[i],
                        "text": texts[i],
                        "similarity": round(float(similarities[i]), 4),
                        "rrf_score": round(float(rrf_scores[i]), 6),
                        "metadata": json.loads(metas[i]),
                        "created_at": created[i],
                    }
                    for i in top_indices
                ]

            elif hybrid and keywords and fusion == "boost":
                # Legacy additive keyword boost
                for i, text in enumerate(texts):
                    text_lower = text.lower()
                    keyword_hits = sum(1 for kw in keywords if kw in text_lower)
                    boost = keyword_hits / len(keywords) * 0.3
                    similarities[i] = min(1.0, similarities[i] + boost)

            # Pure semantic or boost mode — sort by similarity
            top_indices = np.argsort(similarities)[::-1][:limit]

            return [
                {
                    "id": ids[i],
                    "text": texts[i],
                    "similarity": round(float(similarities[i]), 4),
                    "metadata": json.loads(metas[i]),
                    "created_at": created[i],
                }
                for i in top_indices
            ]
        except ImportError:
            # Fallback to Python loop if numpy not available. Binary-quant needs
            # numpy (packbits/popcount) and stores non-JSON bit blobs, so it has
            # no meaningful pure-Python path.
            if self.binary_quant:
                raise RuntimeError("binary_quant=True requires numpy for packed-bit search")
            scored = []
            for row in rows:
                try:
                    emb = json.loads(row["embedding"])
                    sim = cosine_similarity(query_emb, emb)
                    scored.append({
                        "id": row["id"],
                        "text": row["text"],
                        "similarity": round(sim, 4),
                        "metadata": json.loads(row["metadata_json"]),
                        "created_at": row["created_at"],
                    })
                except (json.JSONDecodeError, TypeError):
                    continue
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:limit]

    async def get_by_position(
        self,
        position_value: int,
        *,
        position_key: str = "position",
        group_key: str | None = None,
        group_value=None,
    ) -> dict | None:
        """Return the row whose metadata[position_key] equals position_value.

        When group_key and group_value are both supplied, also constrains
        metadata[group_key] == group_value (so neighbours of a hit can be
        confined to the same conversation/session/document).

        Used by retrieve(adjacent_neighbors=N) to inject ±N positional
        neighbours around each hit. Returns None when nothing matches.
        """
        sql = (
            "SELECT id, text, metadata_json, created_at FROM vector_memory "
            "WHERE json_extract(metadata_json, ?) = ?"
        )
        params: list = [f"$.{position_key}", position_value]
        if group_key is not None and group_value is not None:
            sql += " AND json_extract(metadata_json, ?) = ?"
            params += [f"$.{group_key}", group_value]
        sql += " LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

    async def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as n FROM vector_memory").fetchone()
        return row["n"]

    async def clear(self) -> int:
        cursor = self._conn.execute("DELETE FROM vector_memory")
        self._conn.commit()
        return cursor.rowcount
