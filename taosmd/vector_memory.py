"""Lightweight vector memory store using QMD embeddings (taOSmd).

Stores text passages with their embeddings for semantic search.
Uses QMD's /embed endpoint for on-device NPU-accelerated embedding.
Vectors stored in SQLite for persistence; no external vector DB needed.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path

from . import _db

logger = logging.getLogger(__name__)

# Default token embedding dimension (all-MiniLM-L6-v2 backbone). Kept as a
# module-level constant for backward-compat imports in existing tests.
# Inside VectorMemory, prefer self._token_dim which may differ for pylate models.
_EMBED_DIM = 384

SCHEMA = """
CREATE TABLE IF NOT EXISTS vector_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    valid_to REAL
);
CREATE INDEX IF NOT EXISTS idx_vm_created ON vector_memory(created_at DESC);
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StoreModeMismatch(Exception):
    """Raised when a store is reopened in a different storage format than it
    was built in (dense vs late-interaction vs binary-quant).

    The storage formats are mutually incompatible (a single pooled vector, a
    per-token matrix, and packed sign bits cannot be scored against each
    other), so the fix is never to silently serve wrong-mode results. The
    archive is the zero-loss source, so the resolution is always to re-embed
    the store from the archive in the new mode.
    """


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _bm25_python_rank(query: str, texts: list[str]) -> list[tuple[int, float]]:
    """Dependency-free Okapi BM25; best-first (index, score) over ``texts``.

    Fallback for :meth:`VectorMemory.search_bm25` when bm25s is not
    installed. Matches the bm25s defaults the fusion paths were benchmarked
    with: k1=1.5, b=0.75, Lucene IDF, lowercase + non-word-split tokens.
    """
    import math
    import re

    k1, b = 1.5, 0.75
    docs = [re.findall(r"\w+", t.lower()) for t in texts]
    n = len(docs)
    avgdl = (sum(len(d) for d in docs) / n) if n else 0.0
    avgdl = avgdl or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for term in set(d):
            df[term] = df.get(term, 0) + 1

    q_terms = re.findall(r"\w+", query.lower())
    scored = []
    for i, d in enumerate(docs):
        tf: dict[str, int] = {}
        for term in d:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(d) / avgdl))
        scored.append((i, score))
    scored.sort(key=lambda pair: -pair[1])
    return scored


def pack_sign_bits(embedding: list[float]) -> str:
    """Pack an embedding into a base64 string of sign bits (1 bit/dim).

    Each dimension becomes a single bit (1 if >= 0, else 0), so a 768-dim
    float32 vector (3072 bytes) stores as 96 bytes, the 32x footprint of the
    binary-quant path. Decode + score with :func:`hamming_similarity`.
    """
    import numpy as np

    bits = np.packbits(np.asarray(embedding, dtype=np.float32) >= 0.0)
    return base64.b64encode(bits.tobytes()).decode("ascii")


# Query-time prefix for arctic-embed (asymmetric: queries only, no doc prefix).
_ARCTIC_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _onnx_apply_prefix(onnx_path, text: str, task: str) -> str:
    """Apply a model-specific input prefix to ONNX embed text.

    Asymmetric embedders need the right prefix or retrieval silently
    degrades. nomic-embed prefixes both queries and documents with the
    task name; arctic-embed prefixes only the query. MiniLM and anything
    else gets no prefix. Detection is by model directory name.
    """
    # Env override (E-010 bake-off: let any model set its prefixes explicitly,
    # independent of pooling and of directory-name detection). Active when
    # either var is present; empty string means "no prefix on that side".
    if "TAOSMD_ONNX_QUERY_PREFIX" in os.environ or "TAOSMD_ONNX_DOC_PREFIX" in os.environ:
        if task == "search_query":
            return f"{os.environ.get('TAOSMD_ONNX_QUERY_PREFIX', '')}{text}"
        return f"{os.environ.get('TAOSMD_ONNX_DOC_PREFIX', '')}{text}"
    path = str(onnx_path).lower() if onnx_path else ""
    if "nomic" in path:
        return f"{task}: {text}"
    if "arctic" in path and task == "search_query":
        return f"{_ARCTIC_QUERY_PREFIX}{text}"
    return text


def _onnx_pooling_mode(onnx_path) -> str:
    """Pooling mode for an ONNX embedder: "cls" or "mean".

    arctic-embed pools the CLS token (its 1_Pooling config sets
    pooling_mode_cls_token true, mean false); the MiniLM default and
    everything else mean-pool over the attention mask.
    """
    pm = os.environ.get("TAOSMD_ONNX_POOLING")
    if pm in ("cls", "mean"):
        return pm
    path = str(onnx_path).lower() if onnx_path else ""
    if "arctic" in path:
        return "cls"
    return "mean"


class VectorMemory:
    """SQLite-backed vector store with pluggable embeddings.

    Supports:
    - QMD NPU embeddings (Qwen3-Embed-0.6B on RK3588)
    - sentence-transformers CPU embeddings (all-MiniLM-L6-v2, same as MemPalace)
    """

    def __init__(
        self,
        db_path: str | Path = "data/vector-memory.db",
        qmd_url: str = "http://localhost:7832",
        embed_mode: str = "qmd",  # "qmd", "local", or "onnx"
        local_model: str = "all-MiniLM-L6-v2",
        onnx_path: str = "",
        binary_quant: bool = False,
        late_interaction: bool = False,
        colbert_model: str = "",
    ):
        self._db_path = str(db_path)
        self._qmd_url = qmd_url
        self._onnx_path = onnx_path
        # When a ColBERT model is specified, route through the local (sentence-
        # transformers) embed path and enable late-interaction MaxSim scoring.
        self._colbert_model = colbert_model
        if colbert_model:
            embed_mode = "local"
            local_model = colbert_model
            late_interaction = True
        self._embed_mode = embed_mode
        self._local_model_name = local_model
        # Score retrieval by sign-bit Hamming similarity instead of full-precision
        # cosine. Off by default; opt-in footprint/speed option for memory- or
        # CPU-constrained (SBC) deployments: vectors are stored as packed sign
        # bits (1 bit/dim, 32x smaller) and scored by XOR+popcount. Recall-neutral
        # on LoCoMo-1540 (see docs/benchmarks.md). Must be fixed for a store's
        # lifetime; a binary-quant store persists bit blobs, not float vectors,
        # so you cannot flip an existing DB between modes.
        self.binary_quant = binary_quant
        # Score retrieval by token-level MaxSim (ColBERT-style late interaction)
        # instead of full-precision pooled cosine. Off by default; opt-in
        # benchmark lever: the MiniLM ONNX embedder already computes per-token
        # vectors before mean-pooling, so this stores the full token matrix
        # (seq_len x 384 float16) per memory and scores by
        # mean_q(max_t(q_t . d_t)). Far larger footprint than pooled cosine (a
        # whole matrix per row, not one vector). Mutually exclusive with
        # binary_quant. Shipped on LoCoMo-1540 (see docs/benchmarks.md).
        self.late_interaction = late_interaction
        if self.binary_quant and self.late_interaction:
            raise ValueError(
                "binary_quant and late_interaction are mutually exclusive"
            )
        self._conn: sqlite3.Connection | None = None
        self._http = None
        self._local_model = None
        self._pylate_model = None
        self._onnx_session = None
        self._onnx_tokenizer = None
        # Per-instance token embedding dimension. Default matches the MiniLM
        # backbone (384). Set to the model's actual projected dim at init time
        # when a pylate ColBERT model is loaded (e.g. 128 for colbertv2.0).
        # All token-blob storage and reshape sites use this, not _EMBED_DIM.
        self._token_dim: int = _EMBED_DIM
        # BM25 index cache: rebuilt lazily and invalidated whenever the active
        # corpus changes (add / supersede).  Keyed by fusion mode because
        # bm25_rrf and bm25_lemma_rrf index different text forms.
        # ``_bm25_cache`` maps mode → (retriever, texts, ids) tuple.
        # ``_bm25_dirty`` is set on every write so the next query rebuilds.
        self._bm25_cache: dict[str, tuple] = {}
        self._bm25_dirty: bool = True

    async def init(self, http_client=None) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = _db.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._check_store_mode()
        self._conn.commit()
        self._http = http_client

        # Load local embedding model if requested
        if self._embed_mode == "local":
            if self._colbert_model:
                # Try the pylate path first so the trained projection head is
                # applied. Fall back to sentence-transformers with a warning if
                # pylate is not installed.
                pylate_loaded = False
                try:
                    from pylate import models as pylate_models  # type: ignore[import-untyped]
                    self._pylate_model = pylate_models.ColBERT(
                        model_name_or_path=self._colbert_model
                    )
                    # Determine the projected output dimension by encoding a short
                    # probe string. pylate returns (1, seq, dim); take the last axis.
                    import numpy as np
                    probe_out = self._pylate_model.encode(
                        ["probe"],
                        is_query=False,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    # probe_out may be a list of arrays or a 3-D array.
                    probe_arr = np.asarray(probe_out[0])  # (seq, dim)
                    self._token_dim = int(probe_arr.shape[-1])
                    logger.info(
                        "Loaded ColBERT model via pylate: %s (token dim=%d)",
                        self._colbert_model,
                        self._token_dim,
                    )
                    pylate_loaded = True
                except ImportError:
                    logger.warning(
                        "pylate not installed; falling back to sentence-transformers "
                        "for %s. The projection head will NOT be applied and token "
                        "vectors will be backbone-dim. Run: pip install pylate",
                        self._colbert_model,
                    )
                except Exception as exc:
                    logger.warning(
                        "pylate ColBERT load failed (%s); falling back to "
                        "sentence-transformers for %s",
                        exc,
                        self._colbert_model,
                    )

                if not pylate_loaded:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._local_model = SentenceTransformer(self._colbert_model)
                        logger.info(
                            "Loaded ColBERT model via sentence-transformers (fallback): %s",
                            self._colbert_model,
                        )
                    except ImportError:
                        logger.warning(
                            "sentence-transformers not installed, falling back to QMD"
                        )
                        self._embed_mode = "qmd"
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._local_model = SentenceTransformer(self._local_model_name)
                    logger.info("Loaded local embedding model: %s", self._local_model_name)
                except ImportError:
                    logger.warning("sentence-transformers not installed, falling back to QMD")
                    self._embed_mode = "qmd"
        elif self._embed_mode == "onnx":
            # Resolved before the try so the failure path can name it.
            model_dir = self._onnx_path or "models/minilm-onnx"
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                # Find model.onnx at the dir root or under an onnx/ subdir
                # (the layout arctic-embed and most HF ONNX exports ship).
                model_file = f"{model_dir}/model.onnx"
                if not Path(model_file).exists() and Path(f"{model_dir}/onnx/model.onnx").exists():
                    model_file = f"{model_dir}/onnx/model.onnx"
                self._onnx_session = ort.InferenceSession(
                    model_file,
                    providers=["CPUExecutionProvider"],
                )
                self._onnx_tokenizer = AutoTokenizer.from_pretrained(model_dir)
                logger.info("Loaded ONNX embedding model from %s", model_file)
            except Exception as e:
                # Loud and actionable: a silent embedder swap corrupts retrieval.
                # The store's vectors are tied to the embedder that wrote them, so
                # falling back to a different embedder than the store was built
                # with returns meaningless results. Name the dir and the fix.
                logger.warning(
                    "ONNX embedder failed to load from %s (%s). Falling back to QMD "
                    "remote embeddings. If a local embedder was expected, its ONNX "
                    "files are missing or unreadable; run scripts/setup.sh to fetch "
                    "them. NOTE: a store's vectors are tied to the embedder that "
                    "wrote them, so serving a different embedder than the store was "
                    "built with returns meaningless retrieval.",
                    model_dir, e,
                )
                self._embed_mode = "qmd"

    def _migrate(self) -> None:
        """Bring an existing DB up to the current schema without data loss.

        Adds the nullable ``valid_to`` column to stores created before the
        correction-supersede feature. ``ALTER TABLE ... ADD COLUMN`` only
        appends a NULL-defaulted column; no rows are rewritten or dropped, so
        every existing vector survives and stays active (valid_to IS NULL).
        """
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(vector_memory)")}
        if "valid_to" not in cols:
            self._conn.execute("ALTER TABLE vector_memory ADD COLUMN valid_to REAL")
        # Index creation is deferred to here (not in SCHEMA) so it runs *after*
        # the column is guaranteed to exist; a legacy table only gains the
        # column via the ALTER above, and CREATE INDEX in SCHEMA would fire
        # before that on the no-op CREATE TABLE IF NOT EXISTS path.
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_valid ON vector_memory(valid_to)")

    def _store_mode_signature(self) -> str:
        """The storage-format and vector-space mode of this instance.

        Two kinds of dimension belong here. Format: late_interaction (per-token
        matrix vs single pooled vector) and binary_quant (packed bits vs float).
        Space: the embedder identity, because two embedders (e.g. MiniLM vs
        arctic-embed) produce vectors in incompatible spaces even at the same
        dim, so a stored MiniLM vector is meaningless to an arctic query and the
        store must be re-embedded on a switch. embed_mode (qmd/local/onnx) is
        NOT included: those are transports for the same model, interoperable.
        """
        return (
            f"late_interaction={int(self.late_interaction)};"
            f"binary_quant={int(self.binary_quant)};"
            f"embedder={self._embedder_identity()}"
        )

    def _embedder_identity(self) -> str:
        """A stable name for the embedding model, machine-path independent.

        Uses the onnx model directory's basename (e.g. arctic-embed-s,
        minilm-onnx) in onnx mode, else the local model name, else the qmd
        default. Absolute paths are reduced to the dir name so the same model
        compares equal across machines.
        """
        if self._colbert_model:
            return self._colbert_model
        if self._embed_mode == "onnx" and self._onnx_path:
            return Path(self._onnx_path).name
        if self._embed_mode == "local":
            return self._local_model_name
        return f"qmd:{self._embed_mode}"

    @staticmethod
    def _parse_mode(sig: str) -> dict:
        """Parse a mode signature string into a key->value dict."""
        out = {}
        for part in sig.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k] = v
        return out

    def _check_store_mode(self) -> None:
        """Refuse to open a store in a different format or vector space than it
        holds.

        Writes the mode marker on a fresh store. On reopen, the comparison is
        component-wise over only the keys the recorded marker actually carries,
        so an older marker (e.g. one written before the embedder component was
        added) is honoured for what it knew and then upgraded to the full
        current signature, never spuriously rejected. A legacy store with rows
        and no marker is assumed dense MiniLM (the historical default).
        """
        current = self._store_mode_signature()
        cur = self._parse_mode(current)
        row = self._conn.execute(
            "SELECT value FROM store_meta WHERE key = 'mode'"
        ).fetchone()
        recorded = row["value"] if row else None

        if recorded is not None:
            rec = self._parse_mode(recorded)
            conflict = {k: v for k, v in rec.items() if cur.get(k) != v}
            if conflict:
                raise StoreModeMismatch(
                    f"vector store at {self._db_path} was built with {recorded} "
                    f"but is being opened with {current} (conflict on {conflict}). "
                    f"The storage format or embedding space is incompatible. "
                    f"Re-embed from the archive in the new mode (the archive is "
                    f"zero-loss) rather than mixing them."
                )
            if recorded != current:  # compatible but stale (e.g. legacy marker), upgrade
                self._conn.execute(
                    "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)",
                    (current,),
                )
            return

        # No marker: fresh store, or a legacy store predating the marker. Only
        # the FORMAT is enforceable here (no serve store was ever built in
        # late-interaction or binary-quant mode, so existing rows are dense);
        # the legacy store's embedder is unknowable, so it is not enforced, the
        # marker we write now protects future embedder switches.
        has_rows = self._conn.execute(
            "SELECT 1 FROM vector_memory LIMIT 1"
        ).fetchone() is not None
        legacy_format = {"late_interaction": "0", "binary_quant": "0"}
        if has_rows and any(cur.get(k) != v for k, v in legacy_format.items()):
            raise StoreModeMismatch(
                f"vector store at {self._db_path} has existing rows and no mode "
                f"marker, so its format is assumed dense, but is being opened with "
                f"{current}. Re-embed from the archive in the new mode rather than "
                f"mixing storage formats."
            )
        self._conn.execute(
            "INSERT OR REPLACE INTO store_meta (key, value) VALUES ('mode', ?)",
            (current,),
        )

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
            # Apply any model-specific input prefix (asymmetric embedders).
            embed_text = _onnx_apply_prefix(self._onnx_path, text[:512], task)

            inputs = self._onnx_tokenizer(embed_text, return_tensors="np", padding=True, truncation=True)
            feed = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            # Add token_type_ids if the model expects it
            if any(inp.name == "token_type_ids" for inp in self._onnx_session.get_inputs()):
                feed["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

            outputs = self._onnx_session.run(None, feed)

            # Pool. A model that exposes a ready sentence_embedding wins; else
            # pool per the model's documented mode (CLS for arctic-embed, mean
            # otherwise). Getting this wrong silently degrades a model, so it
            # is selected explicitly, not guessed.
            output_names = [o.name for o in self._onnx_session.get_outputs()]
            if "sentence_embedding" in output_names:
                idx = output_names.index("sentence_embedding")
                emb = outputs[idx][0]
            elif _onnx_pooling_mode(self._onnx_path) == "cls":
                # CLS-token pooling: the first token of the last hidden state.
                emb = outputs[0][0, 0]
            else:
                # Mean pooling over the attention mask.
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

    async def embed_tokens(self, text: str, task: str = "search_document") -> list[list[float]]:
        """Return per-token L2-normalised embeddings (ColBERT-style).

        Supports three backends:
        - pylate ColBERT (preferred when colbert_model is set): applies the
          trained projection head, returning vectors in the model's actual
          trained space (e.g. 128-dim for colbertv2.0).
        - sentence-transformers fallback: uses output_value="token_embeddings"
          which returns the backbone's token output BEFORE any projection.
        - ONNX (MiniLM): extracts per-token hidden states before mean-pooling.
        """
        is_query = (task == "search_query")
        if self._embed_mode == "local" and self._colbert_model:
            if self._pylate_model is not None:
                result = self._embed_tokens_pylate(text, is_query=is_query)
            elif self._local_model is not None:
                result = self._embed_tokens_st(text)
            else:
                result = []
            if result:
                got_dim = len(result[0])
                if got_dim != self._token_dim:
                    raise RuntimeError(
                        f"late_interaction token dim {got_dim} does not match "
                        f"store _token_dim {self._token_dim}. Mixed-dim corpora "
                        "are not supported within one store instance."
                    )
            return result
        if not (self._embed_mode == "onnx" and self._onnx_session is not None):
            raise RuntimeError("late_interaction requires embed_mode=onnx or a ColBERT model")

        import numpy as np

        # Add task prefix for models that use it (nomic-embed) — same convention
        # as _embed_onnx.
        embed_text = text[:512]
        if self._onnx_path and "nomic" in str(self._onnx_path).lower():
            embed_text = f"{task}: {embed_text}"

        inputs = self._onnx_tokenizer(embed_text, return_tensors="np", padding=True, truncation=True)
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if any(inp.name == "token_type_ids" for inp in self._onnx_session.get_inputs()):
            feed["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)

        outputs = self._onnx_session.run(None, feed)

        # outputs[0] is the per-token hidden states (1, seq, dim).
        token_embeddings = np.asarray(outputs[0][0], dtype=np.float32)  # (seq, dim)
        if token_embeddings.shape[-1] != self._token_dim:
            # The stored blobs are reshaped as (-1, self._token_dim) at search
            # time, so a wrong-dim model would silently write garbage and
            # then fail every query (this is exactly how the colbertv2.0
            # probe died: ST drops its 768->128 projection head). Fail at
            # the first embed instead.
            raise RuntimeError(
                f"late_interaction expects {self._token_dim}-dim token output, got "
                f"{token_embeddings.shape[-1]} from {self._onnx_path or self._colbert_model!r}. "
                "ColBERT checkpoints with a projection head (e.g. colbertv2.0) "
                "need a pylate-style loader, not a bare transformer export."
            )
        mask = np.asarray(inputs["attention_mask"][0], dtype=bool)      # (seq,)
        token_embeddings = token_embeddings[mask]                       # drop padding
        # L2-normalise each token vector so dot product == cosine for MaxSim.
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        token_embeddings = token_embeddings / (norms + 1e-8)
        return token_embeddings.tolist()

    def _embed_local(self, text: str) -> list[float]:
        """Embed using local sentence-transformers model (CPU)."""
        try:
            emb = self._local_model.encode(text[:512], convert_to_numpy=True)
            if hasattr(emb, "tolist"):
                return emb.tolist()
            import numpy as np
            return np.mean(emb, axis=0).tolist() if emb.ndim == 2 else emb.tolist()
        except Exception as e:
            logger.debug("Local embedding failed: %s", e)
            return []

    def _embed_tokens_st(self, text: str) -> list[list[float]]:
        """Per-token embeddings via a sentence-transformers model.

        NOTE: ST's output_value="token_embeddings" returns the TRANSFORMER
        module's token output, BEFORE any Pooling/Dense module — a ColBERT
        projection head is NOT applied on this path. What you get is
        backbone-token MaxSim, which is empirically strong but is not the
        model's trained ColBERT space; loading via pylate is required for
        the projected space. Vectors are L2-normalised before return.
        """
        import numpy as np
        try:
            token_embs = self._local_model.encode(
                text[:512],
                output_value="token_embeddings",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            if not isinstance(token_embs, np.ndarray):
                token_embs = np.asarray([np.asarray(v) for v in token_embs])
            if token_embs.ndim != 2 or token_embs.shape[-1] != self._token_dim:
                # Stored blobs are reshaped as (-1, self._token_dim) at search
                # time; a wrong-dim model writes garbage that fails every
                # query later (how the colbertv2.0 probe died). Fail at the
                # first embed with the actionable cause instead.
                raise RuntimeError(
                    f"late_interaction expects {self._token_dim}-dim token output, got "
                    f"shape {token_embs.shape} from {self._colbert_model!r}. "
                    "Models whose backbone hidden size differs (or whose quality "
                    "lives in a projection head, e.g. colbertv2.0) need a "
                    "pylate-style loader."
                )
            return token_embs.tolist()
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("ColBERT token embedding failed: %s", e)
            return []

    def _embed_tokens_pylate(self, text: str, *, is_query: bool = False) -> list[list[float]]:
        """Per-token embeddings via the pylate ColBERT model.

        pylate applies the full model including any Dense/projection head, so
        the returned vectors live in the model's trained ColBERT space (e.g.
        128-dim for colbertv2.0, 128-dim for answerai-colbert-small-v1).

        Vectors are L2-normalised before return (consistent with the ONNX/ST
        paths so MaxSim dot products equal cosine similarities).
        """
        import numpy as np
        try:
            out = self._pylate_model.encode(
                [text[:512]],
                is_query=is_query,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # pylate returns a list of per-document arrays or a 3-D array.
            # Shape of out[0] is (seq_len, dim).
            token_embs = np.asarray(out[0], dtype=np.float32)
            if token_embs.ndim != 2:
                raise RuntimeError(
                    f"pylate encode returned unexpected shape {token_embs.shape}"
                )
            dim = token_embs.shape[-1]
            if dim != self._token_dim:
                raise RuntimeError(
                    f"pylate token dim {dim} does not match store _token_dim "
                    f"{self._token_dim}. Mixed-dim corpora are not supported."
                )
            # L2-normalise each token vector so dot product == cosine for MaxSim.
            norms = np.linalg.norm(token_embs, axis=1, keepdims=True)
            token_embs = token_embs / (norms + 1e-8)
            return token_embs.tolist()
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("pylate token embedding failed: %s", exc)
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

        if self.late_interaction:
            # Late-interaction mode stores the full token matrix (seq_len x dim)
            # as a base64 float16 blob — large (a matrix per memory, not one
            # vector), but that's the point of the MaxSim probe. The shape is
            # recovered on read as nbytes/(self._token_dim*2) tokens x self._token_dim.
            import numpy as np

            token_mat = await self.embed_tokens(text, task="search_document")
            if not token_mat:
                return -1
            blob = np.asarray(token_mat, dtype=np.float16).tobytes()
            embedding_text = base64.b64encode(blob).decode("ascii")
        else:
            embedding = await self.embed(text, task="search_document")
            if not embedding:
                return -1
            # In binary-quant mode store only the packed sign bits (1 bit/dim,
            # 32x smaller) — the full-precision floats are intentionally
            # discarded, that footprint saving is the point of the mode. Cosine
            # mode stores the float vector as JSON as before.
            embedding_text = pack_sign_bits(embedding) if self.binary_quant else json.dumps(embedding)

        now = time.time()
        cursor = self._conn.execute(
            "INSERT INTO vector_memory (text, embedding, metadata_json, created_at) VALUES (?, ?, ?, ?)",
            (text, embedding_text, json.dumps(metadata or {}), now),
        )
        self._conn.commit()
        self._bm25_dirty = True  # corpus changed; invalidate BM25 cache
        return cursor.lastrowid

    def _load_active_rows(
        self,
        project: str | None = None,
        search_agents: list[str] | None = None,
        now: float | None = None,
    ):
        """Fetch active (non-superseded) rows, scoped by project/agent.

        Untagged rows are kept by both filters so pre-project / standalone
        memory is never hidden. Shared by the semantic and BM25-only paths so
        scoping rules cannot drift between them.

        TTL filter: rows whose metadata ``forget_after`` is a number and is
        less than ``now`` are excluded from retrieval exactly like superseded
        rows. The raw row is never deleted (zero-loss); only recall hides it.
        Non-numeric or missing ``forget_after`` values are silently ignored so
        existing memories are never affected. ``now`` defaults to
        ``time.time()``; pass an explicit float in tests to control the clock.
        """
        if now is None:
            now = time.time()

        rows = self._conn.execute(
            "SELECT id, text, embedding, metadata_json, created_at FROM vector_memory "
            "WHERE valid_to IS NULL"
        ).fetchall()

        filtered = []
        agent_set = set(search_agents) if search_agents else None
        for row in rows:
            try:
                meta = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                meta = {}

            # TTL filter: if forget_after is a valid number and has passed,
            # exclude the row from active recall. Non-numeric values are
            # ignored (row stays visible) with a debug log so bad metadata
            # never silently hides memories.
            fa = meta.get("forget_after")
            if fa is not None:
                try:
                    if float(fa) < now:
                        continue
                except (TypeError, ValueError):
                    logger.debug(
                        "ignore non-numeric forget_after=%r on row id=%s",
                        fa,
                        row["id"],
                    )

            if project is not None or search_agents is not None:
                # Project filter: skip rows positively tagged with a different
                # project. Untagged (pre-project) rows are kept so existing
                # standalone memory is never hidden.
                row_project = meta.get("project")
                if project is not None and row_project is not None and row_project != project:
                    continue
                # Agent filter: skip rows positively tagged with an out-of-scope
                # agent. Untagged rows are kept (preserves standalone behaviour).
                row_agent = meta.get("agent")
                if agent_set is not None and row_agent is not None and row_agent not in agent_set:
                    continue

            filtered.append(row)
        return filtered

    def existing_source_ids(self, agent: str | None = None) -> set[str]:
        """Return the user-metadata ``source_id`` values already stored.

        Backs idempotent re-imports over ``POST /ingest/batch`` (#25): items
        whose ``id`` is already present are skipped instead of duplicated.
        Rows tagged with a different agent are excluded when ``agent`` is set;
        untagged rows are included, matching the search-scoping rules.
        """
        out: set[str] = set()
        for row in self._load_active_rows(search_agents=[agent] if agent else None):
            try:
                meta = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            inner = meta.get("metadata")
            sid = inner.get("source_id") if isinstance(inner, dict) else None
            if sid:
                out.add(str(sid))
        return out

    async def search_bm25(
        self,
        query: str,
        limit: int = 5,
        project: str | None = None,
        search_agents: list[str] | None = None,
    ) -> list[dict]:
        """BM25-only retrieval: no query embedding, no semantic scoring.

        The fast path for short-form keyword search (the #25 user-memory
        contract: search-as-you-type, sub-300ms SLA). Uses bm25s when
        installed (same index parameters as the bm25_rrf fusion path, cached
        in ``_bm25_cache``); otherwise falls back to a dependency-free Okapi
        BM25 with identical k1/b. ``similarity`` carries the sigmoid-normalised
        BM25 score so it reads in the same 0-1 confidence range as cosine
        hits. Rows with zero term overlap are dropped rather than padded in.
        """
        if not query:
            return []
        rows = self._load_active_rows(project=project, search_agents=search_agents)
        if not rows:
            return []

        ids = [row["id"] for row in rows]
        texts = [row["text"] for row in rows]
        metas = [row["metadata_json"] for row in rows]
        created = [row["created_at"] for row in rows]

        ranked = self._bm25_rank(query, texts)

        from taosmd.utils.scoring import get_bm25_params, normalize_bm25
        midpoint, steepness = get_bm25_params(query)

        out = []
        for idx, raw in ranked:
            if len(out) >= limit:
                break
            if raw <= 0.0:
                break  # ranked is best-first; everything after has no overlap
            try:
                meta = json.loads(metas[idx])
            except (json.JSONDecodeError, TypeError):
                meta = {}
            out.append({
                "id": ids[idx],
                "text": texts[idx],
                "bm25_score": round(float(raw), 4),
                "similarity": round(normalize_bm25(float(raw), midpoint, steepness), 4),
                "metadata": meta,
                "created_at": created[idx],
            })
        return out

    def _bm25_rank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        """Rank ``texts`` against ``query`` by BM25; best-first (index, score).

        bm25s path reuses the same cache discipline as the fusion modes:
        keyed entry holding (retriever, texts) so any corpus or scope change
        forces a rebuild.
        """
        try:
            import bm25s

            cache_key = "bm25_only"
            cached = self._bm25_cache.get(cache_key)
            if self._bm25_dirty or cached is None or cached[1] != texts:
                corpus_tokens = bm25s.tokenize(texts, stopwords=None, stemmer=None)
                retriever = bm25s.BM25(k1=1.5, b=0.75)
                retriever.index(corpus_tokens)
                self._bm25_cache[cache_key] = (retriever, texts)
                self._bm25_dirty = False
            else:
                retriever = cached[0]
            query_tokens = bm25s.tokenize([query], stopwords=None, stemmer=None)
            k = len(texts)
            indices, scores = retriever.retrieve(query_tokens, k=k)
            return [(int(indices[0][r]), float(scores[0][r])) for r in range(k)]
        except ImportError:
            logger.warning(
                "bm25s not installed; using pure-Python BM25 fallback "
                "(pip install bm25s for the fast path)"
            )
            return _bm25_python_rank(query, texts)

    async def search(
        self,
        query: str,
        limit: int = 5,
        hybrid: bool = True,
        fusion: str = "boost",
        project: str | None = None,
        search_agents: list[str] | None = None,
    ) -> list[dict]:
        """Semantic search with optional hybrid fusion.

        Fusion modes:
          "rrf":             RRF over (semantic ranks, keyword-presence-
                              heuristic ranks). Legacy "hybrid" path.
          "bm25_rrf":        RRF over (semantic ranks, proper BM25 ranks).
                              Industrial BM25 (IDF + TF saturation + length
                              norm via bm25s) on raw text.
          "bm25_lemma_rrf": Same as bm25_rrf but BM25 indexes and queries
                              run on spaCy-lemmatised text (handles
                              meetings/meeting, attending/attend etc.).
                              ~+50 ms/query for spaCy on top of BM25.
                              Falls back to bm25_rrf if spaCy unavailable.
          "mem0_additive":   Mem0-style additive scoring instead of RRF.
                              (semantic + sigmoid-normalised BM25 lemma) / 2.
                              Adaptive sigmoid params from query length.
                              Keeps signal magnitudes calibrated rather than
                              flattening to ranks.
          "boost":           Legacy additive keyword boost.
          "none":            Pure semantic cosine similarity.

        When hybrid=False, fusion mode is ignored and pure semantic is used.

        Args:
            project: When set, only return memories tagged with this project.
            search_agents: When set (with project), only return memories from
                these agent names within the project. Enables cross-agent reads.
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

        # Load all *active* embeddings and compute similarity. Superseded rows
        # (valid_to IS NOT NULL) are soft-hidden from recall, mirroring the
        # KG's active-triple filter, but the raw row is never deleted, so
        # zero-loss is preserved. This active filter covers every scoring path
        # below (cosine, binary-quant, and the pure-Python fallback) because
        # they all consume this single row set. No behaviour change when
        # nothing has been superseded.
        rows = self._load_active_rows(project=project, search_agents=search_agents)

        if not rows:
            return []

        try:
            import numpy as np

            # Parse all stored embeddings. In binary-quant mode the column holds
            # base64 packed sign bits; in late-interaction mode it holds a base64
            # float16 token matrix (decoded to a seq_len x _token_dim float32 array); in
            # cosine mode it holds a JSON float list.
            ids = []
            texts = []
            metas = []
            created = []
            emb_list = []
            for row in rows:
                try:
                    raw = row["embedding"]
                    if self.late_interaction:
                        blob = base64.b64decode(raw)
                        emb = np.frombuffer(blob, dtype=np.float16).reshape(-1, self._token_dim).astype(np.float32)
                        emb = emb if emb.size else None
                    elif self.binary_quant:
                        emb = base64.b64decode(raw)
                    else:
                        emb = json.loads(raw)
                    if emb is not None and len(emb):
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
                # recall-neutral LoCoMo-1540 result was measured on; packing only
                # changes storage/speed, not ranking (see docs/benchmarks.md).
                dim = len(query_emb)
                query_bits = np.packbits(np.asarray(query_emb, dtype=np.float32) >= 0.0)
                doc_bits = np.frombuffer(b"".join(emb_list), dtype=np.uint8).reshape(len(emb_list), -1)
                xor = np.bitwise_xor(doc_bits, query_bits[None, :])
                hamming = np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)
                similarities = 1.0 - hamming / dim
            elif self.late_interaction:
                # ColBERT-style MaxSim over MiniLM token vectors. The query is
                # embedded once to its token matrix (Q x 384); each stored doc is
                # a token matrix (M_d x 384). All vectors are L2-normalised so the
                # dot product is cosine. Per doc:
                #   MaxSim_d = mean_q( max_t( q_t . d_t ) )
                # i.e. each query token matches its single best doc token, then
                # average over query tokens. emb_list holds the doc matrices in
                # the same order as ids/texts/metas, so similarities aligns 1:1.
                qmat = np.asarray(await self.embed_tokens(query, task="search_query"), dtype=np.float32)
                maxsims = np.empty(len(emb_list), dtype=np.float32)
                for i, dmat in enumerate(emb_list):
                    sim_matrix = qmat @ dmat.T  # (Q, M_d)
                    maxsims[i] = sim_matrix.max(axis=1).mean()
                similarities = maxsims
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
                    # Build / reuse cached BM25 index for this fusion mode.
                    # The cache is keyed by fusion mode and valid while the
                    # active corpus is unchanged (_bm25_dirty is False).
                    # Each entry is (retriever, bm25_texts_used) so a
                    # corpus change (add / supersede) forces a full rebuild.
                    cache_key = fusion
                    cached = self._bm25_cache.get(cache_key)
                    if self._bm25_dirty or cached is None or cached[1] != bm25_texts:
                        corpus_tokens = bm25s.tokenize(bm25_texts, stopwords=None, stemmer=None)
                        retriever = bm25s.BM25(k1=1.5, b=0.75)
                        retriever.index(corpus_tokens)
                        self._bm25_cache[cache_key] = (retriever, bm25_texts)
                        self._bm25_dirty = False
                    else:
                        retriever = cached[0]
                    query_tokens = bm25s.tokenize([bm25_query], stopwords=None, stemmer=None)
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

            # Pure semantic or boost mode; sort by similarity
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
            if self.late_interaction:
                raise RuntimeError("late_interaction requires numpy")
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
            "WHERE valid_to IS NULL AND json_extract(metadata_json, ?) = ?"
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

    async def stats(self) -> dict:
        """Overall vector-memory statistics."""
        return {"count": await self.count()}

    async def supersede(self, row_id: int, ended_at: float | None = None) -> bool:
        """Soft-hide a single vector row from active recall.

        Sets ``valid_to`` on the row so ``search()`` (and the adjacent-neighbour
        lookup) no longer return it, mirroring the KG's
        :meth:`TemporalKnowledgeGraph.invalidate`. The raw row (text, embedding,
        and metadata) is *retained*; it is only excluded from recall, never
        deleted. The append-only archive entry is likewise untouched, so
        zero-loss is preserved.

        Already-superseded rows are left as-is (the guard mirrors the KG's
        ``valid_to IS NULL`` condition). Returns True if a row was affected.
        """
        end = ended_at if ended_at is not None else time.time()
        cursor = self._conn.execute(
            "UPDATE vector_memory SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
            (end, row_id),
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            self._bm25_dirty = True  # active corpus changed; invalidate BM25 cache
        return cursor.rowcount > 0

    async def supersede_matching(self, text_or_substring: str, ended_at: float | None = None) -> int:
        """Soft-hide every active row whose stored text contains the substring.

        Used to wire corrections by content: when a fact is corrected, the
        chunk(s) still carrying the *old* value are excluded from active recall
        so the stale fact stops resurfacing in vector search too. As with
        :meth:`supersede`, the raw rows are retained (zero-loss); only their
        ``valid_to`` is stamped. Matching is a plain case-sensitive substring
        test on the stored (already secret-redacted) text. An empty/blank
        ``text_or_substring`` is a no-op (returns 0) so a missing correction
        value can never sweep the whole store. Returns the number of rows
        superseded.
        """
        needle = (text_or_substring or "").strip()
        if not needle:
            return 0
        end = ended_at if ended_at is not None else time.time()
        cursor = self._conn.execute(
            "UPDATE vector_memory SET valid_to = ? "
            "WHERE valid_to IS NULL AND instr(text, ?) > 0",
            (end, text_or_substring),
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            self._bm25_dirty = True  # active corpus changed; invalidate BM25 cache
        return cursor.rowcount

    async def iter_entries(
        self,
        agent: str | None = None,
        include_superseded: bool = True,
    ):
        """Yield (text, metadata_dict) tuples for stored entries.

        Used by :func:`taosmd.api.reconcile` to enumerate the current vector
        corpus (including superseded rows when ``include_superseded=True``) so
        the reconcile logic can distinguish "truly absent" from "intentionally
        superseded". Superseded rows count as *present* in the multiset; their
        content is not re-added, so corrected/stale entries are never resurrected.

        When ``agent`` is given, only rows whose stored metadata ``agent`` field
        matches are returned. This mirrors the per-agent scope used by
        :meth:`add` when metadata ``{"agent": ...}`` is attached at ingest time.
        Rows with no ``agent`` field in their metadata are included only when
        ``agent`` is ``None``.

        Yields ``(text: str, metadata: dict)`` in insertion order (ascending id).
        """
        sql = "SELECT text, metadata_json FROM vector_memory"
        if not include_superseded:
            sql += " WHERE valid_to IS NULL"
        sql += " ORDER BY id ASC"

        for row in self._conn.execute(sql).fetchall():
            try:
                meta = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                meta = {}
            if agent is not None and meta.get("agent") != agent:
                continue
            yield row["text"], meta

    async def clear(self) -> int:
        cursor = self._conn.execute("DELETE FROM vector_memory")
        self._conn.commit()
        return cursor.rowcount
