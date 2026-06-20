# taOSmd memory app: recommended config and toggles (integration handoff)

This is the consolidated configuration surface for embedding taOSmd as the memory app in taOS: which models and retrieval settings to default per hardware tier, and which toggles to expose to the user. Full measurements and provenance are in [benchmarks.md](benchmarks.md#hardware-tiers-recommended-configurations); this page is the short, integration-focused view.

## Always-on defaults (every tier)

- Embedder: arctic-embed-s ONNX on CPU for fresh low-tier installs (the dense default, +0.057 judged retrieval over MiniLM at the same 384 dims and the same latency). MiniLM stays supported and is the model the 97.0% Recall@5 headline was measured on, so existing installs keep it.
- Reranker: ms-marco-MiniLM ONNX on CPU, second-stage over the top-K vector hits.
- Do not enable multihop-decompose (regresses at every model size measured) or the session_date context format (a no-op once the prompt carries absolute dates). Leave both off; no need to surface them.

## Per-tier generator and retrieval defaults

| Tier | Generator (default) | Retrieval recipe | Measured |
|---|---|---|---|
| 12 GB GPU (RTX 3060 class) | qwen3.5:9b Q4_K_M (best quality) or llama3.1:8b (2.4x faster) | k=20, adj=2, llm-query-expansion, fusion rrf; add bge-v2-m3 MaxSim rerank where affordable | LoCoMo 0.557 strict / 0.748 lenient (tri-judge) |
| 8 GB GPU | qwen3.5:9b IQ4_XS (4.81 GB), concurrency 1 | same stack as the 12 GB tier | ~0.55 (from the quant-cliff data) |
| 16 GB Orange Pi 5 Plus (RK3588 NPU) | Qwen3-4B via rkllama on the NPU | adj=2, k=20, llm-exp, RRF; reranker Qwen3-Reranker-0.6B on NPU | LoCoMo 0.490; this is the 97.0% Recall@5 reference stack |
| 4 GB GPU (GTX 1050 Ti) | qwen3:4b Q4 (~2.5 GB) | adj=2, k=10, RRF; skip llm-query-expansion | LoCoMo 0.530 |
| Raspberry Pi 4 (CPU only) | qwen3:1.7b or smaller | adj=1, k=10; skip every flag that adds an LLM call | extrapolation; better used as a storage/retrieval node, offload generation to a peer |

Embedder, reranker, and judge run as CPU ONNX on every tier; GPU or NPU VRAM is for the generator only. Do not run the judge on the same device as the generator.

## User-facing toggles to expose

- Verified-answer mode (opt-in): reranking plus a CoVe-style answer self-verification pass. This is the recommended config for answer quality and is the 74.6% end-to-end LongMemEval-S Judge number (vs 47.2% starved baseline). taOSmd serves memory, so answer generation is the consumer path; keep it off in the core default and recommend it on where the consumer wants graded answers.
- prefer_verified integrity gate (opt-in, off by default): demotes (never deletes) facts the verifier could not support against their source span. A pre-registered sweep eliminated served-hallucination (0.04 to 0.00) at no measured accuracy or recall cost, reproduced at n=250 under a strict judge. Expose it as an integrity-mode toggle; keep the default off until a full tri-judge confirm and an explicit sign-off.
- Late-interaction retrieval (opt-in): token-level MaxSim, lifts evidence recall from 0.64 to 0.85 on LoCoMo at about 110 ms/query on a 16-core CPU, no GPU or reranker needed. A good default-on option for CPU tiers that want better recall without a cross-encoder.
- Binary embedding quantization (opt-in): 1 bit per dimension, 32x smaller vectors, recall-neutral. Expose as a footprint option for memory-constrained or SBC deployments.
- Fusion mode: rrf or mem0_additive (both beat the older mem0-only guidance at full 1540-QA scale); boost is fine on the smallest tiers.

## What not to default on

multihop-decompose (regresses), session_date context format (no-op), HyDE (regresses on small generators), and bare FROM-gguf Modelfiles (they need the full TEMPLATE/RENDERER/PARSER/PARAMETER metadata cloned from `ollama show <model> --modelfile` or output runs away and looks far slower than the kernel is).

For the live measurement log and any new tier validations, see [benchmarks.md](benchmarks.md) and the [research report](research-report.md).
