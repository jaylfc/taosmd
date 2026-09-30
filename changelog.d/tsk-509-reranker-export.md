### Fixed
- `ensure_reranker_model()` now fails loudly when the BGE-v2-m3 ONNX model is missing, since `BAAI/bge-reranker-v2-m3` publishes no ONNX file. Added `scripts/export_reranker_onnx.sh` (and `.ps1`) to export the model via `optimum-cli`. The runner's `--reranker bge-v2-m3` check now verifies `model.onnx` exists (using `taosmd.recipes._reranker_present`) and points to the export script.
