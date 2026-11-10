Fix Git LFS pointer detection in ONNX model loading.

- `taosmd.api._has_onnx_model()` now rejects Git LFS pointer files (small text
  files starting with `version https://git-lfs.github.com/spec/`) instead of
  treating them as valid ONNX models.
- `taosmd.vector_memory.VectorMemory.init()` raises a clear `RuntimeError`
  naming the LFS pointer file and suggesting `hf download <repo>
  onnx/model.onnx` (without `--include`) to fetch the real model.
- The existing broad `except Exception` handler logs the error and falls back
  to QMD embed mode as before.
