### Fixed

ONNX model detection now rejects Git LFS pointer files. `_has_onnx_model()` in `taosmd/api.py` and the ONNX embedder init in `taosmd/vector_memory.py` both check for LFS pointers (small text files starting with `version https://git-lfs.github.com/spec/`) and treat them as missing models. When an LFS pointer is detected, a clear `RuntimeError` is raised telling the user to re-download the real model file with `hf download <repo> onnx/model.onnx` (without `--include`), and the embedder falls back to QMD with a warning that names the pointer file.
