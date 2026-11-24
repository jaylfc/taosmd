### Fixed
- The ONNX embedder loader now refuses a sentence-transformers export whose
  modules.json lists a Dense head but whose graph exposes no `sentence_embedding`
  output (for example MongoDB/mdbr-leaf-ir's official ONNX, which drops the
  384->768 Dense layer after pooling), instead of silently mean-pooling the
  backbone to the wrong dimension, and validates the first embedding against the
  declared output dimension. Names the model dir and the fix: re-export with the
  Dense layer exposing `sentence_embedding`.
### Added
- `mdbr-leaf-ir` is a selectable (non-default) ONNX embedder, using the query
  prefix and mean pooling stated in its model card ("Sentence Transformers"
  quickstart / `config_sentence_transformers.json` prompts), wired via
  `_onnx_apply_prefix` and `_onnx_pooling_mode`.
