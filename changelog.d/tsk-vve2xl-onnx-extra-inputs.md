### Fixed

The ONNX embedder now generic-empty-fills declared media graph inputs
(`image_features`, `video_features`, `audio_features`, etc.) that a text
embedder cannot populate, so models whose ONNX export ships those inputs
(such as EmbeddingGemma-2) embed normally instead of raising on a missing
feed key and silently returning empty vectors. Empty-feeding only applies to
inputs with a symbolic leading dimension and concrete trailing dims at a
supported scalar type; anything that cannot be driven is left unfed and the
load-time probe now raises, logging a WARNING that names the model file and
the unfed inputs before falling back to QMD. `_embed_onnx` failures now log
at WARNING (with the model path) rather than DEBUG, and that empty-vector
result is still treated as an embedder-down signal so ingest continues to
surface the degradation rather than raising.
