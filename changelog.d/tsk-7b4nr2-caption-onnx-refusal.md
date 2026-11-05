### Fixed

- `CaptionTextImageEmbedder` now raises `RuntimeError` when the embedded `VectorMemory` silently falls back from ONNX to the qmd remote backend, instead of proceeding with zero-width embeddings and an empty model provenance.
- `provenance()` now raises `FileNotFoundError` naming the missing file when `model.onnx` or `tokenizer.json` is absent from the ONNX model directory, instead of silently passing.
