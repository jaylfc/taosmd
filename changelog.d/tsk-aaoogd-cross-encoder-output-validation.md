### Fixed
- Cross-encoder now validates ONNX output shapes and raises clear errors for non-logit outputs (3-D last_hidden_state, 2-D pooled hidden states) instead of silently returning wrong scores
- Added support for ettin-reranker-68m-v1: when the model directory contains sentence-transformers config and safetensors with the scoring head, the head is loaded and applied to CLS-pooled hidden states (following the architecture in the model card: Transformer → Pooling(cls) → Dense(GELU) → LayerNorm → Dense(scores))
- For 2-class logit heads (shape batch, 2), the positive-class logit (index 1) is now used as the relevance score
