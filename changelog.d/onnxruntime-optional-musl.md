- `onnxruntime` moved out of the core dependencies into a new `onnx` extra. It
  publishes no musllinux wheel and no sdist, so requiring it outright made
  taosmd — and every project depending on it — impossible to install on any
  musl host (Alpine, postmarketOS). All ONNX imports were already lazy, so the
  core install is unaffected; install `taosmd[onnx]` to get the ONNX embed mode
  and cross-encoder back.
