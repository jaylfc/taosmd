### Fixed

The `live_embed_backend` test guard now verifies an ONNX embed backend by
loading the model with ONNX Runtime instead of only checking that
`model.onnx` exists, so a truncated or empty model file (for example left by
an interrupted `scripts/setup.sh`) makes the guarded tests SKIP instead of
failing at embed time. The QMD service probe result is cached for the process
so a filtered (rather than refused) port is only waited on once per run.
