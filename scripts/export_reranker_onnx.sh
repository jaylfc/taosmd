#!/usr/bin/env bash
# Export BAAI/bge-reranker-v2-m3 to ONNX via optimum-cli.
#
# Output is ~2.2 GB fp32. Needs torch only inside the throwaway venv.
# optimum 2.x dropped the `exporters` extra so `optimum-onnx` is the package.
#
# Usage: bash scripts/export_reranker_onnx.sh <dest_dir>
#   dest_dir default: models/bge-reranker-v2-m3-onnx
#
# The script creates a throwaway venv at ${EXPORT_VENV:-$HOME/.cache/taosmd/rerank-export-venv},
# installs CPU torch from https://download.pytorch.org/whl/cpu plus
# optimum-onnx onnx onnxruntime transformers sentencepiece,
# then runs optimum-cli export onnx.

set -euo pipefail

DEST_DIR="${1:-models/bge-reranker-v2-m3-onnx}"
EXPORT_VENV="${EXPORT_VENV:-$HOME/.cache/taosmd/rerank-export-venv}"
REVISION="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

echo "Exporting BAAI/bge-reranker-v2-m3 (rev ${REVISION}) to ONNX..."
echo "Destination: ${DEST_DIR}"
echo "Throwaway venv: ${EXPORT_VENV}"

# Create throwaway venv
uv venv "${EXPORT_VENV}"

# Install CPU torch + export deps
"${EXPORT_VENV}/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch
"${EXPORT_VENV}/bin/pip" install optimum-onnx onnx onnxruntime transformers sentencepiece

# Run the export
"${EXPORT_VENV}/bin/optimum-cli" export onnx \
    --model BAAI/bge-reranker-v2-m3 \
    --revision "${REVISION}" \
    --task text-classification \
    --opset 17 \
    "${DEST_DIR}"

# Verify model.onnx exists
if [[ ! -f "${DEST_DIR}/model.onnx" ]]; then
    echo "ERROR: ${DEST_DIR}/model.onnx not found after export" >&2
    exit 1
fi

# Print sha256 of model.onnx and any shards
sha256sum "${DEST_DIR}/model.onnx"*

echo "Export complete. Model ready at ${DEST_DIR}"