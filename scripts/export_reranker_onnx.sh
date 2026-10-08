#!/usr/bin/env bash
# export_reranker_onnx.sh — export the BGE reranker v2-m3 to ONNX format
#
# Usage:
#   ./scripts/export_reranker_onnx.sh [OUTPUT_DIR]
#
# If OUTPUT_DIR is not passed as an argument the script will use "models/cross-encoder-onnx".
# Example:
#   ./scripts/export_reranker_onnx.sh ./my-export-dir
#
# What this does:
#   1. Creates a dedicated virtual environment for optimum-cli using uv
#   2. Installs optimum[exporters] in that environment using uv pip
#   3. Runs optimum-cli export from the export venv's own binary
#   4. Exports the BAAI/bge-reranker-v2-m3 model to ONNX format

set -euo pipefail

echo "=== taOSmd reranker ONNX export ==="

# --- Step 0: parse arguments ---------------------------------------------------
OUTPUT_DIR="${1:-models/cross-encoder-onnx}"
EXPORT_VENV="${OUTPUT_DIR}/.optimum-venv"

echo "Output directory: $OUTPUT_DIR"
echo "Export venv: $EXPORT_VENV"

# --- Step 1: create export venv ------------------------------------------------
echo ""
echo "Step 1: Creating export virtual environment..."
uv venv --seed "$EXPORT_VENV"

# --- Step 2: install optimum[exporters] in the export venv ---------------------
echo ""
echo "Step 2: Installing optimum[exporters]..."
# Use uv pip with the export venv's python to install optimum[exporters]
uv pip install --python "$EXPORT_VENV/bin/python" --quiet optimum[exporters]
echo "  optimum[exporters] installed."

# --- Step 3: verify optimum-cli is available in the export venv ----------------
echo ""
echo "Step 3: Verifying optimum-cli availability..."
if [ ! -x "$EXPORT_VENV/bin/optimum-cli" ]; then
  echo "error: optimum-cli not found in export venv at $EXPORT_VENV/bin/optimum-cli" >&2
  exit 1
fi
echo "  optimum-cli found at $EXPORT_VENV/bin/optimum-cli"

# --- Step 4: run optimum-cli export from the export venv's own binary ---------
echo ""
echo "Step 4: Exporting BGE reranker v2-m3 to ONNX..."
echo "  This may take a while and download ~2GB of model files."
echo "  Exporting to: $OUTPUT_DIR"

# Run the export venv's own optimum-cli binary directly
"$EXPORT_VENV/bin/optimum-cli" export onnx --model="BAAI/bge-reranker-v2-m3" --task=text-classification "$OUTPUT_DIR"

echo ""
echo "=== taOSmd reranker ONNX export complete ==="
echo "  Exported to: $OUTPUT_DIR"
echo "  To use this model, set embed_model=cross-encoder-onnx in your recipe or config."