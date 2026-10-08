# export_reranker_onnx.ps1 -- export the BGE reranker v2-m3 to ONNX format
#
# Usage:
#   .\scripts\export_reranker_onnx.ps1 [-OutputDir <dir>]
#
# If OutputDir is not passed as an argument the script will use "models/cross-encoder-onnx".
# Example:
#   .\scripts\export_reranker_onnx.ps1 -OutputDir ./my-export-dir
#
# What this does:
#   1. Creates a dedicated virtual environment for optimum-cli using uv
#   2. Installs "optimum[exporters]" in that environment using uv pip
#   3. Runs optimum-cli export from the export venv's own binary
#   4. Exports the BAAI/bge-reranker-v2-m3 model to ONNX format

param (
    [string]$OutputDir = "models/cross-encoder-onnx"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== taOSmd reranker ONNX export ===" -ForegroundColor Cyan

# --- Step 0: parse arguments ---------------------------------------------------
$OutputDir = $OutputDir.TrimEnd("\/")
$ExportVenv = Join-Path $OutputDir ".optimum-venv"

Write-Host ""
Write-Host "Output directory: $OutputDir"
Write-Host "Export venv: $ExportVenv"

# --- Step 1: create export venv ------------------------------------------------
Write-Host ""
Write-Host "Step 1: Creating export virtual environment..." -ForegroundColor Yellow
uv venv --seed "$ExportVenv"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create export venv at $ExportVenv"
    exit $LASTEXITCODE
}
if (-not (Test-Path "$ExportVenv\Scripts\python.exe")) {
    Write-Error "python.exe not found in export venv at $ExportVenv\Scripts\python.exe"
    exit 1
}

# --- Step 2: install "optimum[exporters]" in the export venv -------------------
Write-Host ""
Write-Host "Step 2: Installing optimum[exporters]..." -ForegroundColor Yellow
# Use uv pip with the export venv's python to install optimum[exporters]
uv pip install --python "$ExportVenv\Scripts\python.exe" --quiet "optimum[exporters]"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install optimum[exporters] in export venv"
    exit $LASTEXITCODE
}
Write-Host "  optimum[exporters] installed."

# --- Step 3: verify optimum-cli is available in the export venv ----------------
Write-Host ""
Write-Host "Step 3: Verifying optimum-cli availability..." -ForegroundColor Yellow
$optimumCliPath = Join-Path $ExportVenv "Scripts\optimum-cli.exe"
if (-not (Test-Path $optimumCliPath)) {
    Write-Error "optimum-cli not found in export venv at $optimumCliPath"
    exit 1
}
Write-Host "  optimum-cli found at $optimumCliPath"

# --- Step 4: run optimum-cli export from the export venv's own binary ---------
Write-Host ""
Write-Host "Step 4: Exporting BGE reranker v2-m3 to ONNX..." -ForegroundColor Yellow
Write-Host "  This may take a while and download ~2GB of model files."
Write-Host "  Exporting to: $OutputDir"

# Run the export venv's own optimum-cli binary directly
& $optimumCliPath export onnx --model="BAAI/bge-reranker-v2-m3" --task=text-classification $OutputDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "optimum-cli export failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=== taOSmd reranker ONNX export complete ===" -ForegroundColor Green
Write-Host "  Exported to: $OutputDir"
Write-Host "  To use this model, set embed_model=cross-encoder-onnx in your recipe or config."
