<# 
.SYNOPSIS
    Export BAAI/bge-reranker-v2-m3 to ONNX via optimum-cli.

.DESCRIPTION
    Output is ~2.2 GB fp32. Needs torch only inside the throwaway venv.
    optimum 2.x dropped the `exporters` extra so `optimum-onnx` is the package.

    Usage: powershell -File scripts/export_reranker_onnx.ps1 [-DestDir <path>] [-ExportVenv <path>]
    DestDir default: models/bge-reranker-v2-m3-onnx
    ExportVenv default: $env:USERPROFILE/.cache/taosmd/rerank-export-venv

    The script creates a throwaway venv, installs CPU torch from
    https://download.pytorch.org/whl/cpu plus optimum-onnx onnx onnxruntime
    transformers sentencepiece, then runs optimum-cli export onnx.
#>

param(
    [string]$DestDir = "models/bge-reranker-v2-m3-onnx",
    [string]$ExportVenv = "$env:USERPROFILE/.cache/taosmd/rerank-export-venv"
)

$Revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

Write-Host "Exporting BAAI/bge-reranker-v2-m3 (rev $Revision) to ONNX..."
Write-Host "Destination: $DestDir"
Write-Host "Throwaway venv: $ExportVenv"

# Create throwaway venv
uv venv $ExportVenv

# Install CPU torch + export deps
& "$ExportVenv/Scripts/pip" install --index-url https://download.pytorch.org/whl/cpu torch
& "$ExportVenv/Scripts/pip" install optimum-onnx onnx onnxruntime transformers sentencepiece

# Run the export
& "$ExportVenv/Scripts/optimum-cli" export onnx `
    --model BAAI/bge-reranker-v2-m3 `
    --revision $Revision `
    --task text-classification `
    --opset 17 `
    $DestDir

# Verify model.onnx exists
if (-not (Test-Path "$DestDir/model.onnx")) {
    Write-Error "ERROR: $DestDir/model.onnx not found after export"
    exit 1
}

# Print sha256 of model.onnx and any shards
Get-FileHash -Algorithm SHA256 "$DestDir/model.onnx"*

Write-Host "Export complete. Model ready at $DestDir"
