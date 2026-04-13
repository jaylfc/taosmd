#!/bin/bash
set -e

echo "================================"
echo "  taOSmd Setup"
echo "  97.2% Recall@5 on LongMemEval"
echo "================================"
echo ""

# Detect platform
ARCH=$(uname -m)
case "$ARCH" in
    aarch64|arm64) PLATFORM="arm64" ;;
    x86_64|amd64)  PLATFORM="x86_64" ;;
    *)             PLATFORM="$ARCH" ;;
esac
echo "→ Platform: $PLATFORM ($(uname -s))"

# Check for GPU
HAS_NVIDIA=false
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU" ]; then
        HAS_NVIDIA=true
        echo "→ GPU detected: $GPU"
    fi
fi

# Detect install location
INSTALL_DIR="${TAOSMD_DIR:-$HOME/taosmd}"

# Clone if not already present
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "✓ taOSmd already cloned at $INSTALL_DIR"
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "→ Cloning taOSmd..."
    git clone https://github.com/jaylfc/taosmd.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install Python dependencies
echo "→ Installing dependencies..."
pip install -e . --quiet 2>/dev/null || pip3 install -e . --quiet

# Install ONNX Runtime (for fast embeddings)
pip install onnxruntime numpy --quiet 2>/dev/null || pip3 install onnxruntime numpy --quiet

# Ensure huggingface-cli is available
if ! command -v huggingface-cli &>/dev/null; then
    pip install huggingface-hub --quiet 2>/dev/null || pip3 install huggingface-hub --quiet
fi

# ================================================
# 1. Embedding model (required — all platforms)
# ================================================
MODEL_DIR="$INSTALL_DIR/models/minilm-onnx"
if [ -f "$MODEL_DIR/model.onnx" ]; then
    echo "✓ Embedding model already downloaded"
else
    echo "→ Downloading all-MiniLM-L6-v2 ONNX (90MB)..."
    huggingface-cli download onnx-models/all-MiniLM-L6-v2-onnx --local-dir "$MODEL_DIR" --quiet
fi

# ================================================
# 2. LLM for fact extraction (platform-specific)
# ================================================
echo ""
echo "→ Setting up LLM for fact extraction..."

if [ "$PLATFORM" = "arm64" ] && [ -e "/sys/class/misc/mali0" ]; then
    # RK3588 with NPU — use rkllama
    echo "  RK3588 NPU detected"
    if command -v rkllama_server &>/dev/null || [ -d "$HOME/rkllama" ]; then
        echo "  ✓ rkllama already installed"
        # Download Qwen3-4B for NPU extraction
        RKLLM_DIR="$HOME/rkllama/models/qwen3-4b-chat"
        if [ -d "$RKLLM_DIR" ] && ls "$RKLLM_DIR"/*.rkllm &>/dev/null 2>&1; then
            echo "  ✓ Qwen3-4B RKLLM model already present"
        else
            echo "  → Downloading Qwen3-4B for NPU (4.6GB)..."
            mkdir -p "$RKLLM_DIR"
            huggingface-cli download dulimov/Qwen3-4B-rk3588-1.2.1-base \
                Qwen3-4B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm \
                --local-dir "$RKLLM_DIR" --quiet 2>/dev/null || echo "  ⚠ Download failed — install manually"
        fi
    else
        echo "  ⚠ rkllama not found. Install from: https://github.com/NotPunchnox/rkllama"
        echo "    Then re-run this script to download NPU models"
    fi
elif [ "$HAS_NVIDIA" = true ]; then
    # x86 with NVIDIA GPU — use Ollama
    echo "  NVIDIA GPU detected — using Ollama"
    if command -v ollama &>/dev/null; then
        echo "  ✓ Ollama already installed"
    else
        echo "  → Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || echo "  ⚠ Ollama install failed — install manually from ollama.com"
    fi
    if command -v ollama &>/dev/null; then
        if ollama list 2>/dev/null | grep -q "qwen2.5:3b"; then
            echo "  ✓ Qwen2.5-3B model already pulled"
        else
            echo "  → Pulling Qwen2.5-3B for GPU extraction (~2GB)..."
            ollama pull qwen2.5:3b 2>/dev/null || echo "  ⚠ Model pull failed — run: ollama pull qwen2.5:3b"
        fi
    fi
else
    # CPU-only — use Ollama with a small model
    echo "  No GPU/NPU detected — using Ollama with CPU model"
    if command -v ollama &>/dev/null; then
        echo "  ✓ Ollama already installed"
    else
        echo "  → Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null || echo "  ⚠ Ollama install failed — install manually from ollama.com"
    fi
    if command -v ollama &>/dev/null; then
        if ollama list 2>/dev/null | grep -q "qwen2.5:1.5b"; then
            echo "  ✓ Qwen2.5-1.5B model already pulled"
        else
            echo "  → Pulling Qwen2.5-1.5B for CPU extraction (~1GB)..."
            ollama pull qwen2.5:1.5b 2>/dev/null || echo "  ⚠ Model pull failed — run: ollama pull qwen2.5:1.5b"
        fi
    fi
fi

# Create data directory
mkdir -p "$INSTALL_DIR/data"

# Run self-test
echo ""
echo "→ Running self-test..."
python -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
import asyncio

async def test():
    from taosmd import KnowledgeGraph, VectorMemory, Archive

    # Test KG
    kg = KnowledgeGraph('$INSTALL_DIR/data/test-kg.db')
    await kg.init()
    await kg.add_triple('test', 'works', 'perfectly')
    facts = await kg.query_entity('test')
    assert len(facts) > 0, 'KG failed'
    await kg.close()
    print('  ✓ Knowledge Graph')

    # Test Vector Memory
    vmem = VectorMemory('$INSTALL_DIR/data/test-vectors.db', embed_mode='onnx', onnx_path='$MODEL_DIR')
    await vmem.init()
    await vmem.add('taOSmd achieves 97.2% on LongMemEval')
    results = await vmem.search('memory benchmark', hybrid=True)
    assert len(results) > 0, 'Vector search failed'
    await vmem.close()
    print('  ✓ Vector Memory (ONNX + hybrid search)')

    # Test Archive
    archive = Archive(archive_dir='$INSTALL_DIR/data/test-archive', index_path='$INSTALL_DIR/data/test-archive-idx.db')
    await archive.init()
    await archive.record('test', {'msg': 'hello'}, summary='test event')
    events = await archive.query(limit=1)
    assert len(events) > 0, 'Archive failed'
    await archive.close()
    print('  ✓ Zero-Loss Archive')

    # Cleanup test files
    import os
    for f in ['test-kg.db', 'test-vectors.db', 'test-archive-idx.db']:
        p = os.path.join('$INSTALL_DIR/data', f)
        if os.path.exists(p):
            os.remove(p)
    import shutil
    test_archive = os.path.join('$INSTALL_DIR/data', 'test-archive')
    if os.path.exists(test_archive):
        shutil.rmtree(test_archive)

asyncio.run(test())
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "  ✓ taOSmd is ready!"
    echo "================================"
    echo ""
    echo "  Platform: $PLATFORM"
    echo "  Location: $INSTALL_DIR"
    echo "  Model:    $MODEL_DIR/model.onnx"

    # GPU-specific suggestions
    if [ "$HAS_NVIDIA" = true ]; then
        echo ""
        echo "  GPU detected! For LLM-powered fact extraction:"
        echo "    1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
        echo "    2. Pull a model: ollama pull qwen2.5:3b"
        echo "    3. Set: export TAOSMD_LLM_URL=http://localhost:11434"
        echo ""
        echo "  This enables 72% recall background extraction"
        echo "  (vs 39% regex-only without a GPU/LLM)"
    fi

    # RK3588-specific suggestions
    if [ "$PLATFORM" = "arm64" ] && [ -d "/sys/class/misc/mali0" ] 2>/dev/null; then
        echo ""
        echo "  RK3588 NPU detected! For NPU-accelerated models:"
        echo "    See: https://github.com/jaylfc/taosmd#optional-full-stack-orange-pi--rk3588"
    fi

    echo ""
    echo "  Quick test:"
    echo "    python -c \"from taosmd import VectorMemory; print('taOSmd loaded')\""
    echo ""
    echo "  Docs: https://github.com/jaylfc/taosmd"
    echo ""
else
    echo ""
    echo "  ✗ Self-test failed. Check the error above."
    exit 1
fi
