#!/bin/bash
set -e

echo "================================"
echo "  taOSmd Setup"
echo "  97.2% Recall@5 on LongMemEval"
echo "================================"
echo ""

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

# Download embedding model
MODEL_DIR="$INSTALL_DIR/models/minilm-onnx"
if [ -f "$MODEL_DIR/model.onnx" ]; then
    echo "✓ Embedding model already downloaded"
else
    echo "→ Downloading all-MiniLM-L6-v2 ONNX model (90MB)..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download onnx-models/all-MiniLM-L6-v2-onnx --local-dir "$MODEL_DIR" --quiet
    else
        pip install huggingface-hub --quiet 2>/dev/null
        huggingface-cli download onnx-models/all-MiniLM-L6-v2-onnx --local-dir "$MODEL_DIR" --quiet
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
    echo "  Location: $INSTALL_DIR"
    echo "  Model:    $MODEL_DIR/model.onnx"
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
