#!/bin/bash
# Run the full benchmark matrix on Fedora with RTX 3060
# Uses EXACT same models as Pi: all-MiniLM-L6-v2 ONNX (CPUExecutionProvider)
set -e

cd ~/taosmd

# Setup venv if needed
if [ ! -d .venv ]; then
    python3 -m venv .venv
    .venv/bin/pip install -e . 2>&1 | tail -3
fi

# Verify imports
.venv/bin/python3 -c "import taosmd; print(f'taOSmd v{taosmd.__version__}, {len(taosmd.__all__)} exports')"
.venv/bin/python3 -c "import onnxruntime; print(f'ONNX Runtime {onnxruntime.__version__}')"

# Verify dataset
DATASET="benchmarks/data/longmemeval_s_full.json"
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    exit 1
fi
.venv/bin/python3 -c "
import json
with open('$DATASET') as f:
    data = json.load(f)
sizes = [len(q.get('haystack_sessions', [])) for q in data]
print(f'Dataset: {len(data)} questions, {sum(sizes)/len(sizes):.0f} sessions/question avg (min={min(sizes)}, max={max(sizes)})')
"

# Verify ONNX model
.venv/bin/python3 -c "
from taosmd import VectorMemory
import asyncio, tempfile, os
async def test():
    tmp = tempfile.mkdtemp()
    vm = VectorMemory(os.path.join(tmp, 'v.db'), embed_mode='onnx', onnx_path='models/minilm-onnx')
    await vm.init()
    emb = await vm.embed('test')
    print(f'ONNX embedding: {len(emb)} dims, first 3: {emb[:3]}')
    await vm.close()
asyncio.run(test())
" 2>&1 | grep -v "W:onnxruntime" | grep -v "PyTorch"

echo ""
echo "Starting full 500-question matrix benchmark..."
echo "Expected runtime: ~20 minutes with RTX 3060"
echo ""

# Run with unbuffered output
PYTHONUNBUFFERED=1 .venv/bin/python3 -u benchmarks/longmemeval_matrix.py --limit 500 2>&1 | grep -v "W:onnxruntime" | grep -v "PyTorch"
