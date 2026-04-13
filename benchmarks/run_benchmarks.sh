#!/bin/bash
# Run all benchmark variants on Fedora in screen sessions
# Usage: bash run_benchmarks.sh

set -e
cd ~/taosmd

# Kill any running benchmarks
pkill -f realworld_llm_benchmark 2>/dev/null || true
sleep 2

echo "=== Starting Pi-equivalent benchmark (qwen3:4b, 500 questions) ==="
screen -dmS bench-4b bash -c "cd ~/taosmd && .venv/bin/python3 -u benchmarks/realworld_llm_benchmark.py --limit 500 --model qwen3:4b 2>&1 | tee /tmp/taosmd_bench_4b.log; echo 'DONE' >> /tmp/taosmd_bench_4b.log"
echo "Started screen session: bench-4b"

echo ""
echo "=== Queuing GPU benchmark (qwen3.5:9b, starts after 4b finishes) ==="
screen -dmS bench-9b bash -c "while screen -ls | grep bench-4b > /dev/null 2>&1; do sleep 30; done; cd ~/taosmd && .venv/bin/python3 -u benchmarks/realworld_llm_benchmark.py --limit 500 --model qwen3.5:9b 2>&1 | tee /tmp/taosmd_bench_9b.log; echo 'DONE' >> /tmp/taosmd_bench_9b.log"
echo "Started screen session: bench-9b (will wait for bench-4b to finish)"

echo ""
echo "Screen sessions:"
screen -ls
echo ""
echo "Monitor with: tail -f /tmp/taosmd_bench_4b.log"
echo "GPU usage: nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"
