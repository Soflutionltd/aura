#!/bin/bash
set -e
echo "=== AURA Installation ==="
echo "Autonomous Unified Reasoning Architecture"
echo ""

if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Pulling Qwen 3.5-35B-A3B (Apache 2.0, MoE, ~20GB)..."
ollama pull qwen3.5:35b-a3b

echo "Creating AURA model..."
ollama create aura -f Modelfile

echo ""
echo "AURA is ready!"
echo "Run:  ollama run aura"
echo "Benchmark:  python3 scripts/self_improve.py"
echo "Daemon:  python3 orchestrator/daemon.py"
