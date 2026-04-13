#!/bin/bash
set -e
echo "=========================================="
echo "  AURA - Complete Node Installation"
echo "  AI + Blockchain + Storage"
echo "=========================================="
echo ""

# 1. Ollama + AURA model
echo "[1/3] Setting up AURA AI model..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "Pulling Qwen 3.5-35B-A3B (Apache 2.0, MoE)..."
ollama pull qwen3.5:35b-a3b
echo "Creating AURA model..."
ollama create aura -f Modelfile
echo "AURA AI model ready."
echo ""

# 2. IPFS
echo "[2/3] Setting up IPFS storage node..."
bash scripts/install_ipfs.sh
echo ""

# 3. Summary
echo "[3/3] Setup complete!"
echo ""
echo "=========================================="
echo "  AURA Node is Ready"
echo "=========================================="
echo ""
echo "  AI:       ollama run aura"
echo "  Storage:  ipfs daemon"
echo "  Daemon:   python3 orchestrator/daemon.py"
echo "  Benchmark: python3 scripts/self_improve.py"
echo ""
echo "  Your machine will automatically:"
echo "  - Improve AURA when idle (mining)"
echo "  - Store and share LoRA deltas (IPFS)"
echo "  - Earn World Wide Currency tokens"
echo ""
echo "  Welcome to the network."
echo "=========================================="
