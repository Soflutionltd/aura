#!/bin/bash
set -e
echo "=========================================="
echo "  AURA - Complete Node Installation"
echo "  The most advanced local AI agent"
echo "=========================================="
echo ""

# ── Detect hardware ──────────────────────────────────────────
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2*1024}')
RAM_GB=$((RAM_BYTES / 1073741824))
echo "Detected: ${RAM_GB} GB RAM"

# Select model tier based on RAM
if [ "$RAM_GB" -ge 32 ]; then
    MODE="ultra"
    MODEL_TAG="Q8_0"
    MODEL_SIZE="26 GB"
    echo "Mode: ULTRA (full MoE Q8 + LeWM + DFlash)"
elif [ "$RAM_GB" -ge 16 ]; then
    MODE="balanced"
    MODEL_TAG="IQ4_NL"
    MODEL_SIZE="14.7 GB"
    echo "Mode: BALANCED (MoE IQ4_NL + LeWM + DFlash)"
elif [ "$RAM_GB" -ge 8 ]; then
    MODE="eco"
    echo "Mode: ÉCO (Gemma 4 E4B lightweight)"
else
    MODE="minimal"
    echo "Mode: MINIMAL (Gemma 4 E2B)"
fi
echo ""

# ── 1. Install Ollama ────────────────────────────────────────
echo "[1/4] Setting up Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "Ollama ready."
echo ""

# ── 2. Download AURA model ───────────────────────────────────
echo "[2/4] Downloading AURA model ($MODEL_SIZE)..."
if [ "$MODE" = "ultra" ] || [ "$MODE" = "balanced" ]; then
    # Gemma 4 26B-A4B Opus Distill (MoE, distilled from Claude Opus 4.6)
    ollama create aura -f Modelfile
elif [ "$MODE" = "eco" ]; then
    ollama pull gemma4:e4b
    # Create AURA alias with system prompt
    cat > /tmp/aura_eco_modelfile << 'EOF'
FROM gemma4:e4b
SYSTEM "Tu es AURA, un assistant IA local intelligent et privé. Sois concis, direct et utile. Réponds dans la langue de l'utilisateur."
EOF
    ollama create aura -f /tmp/aura_eco_modelfile
else
    ollama pull gemma4:e2b
    cat > /tmp/aura_min_modelfile << 'EOF'
FROM gemma4:e2b
SYSTEM "Tu es AURA, un assistant IA local. Sois concis et utile."
EOF
    ollama create aura -f /tmp/aura_min_modelfile
fi
echo "AURA model ready."
echo ""

# ── 3. Install IPFS ──────────────────────────────────────────
echo "[3/4] Setting up IPFS storage node..."
bash scripts/install_ipfs.sh
echo ""

# ── 4. Install acceleration (Ultra/Balanced only) ────────────
if [ "$MODE" = "ultra" ] || [ "$MODE" = "balanced" ]; then
    echo "[4/4] Installing MLX acceleration (DFlash + TurboQuant)..."
    pip3 install mlx mlx-lm 2>/dev/null || true
    if [ ! -d "vendor/dflash-mlx" ]; then
        git clone https://github.com/Aryagm/dflash-mlx.git vendor/dflash-mlx 2>/dev/null || true
    fi
    echo "Acceleration ready."
else
    echo "[4/4] Skipping acceleration (not needed for $MODE)"
fi

echo ""
echo "=========================================="
echo "  AURA Node is Ready ($MODE)"
echo "=========================================="
echo ""
echo "  AI:         ollama run aura"
echo "  Storage:    ipfs daemon"
echo "  Daemon:     python3 orchestrator/daemon.py"
echo "  Benchmark:  python3 scripts/self_improve.py"
echo "  Hardware:   python3 orchestrator/hardware_detector.py"
echo ""
echo "  RAM: ${RAM_GB} GB | Mode: ${MODE}"
echo ""
echo "  Your machine will automatically:"
echo "  - Improve AURA when idle (mining)"
echo "  - Store and share LoRA deltas (IPFS)"
echo "  - Earn World Wide Currency tokens"
echo ""
echo "  Welcome to the network."
echo "=========================================="
