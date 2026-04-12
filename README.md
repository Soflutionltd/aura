<div align="center">

# ✦ AURA
### Autonomous Unified Reasoning Architecture

**The world's first community-driven, self-improving local AI.**
Zero tokens. Zero cost. Zero limits.

![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Gemma 4 31B](https://img.shields.io/badge/Base_Model-Gemma_4_31B-green.svg)
![Ollama](https://img.shields.io/badge/Runtime-Ollama-orange.svg)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Optimized-black.svg)

</div>

---

## The Vision

Every AI model today is a black box controlled by a corporation.

You pay per token. Your data leaves your machine. You have no control.

**AURA is different.** It runs entirely on your hardware. It improves itself using its own intelligence. Every contribution from every machine in the world makes it smarter. No one owns it. Everyone builds it.

This is what AI should have been from the start.

---

## Why AURA is different

| | |
|---|---|
| **∅ Zero token cost** | Runs 100% locally via Ollama. No API keys. No per-request fees. Ever. |
| **↺ Self-improving** | AURA analyzes its own configuration and proposes improvements. The model makes itself better. |
| **◎ Community-driven** | Every contributor's benchmark results and fine-tunes are merged back. Every machine helps. |
| **⊡ Fully private** | Your prompts, your code, your data — all stay on your machine. Nothing is sent anywhere. |
| **⬡ No gatekeepers** | Apache 2.0 license. Use it, fork it, build on it commercially. No restrictions. |
| **◈ Multimodal** | Built on Gemma 4 31B. Handles text, images, and code with a 256K context window. |

---

## Get started in 3 commands

```bash
# 1. Clone the project
git clone https://github.com/Soflution1/aura-llm

# 2. Install AURA
./scripts/install.sh

# 3. Run AURA locally
ollama run aura
```

---

## How to contribute

**01 — Run the self-improvement agent**
Run `python3 scripts/self_improve.py`. AURA analyzes itself and generates benchmark results. Submit a PR. Takes 5 minutes.

**02 — Improve the Modelfile**
Found better parameters for your hardware? Edit the Modelfile, benchmark it, and submit. Your improvement ships to everyone.

**03 — Add a fine-tune**
Specialized AURA for medicine, law, Rust, or your language? Add it to `fine-tunes/` with your training recipe.

**04 — Improve the agent**
The self-improvement agent in `scripts/self_improve.py` is the heart of AURA. Make it smarter. This is the most impactful contribution.

---

## Hardware compatibility

| Hardware | RAM | Speed | Status |
|----------|-----|-------|--------|
| Mac M2 Max | 96 GB | ~30 tok/s | ✅ Tested |
| Mac M3/M4 Ultra | 192 GB | ~60 tok/s | ✅ Compatible |
| RTX 4090 | 24 GB VRAM | ~40 tok/s | ✅ Compatible |
| RTX 5090 | 32 GB VRAM | ~55 tok/s | 🔜 Coming soon |
| Any machine 32GB+ | 32 GB RAM | ~5 tok/s | ✅ Via SSD offload |

---

## Project structure

```
aura/
├── Modelfile              # AURA's core configuration
├── scripts/
│   ├── install.sh         # One-command setup
│   └── self_improve.py    # Self-improvement agent
├── fine-tunes/            # Community fine-tune contributions
├── benchmarks/            # Benchmark results by hardware
└── docs/CONTRIBUTING.md
```

---

<div align="center">

Built by [Soflution Ltd](https://soflution.com) · Apache 2.0

**AURA is not a product. It is a protocol.**
**Anyone can run it. Anyone can improve it. No one owns it.**

</div>
