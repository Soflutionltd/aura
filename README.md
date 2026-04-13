# AURA — Autonomous Unified Reasoning Architecture

The most advanced local AI agent in the world. Free. Private. Self-improving.

AURA is a decentralized, self-evolving AI that runs entirely on your machine. It combines a state-of-the-art Mixture-of-Experts model (distilled from Claude Opus 4.6) with a lightweight World Model simulator, autonomous agent capabilities, and a community-powered federated learning network.

**No cloud. No tokens. No subscriptions. Forever.**

## What makes AURA unique

**World-Enhanced MoE**: AURA uses a LeWorldModel (15M params) to simulate consequences before acting. Instead of trial-and-error (10-15 tool calls), it anticipates outcomes and executes the best path (2-4 calls). Over time, the MoE absorbs these capabilities and becomes a single ultra-intelligent model.

**Self-improving**: A background daemon continuously diagnoses weaknesses, generates training data, fine-tunes via LoRA, and validates improvements. Every cycle makes AURA smarter.

**Federated learning**: Thousands of users contribute anonymized improvements. Privacy-preserving (Differential Privacy), quality-gated (3 validators minimum), and rewarded with World Wide Currency tokens.

**Autonomous agent**: 7 built-in tools (code execution, file system, RAG search, terminal, web search, image analysis, self-improvement). AURA plans, executes, verifies, and iterates autonomously.

**Hardware adaptive**: Automatically detects your system and selects the optimal configuration. From 4GB laptops to 96GB workstations, everyone gets the best AURA for their hardware.

## Quick start

```bash
git clone https://github.com/Soflution1/aura.git && cd aura
bash scripts/install.sh
ollama run aura
```

One command installs everything: Ollama, the AURA model, IPFS, and acceleration layers. The installer detects your hardware and selects the right model tier automatically.

## Architecture

```
User
  |
  v
AURA App (Tauri: Rust + HTML/JS/CSS)
  |
  +-- Gemma 4 26B-A4B Opus Distill (MoE, 4B active params)
  |     Main brain: reasoning, tool calling, conversation
  |
  +-- LeWorldModel (15M params, <100 Mo, <200ms)
  |     Simulates outcomes before acting
  |
  +-- Agent Orchestrator (ReAct loop + World Model)
  |     7 autonomous tools, max 15 iterations
  |
  +-- MemoryPilot (persistent memory across conversations)
  |
  +-- DFlash + TurboQuant (3-5x speed, 4.6x KV compression)
  |
  +-- Self-Improvement Engine (diagnose, generate, train, validate)
  |
  +-- IPFS Node (distributed storage for LoRA deltas)
  |
  +-- World Wide Currency Node (mining + rewards)
```

## Model tiers

| Mode | RAM | Model | Active params | Speed |
|------|-----|-------|---------------|-------|
| Ultra | 32+ GB | Gemma 4 26B-A4B Opus Q8 | 4B | 160-250 t/s |
| Balanced | 16+ GB | Gemma 4 26B-A4B Opus IQ4_NL | 4B | 200+ t/s |
| Éco | 8+ GB | Gemma 4 E4B | 4B | 300+ t/s |
| Minimal | 4+ GB | Gemma 4 E2B | 2B | 400+ t/s |

## Roadmap

**Phase 1 (done)**: Chat + Agent + LeWM + MoE Opus Distill + DFlash + IPFS + WWC

**Phase 2 (done)**: Federated learning + Trajectory KD + DPO + Expert specialization

**Phase 3 (done)**: World-Enhanced MoE convergence (MoE absorbs LeWM capabilities)

## World Wide Currency (WWC)

Decentralized token that rewards contributors. Mining = improving AURA (Proof of Useful Work). Degressive rewards protect against inflation as the network grows.

Blockchain: Substrate (Rust, Apache 2.0). Storage: IPFS (off-chain) + hash on-chain.

## License

Apache 2.0

## Created by

Antoine Pinelli / Soflution Ltd (UK, Companies House 16910478)

https://github.com/Soflution1/aura
https://github.com/Soflution1/wwc
