# AURA -- Autonomous Unified Reasoning Architecture

The most advanced local AI agent in the world. Free. Private. Self-improving. 24/7.

AURA is a decentralized, self-evolving AI that runs entirely on your machine. It combines a state-of-the-art Mixture-of-Experts model with autonomous agent capabilities, real-time internet access, vision, PDF analysis, persistent memory, and a community-powered federated learning network backed by its own blockchain and cryptocurrency.

No cloud. No tokens. No subscriptions. No data leaves your machine. Forever.

## Why AURA

Every AI assistant today runs in the cloud. Your conversations, your files, your ideas pass through someone else's servers. You pay per token. You're rate-limited. You can't modify the model. You can't make it smarter. You're renting intelligence.

AURA is the opposite. It runs on your hardware. It remembers everything. It improves itself every night while you sleep. It crawls the internet to stay informed even when you don't talk to it. And it's backed by a blockchain that rewards everyone who makes it smarter.

## What makes AURA different

**Self-improving 24/7**: Two background daemons run continuously. The Nightly Pipeline analyzes your conversations, detects poor responses, generates improved versions, trains a LoRA adapter with MLX on Apple Silicon, and merges it into the model. The Knowledge Enrichment daemon crawls HackerNews, Reddit, and the web every 6 hours, summarizes findings, stores them in a local vector database, and generates self-training data. AURA genuinely becomes smarter every day. This is not a prompt trick. The model weights actually change.

**Vision and document understanding**: Drop images or PDFs anywhere in the chat window. AURA analyzes images natively with Gemma 4's built-in vision. PDFs are extracted and analyzed as context. Supports drag-and-drop, paste from clipboard, and file upload.

**Agent mode with computer access**: Toggle Agent Mode and AURA can execute shell commands, read and write files, list directories, and search the web. It uses Gemma 4's native tool calling to chain multiple operations autonomously.

**World-Enhanced MoE**: AURA integrates a LeWorldModel (15M params) that simulates consequences before acting. Instead of trial-and-error tool calls, it anticipates outcomes and executes the optimal path. Over time, the MoE absorbs these capabilities through a 3-phase convergence process.

**Persistent memory**: Every conversation is stored in MemoryPilot, a hybrid BM25 + TF-IDF search engine with a knowledge graph. AURA remembers everything across sessions automatically.

**Federated learning**: Users contribute anonymized model improvements to the network. Privacy-preserving differential privacy, quality-gated by 3 independent validators, and rewarded with World Wide Currency tokens.

**Hardware adaptive**: Automatically detects your system (Apple Silicon tier, RAM, GPU) and selects the optimal configuration. From 4 GB laptops to 96 GB workstations, everyone gets the best AURA for their hardware.

## Quick start

```bash
git clone https://github.com/Soflutionltd/aura.git && cd aura
bash scripts/install.sh
```

The installer detects your hardware and configures everything automatically: Ollama, the AURA model (Gemma 4 26B-A4B MoE), MLX acceleration, IPFS, MemoryPilot, and the self-improvement daemons.

To launch the desktop app:

```bash
cd app && npx tauri build --debug
open src-tauri/target/debug/bundle/macos/AURA.app
```

To use via Moltis (WhatsApp, Telegram, Discord, voice):

```bash
brew install moltis-org/tap/moltis
moltis
```

Then open https://localhost:57791 and configure your channels.

## Architecture

```
User (App / WhatsApp / Telegram / Discord / Voice)
  |
  v
Moltis Gateway (Rust, sandboxed, multi-channel)
  |
  v
AURA Core
  |
  +-- Gemma 4 26B-A4B (MoE, 4B active params, Q4_K_M)
  |     Main brain: reasoning, tool calling, vision, thinking
  |
  +-- LeWorldModel (15M params, <200ms)
  |     Simulates outcomes before acting
  |
  +-- Agent Orchestrator (ReAct loop + 5 tools)
  |     execute_command, read_file, write_file, list_directory, web_search
  |
  +-- MemoryPilot (persistent vector + knowledge graph)
  |
  +-- Nightly Self-Improvement Pipeline
  |     Conversation analysis > Feedback detection > Self-critique
  |     > DPO dataset > LoRA training (MLX) > Merge > Reload
  |
  +-- Knowledge Enrichment Daemon (every 6 hours)
  |     HackerNews + Reddit + Web > Summarize > Store > Self-train
  |
  +-- DFlash + MLX (3-5x speedup on Apple Silicon)
  |
  +-- IPFS Node (distributed LoRA delta storage)
  |
  +-- WWC Blockchain Node (Proof of Useful Work rewards)
```

## Model

AURA currently runs on Gemma 4 26B-A4B, a Mixture-of-Experts model with 25.8 billion total parameters but only 4 billion active per token. This gives the intelligence of a large model with the speed of a small one.

Capabilities: text completion, vision (image analysis), tool calling, chain-of-thought reasoning.

Context window: 262,144 tokens.

Quantization: Q4_K_M (18 GB on disk).

On Apple Silicon with MLX + DFlash, expect 60-150+ tokens per second.

## Hardware tiers

| Mode | RAM | Model | Active params | Speed |
|------|-----|-------|---------------|-------|
| Ultra | 32+ GB | Gemma 4 26B-A4B Q8 + DFlash | 4B | 150-250 tok/s |
| Balanced | 16+ GB | Gemma 4 26B-A4B Q4_K_M | 4B | 60-100 tok/s |
| Eco | 8-16 GB | Gemma 4 26B-A4B Q4_K_M + mmap | 4B | 15-30 tok/s |
| Minimal | 4-8 GB | Gemma 4 E4B Q4 | 4B | 10-20 tok/s |
| Flash-MoE | 48+ GB | Qwen3.5-397B via SSD streaming | 17B | 4-5 tok/s |

The hardware detector automatically selects the best tier for your machine.

## Self-improvement: how it works

AURA has two independent improvement systems that run without user interaction.

### Nightly behavioral improvement (3:00 AM daily)

1. Collects all conversations from MemoryPilot
2. Detects poor responses using rejection signals ("non", "c'est pas ca", "try again", etc.)
3. Asks AURA itself to generate better responses for each failure (self-critique)
4. Creates a DPO dataset (good response vs bad response)
5. Trains a LoRA adapter using mlx-lm on Apple Silicon
6. Fuses the LoRA into the base model weights
7. Reloads the improved model in Ollama

The model genuinely changes. This is real weight modification, not retrieval-augmented generation.

### Knowledge enrichment (every 6 hours)

1. Crawls HackerNews top stories filtered by user interests
2. Crawls Reddit (r/LocalLLaMA, r/MachineLearning, r/substrate)
3. Searches DuckDuckGo for each interest topic
4. Uses AURA itself to summarize and extract key facts
5. Stores summaries in MemoryPilot vector database
6. Generates Q&A self-training pairs from new knowledge
7. These pairs feed into the nightly LoRA training

Even if you don't talk to AURA for a week, when you come back it knows what happened in AI, blockchain, and your other interests.

## World Wide Currency (WWC)

WWC is a cryptocurrency that rewards people who improve AURA. Mining means making the AI smarter, not wasting electricity.

The project is split into two components with a critical design principle:

**The blockchain (modifiable)**: A fork of Polkadot SDK (Substrate), 1.5 million lines of Rust. The infrastructure can evolve via community governance: runtime upgrades, new features, bug fixes, bridges, staking. All voted by token holders.

**The cryptocurrency (immutable)**: The monetary rules are permanently locked in the smart contract. No vote, no governance proposal, no admin can ever modify them:

- Unlimited supply, purely algorithmic emission
- Tokens created only by Proof of Useful Work (training AURA)
- Degressive rewards: BASE_REWARD / sqrt(active_miners / 10)
- 10% of every reward is burned (deflationary pressure)
- Validation by 3 independent staked validators
- Validators must stake tokens (slashing on bad validation: 50%)
- Zero admin mint, zero backdoor, zero override

Governance has two tracks: Technical (votable by community) and Monetary (forbidden at code level). Nobody can change the money.

### Repos

- Blockchain: [Soflutionltd/wwc-blockchain](https://github.com/Soflutionltd/wwc-blockchain) (Polkadot SDK fork, 1.5M lines Rust)
- Crypto pallets: [Soflutionltd/wwc](https://github.com/Soflutionltd/wwc) (token, storage, governance)

## Project structure

```
orchestrator/
  daemon.py                  Master daemon (connects all modules)
  self_improve_engine.py     6-step self-improvement pipeline
  nightly_improve.py         Nightly LoRA training pipeline (REAL weight modification)
  knowledge_enrichment.py    Internet crawling + self-training generation
  inference_engine.py        Auto-detect backend: DFlash > MLX > Ollama
  agent_orchestrator.py      ReAct loop + 7 tools + World Model
  federated_server.py        FedProx + Differential Privacy + Trajectory KD
  world_enhanced_moe.py      3-phase MoE convergence (absorbs LeWM)
  lewm_simulator.py          LeWorldModel simulator
  hardware_detector.py       4-tier auto-detection
  expert_specialization.py   MoE expert tracking by task type
  trajectory_distillation.py DPO + SFT from agent trajectories
  auto_updater.py            3-level update system

scripts/
  install.sh                 One-command installer with hardware detection
  install_ipfs.sh            IPFS setup (macOS + Linux)
  ipfs_integration.py        Publish/retrieve LoRA on IPFS
  mlx_acceleration.py        DFlash + TurboQuant setup
  speed_test.py              Benchmark all backends
  benchmark_ci.py            CI/CD benchmark

prompts/
  grok-speed-intelligence.md Optimization research prompt

app/                         AURA desktop app (Tauri: Rust + HTML/JS/CSS)
  src/index.html             UI with image/PDF drag-drop, agent mode
  src/app.js                 Chat, vision, tool calling, streaming
  src/style.css              Clean white theme
  agent-server.js            Local HTTP server for agent tool execution
```

## Roadmap

Phase 1 (complete): Chat + Agent + LeWM + MoE + DFlash + IPFS + WWC pallets

Phase 2 (complete): Federated learning + Trajectory KD + DPO + Expert specialization

Phase 3 (complete): World-Enhanced MoE convergence + Nightly self-improvement + Knowledge enrichment

Phase 4 (in progress): Moltis integration (WhatsApp/Telegram/Discord/Voice) + SuperGemma4 + MLX native inference + Flash-MoE for 100B+ models

Phase 5 (planned): Blockchain mainnet launch + Public testnet + Mobile app + Multi-agent swarm

## Technology stack

- Model: Gemma 4 26B-A4B MoE (Apache 2.0)
- Inference: Ollama + MLX + DFlash (Apple Silicon native)
- App: Tauri (Rust backend + HTML/JS frontend)
- Gateway: Moltis (Rust, sandboxed, multi-channel)
- Memory: MemoryPilot (hybrid BM25 + vector search + knowledge graph)
- Training: mlx-lm (LoRA fine-tuning on Apple Silicon)
- Storage: IPFS (distributed LoRA delta storage)
- Blockchain: Substrate / Polkadot SDK (Rust)
- Crypto: Custom pallets (token + storage + governance)
- CI/CD: GitHub Actions

## Related repositories

- [Soflutionltd/aura](https://github.com/Soflutionltd/aura) -- AURA LLM + orchestrator + app
- [Soflutionltd/wwc](https://github.com/Soflutionltd/wwc) -- WWC cryptocurrency pallets
- [Soflutionltd/wwc-blockchain](https://github.com/Soflutionltd/wwc-blockchain) -- Full blockchain (Polkadot SDK fork)
- [Soflutionltd/moltis](https://github.com/Soflutionltd/moltis) -- Moltis agent gateway (fork)
- [Soflutionltd/ollama](https://github.com/Soflutionltd/ollama) -- Ollama (fork)
- [Soflutionltd/macos-launchy](https://github.com/Soflutionltd/macos-launchy) -- Launchy app launcher (fork)

## License

Apache 2.0

## Created by

Antoine Pinelli / [Soflution Ltd](https://soflution.com) (UK, Companies House 16910478)
