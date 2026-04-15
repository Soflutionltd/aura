<p align="center">
  <img src="https://raw.githubusercontent.com/Soflutionltd/aura/main/app/src-tauri/icons/128x128.png" width="80" alt="AURA">
</p>

<h1 align="center">AURA</h1>
<h3 align="center">Autonomous Unified Reasoning Architecture</h3>

<p align="center">
  <strong>The world's first self-improving local AI agent backed by its own blockchain.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#self-improvement">Self-Improvement</a> ·
  <a href="#blockchain">Blockchain</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/language-Rust%20%7C%20Python%20%7C%20TypeScript-orange" alt="Language">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-green" alt="Platform">
  <img src="https://img.shields.io/badge/model-Gemma%204%2026B--A4B-red" alt="Model">
  <img src="https://img.shields.io/badge/status-Alpha-yellow" alt="Status">
</p>

---

## What is AURA?

AURA is a **fully local, self-improving AI assistant** that runs on your hardware with zero cloud dependency. Unlike cloud AI services (ChatGPT, Claude, Gemini), AURA:

- **Improves itself every night** by analyzing your conversations, detecting mistakes, and training LoRA adapters on Apple Silicon
- **Learns from the internet 24/7** even when you don't talk to it, crawling HackerNews, Reddit, and the web for your interests
- **Has computer access** and can execute commands, manage files, browse the web, and control your Mac
- **Sees images and reads PDFs** natively via drag-and-drop anywhere in the window
- **Talks to you on WhatsApp, Telegram, Discord** via the Moltis gateway with voice support
- **Rewards contributors** with World Wide Currency (WWC) tokens on its own Substrate blockchain
- **Never sends your data anywhere** unless you opt-in to federated learning (anonymized gradients only)

> "Every other AI rents you intelligence. AURA gives you ownership of it."

---

## Why AURA Exists

Every AI assistant today runs in the cloud. Your conversations pass through someone else's servers. You pay per token. You're rate-limited. You can't modify the model. You can't make it smarter. You're renting intelligence from companies that can change the price, the rules, or shut it down at any time.

AURA inverts this model completely. It runs on your machine. It remembers everything. It improves itself while you sleep. And it's backed by a decentralized network that rewards everyone who makes it smarter.

---

## Quick Start

### macOS (Apple Silicon)

```bash
# Clone and install
git clone https://github.com/Soflutionltd/aura.git && cd aura
bash scripts/install.sh

# The installer auto-detects your hardware and installs:
# - Ollama + AURA model (Gemma 4 26B-A4B MoE)
# - oMLX inference server (MLX native, 3-4x faster)
# - MemoryPilot (persistent memory)
# - DFlash (speculative decoding, 3-5x speedup)
# - IPFS node (distributed storage)
# - Self-improvement daemons (nightly + enrichment)

# Launch the desktop app
cd app && npx tauri build && open src-tauri/target/release/bundle/macos/AURA.app

# Or use via WhatsApp/Telegram/Discord
brew install moltis-org/tap/moltis && moltis
```

### Windows / Linux (NVIDIA)

```bash
git clone https://github.com/Soflutionltd/aura.git && cd aura
bash scripts/install.sh
# Uses Ollama with CUDA/llama.cpp (MLX is macOS-only)
```

---

## Architecture

```
                            ┌─────────────────────────────────────┐
                            │           USER INTERFACES           │
                            ├─────────────────────────────────────┤
                            │  AURA.app   WhatsApp   Telegram     │
                            │  (Tauri)    Discord    Voice   API  │
                            └──────────────┬──────────────────────┘
                                           │
                            ┌──────────────▼──────────────────────┐
                            │         MOLTIS GATEWAY              │
                            │  (Rust, sandboxed, multi-channel)   │
                            │  Voice I/O · Memory · Sub-agents    │
                            │  Lifecycle hooks · MCP tools        │
                            └──────────────┬──────────────────────┘
                                           │
              ┌────────────────────────────▼────────────────────────────┐
              │                    AURA CORE                            │
              ├────────────────────────────────────────────────────────┤
              │                                                        │
              │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
              │  │ Gemma 4 26B  │  │ LeWorldModel │  │   Agent      │ │
              │  │ MoE (4B act) │  │  (15M params)│  │ Orchestrator │ │
              │  │ Vision+Tools │  │  <200ms sim  │  │ 5 tools      │ │
              │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
              │         │                 │                  │         │
              │  ┌──────▼─────────────────▼──────────────────▼───────┐ │
              │  │              INFERENCE ENGINE                      │ │
              │  │  oMLX (MLX native) > DFlash > Ollama (fallback)   │ │
              │  │  901 tok/s prefill · 85% cache · KV on SSD        │ │
              │  └───────────────────────────────────────────────────┘ │
              │                                                        │
              └────────────────────────┬───────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
┌────────▼────────┐  ┌────────────────▼───────────┐  ┌──────────────▼──────┐
│   MemoryPilot   │  │   Self-Improvement Engine   │  │   WWC Blockchain    │
│                 │  │                             │  │                     │
│ BM25 + Vector   │  │ Nightly: LoRA training     │  │ Substrate (Rust)    │
│ Knowledge Graph │  │ 6h: Web crawl + self-train │  │ Proof of Useful Work│
│ 8 projects      │  │ Feedback: implicit detect  │  │ Degressive rewards  │
│ Auto-compaction │  │ MiniMax M2.7 methodology   │  │ 10% burn mechanism  │
│ Cross-session   │  │ Multi-cycle optimization   │  │ Staking + slashing  │
└─────────────────┘  └────────────────────────────┘  │ 2-track governance  │
                                                      │ IPFS storage        │
                                                      └─────────────────────┘
```

---

## Performance

### Inference Speed by Backend

| Backend | Prefill (tok/s) | Generation (tok/s) | Cache Efficiency | Platform |
|---------|----------------|-------------------|-----------------|----------|
| **oMLX + DFlash** | **901** | **140-250** | **85%** | macOS Apple Silicon |
| oMLX (MLX native) | 901 | 36-60 | 85% | macOS Apple Silicon |
| Ollama 0.19 (MLX) | ~400 | 60-130 | None | macOS Apple Silicon |
| Ollama (llama.cpp) | ~100 | 15-40 | None | All platforms |
| Flash-MoE (SSD) | N/A | 4.4 | N/A | macOS 48GB+ |

### Hardware Tiers (Auto-Detected)

| Tier | RAM | Model | Active Params | Generation | Use Case |
|------|-----|-------|--------------|------------|----------|
| **Ultra** | 32+ GB | Gemma 4 26B-A4B Q8 + DFlash | 4B | 150-250 tok/s | Full power, instant responses |
| **Balanced** | 16+ GB | Gemma 4 26B-A4B Q4_K_M | 4B | 60-100 tok/s | Daily driver |
| **Eco** | 8-16 GB | Gemma 4 26B-A4B Q4 + mmap | 4B | 15-30 tok/s | SSD streaming, fits anywhere |
| **Minimal** | 4-8 GB | Gemma 4 E4B Q4 | 4B | 10-20 tok/s | Low-end hardware |
| **Flash** | 48+ GB | Qwen3.5-397B via Flash-MoE | 17B | 4-5 tok/s | Maximum intelligence |

### Intelligence Benchmarks (Gemma 4 26B-A4B)

| Benchmark | Score | vs GPT-5 | vs Opus 4.6 |
|-----------|-------|----------|-------------|
| Arena AI ELO | 1441 | Comparable | ~85% |
| MMLU | ~82% | ~95% of GPT-5 | ~80% |
| Coding (SWE-bench) | ~45% | ~75% | ~65% |
| Reasoning | Strong | Competitive | Competitive |
| Vision | Native | Native | Via tools |
| Tool Calling | Native | Native | Native |

> AURA with SuperGemma4 abliterated shows +8.5% total benchmark improvement and +119% on BenchTok vs base Gemma 4.

---

## Self-Improvement

AURA has two independent improvement systems that run **without user interaction**. This is not a prompt trick. The model weights actually change.

### Nightly Behavioral Improvement (3:00 AM)

Inspired by [MiniMax M2.7's self-evolution methodology](https://www.minimax.io/news/minimax-m27-en) which achieved +30% performance through 100+ autonomous optimization cycles.

```
┌─────────────────────────────────────────────────────────┐
│                  NIGHTLY PIPELINE                        │
│                                                          │
│  1. Collect conversations from MemoryPilot               │
│     └─> 100+ recent conversation memories                │
│                                                          │
│  2. Detect poor responses                                │
│     └─> Rejection signals: "non", "try again", "wrong"  │
│     └─> Reformulation detection                          │
│     └─> Correction pattern matching                      │
│                                                          │
│  3. Self-critique: generate better responses             │
│     └─> AURA generates improved versions of its failures │
│                                                          │
│  4. Create DPO dataset (good vs bad response pairs)      │
│     └─> JSONL format for mlx-lm training                 │
│                                                          │
│  5. Train LoRA adapter (mlx-lm on Apple Silicon)         │
│     └─> Rank 8, 200 steps, lr=1e-5                       │
│                                                          │
│  6. Fuse LoRA into base model + reload in Ollama/oMLX    │
│     └─> Model weights permanently modified               │
│                                                          │
│  Result: AURA is measurably smarter every morning.       │
└─────────────────────────────────────────────────────────┘
```

### Knowledge Enrichment (Every 6 Hours)

Even if you don't talk to AURA for a week, it keeps learning:

```
┌─────────────────────────────────────────────────────────┐
│              KNOWLEDGE ENRICHMENT                        │
│                                                          │
│  Sources:                                                │
│  ├─ HackerNews (top 30, filtered by interests)           │
│  ├─ Reddit (r/LocalLLaMA, r/MachineLearning, r/substrate)│
│  ├─ DuckDuckGo (per-interest topic search)               │
│  └─ Custom RSS feeds (configurable)                      │
│                                                          │
│  Pipeline:                                               │
│  1. Crawl sources ──> raw articles                        │
│  2. AURA summarizes ──> key facts extracted               │
│  3. Store in MemoryPilot ──> vector DB                    │
│  4. Generate Q&A pairs ──> self-training data             │
│  5. Feed into nightly LoRA training                       │
│                                                          │
│  Result: AURA stays informed about AI, blockchain,       │
│  tech, and your custom interests. Automatically.         │
└─────────────────────────────────────────────────────────┘
```

### Improvement vs Cloud Models

| Feature | AURA | Claude | ChatGPT | Gemini |
|---------|------|--------|---------|--------|
| Learns from your conversations | Yes (LoRA) | No | No | No |
| Learns from internet autonomously | Yes (6h cycle) | No | No | No |
| Model weights change over time | Yes | No | No | No |
| Works offline | Yes | No | No | No |
| Free forever | Yes | $20/mo | $20/mo | $20/mo |
| Data stays local | Yes | No | No | No |
| Federated learning network | Yes (WWC) | No | No | No |

---

## Features

### Desktop App (AURA.app)

- Clean white interface with AURA branding
- Streaming responses with animated loading dots
- **Image analysis**: drag-drop, paste, or upload images anywhere in the window
- **PDF analysis**: drag-drop PDFs, automatic text extraction via pypdf
- **Agent mode**: toggle button gives AURA access to your terminal, files, and web
- **Auto-update**: checks GitHub Releases every 30min, one-click update + restart
- **Conversation history**: persistent, searchable
- **Web search**: automatic when you ask about current events

### Multi-Channel Access (via Moltis)

- **WhatsApp**: talk to AURA from your phone like a contact
- **Telegram**: bot integration
- **Discord**: server bot
- **Voice**: 8 TTS + 7 STT providers, talk and listen
- **Web UI**: https://localhost:57791
- **Mobile PWA**: add to home screen with push notifications
- **API**: JSON-RPC for programmatic access

### Agent Capabilities

- `execute_command`: run any shell command on the local machine
- `read_file` / `write_file`: manage files anywhere on disk
- `list_directory`: browse the filesystem
- `web_search`: search the internet in real-time
- `extract_pdf`: read and analyze PDF documents
- Tool calling chains up to 10 iterations autonomously

---

## World Wide Currency (WWC)

A cryptocurrency that rewards intelligence, not electricity.

### Design Principle

The project is split into two components with a critical separation:

**The blockchain (modifiable by governance)**: A fork of Polkadot SDK (Substrate), 1.5 million lines of battle-tested Rust. The infrastructure evolves via community votes: runtime upgrades, new features, bridges, staking rules.

**The cryptocurrency (permanently immutable)**: The monetary rules are locked in the smart contract forever. No vote, no admin, no governance proposal can ever modify them.

### Monetary Rules (Immutable)

```
Supply:            Unlimited, purely algorithmic
Emission:          Proof of Useful Work only (training AURA)
Reward formula:    BASE_REWARD / sqrt(active_miners / 10)
Burn rate:         10% of every reward (deflationary)
Validation:        3 independent staked validators required
Staking minimum:   100 WWC to validate
Slashing:          50% of stake on bad validation
Admin functions:   ZERO. No mint, no pause, no override.
```

### Reward Curve

```
Miners:     10     100     1,000    10,000    100,000
Reward:    100     31.6    10       3.16      1 WWC (minimum)
```

### Governance (Two Tracks)

| Track | Scope | Who Votes | Can Modify Monetary Rules? |
|-------|-------|-----------|--------------------------|
| **Technical** | Runtime upgrades, new pallets, bridges, staking | WWC holders | No |
| **Monetary** | Token emission, burn rate, reward formula | Nobody | **FORBIDDEN AT CODE LEVEL** |

### Distribution

| Allocation | Percentage | Vesting |
|-----------|-----------|---------|
| Contributors (miners) | 40% | Immediate |
| Independent Foundation | 20% | 2 years |
| Founders | 15% | 4 years linear |
| Validators | 15% | Immediate |
| Public Sale | 10% | None |

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Model | Gemma 4 26B-A4B MoE | Core intelligence (Apache 2.0) |
| Inference (macOS) | oMLX + DFlash | MLX native, KV cache SSD, 4x speedup |
| Inference (all) | Ollama 0.19 | Fallback, cross-platform |
| Desktop App | Tauri (Rust + HTML/JS) | Native macOS app |
| Agent Gateway | Moltis (Rust) | WhatsApp, Telegram, Discord, voice |
| Memory | MemoryPilot (Rust) | BM25 + vector + knowledge graph |
| Training | mlx-lm (Python) | LoRA fine-tuning on Apple Silicon |
| Blockchain | Substrate / Polkadot SDK (Rust) | WWC network |
| Storage | IPFS | Distributed LoRA delta storage |
| CI/CD | GitHub Actions | Automated testing |
| Quantization | JANG + oQ4 | Adaptive per-layer, best quality |

---

## Project Structure

```
aura/
├── orchestrator/                    # Core intelligence (Python)
│   ├── daemon.py                    # Master daemon, connects all modules
│   ├── nightly_improve.py           # LoRA training pipeline (REAL weight modification)
│   ├── knowledge_enrichment.py      # Internet crawling + self-training
│   ├── self_improve_engine.py       # 6-step self-improvement (586 lines)
│   ├── inference_engine.py          # Auto-detect: oMLX > DFlash > MLX > Ollama
│   ├── agent_orchestrator.py        # ReAct loop + 5 tools + World Model
│   ├── federated_server.py          # FedProx + Differential Privacy
│   ├── world_enhanced_moe.py        # 3-phase MoE convergence
│   ├── lewm_simulator.py            # LeWorldModel simulator
│   ├── hardware_detector.py         # 4-tier auto-detection
│   ├── expert_specialization.py     # MoE expert tracking
│   ├── trajectory_distillation.py   # DPO + SFT from agent trajectories
│   └── auto_updater.py              # 3-level update system
│
├── app/                             # Desktop application (Tauri)
│   ├── src/index.html               # UI: image/PDF drag-drop, agent mode
│   ├── src/app.js                   # Chat, vision, tool calling, streaming
│   ├── src/style.css                # Clean white theme
│   ├── src-tauri/tauri.conf.json    # App config + auto-updater
│   └── agent-server.js              # Local HTTP server for agent tools
│
├── scripts/                         # Installation and utilities
│   ├── install.sh                   # One-command installer
│   ├── install_ipfs.sh              # IPFS setup
│   ├── ipfs_integration.py          # LoRA publish/retrieve on IPFS
│   ├── mlx_acceleration.py          # DFlash + TurboQuant
│   └── speed_test.py                # Benchmark all backends
│
├── prompts/                         # Research and optimization
│   └── grok-speed-intelligence.md   # Speed vs intelligence optimization
│
├── training/                        # Auto-generated training data
│   ├── datasets/                    # DPO + SFT JSONL files
│   └── lora_adapters/               # Trained LoRA weights
│
├── Modelfile                        # Ollama model definition
├── README.md                        # This file
└── LICENSE                          # Apache 2.0
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | Complete | Chat + Agent + LeWM + MoE + DFlash + IPFS + WWC pallets |
| **Phase 2** | Complete | Federated learning + Trajectory KD + DPO + Expert specialization |
| **Phase 3** | Complete | World-Enhanced MoE + Nightly self-improvement + Knowledge enrichment |
| **Phase 4** | In Progress | oMLX inference + Moltis gateway + SuperGemma4 + WhatsApp/Telegram |
| **Phase 5** | Planned | Blockchain testnet + Public beta + Mobile app + Multi-agent swarm |
| **Phase 6** | Planned | Mainnet launch + Token sale + Enterprise API + Browser extension |

---

## Related Repositories

| Repository | Description | Language |
|-----------|-------------|---------|
| [Soflutionltd/aura](https://github.com/Soflutionltd/aura) | AURA core: LLM + orchestrator + app | Python, JS, Rust |
| [Soflutionltd/aura-inference](https://github.com/Soflutionltd/aura-inference) | Inference server (oMLX fork, MLX native) | Python |
| [Soflutionltd/wwc](https://github.com/Soflutionltd/wwc) | WWC cryptocurrency pallets (immutable) | Rust |
| [Soflutionltd/wwc-blockchain](https://github.com/Soflutionltd/wwc-blockchain) | Full blockchain (Polkadot SDK fork, 1.5M lines) | Rust |
| [Soflutionltd/moltis](https://github.com/Soflutionltd/moltis) | Agent gateway: WhatsApp, Telegram, voice | Rust |
| [Soflutionltd/ollama](https://github.com/Soflutionltd/ollama) | Ollama (fallback inference for Windows/Linux) | Go |
| [Soflutionltd/macos-launchy](https://github.com/Soflutionltd/macos-launchy) | macOS app launcher (Launchpad replacement) | Swift |

---

## Research Influences

AURA's design is informed by cutting-edge research and projects:

- **MiniMax M2.7** : Self-evolution methodology (100+ autonomous optimization cycles, +30% performance)
- **Flash-MoE** : Running 397B models on MacBook via SSD streaming + Metal shaders
- **DFlash MLX** : Speculative decoding for 3-5x inference speedup on Apple Silicon
- **Substrate/Polkadot** : Runtime-upgradable blockchain with on-chain governance
- **Flower/FedProx** : Federated learning with differential privacy
- **SuperGemma4** : Abliterated models (+8.5% benchmarks, +119% BenchTok, zero censure)
- **JANG Quantization** : Adaptive per-layer quantization (attention 4-8 bit, MLP 2-6 bit)
- **oMLX** : Two-tier KV cache (RAM + SSD) for 10x prefill speedup
- **Moltis** : Secure Rust agent gateway with sandboxed execution
- **AirLLM** : Layer-by-layer SSD offloading for running 70B on 4GB RAM

---

## Contributing

AURA is open source (Apache 2.0). Contributions are welcome.

Every meaningful contribution earns WWC tokens when the blockchain launches. The more you improve AURA, the more tokens you earn. This is Proof of Useful Work.

```bash
git clone https://github.com/Soflutionltd/aura.git
cd aura
# Read the code, find something to improve, submit a PR
```

---

## License

Apache 2.0. Free to use, modify, and distribute.

---

<p align="center">
  <strong>Created by Antoine Pinelli</strong><br>
  <a href="https://soflution.com">Soflution Ltd</a> (UK, Companies House 16910478)<br>
  <a href="https://lavilla.fr">La Villa Calvi</a> · <a href="https://lavillart.com">LaVillArt</a>
</p>
