# Soflution LLM

**A self-improving open source LLM that runs 100% locally. Zero token cost. Zero cloud dependency.**

Built on Gemma 4 31B, optimized for Apple Silicon and consumer hardware.

## The Vision

- Runs entirely on your local machine via Ollama
- Uses Gemma 4 31B (Apache 2.0) as base
- Self-improves by analyzing its own configuration
- Every contributor's machine makes the model better
- Zero tokens, zero cost, zero data sent externally

## Quick Start

```bash
git clone https://github.com/Soflution1/soflution-llm.git
cd soflution-llm
chmod +x scripts/install.sh
./scripts/install.sh
ollama run soflution-llm
```

## Project Structure

```
soflution-llm/
├── Modelfile              # Ollama model configuration
├── scripts/
│   ├── install.sh         # One-command setup
│   └── self_improve.py    # Self-improvement agent
├── fine-tunes/            # Community fine-tune contributions
├── benchmarks/            # Benchmark results by hardware
└── docs/CONTRIBUTING.md
```

## How to Contribute

1. Run `python3 scripts/self_improve.py` and submit your benchmark results
2. Improve the Modelfile parameters
3. Add fine-tunes specialized for your use case
4. Improve the self-improvement agent

## License

Apache 2.0 - Built by [Soflution Ltd](https://soflution.com)
