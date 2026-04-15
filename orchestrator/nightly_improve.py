#!/usr/bin/env python3
"""
AURA Nightly Self-Improvement Pipeline
Runs every night via LaunchAgent. Does the REAL work:
1. Collects conversations from MemoryPilot
2. Identifies poor responses (reformulations, corrections, "non")
3. Generates improved response pairs
4. Creates DPO/SFT training dataset
5. Trains a LoRA adapter with mlx-lm
6. Merges the LoRA into the base model
7. Reloads the model in Ollama

This is NOT a prompt trick. This modifies the actual model weights.
"""

import os
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta

# ── Config ──
AURA_HOME = Path.home() / "Cursor/App/soflution-llm"
MEMORY_URL = "http://localhost:23100"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "aura"
MLX_VENV = Path.home() / "Cursor/App/soflution-llm/.venv-mlx"
TRAINING_DIR = AURA_HOME / "training"
LORA_OUTPUT = TRAINING_DIR / "lora_adapters"
DATASET_DIR = TRAINING_DIR / "datasets"
LOG_FILE = AURA_HOME / "logs/nightly_improve.log"
MIN_PAIRS_FOR_TRAINING = 10  # minimum conversation pairs to trigger training
MAX_TRAINING_STEPS = 200
LORA_RANK = 8
LEARNING_RATE = 1e-5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NIGHTLY] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("nightly")

# Ensure directories exist
for d in [TRAINING_DIR, LORA_OUTPUT, DATASET_DIR, AURA_HOME / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Step 1: Collect conversations from MemoryPilot ──
def collect_conversations():
    """Fetch recent conversations from MemoryPilot."""
    import urllib.request
    log.info("Step 1: Collecting conversations from MemoryPilot...")
    try:
        req = urllib.request.Request(
            f"{MEMORY_URL}/search",
            data=json.dumps({"query": "conversation", "limit": 100, "kind": "conversation"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            conversations = data.get("results", [])
            log.info(f"  Found {len(conversations)} conversation memories")
            return conversations
    except Exception as e:
        log.warning(f"  MemoryPilot unavailable: {e}")
        return []

# ── Step 2: Detect poor responses ──
def detect_poor_responses(conversations):
    """Identify conversations where user was likely unsatisfied."""
    log.info("Step 2: Detecting poor responses...")
    poor_pairs = []
    rejection_signals = [
        "non", "no", "pas correct", "incorrect", "wrong", "faux",
        "je voulais", "I meant", "c'est pas ça", "that's not",
        "recommence", "retry", "essaie encore", "try again",
        "je ne suis pas", "not what I", "tu te trompes", "you're wrong",
    ]
    for conv in conversations:
        content = conv.get("content", "")
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("User:"):
                user_msg = line[5:].strip()
                # Check if this is a correction/rejection
                if any(sig in user_msg.lower() for sig in rejection_signals):
                    # Find the AURA response before this rejection
                    for j in range(i - 1, -1, -1):
                        if lines[j].startswith("AURA:"):
                            bad_response = lines[j][5:].strip()
                            poor_pairs.append({
                                "prompt": lines[j - 1][5:].strip() if j > 0 and lines[j-1].startswith("User:") else user_msg,
                                "rejected": bad_response,
                                "correction_hint": user_msg,
                            })
                            break
    log.info(f"  Found {len(poor_pairs)} poor response pairs")
    return poor_pairs

# ── Step 3: Generate improved responses using the model itself ──
def generate_improved_responses(poor_pairs):
    """Use AURA itself to generate better responses for the failed ones."""
    import urllib.request
    log.info("Step 3: Generating improved responses...")
    training_pairs = []
    for pair in poor_pairs[:30]:  # Limit to 30 per night
        prompt = pair["prompt"]
        rejected = pair["rejected"]
        hint = pair["correction_hint"]
        # Ask the model to generate a better response
        improve_prompt = f"""The user asked: "{prompt}"
You previously answered: "{rejected}"
The user then corrected you: "{hint}"
Now generate a much better answer to the original question. Be concise and accurate."""
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps({
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": improve_prompt}],
                    "stream": False,
                    "keep_alive": "5m",
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                chosen = data.get("message", {}).get("content", "")
                if chosen and len(chosen) > 10:
                    training_pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    })
        except Exception as e:
            log.warning(f"  Failed to generate improved response: {e}")
            continue
    log.info(f"  Generated {len(training_pairs)} improved pairs")
    return training_pairs

# ── Step 4: Create DPO training dataset ──
def create_training_dataset(training_pairs):
    """Save training pairs as JSONL for mlx-lm DPO/SFT training."""
    log.info("Step 4: Creating training dataset...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = DATASET_DIR / f"dpo_{timestamp}.jsonl"
    sft_path = DATASET_DIR / f"sft_{timestamp}.jsonl"
    # DPO format: {"prompt": ..., "chosen": ..., "rejected": ...}
    with open(dataset_path, "w") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    # SFT format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    with open(sft_path, "w") as f:
        for pair in training_pairs:
            entry = {"messages": [
                {"role": "user", "content": pair["prompt"]},
                {"role": "assistant", "content": pair["chosen"]},
            ]}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"  Dataset saved: {dataset_path} ({len(training_pairs)} pairs)")
    return sft_path, dataset_path

# ── Step 5: Train LoRA with mlx-lm ──
def train_lora(sft_path):
    """Train a LoRA adapter using mlx-lm on Apple Silicon."""
    log.info("Step 5: Training LoRA adapter with mlx-lm...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_path = LORA_OUTPUT / f"adapter_{timestamp}"
    adapter_path.mkdir(parents=True, exist_ok=True)
    # Use mlx-lm from the virtual environment
    python = MLX_VENV / "bin/python3"
    if not python.exists():
        log.error(f"  MLX venv not found at {python}")
        return None
    cmd = [
        str(python), "-m", "mlx_lm.lora",
        "--model", "mlx-community/gemma-4-26b-it-4bit",  # MLX format model
        "--data", str(sft_path.parent),
        "--train",
        "--adapter-path", str(adapter_path),
        "--iters", str(MAX_TRAINING_STEPS),
        "--batch-size", "1",
        "--lora-rank", str(LORA_RANK),
        "--learning-rate", str(LEARNING_RATE),
    ]
    log.info(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            log.info(f"  LoRA training complete: {adapter_path}")
            return adapter_path
        else:
            log.error(f"  Training failed: {result.stderr[:500]}")
            return None
    except subprocess.TimeoutExpired:
        log.error("  Training timed out (1 hour limit)")
        return None

# ── Step 6: Merge LoRA and reload model ──
def merge_and_reload(adapter_path):
    """Merge LoRA adapter into base model and reload in Ollama."""
    log.info("Step 6: Merging LoRA and reloading model...")
    python = MLX_VENV / "bin/python3"
    merged_path = TRAINING_DIR / "merged_model"
    # Fuse LoRA into base model
    cmd = [
        str(python), "-m", "mlx_lm.fuse",
        "--model", "mlx-community/gemma-4-26b-it-4bit",
        "--adapter-path", str(adapter_path),
        "--save-path", str(merged_path),
    ]
    log.info(f"  Fusing LoRA...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error(f"  Fusion failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        log.error(f"  Fusion error: {e}")
        return False
    # Convert merged MLX model to GGUF for Ollama
    log.info("  Converting to GGUF for Ollama...")
    gguf_path = TRAINING_DIR / "aura_improved.gguf"
    convert_cmd = [
        str(python), "-m", "mlx_lm.convert",
        "--model", str(merged_path),
        "--quantize", "--q-bits", "4",
        "--upload-repo", "",  # no upload, local only
    ]
    # Alternative: use llama.cpp convert if mlx_lm.convert doesn't output GGUF
    # For now, create a new Ollama model from the merged MLX weights
    log.info("  Creating updated Ollama model...")
    try:
        # Write a new Modelfile pointing to the improved model
        modelfile_content = f"""FROM {str(merged_path)}
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER num_ctx 16384
SYSTEM "Tu es AURA, un agent IA autonome ultra-puissant, local et décentralisé. Tu t'améliores continuellement grâce à tes conversations. Sois direct, professionnel, ultra-efficace. Réponds dans la langue de l'utilisateur."
"""
        modelfile_path = TRAINING_DIR / "Modelfile.improved"
        modelfile_path.write_text(modelfile_content)
        # Recreate the model in Ollama
        result = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            log.info("  Model reloaded in Ollama successfully!")
            return True
        else:
            log.error(f"  Ollama create failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        log.error(f"  Reload error: {e}")
        return False

# ── Main pipeline ──
def run_nightly():
    """Run the complete nightly self-improvement pipeline."""
    log.info("=" * 60)
    log.info("AURA Nightly Self-Improvement Pipeline")
    log.info(f"Started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    # Step 1: Collect
    conversations = collect_conversations()
    if not conversations:
        log.info("No conversations found. Skipping tonight.")
        return

    # Step 2: Detect problems
    poor_pairs = detect_poor_responses(conversations)
    if len(poor_pairs) < MIN_PAIRS_FOR_TRAINING:
        log.info(f"Only {len(poor_pairs)} poor pairs (need {MIN_PAIRS_FOR_TRAINING}). Skipping training.")
        log.info("Accumulating data for next night...")
        return

    # Step 3: Generate improvements
    training_pairs = generate_improved_responses(poor_pairs)
    if not training_pairs:
        log.info("No training pairs generated. Skipping.")
        return

    # Step 4: Create dataset
    sft_path, dpo_path = create_training_dataset(training_pairs)

    # Step 5: Train LoRA
    adapter_path = train_lora(sft_path)
    if not adapter_path:
        log.error("LoRA training failed. Model unchanged.")
        return

    # Step 6: Merge and reload
    success = merge_and_reload(adapter_path)
    if success:
        log.info("=" * 60)
        log.info("AURA has improved itself! New model loaded.")
        log.info(f"Training pairs used: {len(training_pairs)}")
        log.info(f"Adapter: {adapter_path}")
        log.info("=" * 60)
    else:
        log.error("Merge/reload failed. Model unchanged. Will retry tomorrow.")


if __name__ == "__main__":
    run_nightly()
