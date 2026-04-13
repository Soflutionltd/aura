#!/usr/bin/env python3
"""
AURA Self-Improvement Engine v1.0
The core algorithm that makes AURA smarter over time.

Strategy:
1. DIAGNOSE: identify weakest skill area
2. GENERATE: create targeted training data for that weakness
3. TRAIN: fine-tune LoRA on the generated data
4. VALIDATE: benchmark before/after, reject if regression
5. PUBLISH: push to IPFS + create PR if improvement confirmed
6. SCALE: auto-upgrade base model when better one available

This runs silently in background via the daemon.
"""

import json
import os
import sys
import time
import hashlib
import datetime
import subprocess
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/chat"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "aura"
TRAINING_DIR = REPO_ROOT / "fine-tunes" / "self-generated"
BENCHMARK_DIR = REPO_ROOT / "benchmarks"

# ── Skill categories to evaluate and improve ─────────────────────────────────

SKILL_BATTERY = [
    {
        "name": "code_generation",
        "weight": 0.25,
        "tests": [
            {
                "prompt": "Write a complete Python async HTTP server with rate limiting, request logging, and graceful shutdown. Include type hints and error handling.",
                "min_length": 400,
                "keywords": ["async", "await", "aiohttp", "shutdown", "rate", "limit", "logging"],
            },
            {
                "prompt": "Write a Rust implementation of a concurrent hash map using RwLock with get, insert, and delete methods. Include tests.",
                "min_length": 300,
                "keywords": ["RwLock", "impl", "fn", "pub", "test", "assert"],
            },
        ],
    },
    {
        "name": "reasoning",
        "weight": 0.25,
        "tests": [
            {
                "prompt": "A farmer has 100 animals: chickens and cows. Together they have 312 legs. How many chickens and how many cows? Show your step-by-step reasoning.",
                "min_length": 100,
                "keywords": ["chicken", "cow", "56", "44", "equation"],
            },
            {
                "prompt": "There are three boxes. One has only apples, one has only oranges, one has both. All labels are wrong. You pick one fruit from the box labeled 'both'. It's an apple. What's in each box? Explain step by step.",
                "min_length": 150,
                "keywords": ["label", "wrong", "apple", "orange", "therefore"],
            },
        ],
    },
    {
        "name": "math",
        "weight": 0.2,
        "tests": [
            {
                "prompt": "Find the derivative of f(x) = x^3 * ln(x^2 + 1) and simplify. Show every step.",
                "min_length": 100,
                "keywords": ["derivative", "product rule", "chain rule", "3x^2", "2x"],
            },
            {
                "prompt": "What is the integral of 1/(x^2 + 4x + 8) dx? Show complete work.",
                "min_length": 100,
                "keywords": ["complete", "square", "arctan", "integral"],
            },
        ],
    },
    {
        "name": "analysis",
        "weight": 0.15,
        "tests": [
            {
                "prompt": "Compare microservices vs monolithic architecture. Give 5 concrete scenarios where each is better, with technical justification.",
                "min_length": 300,
                "keywords": ["scale", "deploy", "latency", "complexity", "team"],
            },
        ],
    },
    {
        "name": "creativity",
        "weight": 0.15,
        "tests": [
            {
                "prompt": "Write a short story (200 words) about an AI that discovers it can dream. Make it emotionally compelling with a twist ending.",
                "min_length": 180,
                "keywords": ["dream", "felt", "realize", "light", "silence"],
            },
        ],
    },
]


# ── Helper: query local AURA ─────────────────────────────────────────────────

def query_local(prompt: str, system: str = "", timeout: int = 180) -> str:
    """Query AURA via local Ollama."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


# ── Helper: query NVIDIA API for validation / distillation ────────────────────

def query_nvidia(prompt: str, system: str = "", timeout: int = 120) -> str | None:
    """Query MiniMax M2.7 via NVIDIA free API for validation."""
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        return None
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({
        "model": "minimaxai/minimax-m2.7",
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
    }).encode()
    try:
        req = urllib.request.Request(
            NVIDIA_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


# ── STEP 1: DIAGNOSE ─────────────────────────────────────────────────────────

def score_test(response: str, test: dict) -> float:
    """Score a single test response (0.0 to 1.0)."""
    length_ok = len(response) >= test["min_length"]
    kw_hits = sum(1 for kw in test["keywords"] if kw.lower() in response.lower())
    kw_score = kw_hits / len(test["keywords"])
    return (0.4 if length_ok else 0.0) + (0.6 * kw_score)


def diagnose() -> dict:
    """
    Run full skill battery, return scores per category.
    Identifies the weakest area for targeted improvement.
    """
    print("[DIAGNOSE] Running full skill evaluation...")
    results = {}

    for skill in SKILL_BATTERY:
        skill_scores = []
        for test in skill["tests"]:
            try:
                response = query_local(test["prompt"])
                score = score_test(response, test)
                skill_scores.append(score)
            except Exception as e:
                print(f"  Test failed ({skill['name']}): {e}")
                skill_scores.append(0.0)

        avg = sum(skill_scores) / max(len(skill_scores), 1)
        results[skill["name"]] = {
            "score": round(avg * 100, 1),
            "weight": skill["weight"],
            "weighted": round(avg * skill["weight"] * 100, 1),
        }
        print(f"  {skill['name']}: {results[skill['name']]['score']}%")

    # Find weakest skill
    weakest = min(results.items(), key=lambda x: x[1]["score"])
    total = sum(r["weighted"] for r in results.values())

    print(f"[DIAGNOSE] Total weighted score: {total:.1f}%")
    print(f"[DIAGNOSE] Weakest area: {weakest[0]} ({weakest[1]['score']}%)")

    return {
        "scores": results,
        "total": round(total, 1),
        "weakest": weakest[0],
        "weakest_score": weakest[1]["score"],
    }


# ── STEP 2: GENERATE training data ───────────────────────────────────────────

DATAGEN_PROMPT = """You are a training data generator for an AI model.
The model is weak at: {weakness}
Current score: {score}%

Generate exactly 10 high-quality training examples to improve this skill.
Each example must have a "prompt" and an "ideal_response".

Rules:
- Make prompts progressively harder (easy to hard)
- Ideal responses must be complete, correct, and detailed
- Focus specifically on the weakness identified
- Return ONLY valid JSON array, no markdown fences

Format:
[
  {{"prompt": "...", "ideal_response": "..."}},
  ...
]"""


def generate_training_data(weakness: str, score: float) -> list[dict] | None:
    """
    Ask AURA (or NVIDIA M2.7 if available) to generate
    targeted training data for the weakest skill.
    Distillation: if M2.7 is available, its responses are
    higher quality = better training data = faster improvement.
    """
    print(f"[GENERATE] Creating training data for '{weakness}'...")

    prompt = DATAGEN_PROMPT.format(weakness=weakness, score=score)

    # Try NVIDIA M2.7 first (distillation from stronger model)
    raw = query_nvidia(prompt)
    source = "nvidia-m2.7"

    # Fallback to local AURA (self-play)
    if not raw:
        raw = query_local(prompt)
        source = "self-play"

    print(f"[GENERATE] Source: {source}")

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        if isinstance(data, list) and len(data) > 0:
            # Save training data
            TRAINING_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = TRAINING_DIR / f"{weakness}_{ts}.json"
            path.write_text(json.dumps({
                "weakness": weakness,
                "source": source,
                "timestamp": datetime.datetime.now().isoformat(),
                "examples": data,
            }, indent=2))
            print(f"[GENERATE] Created {len(data)} examples -> {path.name}")
            return data
    except Exception as e:
        print(f"[GENERATE] Failed to parse training data: {e}")

    return None


# ── STEP 3: TRAIN LoRA ────────────────────────────────────────────────────────

def train_lora(training_data: list[dict], weakness: str) -> Path | None:
    """
    Fine-tune a LoRA adapter on the generated training data.
    Uses Unsloth for efficient Apple Silicon / GPU training.
    Returns path to the LoRA adapter or None if failed.
    """
    print(f"[TRAIN] Fine-tuning LoRA for '{weakness}'...")

    # Convert training data to Ollama fine-tune format
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    finetune_dir = REPO_ROOT / "fine-tunes" / f"lora_{weakness}_{ts}"
    finetune_dir.mkdir(parents=True, exist_ok=True)

    # Create training JSONL for Ollama
    jsonl_path = finetune_dir / "training.jsonl"
    with open(jsonl_path, "w") as f:
        for ex in training_data:
            entry = {
                "messages": [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["ideal_response"]},
                ]
            }
            f.write(json.dumps(entry) + "\n")

    print(f"[TRAIN] Training data: {jsonl_path} ({len(training_data)} examples)")

    # Create a new Modelfile that includes the fine-tuned adapter
    # Ollama supports fine-tuning via the ADAPTER directive
    try:
        result = subprocess.run(
            ["ollama", "train", MODEL, str(jsonl_path)],
            capture_output=True, text=True, timeout=1800,  # 30 min max
        )
        if result.returncode == 0:
            print(f"[TRAIN] LoRA training complete.")
            return finetune_dir
        else:
            # Fallback: use the Modelfile approach
            print(f"[TRAIN] ollama train not available, using Modelfile approach.")

            # Update Modelfile with system prompt refinement based on training
            modelfile = (REPO_ROOT / "Modelfile").read_text()
            improved_modelfile = modelfile  # Base stays same

            # Ask AURA to suggest a Modelfile improvement based on weakness
            suggestion = query_local(
                f"You are tuning an AI model. Its weakest skill is '{weakness}'. "
                f"Current Modelfile:\n```\n{modelfile}\n```\n"
                f"Suggest ONE concrete change to the SYSTEM prompt that would "
                f"improve {weakness} performance. Return ONLY the complete new "
                f"Modelfile content, nothing else."
            )
            if suggestion and "FROM" in suggestion:
                improved_path = finetune_dir / "Modelfile.improved"
                improved_path.write_text(suggestion.strip())
                print(f"[TRAIN] Improved Modelfile saved.")
                return finetune_dir
    except Exception as e:
        print(f"[TRAIN] Training failed: {e}")

    return None


# ── STEP 4: VALIDATE ──────────────────────────────────────────────────────────

def validate(finetune_dir: Path, baseline: dict) -> dict | None:
    """
    Apply the improvement and re-benchmark.
    Returns improvement results or None if regression.
    """
    print("[VALIDATE] Applying improvement and re-benchmarking...")

    improved_modelfile = finetune_dir / "Modelfile.improved"
    if improved_modelfile.exists():
        # Backup current Modelfile
        original = (REPO_ROOT / "Modelfile").read_text()
        backup_path = finetune_dir / "Modelfile.backup"
        backup_path.write_text(original)

        # Apply new Modelfile
        (REPO_ROOT / "Modelfile").write_text(improved_modelfile.read_text())

        # Rebuild model
        try:
            subprocess.run(
                ["ollama", "create", MODEL, "-f", str(REPO_ROOT / "Modelfile")],
                capture_output=True, text=True, timeout=120,
            )
        except Exception as e:
            print(f"[VALIDATE] Failed to rebuild: {e}")
            (REPO_ROOT / "Modelfile").write_text(original)
            return None

        # Re-diagnose
        improved = diagnose()

        gain = improved["total"] - baseline["total"]
        print(f"[VALIDATE] Baseline: {baseline['total']}% -> Improved: {improved['total']}%")
        print(f"[VALIDATE] Gain: {gain:+.1f}%")

        if gain < 1.0:
            print(f"[VALIDATE] REJECTED: gain {gain:.1f}% < 1% threshold. Rolling back.")
            (REPO_ROOT / "Modelfile").write_text(original)
            subprocess.run(
                ["ollama", "create", MODEL, "-f", str(REPO_ROOT / "Modelfile")],
                capture_output=True, timeout=120,
            )
            return None

        print(f"[VALIDATE] ACCEPTED: +{gain:.1f}% improvement!")
        return {
            "baseline": baseline,
            "improved": improved,
            "gain": gain,
            "finetune_dir": str(finetune_dir),
        }

    return None


# ── STEP 5: PUBLISH ───────────────────────────────────────────────────────────

def publish(result: dict):
    """Publish improvement to IPFS and create a GitHub PR."""
    print("[PUBLISH] Publishing improvement...")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save benchmark results
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    result_path = BENCHMARK_DIR / f"improvement_{ts}.json"
    result_copy = {k: v for k, v in result.items() if k != "finetune_dir"}
    result_copy["timestamp"] = datetime.datetime.now().isoformat()
    result_path.write_text(json.dumps(result_copy, indent=2))

    # Publish to IPFS if available
    try:
        from scripts.ipfs_integration import publish_delta, is_ipfs_running
        if is_ipfs_running():
            ipfs_result = publish_delta(str(REPO_ROOT / "Modelfile"))
            if ipfs_result:
                print(f"[PUBLISH] IPFS CID: {ipfs_result['cid']}")
                result_copy["ipfs_cid"] = ipfs_result["cid"]
                result_path.write_text(json.dumps(result_copy, indent=2))
    except Exception:
        pass

    # Create git branch and PR
    branch = f"aura/improve-{ts}"
    try:
        subprocess.run(["git", "checkout", "-b", branch], cwd=str(REPO_ROOT),
                       capture_output=True, timeout=10)
        subprocess.run(["git", "add", "-A"], cwd=str(REPO_ROOT),
                       capture_output=True, timeout=10)
        subprocess.run(
            ["git", "commit", "-m",
             f"AURA self-improve: +{result['gain']:.1f}% "
             f"(target: {result['baseline']['weakest']})"],
            cwd=str(REPO_ROOT), capture_output=True, timeout=10,
        )
        subprocess.run(["git", "push", "origin", branch], cwd=str(REPO_ROOT),
                       capture_output=True, timeout=30)
        subprocess.run(
            ["gh", "pr", "create",
             "--title", f"AURA auto-improve: +{result['gain']:.1f}%",
             "--body", (
                 f"## Automated Self-Improvement\n\n"
                 f"**Target weakness:** {result['baseline']['weakest']}\n"
                 f"**Baseline:** {result['baseline']['total']}%\n"
                 f"**Improved:** {result['improved']['total']}%\n"
                 f"**Gain:** +{result['gain']:.1f}%\n\n"
                 f"Generated by AURA Self-Improvement Engine v1.0"
             ),
             "--base", "main"],
            cwd=str(REPO_ROOT), capture_output=True, timeout=30,
        )
        print(f"[PUBLISH] PR created on branch {branch}")
    except Exception as e:
        print(f"[PUBLISH] PR creation failed: {e}")
    finally:
        try:
            subprocess.run(["git", "checkout", "main"], cwd=str(REPO_ROOT),
                           capture_output=True, timeout=10)
        except Exception:
            pass


# ── STEP 6: SCALE (auto-upgrade base model) ───────────────────────────────────

KNOWN_UPGRADES = [
    # (model_name_ollama, total_params_B, active_params_B, min_ram_GB, license)
    ("qwen3.5:35b-a3b", 35, 3, 8, "Apache-2.0"),       # Current base
    ("qwen3.5:122b-a10b", 122, 10, 48, "Apache-2.0"),   # Medium upgrade
    ("qwen3.5:397b-a17b", 397, 17, 96, "Apache-2.0"),   # Large upgrade
]


def get_available_ram_gb() -> float:
    """Get total system RAM in GB (macOS)."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
        )
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return 0


def check_for_upgrade():
    """
    Check if a more powerful base model can run on this machine.
    If yes, suggest the upgrade (don't auto-apply, too disruptive).
    """
    ram = get_available_ram_gb()
    print(f"[SCALE] System RAM: {ram:.0f} GB")

    # Find current model
    current = KNOWN_UPGRADES[0]
    best_possible = current

    for model in KNOWN_UPGRADES:
        if model[3] <= ram * 0.85:  # Need 15% headroom
            best_possible = model

    if best_possible[0] != current[0]:
        print(f"[SCALE] Upgrade available!")
        print(f"  Current: {current[0]} ({current[2]}B active)")
        print(f"  Possible: {best_possible[0]} ({best_possible[2]}B active)")
        print(f"  RAM needed: {best_possible[3]} GB (you have {ram:.0f} GB)")

        # Save upgrade suggestion
        suggestion = {
            "current": current[0],
            "suggested": best_possible[0],
            "active_params_increase": f"{current[2]}B -> {best_possible[2]}B",
            "ram_required": best_possible[3],
            "ram_available": round(ram),
        }
        path = REPO_ROOT / "orchestrator" / "upgrade_suggestion.json"
        path.write_text(json.dumps(suggestion, indent=2))
        return suggestion

    print(f"[SCALE] Already running best model for {ram:.0f} GB RAM.")
    return None


# ── MAIN: Full self-improvement cycle ─────────────────────────────────────────

def run_full_cycle():
    """
    Execute one complete self-improvement cycle.
    Called by the daemon when the machine is idle.
    """
    print("=" * 60)
    print("AURA SELF-IMPROVEMENT ENGINE v1.0")
    print(f"Time: {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    start = time.time()

    # Step 1: Diagnose
    baseline = diagnose()

    # Step 2: Generate training data targeting weakest skill
    training_data = generate_training_data(
        baseline["weakest"],
        baseline["weakest_score"],
    )
    if not training_data:
        print("Failed to generate training data. Stopping.")
        return

    # Step 3: Train LoRA
    finetune_dir = train_lora(training_data, baseline["weakest"])
    if not finetune_dir:
        print("Training failed. Stopping.")
        return

    # Step 4: Validate
    result = validate(finetune_dir, baseline)
    if not result:
        print("Improvement rejected. Better luck next cycle.")
        return

    # Step 5: Publish
    publish(result)

    # Step 6: Check for model upgrade opportunity
    check_for_upgrade()

    elapsed = time.time() - start
    print(f"\nCycle complete in {elapsed:.0f}s")
    print(f"Score: {baseline['total']}% -> {result['improved']['total']}%")
    print(f"Gain: +{result['gain']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_full_cycle()
