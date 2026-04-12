#!/usr/bin/env python3
"""
AURA Daemon: background self-improvement agent.
Runs silently when the Mac is idle, asks AURA to improve itself,
benchmarks the result, and opens a PR if performance goes up.
Zero tokens. Zero cost. Pure local intelligence.
"""

import json
import subprocess
import sys
import time
import datetime
import os
import signal
import shutil
import hashlib
import urllib.request
import urllib.error
import logging
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "aura"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELFILE_PATH = REPO_ROOT / "Modelfile"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
LOG_DIR = REPO_ROOT / "orchestrator" / "logs"

CPU_IDLE_THRESHOLD = 20          # percent, below = idle
MAX_RUN_SECONDS = 600            # 10 min hard cap per cycle
BENCHMARK_GAIN_REQUIRED = 1.0    # must gain at least 1%
CYCLE_INTERVAL_SECONDS = 1800    # check every 30 min
BRANCH_PREFIX = "aura/auto-improve"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"daemon_{datetime.date.today().isoformat()}.log"
        ),
    ],
)
log = logging.getLogger("aura-daemon").info

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    log(f"Received signal {signum}, shutting down gracefully...")
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── System checks ─────────────────────────────────────────────────────────────

def is_on_ac_power() -> bool:
    """Return True if Mac is plugged into AC power."""
    try:
        out = subprocess.check_output(["pmset", "-g", "ps"], text=True)
        return "AC Power" in out
    except Exception:
        return False


def cpu_usage() -> float:
    """Return current CPU usage percent."""
    try:
        out = subprocess.check_output(
            ["top", "-l", "1", "-n", "0", "-stats", "cpu"],
            text=True, timeout=10
        )
        for line in out.splitlines():
            if "CPU usage" in line:
                for part in line.split(","):
                    if "idle" in part:
                        idle = float(part.strip().split("%")[0])
                        return 100.0 - idle
        return 100.0
    except Exception:
        return 100.0


def is_ollama_running() -> bool:
    """Check if Ollama is reachable and AURA model is available."""
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL)
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        return any(MODEL in m for m in models)
    except Exception:
        return False


# ── Ollama + Git helpers ──────────────────────────────────────────────────────

def query_aura(prompt: str, timeout: int = 120) -> str:
    """Send a prompt to AURA via Ollama, return response text."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 8192},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["message"]["content"]


def git(*args):
    """Run a git command in the repo and return stdout."""
    return subprocess.check_output(
        ["git"] + list(args),
        cwd=str(REPO_ROOT),
        text=True,
        timeout=30,
    ).strip()


# ── Benchmark suite ───────────────────────────────────────────────────────────

BENCHMARKS = [
    {
        "name": "code_rust",
        "prompt": (
            "Write a complete Rust function that implements a"
            " thread-safe LRU cache with get and put methods."
            " Include proper error handling."
        ),
        "min_length": 200,
        "keywords": ["fn", "impl", "struct", "pub", "->"],
    },
    {
        "name": "reasoning",
        "prompt": (
            "Explain step by step why quicksort has O(n log n)"
            " average time complexity but O(n^2) worst case."
            " Be precise and mathematical."
        ),
        "min_length": 150,
        "keywords": ["pivot", "partition", "O(n", "log"],
    },
    {
        "name": "code_python",
        "prompt": (
            "Write a Python async web scraper that respects"
            " robots.txt and rate limits. Include type hints."
        ),
        "min_length": 200,
        "keywords": ["async", "await", "def", "robots"],
    },
    {
        "name": "analysis",
        "prompt": (
            "Compare B-trees and LSM-trees for database storage"
            " engines. When would you choose each?"
        ),
        "min_length": 150,
        "keywords": ["read", "write", "disk", "memory"],
    },
]


def run_benchmark() -> dict:
    """Run all benchmarks. Returns {per_task: {name: score}, total: float}."""
    scores = {}
    for b in BENCHMARKS:
        try:
            resp = query_aura(b["prompt"], timeout=180)
            length_ok = len(resp) >= b["min_length"]
            kw_hits = sum(1 for kw in b["keywords"] if kw.lower() in resp.lower())
            kw_score = kw_hits / len(b["keywords"])
            scores[b["name"]] = round((0.5 if length_ok else 0.0) + 0.5 * kw_score, 1) * 100
        except Exception as e:
            log(f"  Benchmark '{b['name']}' failed: {e}")
            scores[b["name"]] = 0.0
    total = round(sum(scores.values()) / max(len(scores), 1), 1)
    return {"per_task": scores, "total": total}


# ── Self-improvement proposal ─────────────────────────────────────────────────

IMPROVEMENT_PROMPT = """You are AURA, a self-improving AI. Analyze your own Modelfile and suggest exactly ONE concrete improvement.

Current Modelfile:
```
{modelfile}
```

Current benchmark scores: {scores}

Rules:
- Suggest ONE change only (parameter tweak OR system prompt improvement)
- Return ONLY valid JSON, no markdown fences, no explanation outside the JSON
- Schema:
{{
  "change_type": "parameter" | "system_prompt",
  "description": "what you changed and why",
  "new_modelfile": "complete new Modelfile content"
}}
"""


def propose_improvement(baseline: dict) -> dict | None:
    """Ask AURA to propose one improvement. Returns parsed JSON or None."""
    modelfile = MODELFILE_PATH.read_text()
    prompt = IMPROVEMENT_PROMPT.format(
        modelfile=modelfile,
        scores=json.dumps(baseline, indent=2),
    )
    try:
        raw = query_aura(prompt, timeout=180).strip()
        # Strip markdown fences if AURA wraps its response
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        log(f"  Could not parse improvement proposal: {e}")
        return None


# ── Core improvement cycle ────────────────────────────────────────────────────

def run_cycle():
    """Execute one self-improvement cycle."""
    log("=== AURA self-improvement cycle starting ===")
    start = time.time()

    # 1. Preconditions
    if not is_on_ac_power():
        log("Not on AC power. Skipping.")
        return
    usage = cpu_usage()
    if usage > CPU_IDLE_THRESHOLD:
        log(f"CPU at {usage:.1f}%%, too busy. Skipping.")
        return
    if not is_ollama_running():
        log("Ollama not running or AURA model not found. Skipping.")
        return
    log(f"CPU at {usage:.1f}%%, AC power, Ollama OK. Proceeding.")

    # 2. Pull latest
    try:
        git("pull", "--rebase", "origin", "main")
        log("Pulled latest from origin/main.")
    except Exception as e:
        log(f"Git pull failed (non-fatal): {e}")

    # 3. Baseline benchmark
    log("Running baseline benchmark...")
    baseline = run_benchmark()
    log(f"Baseline: {baseline['total']} | {baseline['per_task']}")

    # 4. Time check
    if time.time() - start > MAX_RUN_SECONDS:
        log("Time limit reached after baseline. Stopping.")
        return

    # 5. Ask AURA for one improvement
    log("Asking AURA for an improvement proposal...")
    proposal = propose_improvement(baseline)
    if not proposal or "new_modelfile" not in proposal:
        log("No valid proposal. Stopping cycle.")
        return
    log(f"Proposal: {proposal.get('description', 'N/A')}")

    # 6. Backup current Modelfile, apply proposal
    backup = MODELFILE_PATH.read_text()
    MODELFILE_PATH.write_text(proposal["new_modelfile"])
    log("Applied proposed Modelfile.")

    # 7. Recreate the AURA model in Ollama with new Modelfile
    try:
        subprocess.run(
            ["ollama", "create", MODEL, "-f", str(MODELFILE_PATH)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        log("Rebuilt AURA model with new Modelfile.")
    except Exception as e:
        log(f"Failed to rebuild model: {e}. Rolling back.")
        MODELFILE_PATH.write_text(backup)
        return

    # 8. Re-benchmark
    if time.time() - start > MAX_RUN_SECONDS:
        log("Time limit reached before re-benchmark. Rolling back.")
        MODELFILE_PATH.write_text(backup)
        return

    log("Running post-improvement benchmark...")
    improved = run_benchmark()
    log(f"Improved: {improved['total']} | {improved['per_task']}")

    gain = improved["total"] - baseline["total"]
    log(f"Gain: {gain:+.1f}%%")

    # 9. Decision: keep or rollback
    if gain < BENCHMARK_GAIN_REQUIRED:
        log(f"Gain {gain:.1f}%% < {BENCHMARK_GAIN_REQUIRED}%%. Rolling back.")
        MODELFILE_PATH.write_text(backup)
        subprocess.run(
            ["ollama", "create", MODEL, "-f", str(MODELFILE_PATH)],
            capture_output=True, timeout=120,
        )
        return

    log(f"Gain {gain:.1f}%% >= {BENCHMARK_GAIN_REQUIRED}%%. Keeping improvement!")

    # 10. Save benchmark results
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "baseline": baseline,
        "improved": improved,
        "gain": gain,
        "proposal": proposal.get("description", ""),
    }
    result_path = BENCHMARKS_DIR / f"improvement_{ts}.json"
    result_path.write_text(json.dumps(result, indent=2))

    # 11. Create branch and PR
    branch = f"{BRANCH_PREFIX}-{ts}"
    try:
        git("checkout", "-b", branch)
        git("add", "Modelfile", str(result_path.relative_to(REPO_ROOT)))
        git(
            "commit", "-m",
            f"AURA auto-improve: +{gain:.1f}%% | {proposal.get('description', '')[:80]}",
        )
        git("push", "origin", branch)
        # Create PR via GitHub CLI
        subprocess.run(
            [
                "gh", "pr", "create",
                "--title", f"AURA auto-improve: +{gain:.1f}%% benchmark gain",
                "--body", (
                    f"## Automated improvement\n\n"
                    f"**Change:** {proposal.get('description', 'N/A')}\n\n"
                    f"**Baseline score:** {baseline['total']}\n"
                    f"**New score:** {improved['total']}\n"
                    f"**Gain:** +{gain:.1f}%%\n\n"
                    f"This PR was generated automatically by the AURA daemon."
                ),
                "--base", "main",
            ],
            cwd=str(REPO_ROOT),
            check=True, capture_output=True, text=True, timeout=30,
        )
        log(f"PR created on branch {branch}.")
    except Exception as e:
        log(f"Failed to create PR: {e}")
    finally:
        # Return to main branch
        try:
            git("checkout", "main")
        except Exception:
            pass

    log("=== Cycle complete ===")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log("AURA Daemon starting.")
    log(f"Repo: {REPO_ROOT}")
    log(f"Cycle interval: {CYCLE_INTERVAL_SECONDS}s | CPU threshold: {CPU_IDLE_THRESHOLD}%%")

    while not _shutdown:
        try:
            run_cycle()
        except Exception as e:
            log(f"Cycle crashed: {e}")

        # Wait for next cycle, checking shutdown flag every 10s
        waited = 0
        while waited < CYCLE_INTERVAL_SECONDS and not _shutdown:
            time.sleep(10)
            waited += 10

    log("AURA Daemon stopped.")


if __name__ == "__main__":
    main()
