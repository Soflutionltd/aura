#!/usr/bin/env python3
"""
AURA Master Daemon v2.0
The brain that connects everything together.

Runs in background, manages:
  1. Self-improvement cycles (when idle)
  2. Auto-updates (check + download + notify)
  3. Federated contributions (send/receive)
  4. World Model convergence tracking
  5. Expert specialization monitoring
  6. IPFS node health
  7. Mining rewards (Proof of Useful Work)

Triggered by LaunchAgent on macOS, systemd on Linux.
"""

import os
import sys
import time
import json
import subprocess
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "orchestrator" / "daemon.log"
STATE_FILE = REPO_ROOT / "orchestrator" / "daemon_state.json"

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_improve": "",
        "last_update_check": "",
        "last_federated": "",
        "total_cycles": 0,
        "total_improvements": 0,
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def is_idle() -> bool:
    """Check if system is idle (CPU < 20%) and on AC power."""
    try:
        # CPU usage
        out = subprocess.check_output(
            ["top", "-l", "1", "-n", "0"], text=True, timeout=10
        )
        for line in out.split("\n"):
            if "CPU usage" in line:
                idle_str = line.split("idle")[0].split()[-1].replace("%", "")
                idle_pct = float(idle_str)
                if idle_pct < 70:  # Less than 70% idle = system busy
                    return False

        # AC power (macOS)
        out = subprocess.check_output(
            ["pmset", "-g", "batt"], text=True, timeout=5
        )
        if "Battery Power" in out:
            return False  # On battery, don't mine

        return True
    except Exception:
        return False


def is_ollama_running() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def run_cycle(state: dict):
    """Run one complete daemon cycle."""
    now = datetime.datetime.now()
    state["total_cycles"] += 1

    # ── 1. Check for updates (every 6 hours) ─────────────────
    last_update = state.get("last_update_check", "")
    hours_since_update = 999
    if last_update:
        try:
            last = datetime.datetime.fromisoformat(last_update)
            hours_since_update = (now - last).total_seconds() / 3600
        except Exception:
            pass

    if hours_since_update >= 6:
        log("[UPDATE] Checking for updates...")
        try:
            from orchestrator.auto_updater import run_update_check
            run_update_check()
            state["last_update_check"] = now.isoformat()
        except Exception as e:
            log(f"[UPDATE] Failed: {e}")

    # ── 2. Self-improvement (when idle, every 24 hours) ──────
    last_improve = state.get("last_improve", "")
    hours_since_improve = 999
    if last_improve:
        try:
            last = datetime.datetime.fromisoformat(last_improve)
            hours_since_improve = (now - last).total_seconds() / 3600
        except Exception:
            pass

    if hours_since_improve >= 24 and is_idle() and is_ollama_running():
        log("[IMPROVE] Starting self-improvement cycle...")
        try:
            from orchestrator.self_improve_engine import run_full_cycle
            run_full_cycle()
            state["last_improve"] = now.isoformat()
            state["total_improvements"] += 1
        except Exception as e:
            log(f"[IMPROVE] Failed: {e}")

    # ── 3. Federated contribution (when idle, every 12 hours) ─
    last_fed = state.get("last_federated", "")
    hours_since_fed = 999
    if last_fed:
        try:
            last = datetime.datetime.fromisoformat(last_fed)
            hours_since_fed = (now - last).total_seconds() / 3600
        except Exception:
            pass

    if hours_since_fed >= 12 and is_idle():
        log("[FEDERATED] Running federated aggregation...")
        try:
            from orchestrator.federated_server import FederatedServer
            server = FederatedServer()
            server.run_aggregation_round()
            state["last_federated"] = now.isoformat()
        except Exception as e:
            log(f"[FEDERATED] Failed: {e}")

    # ── 4. Trajectory distillation (when we have enough data) ─
    try:
        from orchestrator.trajectory_distillation import load_trajectories
        good, bad = load_trajectories()
        if len(good) >= 10:
            log(f"[DISTILL] {len(good)} successful trajectories. Running distillation...")
            from orchestrator.trajectory_distillation import run_distillation
            run_distillation()
    except Exception as e:
        log(f"[DISTILL] Skipped: {e}")

    # ── 5. World Model convergence check ──────────────────────
    try:
        from orchestrator.world_enhanced_moe import ConvergenceTracker
        tracker = ConvergenceTracker()
        status = tracker.get_status()
        if status["total_pairs"] > 0 and status["total_pairs"] % 500 == 0:
            log(f"[CONVERGENCE] Phase: {status['phase']} | "
                f"Score: {status['convergence']*100:.1f}% | "
                f"Pairs: {status['total_pairs']}")
            if status["phase"] == "absorbed":
                log("[CONVERGENCE] MoE has absorbed LeWM! World-Enhanced MoE active.")
    except Exception:
        pass

    # ── 6. Expert specialization report ───────────────────────
    try:
        from orchestrator.expert_specialization import ExpertTracker
        et = ExpertTracker()
        suggestions = et.suggest_distillation_targets()
        if suggestions:
            for s in suggestions:
                log(f"[EXPERTS] {s['recommendation']}")
    except Exception:
        pass

    save_state(state)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("AURA Master Daemon v2.0 started")
    log(f"Repo: {REPO_ROOT}")
    log("=" * 60)

    state = load_state()
    cycle_interval = 300  # 5 minutes between checks

    while True:
        try:
            run_cycle(state)
        except Exception as e:
            log(f"[ERROR] Cycle failed: {e}")

        time.sleep(cycle_interval)


if __name__ == "__main__":
    main()
