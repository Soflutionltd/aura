#!/usr/bin/env python3
"""
AURA Auto-Updater
Checks for new model versions, downloads in background,
and notifies the app to restart when ready.

Three update levels:
  Level 1: LoRA patches (weekly, ~50 Mo, no restart needed)
  Level 2: LoRA merge (monthly, ~50 Mo, no restart needed)
  Level 3: Base model upgrade (rare, 20+ Go, restart required)
"""

import json
import os
import subprocess
import time
import datetime
import hashlib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434"
UPDATE_CHECK_URL = "https://raw.githubusercontent.com/Soflution1/aura/main/orchestrator/latest_version.json"
UPDATE_STATE_FILE = REPO_ROOT / "orchestrator" / "update_state.json"
NOTIFICATION_FILE = REPO_ROOT / "orchestrator" / "pending_update.json"
MODEL = "aura"


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [UPDATER] {msg}")


def get_current_version() -> dict:
    """Read current installed version info."""
    if UPDATE_STATE_FILE.exists():
        return json.loads(UPDATE_STATE_FILE.read_text())
    return {"version": "0.1.0", "model": "qwen3.5:35b-a3b", "lora_version": 0}


def save_current_version(state: dict):
    UPDATE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    UPDATE_STATE_FILE.write_text(json.dumps(state, indent=2))


def check_for_updates() -> dict | None:
    """Check GitHub for latest version info."""
    try:
        req = urllib.request.Request(UPDATE_CHECK_URL)
        with urllib.request.urlopen(req, timeout=10) as r:
            latest = json.loads(r.read())
        current = get_current_version()

        updates = {}

        # Check LoRA patch
        if latest.get("lora_version", 0) > current.get("lora_version", 0):
            updates["lora"] = {
                "type": "lora_patch",
                "level": 1,
                "from_version": current.get("lora_version", 0),
                "to_version": latest["lora_version"],
                "url": latest.get("lora_url", ""),
                "size_mb": latest.get("lora_size_mb", 50),
                "restart_required": False,
            }

        # Check base model upgrade
        if latest.get("model") != current.get("model"):
            updates["model"] = {
                "type": "model_upgrade",
                "level": 3,
                "from_model": current.get("model"),
                "to_model": latest["model"],
                "url": latest.get("model_url", ""),
                "size_mb": latest.get("model_size_mb", 20000),
                "min_ram_gb": latest.get("min_ram_gb", 8),
                "restart_required": True,
            }

        if updates:
            log(f"Updates available: {list(updates.keys())}")
            return updates
        return None

    except Exception as e:
        log(f"Update check failed: {e}")
        return None


def get_system_ram_gb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return 0


def download_lora_patch(update: dict) -> bool:
    """Download a LoRA patch from IPFS or GitHub."""
    log(f"Downloading LoRA patch v{update['to_version']} ({update['size_mb']} Mo)...")
    try:
        # Try IPFS first
        if update.get("ipfs_cid"):
            result = subprocess.run(
                ["ipfs", "get", update["ipfs_cid"], "-o",
                 str(REPO_ROOT / "fine-tunes" / f"lora_v{update['to_version']}")],
                capture_output=True, timeout=300,
            )
            if result.returncode == 0:
                log("Downloaded via IPFS.")
                return True

        # Fallback to URL
        if update.get("url"):
            output = REPO_ROOT / "fine-tunes" / f"lora_v{update['to_version']}.bin"
            urllib.request.urlretrieve(update["url"], str(output))
            log(f"Downloaded via HTTP: {output.name}")
            return True

        return False
    except Exception as e:
        log(f"LoRA download failed: {e}")
        return False


def download_model_upgrade(update: dict) -> bool:
    """Download a new base model. This is the big one (20+ Go)."""
    ram = get_system_ram_gb()
    min_ram = update.get("min_ram_gb", 8)

    if ram < min_ram:
        log(f"Not enough RAM for upgrade: {ram:.0f} GB < {min_ram} GB required. Skipping.")
        return False

    new_model = update["to_model"]
    log(f"Downloading new base model: {new_model} ({update['size_mb']} Mo)...")
    log("This will take a while. Downloading in background...")

    try:
        result = subprocess.run(
            ["ollama", "pull", new_model],
            capture_output=True, text=True, timeout=7200,  # 2 hours max
        )
        if result.returncode == 0:
            log(f"Model {new_model} downloaded successfully.")
            return True
        else:
            log(f"Model download failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        log(f"Model download failed: {e}")
        return False


def apply_lora_patch(update: dict) -> bool:
    """Apply a LoRA patch to the current model. No restart needed."""
    log("Applying LoRA patch...")
    try:
        # Rebuild AURA model with updated Modelfile
        subprocess.run(
            ["ollama", "create", MODEL, "-f", str(REPO_ROOT / "Modelfile")],
            capture_output=True, text=True, timeout=120,
        )
        # Update version state
        state = get_current_version()
        state["lora_version"] = update["to_version"]
        save_current_version(state)
        log(f"LoRA patch applied. Now at v{update['to_version']}.")
        return True
    except Exception as e:
        log(f"LoRA apply failed: {e}")
        return False


def apply_model_upgrade(update: dict) -> bool:
    """Switch to a new base model. Requires app restart."""
    new_model = update["to_model"]
    log(f"Switching base model to {new_model}...")

    try:
        # Update Modelfile to use new base
        modelfile = (REPO_ROOT / "Modelfile").read_text()
        old_from = None
        for line in modelfile.splitlines():
            if line.startswith("FROM "):
                old_from = line
                break

        if old_from:
            new_modelfile = modelfile.replace(old_from, f"FROM {new_model}")
            (REPO_ROOT / "Modelfile").write_text(new_modelfile)

        # Rebuild AURA model
        subprocess.run(
            ["ollama", "create", MODEL, "-f", str(REPO_ROOT / "Modelfile")],
            capture_output=True, text=True, timeout=120,
        )

        # Update version state
        state = get_current_version()
        state["model"] = new_model
        save_current_version(state)

        log(f"Model upgraded to {new_model}.")
        return True
    except Exception as e:
        log(f"Model upgrade failed: {e}")
        return False


def notify_app_restart(update_info: dict):
    """
    Write a notification file that the AURA app reads.
    The app shows a popup: "Update installed. Restart to apply."
    """
    NOTIFICATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    notification = {
        "type": "update_ready",
        "message": "AURA has been updated. Restart to apply.",
        "details": f"New model: {update_info.get('to_model', 'improved')}",
        "timestamp": datetime.datetime.now().isoformat(),
        "restart_required": True,
    }
    NOTIFICATION_FILE.write_text(json.dumps(notification, indent=2))
    log("Restart notification written. App will show popup.")

    # Also trigger a macOS notification
    try:
        subprocess.run([
            "osascript", "-e",
            'display notification "A new version is ready. Please restart AURA." '
            'with title "AURA Update" subtitle "Intelligence upgraded"'
        ], timeout=5)
    except Exception:
        pass


# ── Main update cycle ─────────────────────────────────────────────────────────

def run_update_check():
    """Full update check and apply cycle."""
    log("Checking for updates...")

    updates = check_for_updates()
    if not updates:
        log("No updates available.")
        return

    # Level 1/2: LoRA patches (apply silently, no restart)
    if "lora" in updates:
        update = updates["lora"]
        if download_lora_patch(update):
            apply_lora_patch(update)
            # No restart needed for LoRA patches

    # Level 3: Model upgrade (download + notify for restart)
    if "model" in updates:
        update = updates["model"]
        if download_model_upgrade(update):
            if apply_model_upgrade(update):
                notify_app_restart(update)

    log("Update check complete.")


if __name__ == "__main__":
    run_update_check()
