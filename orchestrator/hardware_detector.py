#!/usr/bin/env python3
"""
AURA Hardware Detector + Smart Model Selector
Detects system capabilities and selects the optimal model configuration.

Mode Ultra: full MoE + LeWM + DFlash (powerful machines)
Mode Éco: smaller model, no LeWM (lightweight machines)
Mode Agent: loads Code World Model for heavy coding tasks
"""

import subprocess
import platform
import json
from pathlib import Path


class HardwareProfile:
    """Detected hardware capabilities."""

    def __init__(self):
        self.os = platform.system()
        self.arch = platform.machine()
        self.ram_gb = 0
        self.gpu_type = "none"  # "apple_silicon", "nvidia", "amd", "none"
        self.gpu_name = ""
        self.gpu_vram_gb = 0
        self.cpu_cores = 0
        self.has_npu = False
        self.detect()

    def detect(self):
        """Auto-detect hardware capabilities."""
        # RAM
        try:
            if self.os == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                self.ram_gb = round(int(out.strip()) / (1024**3))
            elif self.os == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb = int(line.split()[1])
                            self.ram_gb = round(kb / (1024**2))
                            break
        except Exception:
            pass

        # GPU detection
        if self.os == "Darwin" and self.arch == "arm64":
            self.gpu_type = "apple_silicon"
            self.gpu_vram_gb = self.ram_gb  # Unified memory
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                )
                self.gpu_name = out.strip()
            except Exception:
                self.gpu_name = "Apple Silicon"
        else:
            # Check NVIDIA
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"], text=True
                )
                parts = out.strip().split(",")
                self.gpu_type = "nvidia"
                self.gpu_name = parts[0].strip()
                self.gpu_vram_gb = round(int(parts[1].strip()) / 1024)
            except Exception:
                pass

        # CPU cores
        try:
            if self.os == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True)
                self.cpu_cores = int(out.strip())
            else:
                import os
                self.cpu_cores = os.cpu_count() or 0
        except Exception:
            pass

    def to_dict(self):
        return {
            "os": self.os, "arch": self.arch, "ram_gb": self.ram_gb,
            "gpu_type": self.gpu_type, "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb, "cpu_cores": self.cpu_cores,
        }


# ── Model configurations by tier ─────────────────────────────────────────────

MODEL_TIERS = {
    "ultra": {
        "name": "Mode Ultra",
        "min_ram_gb": 32,
        "model": "hf.co/TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill-GGUF:Q8_0",
        "model_size_gb": 26,
        "lewm_enabled": True,
        "dflash_enabled": True,
        "turboquant_enabled": True,
        "context_size": 32768,
        "description": "Full power: MoE Q8 + LeWM + DFlash",
    },
    "balanced": {
        "name": "Mode Balanced",
        "min_ram_gb": 16,
        "model": "hf.co/TeichAI/gemma-4-26B-A4B-it-Claude-Opus-Distill-GGUF:IQ4_NL",
        "model_size_gb": 14.7,
        "lewm_enabled": True,
        "dflash_enabled": True,
        "turboquant_enabled": True,
        "context_size": 16384,
        "description": "Optimized: MoE IQ4_NL + LeWM + DFlash",
    },
    "eco": {
        "name": "Mode Éco",
        "min_ram_gb": 8,
        "model": "gemma4:e4b",
        "model_size_gb": 5,
        "lewm_enabled": False,
        "dflash_enabled": False,
        "turboquant_enabled": False,
        "context_size": 8192,
        "description": "Lightweight: Gemma 4 E4B for 8GB machines",
    },
    "minimal": {
        "name": "Mode Minimal",
        "min_ram_gb": 4,
        "model": "gemma4:e2b",
        "model_size_gb": 2,
        "lewm_enabled": False,
        "dflash_enabled": False,
        "turboquant_enabled": False,
        "context_size": 4096,
        "description": "Minimal: Gemma 4 E2B for old machines",
    },
}


def select_best_tier(hw: HardwareProfile) -> dict:
    """Select the best model tier for this hardware."""
    ram = hw.ram_gb
    if hw.gpu_type == "apple_silicon":
        effective_vram = ram  # Unified memory = all RAM is VRAM
    else:
        effective_vram = hw.gpu_vram_gb if hw.gpu_vram_gb > 0 else ram

    for tier_name in ["ultra", "balanced", "eco", "minimal"]:
        tier = MODEL_TIERS[tier_name]
        if effective_vram >= tier["min_ram_gb"] * 1.15:  # 15% headroom
            return {**tier, "tier": tier_name}

    # Fallback
    return {**MODEL_TIERS["minimal"], "tier": "minimal"}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hw = HardwareProfile()
    print(f"Hardware detected:")
    for k, v in hw.to_dict().items():
        print(f"  {k}: {v}")

    tier = select_best_tier(hw)
    print(f"\nRecommended: {tier['name']}")
    print(f"  Model: {tier['model']}")
    print(f"  Size: {tier['model_size_gb']} GB")
    print(f"  LeWM: {'ON' if tier['lewm_enabled'] else 'OFF'}")
    print(f"  DFlash: {'ON' if tier['dflash_enabled'] else 'OFF'}")
    print(f"  Context: {tier['context_size']} tokens")
    print(f"  {tier['description']}")
