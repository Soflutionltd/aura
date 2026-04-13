#!/usr/bin/env python3
"""
AURA MLX Acceleration Layer
Integrates DFlash speculative decoding + TurboQuant KV cache compression
for maximum inference speed on Apple Silicon.

DFlash: 3-5x speedup via block speculative decoding
TurboQuant: 4.6x KV cache compression for longer contexts

Combined: fast inference + long context on consumer Macs.
"""

import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_mlx_installed() -> bool:
    """Check if MLX and required packages are installed."""
    try:
        import mlx
        return True
    except ImportError:
        return False


def check_dflash_installed() -> bool:
    """Check if dflash-mlx is installed."""
    try:
        from dflash_mlx import DFlashGenerator
        return True
    except ImportError:
        return False


def install_acceleration():
    """Install MLX + DFlash + TurboQuant for Apple Silicon."""
    print("[MLX] Installing acceleration stack for Apple Silicon...")

    packages = [
        "mlx",
        "mlx-lm",
        "mlx-vlm",
    ]

    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", pkg],
            capture_output=True,
        )

    # Install dflash-mlx from source
    dflash_dir = REPO_ROOT / "vendor" / "dflash-mlx"
    if not dflash_dir.exists():
        print("  Cloning dflash-mlx...")
        subprocess.run(
            ["git", "clone", "https://github.com/Aryagm/dflash-mlx.git",
             str(dflash_dir)],
            capture_output=True,
        )
    print("  Installing dflash-mlx...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(dflash_dir)],
        capture_output=True,
    )

    print("[MLX] Acceleration stack installed.")


def start_accelerated_server(
    model: str = "Qwen/Qwen3.5-35B-A3B",
    port: int = 11435,
    use_dflash: bool = True,
    use_turboquant: bool = True,
    context_size: int = 32768,
):
    """
    Start an accelerated MLX inference server for AURA.
    OpenAI-compatible API on localhost.

    With DFlash: 3-5x faster token generation
    With TurboQuant: 4.6x less memory for KV cache
    """
    print(f"[MLX] Starting accelerated server on port {port}")
    print(f"  Model: {model}")
    print(f"  DFlash: {'ON' if use_dflash else 'OFF'}")
    print(f"  TurboQuant: {'ON' if use_turboquant else 'OFF'}")
    print(f"  Context: {context_size} tokens")

    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", model,
        "--port", str(port),
    ]

    # Add TurboQuant KV cache compression
    if use_turboquant:
        cmd.extend([
            "--kv-bits", "3.5",
            "--kv-quant-scheme", "turboquant",
        ])

    # For DFlash, we use the dflash_mlx library directly
    # It wraps the model with speculative decoding
    if use_dflash and check_dflash_installed():
        print("  Using DFlash speculative decoding...")
        from dflash_mlx import DFlashGenerator

        # DFlash needs a draft model (smaller, faster)
        # For Qwen3.5-35B, we use a smaller Qwen as draft
        generator = DFlashGenerator(
            target_model=model,
            draft_model="Qwen/Qwen3-4B",
        )
        print(f"  DFlash ready: {model} + Qwen3-4B draft")
        return generator

    # Fallback: standard MLX server without DFlash
    print("  Starting standard MLX server (no DFlash)...")
    process = subprocess.Popen(cmd)
    print(f"  Server running on http://localhost:{port}")
    return process


def get_system_info() -> dict:
    """Get Apple Silicon hardware info for optimization."""
    import platform
    info = {
        "chip": platform.processor(),
        "os": platform.platform(),
    }

    # Get RAM
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        )
        info["ram_gb"] = round(int(out.strip()) / (1024**3))
    except Exception:
        info["ram_gb"] = 0

    # Get GPU cores
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True
        )
        info["perf_cores"] = int(out.strip())
    except Exception:
        info["perf_cores"] = 0

    return info


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AURA MLX Acceleration")
    parser.add_argument("command", choices=["install", "serve", "info", "bench"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--no-dflash", action="store_true")
    parser.add_argument("--no-turboquant", action="store_true")
    args = parser.parse_args()

    if args.command == "install":
        install_acceleration()

    elif args.command == "serve":
        start_accelerated_server(
            model=args.model,
            port=args.port,
            use_dflash=not args.no_dflash,
            use_turboquant=not args.no_turboquant,
        )

    elif args.command == "info":
        info = get_system_info()
        print(f"Chip: {info['chip']}")
        print(f"RAM: {info['ram_gb']} GB")
        print(f"Performance cores: {info['perf_cores']}")
        if info["ram_gb"] >= 96:
            print("Recommended: Qwen3.5-122B-A10B with DFlash")
        elif info["ram_gb"] >= 32:
            print("Recommended: Qwen3.5-35B-A3B with DFlash")
        elif info["ram_gb"] >= 16:
            print("Recommended: Qwen3.5-9B with DFlash")
        else:
            print("Recommended: Qwen3.5-2B")

    elif args.command == "bench":
        print("Running acceleration benchmark...")
        info = get_system_info()
        print(f"Hardware: {info['chip']} / {info['ram_gb']} GB")
        if check_dflash_installed():
            from dflash_mlx import DFlashGenerator
            gen = DFlashGenerator(target_model=args.model)
            result = gen.generate(
                "Write a quicksort in Python.",
                max_new_tokens=128,
            )
            print(f"Output: {len(result.text)} chars")
            print(f"Speed: {result.tokens_per_second:.1f} tok/s")
        else:
            print("DFlash not installed. Run: python3 mlx_acceleration.py install")
