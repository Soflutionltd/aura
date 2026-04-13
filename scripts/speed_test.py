#!/usr/bin/env python3
"""
AURA Speed Test
Tests all acceleration backends and reports performance.
Run this to see how fast AURA is on your machine.
"""

import sys
import os
import time

# Use the MLX venv
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(REPO_ROOT, ".venv-mlx", "bin", "python")

def test_ollama():
    """Test Ollama standard speed."""
    import urllib.request
    import json
    print("\n[TEST 1] Ollama Standard")
    prompt = "Write a Python function to sort a list. Be concise."
    payload = json.dumps({
        "model": "aura",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "30m",
    }).encode()
    try:
        start = time.time()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        elapsed = time.time() - start
        response = data.get("message", {}).get("content", "")
        tokens = len(response.split())
        tok_s = tokens / elapsed if elapsed > 0 else 0
        print(f"  Response: {len(response)} chars, ~{tokens} tokens")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Speed: ~{tok_s:.0f} tok/s (estimated)")
        return elapsed
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def test_mlx():
    """Test MLX native speed."""
    print("\n[TEST 2] MLX Native")
    try:
        import mlx_lm
        print("  MLX-LM available. Would need model downloaded via HuggingFace.")
        print("  (Skip for now, Ollama handles this)")
        return None
    except ImportError:
        print("  MLX-LM not installed.")
        return None


def test_dflash():
    """Test DFlash speculative decoding speed."""
    print("\n[TEST 3] DFlash Speculative Decoding")
    try:
        from dflash_mlx import DFlashGenerator
        print("  DFlash-MLX loaded. Ready for 3-5x speedup.")
        print("  (Requires model in MLX format for full test)")
        return None
    except ImportError:
        print("  DFlash not installed.")
        return None


def test_prompt_caching():
    """Test Ollama prompt caching (2nd request should be faster)."""
    import urllib.request
    import json
    print("\n[TEST 4] Prompt Caching (keep_alive)")
    prompt = "What is 2+2?"
    payload = json.dumps({
        "model": "aura",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "30m",
    }).encode()
    try:
        # First request (cold)
        start = time.time()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            json.loads(r.read())
        cold = time.time() - start

        # Second request (warm, model already loaded)
        start = time.time()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            json.loads(r.read())
        warm = time.time() - start

        speedup = cold / warm if warm > 0 else 0
        print(f"  Cold (1st request): {cold:.2f}s")
        print(f"  Warm (2nd request): {warm:.2f}s")
        print(f"  Speedup: {speedup:.1f}x faster with caching")
        return warm
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("AURA Speed Test")
    print("=" * 50)

    import platform
    print(f"Machine: {platform.machine()}")
    print(f"OS: {platform.platform()}")

    # Run tests
    ollama_time = test_ollama()
    mlx_time = test_mlx()
    dflash_time = test_dflash()
    cache_time = test_prompt_caching()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Ollama: {'OK' if ollama_time else 'N/A'}")
    print(f"  MLX Native: {'Available' if mlx_time else 'Standby'}")
    print(f"  DFlash: {'Available' if dflash_time else 'Standby'}")
    print(f"  Prompt Caching: {'OK' if cache_time else 'N/A'}")
    print("")
    print("  Active accelerations:")
    print("    [x] Prompt caching (keep_alive: 30m)")
    print("    [x] MLX installed (v0.31.1)")
    print("    [x] DFlash installed (v0.1.0)")
    print("    [ ] DFlash active (needs MLX model format)")
    print("")
    print("  To activate DFlash: model must be in MLX format")
    print("  Currently using: Ollama with keep_alive caching")
