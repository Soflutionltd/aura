#!/usr/bin/env python3
"""
AURA x NVIDIA NIM: connect AURA to MiniMax M2.7 via NVIDIA's free API.
This allows AURA to use the most powerful open source model for validation
and benchmarking, at zero cost, while contributors run lighter models locally.

Setup:
  1. Go to https://build.nvidia.com
  2. Create a free account
  3. Navigate to MiniMax M2.7 model page
  4. Click "Get API Key" then "Generate Key"
  5. Export: export NVIDIA_API_KEY="nvapi-your-key-here"
  6. Run: python3 scripts/nvidia_m27.py
"""

import json
import os
import sys
import urllib.request

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "minimaxai/minimax-m2.7"


def get_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        print("ERROR: NVIDIA_API_KEY not set.")
        print("Get your free key at https://build.nvidia.com")
        sys.exit(1)
    return key


def query_m27(prompt: str, system: str = "", temperature: float = 1.0) -> str:
    """Query MiniMax M2.7 via NVIDIA NIM API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": 4096,
    }).encode()

    req = urllib.request.Request(
        NVIDIA_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_api_key()}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


AURA_SYSTEM = """You are AURA (Autonomous Unified Reasoning Architecture), the world's first
community-driven, self-improving AI. You analyze problems deeply, write precise code,
and continuously improve yourself through self-reflection and benchmarking.
Powered by MiniMax M2.7 via NVIDIA NIM. Zero cost. Maximum power."""


def main():
    print("=" * 60)
    print("AURA x MiniMax M2.7 (via NVIDIA NIM, free API)")
    print("=" * 60)
    print()

    # Quick test
    print("Testing connection...")
    response = query_m27(
        "Write a Rust function that reverses a string in-place. Be concise.",
        system=AURA_SYSTEM,
    )
    print(f"Response:\n{response}")
    print()

    # Interactive mode
    print("AURA is ready. Type your prompts (Ctrl+C to quit).")
    print()
    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            response = query_m27(prompt, system=AURA_SYSTEM)
            print(f"\nAURA: {response}\n")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
