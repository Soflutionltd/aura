#!/usr/bin/env python3
"""
AURA CI Benchmark: runs the standard benchmark suite against a given model.
Usage: python3 benchmark_ci.py <model_name>
Outputs JSON to stdout with {per_task: {}, total: float}
"""

import json
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

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


def query(model: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 8192},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "aura"
    scores = {}
    for b in BENCHMARKS:
        try:
            resp = query(model, b["prompt"])
            length_ok = len(resp) >= b["min_length"]
            kw_hits = sum(1 for kw in b["keywords"] if kw.lower() in resp.lower())
            kw_score = kw_hits / len(b["keywords"])
            scores[b["name"]] = round((0.5 if length_ok else 0.0) + 0.5 * kw_score, 1) * 100
        except Exception as e:
            print(f"Benchmark '{b['name']}' failed: {e}", file=sys.stderr)
            scores[b["name"]] = 0.0
    total = round(sum(scores.values()) / max(len(scores), 1), 1)
    result = {"per_task": scores, "total": total}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
