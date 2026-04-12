#!/usr/bin/env python3
"""Soflution LLM Self-Improvement Agent - zero tokens, zero cost."""

import json, urllib.request, platform, datetime
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "soflution-llm"
REPO_ROOT = Path(__file__).parent.parent

def query(prompt):
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["message"]["content"]

def main():
    print("🤖 Soflution LLM Self-Improvement Agent")
    print(f"Model: {MODEL} | Hardware: {platform.machine()}")
    
    tests = [
        ("code", "Write a Rust function that reverses a string in-place."),
        ("reasoning", "What is the best algorithm to find primes up to N and why?"),
    ]
    
    results = {}
    for name, prompt in tests:
        print(f"Testing {name}...")
        response = query(prompt)
        results[name] = {"prompt": prompt, "length": len(response), "preview": response[:200]}
    
    modelfile = (REPO_ROOT / "Modelfile").read_text()
    analysis = query(f"Analyze your own Modelfile and suggest improvements:\n\n{modelfile}")
    
    out = {"timestamp": datetime.datetime.now().isoformat(), "hardware": platform.machine(), "results": results, "self_analysis": analysis}
    (REPO_ROOT / "benchmarks" / f"benchmark_{platform.machine()}.json").write_text(json.dumps(out, indent=2))
    print("✅ Done! Submit a PR with your benchmark results.")

if __name__ == "__main__":
    main()
