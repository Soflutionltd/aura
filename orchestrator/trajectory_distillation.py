#!/usr/bin/env python3
"""
AURA Agent Trajectory Distillation
Converts successful agent trajectories into training data
for the federated learning pipeline.

Only successful trajectories are used (quality filter).
Applies DPO-style preference optimization:
  - Good trajectory = chosen response
  - Failed trajectory = rejected response

This is what makes AURA learn to ACT, not just TALK.
"""

import json
from pathlib import Path

TRAJECTORY_DIR = Path(__file__).resolve().parent.parent / "trajectories"
TRAINING_DIR = Path(__file__).resolve().parent.parent / "fine-tunes" / "agent-distill"


def load_trajectories() -> tuple[list[dict], list[dict]]:
    """Load successful and failed trajectories."""
    good, bad = [], []
    for f in TRAJECTORY_DIR.glob("traj_*.json"):
        data = json.loads(f.read_text())
        if data.get("success"):
            good.append(data)
        else:
            bad.append(data)
    return good, bad


def generate_dpo_pairs(good: list[dict], bad: list[dict]) -> list[dict]:
    """
    Generate DPO (Direct Preference Optimization) training pairs.
    Each pair: same prompt, chosen (good trajectory), rejected (bad trajectory).
    """
    pairs = []
    for g in good:
        # Find a failed trajectory with similar prompt (or use any)
        prompt = g["prompt"]
        matching_bad = [b for b in bad if len(b.get("steps", [])) > 0]

        chosen = format_trajectory_as_response(g)
        rejected = format_trajectory_as_response(matching_bad[0]) if matching_bad else None

        pair = {
            "prompt": prompt,
            "chosen": chosen,
            "tool_count_chosen": g.get("tool_count", 0),
            "elapsed_chosen": g.get("elapsed_seconds", 0),
        }
        if rejected:
            pair["rejected"] = rejected
        pairs.append(pair)

    return pairs


def format_trajectory_as_response(traj: dict) -> str:
    """Format a trajectory into a training response string."""
    parts = []
    for step in traj.get("steps", []):
        tool = step.get("tool", "unknown")
        args = json.dumps(step.get("args", {}))
        result = step.get("result", "")
        parts.append(f'Action: {{"tool": "{tool}", "args": {args}}}')
        parts.append(f"Result: {result}")
    if traj.get("final_answer"):
        parts.append(f"Final answer: {traj['final_answer'][:2000]}")
    return "\n".join(parts)


def generate_sft_data(good: list[dict]) -> list[dict]:
    """
    Generate SFT (Supervised Fine-Tuning) data from successful trajectories.
    Each entry: user prompt -> complete agent response with tool usage.
    """
    entries = []
    for traj in good:
        response = format_trajectory_as_response(traj)
        entries.append({
            "messages": [
                {"role": "user", "content": traj["prompt"]},
                {"role": "assistant", "content": response},
            ]
        })
    return entries


def run_distillation():
    """
    Full distillation pipeline:
    1. Load trajectories (good + bad)
    2. Generate DPO pairs (preference optimization)
    3. Generate SFT data (supervised fine-tuning)
    4. Save training files for federated learning
    """
    good, bad = load_trajectories()
    print(f"[DISTILL] Loaded {len(good)} successful + {len(bad)} failed trajectories")

    if not good:
        print("[DISTILL] No successful trajectories yet. Skipping.")
        return

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # DPO pairs
    dpo_pairs = generate_dpo_pairs(good, bad)
    dpo_path = TRAINING_DIR / "dpo_pairs.json"
    dpo_path.write_text(json.dumps(dpo_pairs, indent=2))
    print(f"[DISTILL] Generated {len(dpo_pairs)} DPO pairs -> {dpo_path.name}")

    # SFT data
    sft_data = generate_sft_data(good)
    sft_path = TRAINING_DIR / "sft_agent.jsonl"
    with open(sft_path, "w") as f:
        for entry in sft_data:
            f.write(json.dumps(entry) + "\n")
    print(f"[DISTILL] Generated {len(sft_data)} SFT entries -> {sft_path.name}")

    # Stats
    avg_tools = sum(t.get("tool_count", 0) for t in good) / max(len(good), 1)
    avg_time = sum(t.get("elapsed_seconds", 0) for t in good) / max(len(good), 1)
    print(f"[DISTILL] Avg tools per success: {avg_tools:.1f}")
    print(f"[DISTILL] Avg time per success: {avg_time:.1f}s")
    print(f"[DISTILL] Success rate: {len(good)}/{len(good)+len(bad)} "
          f"({len(good)*100/max(len(good)+len(bad),1):.0f}%)")


if __name__ == "__main__":
    run_distillation()
