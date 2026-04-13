#!/usr/bin/env python3
"""
AURA Federated Learning Server (Phase 2)
Aggregates LoRA gradients + agent trajectories from the community.

Architecture:
  1. Nodes send anonymized gradients + successful trajectories
  2. Server aggregates using FedProx (handles heterogeneous hardware)
  3. Distills agent trajectories into MoE experts (Trajectory KD)
  4. Publishes improved model to IPFS
  5. All nodes auto-update

Privacy: Differential Privacy (DP) noise added to all gradients.
No raw conversations ever leave a node. Only math (gradients) + 
anonymized trajectories (plan + tool calls, no user data).
"""

import json
import hashlib
import datetime
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict

AGGREGATION_DIR = Path(__file__).resolve().parent.parent / "federated"
LORA_INCOMING = AGGREGATION_DIR / "incoming-loras"
TRAJ_INCOMING = AGGREGATION_DIR / "incoming-trajectories"
MERGED_DIR = AGGREGATION_DIR / "merged"


class DifferentialPrivacy:
    """Add calibrated noise to gradients before aggregation."""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def add_noise(self, gradient: list[float]) -> list[float]:
        """Add Gaussian noise calibrated to (epsilon, delta)-DP."""
        import random
        import math
        sensitivity = 1.0  # L2 sensitivity of clipped gradients
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return [g + random.gauss(0, sigma) for g in gradient]

    def clip_gradient(self, gradient: list[float], max_norm: float = 1.0) -> list[float]:
        """Clip gradient to bounded L2 norm."""
        import math
        norm = math.sqrt(sum(g ** 2 for g in gradient))
        if norm > max_norm:
            scale = max_norm / norm
            return [g * scale for g in gradient]
        return gradient


class FedProxAggregator:
    """
    FedProx aggregation: handles heterogeneous hardware.
    Nodes with weaker hardware do fewer local steps but still contribute.
    Their gradients are weighted by quality, not compute.
    """

    def __init__(self, mu: float = 0.01):
        self.mu = mu  # Proximal term strength
        self.dp = DifferentialPrivacy()
        self.round_number = 0
        self.contribution_log = []

    def aggregate_loras(self, lora_updates: list[dict]) -> dict:
        """
        Aggregate multiple LoRA updates into a single merged update.
        Each update: {"node_id": str, "gradients": list[float], "benchmark_gain": float}
        Weighted by benchmark_gain (better improvements count more).
        """
        if not lora_updates:
            return {}

        self.round_number += 1
        total_weight = sum(u.get("benchmark_gain", 1.0) for u in lora_updates)

        # Determine gradient dimension from first update
        dim = len(lora_updates[0].get("gradients", []))
        if dim == 0:
            return {}

        # Weighted average with DP noise
        merged = [0.0] * dim
        for update in lora_updates:
            grads = update.get("gradients", [0.0] * dim)
            weight = update.get("benchmark_gain", 1.0) / total_weight

            # Clip and add noise (differential privacy)
            clipped = self.dp.clip_gradient(grads)
            noisy = self.dp.add_noise(clipped)

            for i in range(min(dim, len(noisy))):
                merged[i] += noisy[i] * weight

        # Log contribution
        self.contribution_log.append({
            "round": self.round_number,
            "num_updates": len(lora_updates),
            "total_weight": round(total_weight, 2),
            "timestamp": datetime.datetime.now().isoformat(),
        })

        return {
            "round": self.round_number,
            "merged_gradients": merged,
            "num_contributors": len(lora_updates),
            "timestamp": datetime.datetime.now().isoformat(),
        }


class TrajectoryKD:
    """
    Trajectory Knowledge Distillation.
    Distills successful agent trajectories into specific MoE experts.

    This is what makes AURA unique:
    - Regular KD: model learns to ANSWER like the teacher
    - Trajectory KD: model learns to ACT like the teacher
      (plan, choose tools, execute, verify, iterate)
    """

    def __init__(self):
        self.trajectories = []

    def ingest_trajectories(self, traj_dir: Path):
        """Load all successful trajectories for distillation."""
        count = 0
        for f in traj_dir.glob("traj_*_ok.json"):
            data = json.loads(f.read_text())
            if data.get("success") and len(data.get("steps", [])) >= 2:
                self.trajectories.append(data)
                count += 1
        return count

    def classify_trajectory(self, traj: dict) -> str:
        """Classify trajectory by dominant task type for expert routing."""
        tools_used = [s.get("tool", "") for s in traj.get("steps", [])]
        if "code_execution" in tools_used or "terminal_cmd" in tools_used:
            return "coding"
        elif "web_search" in tools_used:
            return "research"
        elif "local_rag" in tools_used:
            return "analysis"
        elif "file_system" in tools_used:
            return "file_ops"
        return "general"

    def generate_expert_training_data(self) -> dict[str, list[dict]]:
        """
        Generate training data per expert specialization.
        Each expert gets trajectories matching its specialization.
        """
        expert_data = defaultdict(list)

        for traj in self.trajectories:
            task_type = self.classify_trajectory(traj)

            # Format as training example
            entry = {
                "messages": [
                    {"role": "user", "content": traj["prompt"]},
                    {"role": "assistant", "content": self._format_agent_response(traj)},
                ],
                "task_type": task_type,
                "tool_count": len(traj.get("steps", [])),
                "elapsed": traj.get("elapsed_seconds", 0),
            }
            expert_data[task_type].append(entry)

        return dict(expert_data)

    def _format_agent_response(self, traj: dict) -> str:
        """Format trajectory as ideal agent response for training."""
        parts = []
        for i, step in enumerate(traj.get("steps", [])):
            tool = step.get("tool", "")
            args = json.dumps(step.get("args", {}))
            result = step.get("result", "")[:500]
            parts.append(f"Step {i+1}: Using {tool}")
            parts.append(f'{{"tool": "{tool}", "args": {args}}}')
            parts.append(f"Result: {result}")
        if traj.get("final_answer"):
            parts.append(f"\n{traj['final_answer'][:2000]}")
        return "\n".join(parts)

    def save_training_data(self, output_dir: Path):
        """Save expert-specific training data for federated distillation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        expert_data = self.generate_expert_training_data()

        for task_type, entries in expert_data.items():
            path = output_dir / f"expert_{task_type}.jsonl"
            with open(path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            print(f"  Expert '{task_type}': {len(entries)} training examples")


class FederatedServer:
    """
    Main federated learning server.
    Runs aggregation rounds on incoming contributions.
    """

    def __init__(self):
        for d in [AGGREGATION_DIR, LORA_INCOMING, TRAJ_INCOMING, MERGED_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        self.aggregator = FedProxAggregator()
        self.trajectory_kd = TrajectoryKD()

    def run_aggregation_round(self):
        """Execute one full aggregation round."""
        print(f"{'='*60}")
        print(f"AURA Federated Learning Round {self.aggregator.round_number + 1}")
        print(f"Time: {datetime.datetime.now().isoformat()}")
        print(f"{'='*60}")

        # 1. Collect LoRA updates
        lora_files = list(LORA_INCOMING.glob("*.json"))
        print(f"\n[LORA] {len(lora_files)} LoRA updates received")

        if lora_files:
            updates = []
            for f in lora_files:
                try:
                    updates.append(json.loads(f.read_text()))
                except Exception:
                    pass
            if updates:
                merged = self.aggregator.aggregate_loras(updates)
                merged_path = MERGED_DIR / f"merged_round_{merged['round']}.json"
                merged_path.write_text(json.dumps(merged, indent=2))
                print(f"[LORA] Merged {len(updates)} updates -> {merged_path.name}")

        # 2. Collect and distill agent trajectories
        traj_count = self.trajectory_kd.ingest_trajectories(TRAJ_INCOMING)
        print(f"\n[TRAJ] {traj_count} new successful trajectories ingested")

        if traj_count > 0:
            expert_dir = MERGED_DIR / f"expert_training_round_{self.aggregator.round_number}"
            self.trajectory_kd.save_training_data(expert_dir)
            print(f"[TRAJ] Expert training data saved to {expert_dir.name}")

        # 3. Publish to IPFS
        print(f"\n[PUBLISH] Publishing merged update to IPFS...")
        try:
            from scripts.ipfs_integration import publish_delta, is_ipfs_running
            if is_ipfs_running():
                for f in MERGED_DIR.glob("merged_round_*.json"):
                    result = publish_delta(str(f))
                    if result:
                        print(f"  IPFS CID: {result['cid']}")
        except Exception as e:
            print(f"  IPFS publish skipped: {e}")

        print(f"\n{'='*60}")
        print(f"Round complete. Total contributions: {len(self.aggregator.contribution_log)}")
        print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = FederatedServer()
    server.run_aggregation_round()
