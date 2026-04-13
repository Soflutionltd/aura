#!/usr/bin/env python3
"""
AURA World-Enhanced MoE Convergence Engine (Phase 3)

The ultimate goal: the MoE absorbs the capabilities of the LeWorldModel
through federated distillation. After 2-3 months of community contributions:

Month 1: MoE + LeWM run side by side (hybrid)
Month 2: MoE starts predicting what LeWM would simulate (partial absorption)
Month 3: MoE fully anticipates outcomes without needing LeWM

Result: a single model that thinks before acting, no extra module needed.
This is what nobody else does. AURA becomes the first self-evolving
World-Enhanced MoE.

Technical approach:
  1. Record LeWM simulation outputs alongside MoE generations
  2. Train MoE to predict LeWM outputs as an auxiliary task
  3. Gradually reduce LeWM usage as MoE accuracy improves
  4. Eventually disable LeWM entirely (optional, user choice)
"""

import json
import datetime
from pathlib import Path
from collections import defaultdict

CONVERGENCE_DIR = Path(__file__).resolve().parent.parent / "world-convergence"


class SimulationPair:
    """A paired record: what LeWM predicted vs what actually happened."""

    def __init__(self, plan: dict, lewm_prediction: dict,
                 actual_result: dict, moe_would_predict: dict = None):
        self.plan = plan
        self.lewm_prediction = lewm_prediction
        self.actual_result = actual_result
        self.moe_would_predict = moe_would_predict
        self.lewm_was_correct = self._check_accuracy(lewm_prediction, actual_result)

    def _check_accuracy(self, prediction: dict, actual: dict) -> bool:
        """Did LeWM correctly predict the outcome?"""
        pred_success = prediction.get("success_probability", 0) > 0.5
        actual_success = actual.get("success", False)
        return pred_success == actual_success

    def to_training_example(self) -> dict:
        """Convert to training example for MoE absorption."""
        return {
            "input": json.dumps(self.plan),
            "lewm_output": json.dumps(self.lewm_prediction),
            "actual_output": json.dumps(self.actual_result),
            "lewm_correct": self.lewm_was_correct,
        }


class ConvergenceTracker:
    """
    Tracks how well the MoE is learning to predict like LeWM.
    When accuracy is high enough, LeWM can be disabled.
    """

    def __init__(self):
        CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
        self.stats_path = CONVERGENCE_DIR / "convergence_stats.json"
        self.pairs_dir = CONVERGENCE_DIR / "simulation_pairs"
        self.pairs_dir.mkdir(exist_ok=True)
        self.stats = self._load_stats()

    def _load_stats(self) -> dict:
        if self.stats_path.exists():
            return json.loads(self.stats_path.read_text())
        return {
            "total_pairs": 0,
            "lewm_correct": 0,
            "moe_correct": 0,
            "convergence_score": 0.0,
            "phase": "hybrid",  # hybrid -> partial -> absorbed
            "history": [],
        }

    def _save_stats(self):
        self.stats_path.write_text(json.dumps(self.stats, indent=2))

    def record_pair(self, pair: SimulationPair):
        """Record a simulation pair for convergence tracking."""
        self.stats["total_pairs"] += 1
        if pair.lewm_was_correct:
            self.stats["lewm_correct"] += 1

        # Save pair for training
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.pairs_dir / f"pair_{ts}.json"
        path.write_text(json.dumps(pair.to_training_example(), indent=2))

        # Update convergence score every 100 pairs
        if self.stats["total_pairs"] % 100 == 0:
            self._update_convergence()

        self._save_stats()

    def _update_convergence(self):
        """Calculate how close the MoE is to absorbing LeWM."""
        total = self.stats["total_pairs"]
        if total == 0:
            return

        lewm_accuracy = self.stats["lewm_correct"] / total
        moe_accuracy = self.stats["moe_correct"] / max(total, 1)

        # Convergence = how well MoE predicts compared to LeWM
        if lewm_accuracy > 0:
            convergence = moe_accuracy / lewm_accuracy
        else:
            convergence = 0.0

        self.stats["convergence_score"] = round(min(convergence, 1.0), 3)

        # Determine phase
        if convergence >= 0.95:
            self.stats["phase"] = "absorbed"
        elif convergence >= 0.7:
            self.stats["phase"] = "partial"
        else:
            self.stats["phase"] = "hybrid"

        self.stats["history"].append({
            "pairs": total,
            "lewm_acc": round(lewm_accuracy, 3),
            "moe_acc": round(moe_accuracy, 3),
            "convergence": self.stats["convergence_score"],
            "phase": self.stats["phase"],
            "timestamp": datetime.datetime.now().isoformat(),
        })

    def should_use_lewm(self) -> bool:
        """Should we still use LeWM or has MoE absorbed it?"""
        phase = self.stats.get("phase", "hybrid")
        if phase == "absorbed":
            return False  # MoE is good enough on its own
        return True

    def get_status(self) -> dict:
        """Get convergence status for display."""
        total = self.stats["total_pairs"]
        return {
            "phase": self.stats["phase"],
            "convergence": self.stats["convergence_score"],
            "total_pairs": total,
            "lewm_accuracy": round(self.stats["lewm_correct"] / max(total, 1), 3),
            "moe_accuracy": round(self.stats["moe_correct"] / max(total, 1), 3),
            "lewm_active": self.should_use_lewm(),
        }


class WorldEnhancedMoETrainer:
    """
    Generates training data to teach the MoE to predict like LeWM.
    This is the absorption process.
    """

    def __init__(self):
        self.training_dir = CONVERGENCE_DIR / "moe_training"
        self.training_dir.mkdir(parents=True, exist_ok=True)

    def generate_absorption_data(self, pairs_dir: Path) -> list[dict]:
        """
        Convert simulation pairs into MoE training data.
        The MoE learns: given a plan, predict success/risk/steps
        (what LeWM does, but inside the MoE itself).
        """
        training_data = []

        for f in pairs_dir.glob("pair_*.json"):
            try:
                pair = json.loads(f.read_text())
                # Only train on pairs where LeWM was correct
                if pair.get("lewm_correct"):
                    training_data.append({
                        "messages": [
                            {"role": "system", "content": (
                                "You are a planning simulator. Given an action plan, "
                                "predict the outcome: success probability (0-1), "
                                "risk level (low/medium/high), and estimated steps. "
                                "Respond with JSON only."
                            )},
                            {"role": "user", "content": pair["input"]},
                            {"role": "assistant", "content": pair["lewm_output"]},
                        ]
                    })
            except Exception:
                pass

        return training_data

    def save_training_batch(self, batch_name: str = "latest"):
        """Save absorption training data for LoRA fine-tuning."""
        tracker = ConvergenceTracker()
        data = self.generate_absorption_data(tracker.pairs_dir)

        if not data:
            print("[ABSORB] No valid pairs yet. Need more agent runs.")
            return

        path = self.training_dir / f"absorb_{batch_name}.jsonl"
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"[ABSORB] Saved {len(data)} training examples -> {path.name}")
        print(f"[ABSORB] Status: {tracker.get_status()}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    tracker = ConvergenceTracker()
    status = tracker.get_status()

    print("AURA World-Enhanced MoE Status")
    print(f"  Phase: {status['phase']}")
    print(f"  Convergence: {status['convergence']*100:.1f}%")
    print(f"  Total pairs: {status['total_pairs']}")
    print(f"  LeWM accuracy: {status['lewm_accuracy']*100:.1f}%")
    print(f"  MoE accuracy: {status['moe_accuracy']*100:.1f}%")
    print(f"  LeWM active: {status['lewm_active']}")

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        trainer = WorldEnhancedMoETrainer()
        trainer.save_training_batch()
